package main

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	inputSize    = 3
	hiddenSize   = 4
	outputSize   = 2
	learningRate = 0.1
	epochs       = 1000
	frameDelay   = 100 * time.Millisecond
)

type NeuralNetwork struct {
	weights1 [][]float64
	weights2 [][]float64
	bias1    []float64
	bias2    []float64
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func secureRandom() float64 {
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		panic("cannot generate random number: " + err.Error())
	}
	return float64(binary.LittleEndian.Uint64(b[:])) / (1 << 63)
}

func initializeNetwork() *NeuralNetwork {
	nn := &NeuralNetwork{
		weights1: make([][]float64, inputSize),
		weights2: make([][]float64, hiddenSize),
		bias1:    make([]float64, hiddenSize),
		bias2:    make([]float64, outputSize),
	}

	for i := range nn.weights1 {
		nn.weights1[i] = make([]float64, hiddenSize)
		for j := range nn.weights1[i] {
			nn.weights1[i][j] = secureRandom() - 0.5
		}
	}

	for i := range nn.weights2 {
		nn.weights2[i] = make([]float64, outputSize)
		for j := range nn.weights2[i] {
			nn.weights2[i][j] = secureRandom() - 0.5
		}
	}

	for i := range nn.bias1 {
		nn.bias1[i] = secureRandom() - 0.5
	}

	for i := range nn.bias2 {
		nn.bias2[i] = secureRandom() - 0.5
	}

	return nn
}

func (nn *NeuralNetwork) forward(input []float64) ([]float64, []float64) {
	hidden := make([]float64, hiddenSize)
	for i := range hidden {
		sum := 0.0
		for j, val := range input {
			sum += val * nn.weights1[j][i]
		}
		hidden[i] = sigmoid(sum + nn.bias1[i])
	}

	output := make([]float64, outputSize)
	for i := range output {
		sum := 0.0
		for j, val := range hidden {
			sum += val * nn.weights2[j][i]
		}
		output[i] = sigmoid(sum + nn.bias2[i])
	}

	return hidden, output
}

func (nn *NeuralNetwork) backward(input, hidden, output, target []float64) {
	// Output layer
	outputErrors := make([]float64, outputSize)
	outputDeltas := make([]float64, outputSize)
	for i := range output {
		outputErrors[i] = target[i] - output[i]
		outputDeltas[i] = outputErrors[i] * sigmoidDerivative(output[i])
	}

	// Hidden layer
	hiddenErrors := make([]float64, hiddenSize)
	hiddenDeltas := make([]float64, hiddenSize)
	for i := range hidden {
		for j := range output {
			hiddenErrors[i] += outputDeltas[j] * nn.weights2[i][j]
		}
		hiddenDeltas[i] = hiddenErrors[i] * sigmoidDerivative(hidden[i])
	}

	// Update weights and biases
	for i := range nn.weights2 {
		for j := range output {
			nn.weights2[i][j] += learningRate * outputDeltas[j] * hidden[i]
		}
	}
	for i := range nn.bias2 {
		nn.bias2[i] += learningRate * outputDeltas[i]
	}

	for i := range nn.weights1 {
		for j := range hidden {
			nn.weights1[i][j] += learningRate * hiddenDeltas[j] * input[i]
		}
	}
	for i := range nn.bias1 {
		nn.bias1[i] += learningRate * hiddenDeltas[i]
	}
}

func drawDetailedNetwork(nn *NeuralNetwork, input, hidden, output []float64, target []float64, loss float64, epoch int) {
	fmt.Print("\033[2J\033[H") // Clear screen and move cursor to top-left
	fmt.Printf("Epoch: %d | Loss: %.4f\n\n", epoch, loss)

	// Calculate max width for each layer
	inputWidth := 8  // "Input" + number
	hiddenWidth := 9 // "Hidden" + number
	outputWidth := 9 // "Output" + number

	// Function to create a neuron representation
	neuron := func(value float64, label string) string {
		return fmt.Sprintf("(%s%.2f)", label, value)
	}

	// Draw input layer
	fmt.Println("Input Layer:")
	for i, val := range input {
		fmt.Printf("%s%s\n", strings.Repeat(" ", inputWidth), neuron(val, "x"))
		if i < len(input)-1 {
			fmt.Printf("%s|\n", strings.Repeat(" ", inputWidth+1))
		}
	}

	// Draw connections to hidden layer
	fmt.Println(strings.Repeat("-", 40))
	for i := range input {
		for j := range hidden {
			weight := nn.weights1[i][j]
			if weight >= 0 {
				fmt.Print("/")
			} else {
				fmt.Print("\\")
			}
		}
		fmt.Println()
	}

	// Draw hidden layer
	fmt.Println("\nHidden Layer:")
	for i, val := range hidden {
		fmt.Printf("%s%s\n", strings.Repeat(" ", hiddenWidth), neuron(val, "h"))
		if i < len(hidden)-1 {
			fmt.Printf("%s|\n", strings.Repeat(" ", hiddenWidth+1))
		}
	}

	// Draw connections to output layer
	fmt.Println(strings.Repeat("-", 40))
	for i := range hidden {
		for j := range output {
			weight := nn.weights2[i][j]
			if weight >= 0 {
				fmt.Print("/")
			} else {
				fmt.Print("\\")
			}
		}
		fmt.Println()
	}

	// Draw output layer
	fmt.Println("\nOutput Layer:")
	for i, val := range output {
		fmt.Printf("%s%s -> Target: %.2f\n", strings.Repeat(" ", outputWidth), neuron(val, "y"), target[i])
		if i < len(output)-1 {
			fmt.Printf("%s|\n", strings.Repeat(" ", outputWidth+1))
		}
	}

	// Draw backpropagation
	fmt.Println("\nBackpropagation:")
	for i := range output {
		error := target[i] - output[i]
		fmt.Printf("%sError: %.4f\n", strings.Repeat(" ", outputWidth), error)
		if i < len(output)-1 {
			fmt.Printf("%s|\n", strings.Repeat(" ", outputWidth+1))
		}
	}
}

func main() {
	nn := initializeNetwork()

	// Training data (example problem with 3 inputs and 2 outputs)
	inputs := [][]float64{
		{0, 0, 1},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 1},
	}
	targets := [][]float64{
		{0, 1},
		{1, 1},
		{1, 0},
		{0, 0},
	}

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i, input := range inputs {
			hidden, output := nn.forward(input)
			loss := 0.0
			for j, target := range targets[i] {
				loss += math.Pow(target-output[j], 2)
			}
			loss /= float64(len(targets[i]))
			totalLoss += loss

			nn.backward(input, hidden, output, targets[i])

			drawDetailedNetwork(nn, input, hidden, output, targets[i], loss, epoch)
			time.Sleep(frameDelay)
		}

		averageLoss := totalLoss / float64(len(inputs))
		fmt.Printf("\nAverage Loss: %.4f\n", averageLoss)
		time.Sleep(frameDelay * 2)
	}
}