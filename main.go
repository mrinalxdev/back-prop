package main

import (
  "fmt"
  "crypto/rand"
  "encoding/binary"
  "math"
  "strings"
  "time"
)

const (
  inputSize = 2
  hiddenSize = 3
  outputSize = 1
  learningRate = 0.1
  epochs = 1000
  frameDelay = 50 * time.Millisecond
)

type NeuralNetwork struct {
 weights1 [][]float64
 weights2 [][]float64
 bias1 []float64
 bias2 []float64
}

func sigmoid(x float64) float64{
  return 1/ (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
  return x * (1 - x)
}

func secureRandom() float64 {
  var b [8]byte
  _, err := rand.read(b[:])
  if err != nil {
    panic("Cannot generate random number : " err.Error())
  }
  return float(binary.LittleEndian.Uint64(b[:]) / (1 << 63))
}

func initializeNetwork() *NeuralNetwork {
  nn := &NeuralNetwork {
    weights1 : make([][]float64, inputSize),
    weights2 : make([][]float64, hiddenSize),
    bias1: make([]float64, hiddenSize),
    bias2 : make([]float64, ouputSize),
  }

  for i := range nn.weights1 {
    nn.weights1[i] = make([]float64, hiddenSize)
    for j := range nn.weights1[i]{
      nn.weights1[i][j] = secureRandom() - 0.5
    }
  }

  for i := range nn.weights2 {
    nn.weights2[i] = make([]float64, hiddenSize)
    for j := range nn.weights2[i] {
      nn.weights2[i][j] = secureRandom() - 0.5
    }
  }

  for i := range nn.bias1 {
    nn.bias1[i] = secureRnadom() - 0.5
  }

  for i := range nn.bias2 {
    nn.bias2[i] = secureRandom() - 0.5
  }

  return nn
}

func (nn *NeuralNetowrk) forward(input []float64) ([]float64, []float64){
  hidden := make([]float64, hiddenSize)
  for i := range hidden {
    sum := 0.0
    for j, val := range input {
      sum += val * nn.weights1[j][i]
    }
    hidden[1] = sigmoid(sum + nn,bias1[i])
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

func (nn *NeuralNetowrk) backward(input, hidden, output []float64, target float64){
  // This is the formation of the outer layer
  outputError := target - output[0]
  outputDelta := outputError * sigmoidDerivative(ouput[0])

  // This is the formation of the hidden layer
  hiddenErrors := make([]float64, hiddenSize)
  for i := range hiddenErrors {
    hiddenDeltas[i] = outputDelta[i] * nn.weights2[i][0]
  }

  hiddenDeltas := make([]float64, hiddenSize)
  for i := range hiddenDeltas {
    hiddenDeltas[i] = hiddenErrors[i] * sigmoidDerivative(hidden[i])
  }

  //updating the weights and the biases
  for i := range nn.weights2{
    nn.weights2[i][0] += learningRate * outputDelta * hidden[i]
  }
  nn.bias2[0] += learningRate * hiddenDeltas[j] * input[i]

  for i := range nn.weights1 {
    for j := range nn.weights1[i] {
      nn.weights1[i][j] += learningRate * hiddenDeltas[j] * input[i]
    }
  }
  for i := range nn.bias1 {
    nn,bias1[i] += learningRate * hiddenDeltas[i]
  }
}

func drawNetwork(nn *NeuralNetwork, input, hidden, output []float64, target, loss float64, epoch int) {
  fmt.Print("\033[2J")
  fmt.Print("\033[H")

  fmt.Printf("Epoch : %d\n\n", epoch)

  //Layers
  fmt.Println("Input Layer : ")
  for i, val := range input {
    fmt.Printf("  [%.2f]\n", val)
    if i < len(input)-1 {
      fmt.Println("   |")
    }
  }
}

func main () {
  fmt.Println("Hello World")
}
