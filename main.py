import numpy as np
# from sklearn import datasets
def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))  

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()
  
class Neuron:
  def __init__(self, inputSize):
    self.inputSize = inputSize
    self.weights = []
    self.bias = np.random.normal()


    for i in range (0, inputSize):
      self.weights.append(np.random.normal())

  # Return dot product of inputs and weights
  def feedforward(self, input):
    self.output = np.dot(input, self.weights) + self.bias
    return sigmoid(self.output)

  # Calculate derivative of each weight
  def backProp(self, loss, output):
    for i in range(0, len(self.weights)):
      print("poopy")
    # Derivative of loss with respect to the output of the network
    loss_d_output = -2(1 - output)
    
    #output_d_
    
class Layer:
  def __init__(self, size, inputLayer, parent):
    self.neurons = []
    self.size = size
    self.inputLayer = inputLayer
    self.input = []
    self.parent = parent
    self.output = []

    for i in range(self.size):
      self.neurons.append(Neuron(self.parent.layerSizes[self.inputLayer]))

  def getInput(self):
    self.input.clear()
    self.input = self.parent.layerOutputs[self.inputLayer]

  def feedforward(self):
    self.getInput()
    self.output.clear()
    for i in range (0, len(self.neurons)):
      self.output.append(self.neurons[i].feedforward(self.input))
    print (self.output)
    return self.output

# Holds layers
class NeuralNetwork:
  def __init__(self, input, layerSizes):
    self.layers = [input]
    self.layerSizes = [len(input)] + layerSizes
    self.layerOutputs = [input]
    self.input = input

    for i in range (0, len(layerSizes)):
      self.layers.append(Layer(layerSizes[i], i, self))

  def feedforward(self):
    self.layerOutputs.clear()
    self.layerOutputs.append(self.input)
    for i in range(1, len(self.layers)):
      self.layerOutputs.append(self.layers[i].feedforward())

testNet = NeuralNetwork([2, 3], [2, 1])
testNet.feedforward()
print("output: ",testNet.layers[2].output)