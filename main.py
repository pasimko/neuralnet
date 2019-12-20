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
  def __init__(self, input):
    self.input = input
    self.weights = []
    self.bias = np.random.normal()

    for i in range (0, len(input)):
      self.weights.append(np.random.normal())

  # Return dot product of inputs and weights
  def feedforward(self):
    self.output = np.dot(self.input, self.weights) + self.bias
    return sigmoid(self.output)

  # Calculate derivative of each weight
  def backProp(self, loss, output):
    # Derivative of loss with respect to the output of the network
    loss_d_output = -2(1 - output)
    

class Layer:
  def __init__(self, size, inputLayer, parent):
    self.neurons = [];
    self.size = size
    for i in range(size):
      self.neurons.append(Neuron(parent.layerOutputs[inputLayer]))
  def feedforward(self):
    for i in range (0, len(self.neurons)):
      self.neurons[i].feedforward()

class OurNeuralNetwork:
  def __init__(self, input, layerSizes):
    self.layers = [input]
    self.layerSizes = layerSizes
    self.layerOutputs = [input]
  
    for i in range (0, len(layerSizes)):
      self.layers.append(Layer(layerSizes[i], i, self))
  def feedforward(self, x):
    for i in range(0, len(self.layers)):
      self.layers[i].feedforward()

testNet = OurNeuralNetwork([-2, -1], [3, 1])
testNet.feedforward()