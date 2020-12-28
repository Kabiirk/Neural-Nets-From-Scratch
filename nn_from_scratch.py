import numpy as np
import pandas as pd
import matplotlib
import sys

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_output = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for neuron_input, weight in zip(inputs, neuron_weights):
        neuron_output += neuron_input*weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

#1st elem we pass is how the element is gonna be indexed
output = np.dot(weights, inputs) + biases
print(output)

"""# **Batches etc**"""

import numpy as np

#batches allow us to paralellize operations
#batches also help us in generalization

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] #3x4

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] #3x4

biases = [2, 3, 0.5]

#Layer 2
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, 0.33],
           [-0.44, 0.73, -0.13]] #3x4

biases2 = [-1, 2, -0.5]

#output = np.dot(weights, inputs) + biases
#gives issue, we need to change shape by transposition

#Layers
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

import numpy as np
#same code as above but better
np.random.seed(0)

X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)#4 = Features, 5 = No. of Neurons
layer2 = Layer_Dense(5,2) # L1 has 5 op so l2 shld have 5 ip

print("Layer 1")
layer1.forward(X)
print(layer1.output)
print("\n")
print("Layer 2")
layer2.forward(layer1.output)
print(layer2.output)

"""# **Activation Fucntions**

#### ReLU Function
"""

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    '''
    do the same thing as uncommented code below but better !!! LMAOOOOOO
    if(i>0):
        output.append(i)
    else:
        output.append(0)
    '''
    output.append(max(0, i))

print(output)

import numpy as np
import matplotlib.pyplot as plt
 
#same code as above but better
np.random.seed(0)

#Create Dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

X, y = create_data(100, 3)

'''
plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()
'''

layer1 = Layer_Dense(2,5)
#creation of activation object + using in layers
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

#### Softmax Function
# ReLU Activatio function too inaccurate for backprop
# Since it clips all negative values to 0
# So a network has no idea on how "Wrong" or "Right" an output is 

# Softmax :
# 
# y = exp(x)
# e ~ 2.7182

# Steps:
# imput -> exponentiation -> normalization -> output
#          |______________________________|
#                         |
#                      Softmax

# Example:
#                                                  exp(1)                    exp(2)                    exp(3)
# [1,2,3] -> [exp(1), exp(2), exp(3)] -> [------------------------, ------------------------, ------------------------ ] -> [0.09, 0.24, 0.67]
#                                         exp(1) + exp(2) + exp(3)  exp(1) + exp(2) + exp(3)  exp(1) + exp(2) + exp(3)

# Therefore sofmatx is:
#          exp(z(i,j))
# S(i,j) = -------------
#          ∑ exp(z(i,j))
#          (l=1 -> L)

import math

layer_output = [4.8, 1.21, 2.385]

# 1. Exponentiation
E = math.e # ~ 2.7182

exp_values = []

for output in layer_output:
    exp_values.append(E**output)

print(exp_values)

# 2. Normalization

# Here :
#                      (Single neuron Value)
# (normalized value) = --------------------------
#                      (sum of all neuron values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values)) # should be ~ 1

# Same code with numpy 
# Numpy makes it shorter

# 1. Exponentiation
exp_values = np.exp(layer_output) # Applies to the whole array
print(exp_values)

# 2. Normalization
norm_values = exp_values/np.sum(exp_values)
print(norm_values)
print(sum(norm_values)) # should be ~ 1

# Making output as a Batch
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

# print(np.sum(layer_outputs, axis=1, keepdims=True)) 
# axis=0 -> sum of columns
# axis=1 -> sum of rows
# keepdims=True -> retains original shape of original "Dimenesion" or orientation
#
# The output of this comand is:
# [[8.395]
#  [7.29 ]
#  [2.487]]

# Therefore to normalize properly we do:
norm_values = exp_values/np.sum(exp_values, axis=1, keepdims=True)
print(norm_values)

# Combining it all together

# ===== Some Code reused from ReLU CodeBlock =====

# Create Dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# ===== End of code used from ReLU Codeblock =====

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # As x increases in exp(x), values start becoming very large
        # this can give us an overflow error
        # therefore to manage these values,
        # we take all values of output layer before exponentiation
        # then we subtract the largest value in that layer from everi value in that layer
        # this means that the largest value now becomes 0; and our range of possiblities
        # are now ranging from 0 to 1, after exponentiation
        # Our output is still the same i.e. unaffected
        # Doing this protects us from an overfloe error
        probablities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities

X, y = create_data(100, 3)


dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
# input should be same as output of previous layer i.e. 3
# This layer also has 3 outputs
activation2 = Activation_Softmax()

# Running the Neural Net
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5]) # Print first 5 of multiple output
