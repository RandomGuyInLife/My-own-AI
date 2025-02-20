import numpy as np
import math

class Activation:
    def forward(self, inputs):
        return inputs

class NeuronLayer():
    def __init__(self, n_inputs, n_neurons, activation: Activation):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        self.inputs = n_inputs
        self.neurons = n_neurons
    def forward(self, inputs, useActivation=True):
        o = np.dot(inputs, self.weights) + self.biases
        if useActivation : self.output = self.activation.forward(o)
        else : self.output = o
    def train(self):
        self.prevWeights = self.weights.copy()
        self.prevBiases = self.biases.copy()
        self.weights += 0.05 * np.random.randn(self.inputs, self.neurons)
        self.biases += 0.05 * np.random.randn(1, self.neurons)
    def revert(self):
        self.weights = self.prevWeights.copy()
        self.biases = self.prevBiases.copy()

class ActivationReLU(Activation):
    def forward(self, inputs):
        return np.maximum(0, inputs)

class ActivationSoftmax(Activation):
    def forward(self, inputs):
        return np.round(inputs[0])[0]
    
class ActivationSiLU(Activation):
    def forward(self, inputs: np.array):
        return inputs/(1+(math.e**(-inputs)))

class ActivationSigmoid(Activation):
    def forward(self, inputs: np.array):
        return 1/(1+(math.e**(-inputs)))

class Loss:
    def calculate(self, output, y, a=False):
        sample_losses = self.forward(output, y, a)

        return np.average(sample_losses)
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true, a):
        samples = len(y_pred)
        losses = []
        for o in range(samples):
            losses.append(abs(y_true[o]-y_pred[o]))
            if a: print(y_true[o], "-", y_pred[o])
        return losses
