import numpy as np

class NeuronLayer():
    def __init__(self, inputs, neurons):
        self.neuronCount = neurons
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

provided_inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

expected_outputs = [
    [0],
    [1],
    [1],
    [0]
]

layer1 = NeuronLayer(2, 2)
layer2 = NeuronLayer(2, 4)
layer3 = NeuronLayer(4, 8)
layer4 = NeuronLayer(8, 8)
layer5 = NeuronLayer(8, 4)
layer6 = NeuronLayer(4, 2)
layer7 = NeuronLayer(2, 1)

for i in range(4):
    layer1.forward(provided_inputs[i])
    layer2.forward(layer1.output)
    layer3.forward(layer2.output)
    layer4.forward(layer3.output)
    layer5.forward(layer4.output)
    layer6.forward(layer5.output)
    layer7.forward(layer6.output)
    print(layer7.output)