import numpy as np
import os
from utils.classes import NeuronLayer, ActivationReLU, ActivationSoftmax, Loss_CategoricalCrossentropy

def clear():
    os.system( 'cls' )

'''provided_inputs = [
    [dogImage1],
    [catImage1],
    [catImage2],
    [dogImage2],
    [catImage3],
    [dogImage3]
]'''

expected_outputs = [
    [0],
    [1],
    [1],
    [0],
    [1],
    [0]
]

neuronLayer1 = NeuronLayer(1,6)
neuronLayer2 = NeuronLayer(6,18)
neuronLayer3 = NeuronLayer(18,18)
neuronLayer4 = NeuronLayer(18,6)
neuronLayer5 = NeuronLayer(6,2)
activationR = ActivationReLU()
activationS = ActivationSoftmax()
loss = Loss_CategoricalCrossentropy()
print(loss.calculate([[1, 0.1], [1, 0.1], [0.1, 1], [0.1, 1], [1, 0.1], [0.1, 1]], np.array(expected_outputs)))