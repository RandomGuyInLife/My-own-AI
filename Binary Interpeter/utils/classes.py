import numpy as np

class NeuronLayer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def forward(self, inputs: list[list[float]]):
        outputs = 0
        for i in range(inputs[0].__len__()):
            outputs += inputs[0][i]
        self.output = (outputs/inputs[0].__len__()).__round__()

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        
        return np.mean(sample_losses)
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        losses = []
        for o in range(samples):
            losses.append(abs(y_true[o]-y_pred[o]))
        return losses
