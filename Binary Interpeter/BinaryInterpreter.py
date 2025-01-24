import numpy as np
import os
from utils.classes import NeuronLayer, ActivationReLU, ActivationSoftmax, Loss_CategoricalCrossentropy

def clear():
    os.system( 'clear' )

provided_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

expected_outputs = [
    [0],
    [1],
    [1],
    [0]
]

dense1 = NeuronLayer(2,4)
activation1 = ActivationReLU()

dense2 = NeuronLayer(4,4)
activation2 = ActivationSoftmax()

dense1.forward(provided_inputs)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = loss_function.calculate(dense2.output, expected_outputs)
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    if iteration%2000 == 0:
        clear()
        print("Training the AI! Progress:", iteration/1000)

    dense1.weights += 0.05 * np.random.randn(2,4)
    dense1.biases += 0.05 * np.random.randn(1,4)
    dense2.weights += 0.05 * np.random.randn(4,1)
    dense2.biases += 0.05 * np.random.randn(1,1)

    dense1.forward(provided_inputs)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    
    loss = loss_function.calculate(dense2.output, expected_outputs)

    if loss < lowest_loss:
        #print("Found a better weight/bias combination! Loss: ", loss, ", Output:", dense2.output)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

clear()
print("AI trained and ready!")

running = True

while running:
    input1 = input("Please enter a single byte (1 or 0) or type exit:")
    if input1 == "exit":
        running = False
        break
    elif input1 == "1" or input1 == "0":
        input2 = input("Please enter another single byte (1 or 0) or type exit:")
        if input2 == "exit":
            running = False
            break
        elif input2 == "1" or input2 == "0":
            dense1.forward([int(input1), int(input2)])
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            print("Those two bites result in a ", activation2.output, "!")
        else:
            print("Your passed an invalid input. The input was ", input2, ". Restarting!")
    else:
        print("Your passed an invalid input. The input was ", input1, ". Restarting!")