import numpy as np
import os
from utils.classes import NeuronLayer, Activation, ActivationSoftmax, Loss_CategoricalCrossentropy

def clear():
    os.system( 'clear' )

def check_int(s: str):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

provided_inputs = [
    [0, 0],
    [1, 1],
    [1, 2],
    [2, 2],
    [3, 3],
    [4, 4],
    [5, 5],
    [10, 10],
    [90, 10]
]

expected_outputs = [
    [0],
    [2],
    [3],
    [4],
    [6],
    [8],
    [10],
    [20],
    [100]
]

pil = provided_inputs.__len__()
activation = Activation()
activation2 = ActivationSoftmax()
dense1 = NeuronLayer(2,pil, activation)
dense2 = NeuronLayer(pil,2*pil, activation)
dense3 = NeuronLayer(2*pil,2*pil, activation)
dense4 = NeuronLayer(2*pil,1, activation2)

dense1.forward(provided_inputs)
dense2.forward(dense1.output)
dense3.forward(dense2.output)
dense4.forward(dense3.output, useActivation=False)

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = loss_function.calculate(dense4.output, expected_outputs)

for iteration in range(100000):
    if iteration%2000 == 0:
        clear()
        print("Training the AI! Progress:", iteration/1000)

    dense1.train()
    dense2.train()
    dense3.train()
    dense4.train()

    dense1.forward(provided_inputs)
    dense2.forward(dense1.output)
    dense3.forward(dense2.output)
    dense4.forward(dense3.output, useActivation=False)
    
    loss = loss_function.calculate(dense4.output, expected_outputs)

    if loss < lowest_loss:
        #print("Found a better weight/bias combination! Loss: ", loss, ", Output:", dense2.output, ", Iteration:", iteration)
        lowest_loss = loss
    else:
        dense1.revert()
        dense2.revert()
        dense3.revert()
        dense4.revert()

clear()
print("AI trained and ready!")

running = True

while running:
    input1 = input("Please enter a number or type exit: ")
    if input1 == "exit":
        running = False
        break
    elif check_int(input1):
        input2 = input("Please enter another number or type exit: ")
        if input2 == "exit":
            running = False
            break
        elif check_int(input2):
            dense1.forward([int(input1), int(input2)])
            dense2.forward(dense1.output)
            dense3.forward(dense2.output)
            dense4.forward(dense3.output, useActivation=True)
            print("Those two numbers combine and result in ", dense4.output, "!")
        else:
            print("Your passed an invalid input. The input was ", input2, ". Restarting!")
    else:
        print("Your passed an invalid input. The input was ", input1, ". Restarting!")