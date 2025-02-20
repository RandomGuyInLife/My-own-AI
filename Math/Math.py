import numpy as np
import os
from utils.classes import *
#from tqdm import tqdm
from alive_progress import alive_bar

def clear():
    os.system( 'clear' )

def check_int(s: str):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def int_operation(s: str):
    if s == "+": return 0
    elif s == "-": return 1
    elif s == "*": return 2
    elif s == "/": return 3
    else: print("TF?!?!?! (The frick)")

#+ 0
#- 1
#* 2
#/ 3

provided_inputs = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 2, 0],
    [1, 3, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 2, 1],
    [1, 3, 1]
]

expected_outputs = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [2],
    [0],
    [1],
    [1]
]

for i in range(100):
    firstNum = np.random.randint(-(i+1), (i+1))
    operator = np.random.randint(0, 3)
    secondNum = np.random.randint(-(i+1), (i+1))
    output = 0

    if (operator == 0): output = firstNum + secondNum
    elif (operator == 1): output = firstNum - secondNum
    elif (operator == 2): output = firstNum * secondNum
    elif (operator == 3): output = firstNum / secondNum

    provided_inputs.append([firstNum, operator, secondNum])
    expected_outputs.append([output])

pil = provided_inputs.__len__()
activation = ActivationSigmoid()
activation2 = ActivationSoftmax()
dense1 = NeuronLayer(3,pil, activation)
dense2 = NeuronLayer(pil,2*pil, activation)
dense3 = NeuronLayer(2*pil,2*pil, activation)
dense4 = NeuronLayer(2*pil,3*pil, activation)
dense5 = NeuronLayer(3*pil,3*pil,activation)
dense6 = NeuronLayer(3*pil,3*pil,activation)
dense7 = NeuronLayer(3*pil,2*pil, activation)
dense8 = NeuronLayer(2*pil,1, activation2)

dense1.forward(provided_inputs)
dense2.forward(dense1.output)
dense3.forward(dense2.output)
dense4.forward(dense3.output)
dense5.forward(dense4.output)
dense6.forward(dense5.output)
dense7.forward(dense6.output)
dense8.forward(dense7.output, useActivation=False)

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = loss_function.calculate(dense8.output, expected_outputs, True)

iterations = 10000

with alive_bar(iterations, title='Training the AI!', length=20, bar='fish') as bar:
    for iteration in range(iterations):

        dense1.train()
        dense2.train()
        dense3.train()
        dense4.train()
        dense5.train()
        dense6.train()
        dense7.train()
        dense8.train()

        dense1.forward(provided_inputs)
        dense2.forward(dense1.output)
        dense3.forward(dense2.output)
        dense4.forward(dense3.output)
        dense5.forward(dense4.output)
        dense6.forward(dense5.output)
        dense7.forward(dense6.output)
        dense8.forward(dense7.output, useActivation=False)
    
        loss = loss_function.calculate(dense8.output, expected_outputs)

        if loss < lowest_loss:
            #print("Found a better weight/bias combination! Loss: ", loss, ", Output:", dense2.output, ", Iteration:", iteration)
            lowest_loss = loss
        else:
            dense1.revert()
            dense2.revert()
            dense3.revert()
            dense4.revert()
            dense5.revert()
            dense6.revert()
            dense7.revert()
            dense8.revert()
        bar()

#clear()
print("AI trained and ready! Lowest recored loss:", lowest_loss)
print("Trained on", pil, "different calculations!")

running = True

while running:
    input1 = input("Please enter a number or type exit: ")
    if input1 == "exit":
        running = False
        break
    elif check_int(input1):
        operation = input("Please enter an operation or type exit: ")
        if operation == "exit":
            running = False
            break
        elif operation == "-" or operation == "+" or operation == "/" or operation == "*":
            input2 = input("Please enter another number or type exit: ")
            if input2 == "exit":
                running = False
                break
            elif check_int(input2):
                dense1.forward([int(input1), int_operation(operation), int(input2)])
                dense2.forward(dense1.output)
                dense3.forward(dense2.output)
                dense4.forward(dense3.output)
                dense5.forward(dense4.output)
                dense6.forward(dense5.output)
                dense7.forward(dense6.output)
                dense8.forward(dense7.output)
                print("Those two numbers combine and result in ", dense8.output, "!")
            else:
                print("You passed an invalid input. The input was ", input2, ". Restarting!")
        else:
            print("You passed an invalid input. The input was ", operation, ". Restarting!")
    else:
        print("You passed an invalid input. The input was ", input1, ". Restarting!")