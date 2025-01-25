# Binary Interpreter

This is an EXTREMELY weak but funtional AI that can interpret 4 different binary combinations and return the first binary value.  
There are the inputs to expected outputs:  
[0, 0] -> 0  
[1, 0] -> 1  
[0, 1] -> 1  
[1, 1] -> 0  

Now how does this work? If you look though the code you will see three parts to the code: [Defining](#defining), [Training](#training), and [Running](#defining).

## Defining

```Python
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

dense2 = NeuronLayer(4,1)
activation2 = ActivationSoftmax()
```

This section of the code is where I defined everything that will shape the AI. The ```provided_inputs``` and ```expected_inputs``` values will be used during [training](#training) to make sure we get the right values we want.  
```dense1``` and ```dense2``` are our [Neural Layers](./utils/classes.py#L3). These are where we input data and get a processed output that simulates a neural network.  
According to the code ```dense1``` is a neural layer that takes two inputs and returns 1 outputs. ```dense2``` takes in the 4 inputs and returns another 4 outputs.  
Last, but not least, the ```activation1``` and ```activation2``` are [Activation Classes](./utils/classes.py#L10-L19). These classes process information in-between neural layers.

## Training

```Python
dense1.forward(provided_inputs)
activation1.forward(dense1.output)

dense2.forward(activation1.output)

loss_function = Loss_CategoricalCrossentropy()

lowest_loss = loss_function.calculate(dense2.output, expected_outputs)
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for _ in range(100000):

    dense1.weights += 0.05 * np.random.randn(2,4)
    dense1.biases += 0.05 * np.random.randn(1,4)
    dense2.weights += 0.05 * np.random.randn(4,1)
    dense2.biases += 0.05 * np.random.randn(1,1)

    dense1.forward(provided_inputs)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    
    loss = loss_function.calculate(dense2.output, expected_outputs)

    if loss < lowest_loss:
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
```

This is the section of code where we train the AI on our inputs and outputs that we want. The first 3 lines process the inputs with the random weights and biases assigned to each neuron when we [defined](#defining) them. Then I defined ```loss_function``` which calculates how far off the output was from the expected output. Lastly, because we haven't tried any other combination of weights and biases we can assume that we have the lowest loss and the best weights and biases, so we save those values in their own variables ```lowest_loss```, ```best_dense1_weights```, ```best_dense1_biases```,```best_dense2_weights```, ```best_dense2_biases``` respectively.  
Now lets actually train the AI. I used a ```for``` loop that repeats 100,000 times. Each time, the code tries a new combination of weights and biases. The output and loss is then recalculated. If the loss is less than the previous recorded loss the best weights and biases and the lowest loss are updated. If the loss is greater than or equal to the lowest loss then the weights and biases are reset to the best value and the loop repeats.

## Running

```Python
running = True

while running:
    input1 = input("Please enter a single byte (1 or 0) or type exit: ")
    if input1 == "exit":
        running = False
        break
    elif input1 == "1" or input1 == "0":
        input2 = input("Please enter another single byte (1 or 0) or type exit: ")
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
```

This is a very simple ```while``` loop which allows the user to send an input to and received and output from the AI. The loop asks for 2 inputs. These can be either 0, 1, or exit. If 0 or 1 are provided then the code moves on to asking for the next input or sending the inputs to the AI. If exit is provided then the ```running``` value is set to ```False``` and the while loop is exited. If any other input is provided then an error message is outputted and the loop restarts.
