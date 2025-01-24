# Binary Interpreter

This is an EXTREMELY weak but funtional AI that can interpret 4 different binary combinations and return the first binary value.  
There are the inputs to expected outputs:  
[0, 0] -> 0  
[1, 0] -> 1  
[0, 1] -> 1  
[1, 1] -> 0  

Now how does this work? If you look though the code you will see three parts to the code: [Defining](#defining), [Training](), and [Running]().

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

dense2 = NeuronLayer(4,4)
activation2 = ActivationSoftmax()
```

This section of the code is where we define everything that will shape the AI. The ```provided_inputs``` and ```expected_inputs``` values will be used during [training]() to make sure we get the right values we want.  
```dense1``` and ```dense2``` are our [Neural Layers](). These are where we input data and get a processed output that simulates a neural network.  
According to the code ```dense1``` is a neural layer that takes two 
Last, but not least, the ```activation1``` and ```activation2``` are [Activation Classes](). These classes process information in-between neural layers.