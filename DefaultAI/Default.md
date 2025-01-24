# Default AI

This is the frame for any AI I could possbly make.   
This file contains two classes that can be used to build an AI. These are the [NeuronLayer](#neuronlayer) and the [ActivationReLU](#activationrelu) classes.

## NeuronLayer

The [NeuronLayer](./AI.py#L3) class takes in two numbers: inputs and neurons. Inputs defines the number of inputs this layer will process and the number of wieghts each neuron will have. The neuron input determines the numer of neuron in the layer and the number of outputs that that layer will have.

## ActivationReLU

The [ActivationReLU](./AI.py#L10) class is an example of an activation class. An activation class or
function will take in the output of a layer and process it to your liking. This example is run to keep the output >= 0