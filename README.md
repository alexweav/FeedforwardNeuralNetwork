# FeedforwardNeuralNetwork
A class for constructing and evaluating feedforward neural networks

Provides simple tools for building, evaluating, and training a feedforward, fully connected neural network.

##About Neural Networks

An artificial neural network is a machine learning model based on actual neurons.  The behavior of a neuron is straightforward: It takes a number of numerical inputs, multiplies each input by a respective weight, and outputs some numerical value related to the inputs.

The neurons used in this model are known as *Sigmoid Neurons*.  Sigmoid neurons do the following:

1.  Take a vector of numerical inputs.
2.  Multiply each numerical input by some weight.  The weights for two different inputs may be different values.
3.  Take the summation of the weighted input values. This can also be represented by a dot product between an input vector and a weight vector.
4.  Add some value to the summation, known as the *bias* or the *threshold*.
4.  Pass the resulting value through the *Sigmoid* function.

![alt tag](https://www.cs.uaf.edu/2007/fall/cs441/proj1notes/schamel/CS%20441%20Project%20%231%20Webpage_html_m74fe8c2a.png)

This allows a single neuron to make decisions based on the settings of its weights and biases.  It is intuitive that a network of neurons, connected to each other, can make very precise decisions on its inputs, making them ideal for advanced categorization processes such as image recognition.

A *Feedforward Neural Network* is a type of network in which information flows in a single direction through layers of neurons.  A single column of neurons is taken to be a layer.  Each neuron in a layer outputs information to each neuron in the following layer, and so on.  The first layer represents the inputs of the network.  The input layer simply acts as a container for the inputs of the network, and does not make decisions.  In any network, there must be at least one decision making layer, as well as exactly one input layer (which is located at the "beginning" of the network).  The output of the final layer is taken to be the output of the entire network.  Any layer that is not an input layer or an output layer is called a *hidden layer*.

![alt tag](http://www.hindawi.com/journals/aai/2011/686258.fig.001.jpg)

In a given network, the decisions are made by setting the weights and biases to specific values.  There exists an algorithm known as *backpropagation*, which takes a set of inputs and a set of expected outputs, and alters the weights and biases to better reflect the desired output.  This is used to train a given network.

##Implementation-Specific Details

Information source: *Neural Networks and Deep Learning* - http://neuralnetworksanddeeplearning.com/

More description will be added.
