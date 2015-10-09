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

In this class, neural networks are constructed from an array of integers.  The array must have at least two elements.  The first element [0] represents the size of the input vector.  Element [1] represents the number of nodes in the first processing layer, element [2] represents the number of nodes in the second processing layer, and so on.  The output vector will be the same length as the number of nodes in the final layer.

Input and output vectors are given as column vectors using my Matrix class, which is included in the implementation. The following image represents what the matrices representing the inputs and outputs will look like.

![alt tag](https://upload.wikimedia.org/math/a/e/0/ae099f03b525727414195676df0e23ea.png)

When evaluating the network, a single column vector is given, which will be used as the input to the network.  The output vector will be returned.  When training the network, both input and output column vectors are given by the user, and the neural network learns by evaluating the given input and comparing it to the given output.

When training, a real number constant known as the *training rate* is used.  This number is a constant multiplier that is used inside the training process.  A higher training rate means that the network will compensate to a change faster, however there is a higher likelihood that it will "overshoot" the desired state.  A lower training rate lowers that likelihood, but it will take longer to train the network in some cases.  The ideal training rate varies from problem to problem, and it is highly advised that the user experiments with this value to find the most effective training set.

Information source: *Neural Networks and Deep Learning* - http://neuralnetworksanddeeplearning.com/

More description will be added.
