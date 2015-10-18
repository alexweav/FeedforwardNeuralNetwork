using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Matrices;

namespace NeuralNetwork {
    class FeedforwardNeuralNetwork {

        private int numLayers;
        private int[] layerSizes;
        private Matrix[] weightMatrices;
        private Matrix[] biases;
        private float learningRate = 0.8F;

        //Construction of a feedforward network is from a sequence of positive integer values
        //The first integer in the sequence represents the number of neurons in the first layer of the network
        //The second integer is the number of neurons in the second layer, and so on
        //For a proper network, the number of nodes on the first layer needs to be equal to the length of the input vector
        //and the number of nodes on the last layer needs to be equal to the length of the output vector
        //It is assumed that each node on a given layer is connected directly and only to each node on the next layer
        public FeedforwardNeuralNetwork(int[] layerSizes) {
            this.layerSizes = layerSizes;
            numLayers = layerSizes.Length;
            weightMatrices = new Matrix[numLayers];
            biases = new Matrix[numLayers];
            //The first layer is the input layer.  The input layer is simply a row of neurons that outputs whatever the input to the algorithm is.
            //Therefore, since the nodes in the input layer have no incoming connections and are not calculated, they have no weight matrices or biases.
            weightMatrices[0] = null;
            biases[0] = null;
            //Initializes the weight matrices and bias vectors for each layer to the proper size
            for (int i = 1; i < numLayers; ++i) {
                weightMatrices[i] = new Matrix(layerSizes[i], layerSizes[i-1]);
                biases[i] = new Matrix(layerSizes[i], 1);
            }
            InitializeWeightMatrices();
            InitializeBiases();
        }

        //Sets every weight to a random value on a Gaussian distribution with mean 0 and standard deviation 1 
        private void InitializeWeightMatrices() {
            GaussianRandom random = new GaussianRandom(0, 1);
            for (int l = 1; l < numLayers; ++l) {
                for (int i = 1; i <= weightMatrices[l].NumRows; ++i) {
                    for(int j = 1; j <= weightMatrices[l].NumColumns; ++j) {
                        weightMatrices[l][i, j] = (float)random.NextDouble();
                    }
                }
            }
        }

        //Sets every bias to a random value on a Gaussian distribution with mean 0 and standard deviation 1
        private void InitializeBiases() {
            GaussianRandom random = new GaussianRandom(0, 1);
            for (int l = 1; l < numLayers; ++l) {
                for (int i = 1; i <= biases[l].NumRows; ++i) {
                    biases[l][i, 1] = (float)random.NextDouble();
                }
            }
        }

        //Takes the given input vector and passes it through the network
        //Returns the network's output vector
        public Matrix EvaluateNetwork(Matrix inputs) {
            if (inputs == null) {
                throw new NullReferenceException("Cannot evaluate a null input vector.");
            }
            if (inputs.NumColumns != 1 || inputs.NumRows != layerSizes[0]) {
                throw new ArgumentException("Input vector must be a nx1 matrix and n must equal the number of nodes in the first layer of the network.");
            }
            Matrix layerOutput = inputs;
            for (int i = 1; i < numLayers; ++i) {
                layerOutput = Sigmoid.Sigma((weightMatrices[i] * layerOutput) + biases[i]);
            }
            return layerOutput;
        }

        

        //Takes an input vector and an expected output vector
        //Uses the discrepancy between the two to train the network via backpropagation
        public void TrainNetwork(Matrix inputs, Matrix expectedResult) {
            //First, we evaluate the network.
            //This is slightly different from the normal evaluation method:
            //We must track the output of each layer, as it is used later in calculation
            //The outputs of the layers are stored in an array called outputs
            //outputs[0] is the output of the input layer (and therefore the input of the system)
            //We must also track the sigmaPrimes of each layer, which is simply the weighted sum of each layer passed through the derivative of the sigmoid function rather than the normal sigmoid function
            //These are used later in calculation and are stored in a similar manner to the actual outputs
            Matrix[] sigmaPrimes = new Matrix[numLayers];
            sigmaPrimes[0] = null;
            Matrix[] outputs = new Matrix[numLayers];
            outputs[0] = inputs;
            for (int i = 1; i < numLayers; ++i) {
                Matrix z = (weightMatrices[i] * outputs[i - 1]) + biases[i];
                sigmaPrimes[i] = Sigmoid.SigmaPrime(z);
                outputs[i] = Sigmoid.Sigma(z);
            }
            Matrix actualResult = outputs[numLayers - 1];
            Matrix[] layerDeltas = new Matrix[numLayers];
            layerDeltas[0] = null;
            //calculate the node delta heuristic for each layer
            //The layerDelta is a vector for which layerDelta[i] is the node delta for the ith node in the layer
            //layerDeltas is an array of the layerDelta vectors.  layerDeltas[0] is the layer delta for the input layer, layerDeltas[1] is for the first layer, etc.
            //The input layer has no layer delta defined on it, so therefore layerDeltas[0] is null
            //The layer delta for the final layer is calculated as follows:
            layerDeltas[numLayers - 1] = Matrix.HadamardProduct((actualResult - expectedResult), sigmaPrimes[numLayers - 1]);
            //The layer delta for every other layer depends on the layer delta of the layer after it, and are calculated as follows:
            for(int i = numLayers - 2; i > 0; --i) {
                layerDeltas[i] = Matrix.HadamardProduct(Matrix.Transpose(weightMatrices[i + 1]) * layerDeltas[i + 1], sigmaPrimes[i]);
            }
            //now we update the biases, from the layer deltas
            for (int i = 1; i < numLayers; ++i) {
                biases[i] = biases[i] - learningRate * (layerDeltas[i]);
            }
            //and update the weights from the layer deltas
            for (int i = 1; i < numLayers; ++i) {
                weightMatrices[i] = weightMatrices[i] - learningRate * (layerDeltas[i] * Matrix.Transpose(outputs[i - 1]));
            }
            
        }

        //Assume we have two output vectors of a neural network.
        //expectedResult is the output vector that we want
        //actualResult is the output vector from the network itself
        //This obtains the quadratic cost of the run, defined as:
        //    (1/2) * sum over j of (expectedResult[j] - actualResult[j])^2 where j ranges over the length of the vectors
        public double GetCost(Matrix expectedResult, Matrix actualResult) {
            if (expectedResult == null || actualResult == null) {
                throw new NullReferenceException("Cannot get cost on null matrices.");
            }
            if (expectedResult.NumRows != actualResult.NumRows || expectedResult.NumColumns != 1 || actualResult.NumColumns != 1) {
                throw new ArgumentException("The expected result and the actual result must be vectors of the same size to calculate cost.");
            }
            double quadraticTotal = 0;  //will store sum of (expectedResult[j] - actualResult[j])^2 for all j ranging over length of input vectors
            for (int i = 1; i < expectedResult.NumRows; ++i) {
                quadraticTotal += (expectedResult[i, 1] - actualResult[i, 1]) * (expectedResult[i, 1] - actualResult[i, 1]);
            }
            return quadraticTotal / 2;
        }
    }
}
