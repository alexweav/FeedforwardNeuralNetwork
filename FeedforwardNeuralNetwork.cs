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
        private float learningRate;
        private float regParameter;

        //Construction of a feedforward network is from a sequence of positive integer values
        //The first integer in the sequence represents the number of neurons in the first layer of the network
        //The second integer is the number of neurons in the second layer, and so on
        //For a proper network, the number of nodes on the first layer needs to be equal to the length of the input vector
        //and the number of nodes on the last layer needs to be equal to the length of the output vector
        //It is assumed that each node on a given layer is connected directly and only to each node on the next layer
        public FeedforwardNeuralNetwork(int[] layerSizes, float learningRate, float regularizationParameter) {
            this.learningRate = learningRate;
            this.regParameter = regularizationParameter;
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
                weightMatrices[i] = new Matrix(layerSizes[i], layerSizes[i - 1]);
                biases[i] = new Matrix(layerSizes[i], 1);
            }
            InitializeWeightMatrices();
            InitializeBiases();
        }

        public float LearningRate {
            get {
                return this.learningRate;
            }
            set {
                if (value <= 0) {
                    throw new ArgumentException("Learning rate must be greater than 0.");
                }
                this.learningRate = value;
            }
        }

        public float RegularizationParameter { 
            get { 
                return this.regParameter;
            }
            set {
                if (value <= 0) {
                    throw new ArgumentException("Regularization paramter must be greater than 0.");
                }
                this.regParameter = value;
            }
        }

        //Sets every weight to a random value on a Gaussian distribution with mean 0 and standard deviation 1 
        private void InitializeWeightMatrices() {
            GaussianRandom random = new GaussianRandom(0, 1);
            for (int l = 1; l < numLayers; ++l) {
                for (int i = 1; i <= weightMatrices[l].NumRows; ++i) {
                    for (int j = 1; j <= weightMatrices[l].NumColumns; ++j) {
                        random.StdDev = Math.Pow(weightMatrices[l].NumColumns, -0.5);
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
                    random.StdDev = Math.Pow(weightMatrices[l].NumColumns, -0.5);
                    biases[l][i, 1] = (float)random.NextDouble();
                }
            }
        }

        //Takes the given input vector and passes it through the network
        //Returns the network's output vector
        public Matrix Evaluate(Matrix inputs) {
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

        public void TrainEpochs(TrainingExample[] examples, int numEpochs) {
            Console.WriteLine("\n");
            for (int i = 0; i < numEpochs; ++i) {
                Console.WriteLine("Beginning epoch " + (i + 1) + " of " + numEpochs + ".");
                TrainDataSet(examples);
            }
        }

        public void TrainDataSet(TrainingExample[] examples) {
            Random rand = new Random();
            examples = examples.OrderBy(x => rand.Next()).ToArray();
            for (int i = 0; i < examples.Length; ++i) {
                TrainIteration(examples[i].input, examples[i].expectedOutput, 1 - (learningRate * regParameter / examples.Length));
                //If there are over 5000 examples, provide status updates in console
                if (examples.Length >= 5000) {
                    if(i == 0) Console.WriteLine("\n");
                    if (i % 1000 == 0 && i != 0) {
                        Console.WriteLine(i + "/" + examples.Length + " objects trained. Epoch " + Math.Round((double)i/examples.Length, 3) * 100 + "% complete.");
                    }
                }
                
            }
            Console.WriteLine(examples.Length + "/" + examples.Length + " objects trained. Epoch 100% complete.");
            Console.WriteLine("\n");
        }

        //Takes an input vector and an expected output vector
        //Uses the discrepancy between the two to train the network via backpropagation
        //regFactor is the regularization constant 1 - (lambda * learningRate/(2 * size of training set))
        //A regFactor of 1 is taken to indicate the absence of regularization
        public void TrainIteration(Matrix inputs, Matrix expectedResult, float regFactor) {
            if (regFactor <= 0.0F || regFactor > 1.0F) {
                throw new ArgumentException("Regularization factor must be on the range (0, 1].");
            }
            Matrix[] weightedLayerSums = new Matrix[numLayers];
            Matrix[] layerOutputs = new Matrix[numLayers];
            Matrix[] newWeightMatrices = new Matrix[numLayers];
            newWeightMatrices[0] = null;
            layerOutputs[0] = inputs;
            weightedLayerSums[0] = null;
            //First we evaulate the network (feedforward) and track all output vectors and weighted sums as we go
            for (int i = 1; i < numLayers; ++i) {
                weightedLayerSums[i] = weightMatrices[i] * layerOutputs[i - 1] + biases[i];
                layerOutputs[i] = Sigmoid.Sigma(weightedLayerSums[i]);
            }
            Matrix[] layerDeltas = new Matrix[numLayers];
            //Calculate the gradient (quadratic cost)
            /*Matrix gradient = QuadraticGradient(layerOutputs[numLayers - 1], expectedResult);
            layerDeltas[numLayers - 1] = Matrix.HadamardProduct(gradient, Sigmoid.SigmaPrime(weightedLayerSums[numLayers - 1]));
            //Then, make a backwards pass through the network, calculating the weight changes and applying them
            for (int i = numLayers - 2; i > 0; --i) {
                layerDeltas[i] = Matrix.HadamardProduct(Matrix.Transpose(weightMatrices[i + 1]) * layerDeltas[i + 1], Sigmoid.SigmaPrime(weightedLayerSums[i]));
            }
            //Gradient descent
            for (int i = numLayers - 1; i > 0; --i) {
                weightMatrices[i] = weightMatrices[i] - (learningRate * layerDeltas[i]) * Matrix.Transpose(layerOutputs[i - 1]);
                biases[i] = biases[i] - learningRate * (layerDeltas[i]);
            }*/

            for (int i = numLayers - 1; i > 0; --i) {
                if (i == numLayers - 1) {
                    Matrix gradient = QuadraticGradient(layerOutputs[i], expectedResult);
                    layerDeltas[i] = Matrix.HadamardProduct(gradient, Sigmoid.SigmaPrime(weightedLayerSums[i]));
                } else {
                    layerDeltas[i] = Matrix.HadamardProduct(Matrix.Transpose(weightMatrices[i + 1]) * layerDeltas[i + 1], Sigmoid.SigmaPrime(weightedLayerSums[i]));
                }
                newWeightMatrices[i] = (regFactor * weightMatrices[i]) - (learningRate * layerDeltas[i]) * Matrix.Transpose(layerOutputs[i - 1]);
                biases[i] = biases[i] - learningRate * (layerDeltas[i]);
            }
            weightMatrices = newWeightMatrices;
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

        private Matrix QuadraticGradient(Matrix activations, Matrix expected) {
            return activations - expected;
        }

        private Matrix CrossEntropyGradient(Matrix activations, Matrix expected) {
            Matrix output = new Matrix(activations.NumRows, 1);
            for (int i = 1; i <= activations.NumRows; ++i) {
                output[i, 1] = (activations[i, 1] - expected[i, 1]) / ((activations[i, 1] + 1) * activations[i, 1]);
            }
            return output;
        }

    }
}
