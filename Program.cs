using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Matrices;

namespace NeuralNetwork {
    class Program {
        //This is simply a main function to demonstrate the use of the network
        //Not to be included in a program which employs the network
        static void Main(string[] args) {
            //First layer is input layer
            //Initialize a 2-layer ANN
            //Two inputs, one output
            int[] layers = new int[] { 2, 1, 1 };
            FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers);
            //Train this many cycles
            int numTrainingEpochs = 10000;
            for (int i = 0; i < numTrainingEpochs; ++i) {
                //Sets input matrices and output matrices and trains the network accordingly for all combinations
                float[,] trainingArray = new float[,] { { 0 },
                                                        { 0 } };
                Matrix trainingInput = new Matrix(trainingArray);
                float[,] expected = new float[,] { { 1 } };
                Matrix expectedOutput = new Matrix(expected);
                fnn.TrainNetwork(trainingInput, expectedOutput);
                trainingArray = new float[,] { { 0 },
                                               { 1 } };
                trainingInput = new Matrix(trainingArray);
                expected = new float[,] { { 0 } };
                expectedOutput = new Matrix(expected);
                fnn.TrainNetwork(trainingInput, expectedOutput);
                trainingArray = new float[,] { { 1 },
                                               { 0 } };
                trainingInput = new Matrix(trainingArray);
                expected = new float[,] { { 1 } };
                expectedOutput = new Matrix(expected);
                fnn.TrainNetwork(trainingInput, expectedOutput);
                trainingArray = new float[,] { { 1 },
                                               { 1 } };
                trainingInput = new Matrix(trainingArray);
                expected = new float[,] { { 1 } };
                expectedOutput = new Matrix(expected);
                fnn.TrainNetwork(trainingInput, expectedOutput);
            }
            //After training, evaluates the network with respect to all possible inputs
            //Prints their outputs to the console to see the result of training
            float[,] inputArray = new float[,] { { 0 },
                                                 { 0 } };
            Matrix input = new Matrix(inputArray);
            Matrix output = fnn.EvaluateNetwork(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 0 },
                                        { 1 } };
            input = new Matrix(inputArray);
            output = fnn.EvaluateNetwork(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 1 },
                                        { 0 } };
            input = new Matrix(inputArray);
            output = fnn.EvaluateNetwork(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 1 },
                                        { 1 } };
            input = new Matrix(inputArray);
            output = fnn.EvaluateNetwork(input);
            Console.WriteLine(output[1, 1]);
        }
    }
}
