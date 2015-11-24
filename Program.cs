using System;
using System.IO;
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
            LogicGatesExample();
            //DecimalBinaryExample();
            //MNISTExample();
            //SineExample();
        }

        public static void MNISTExample() {
            try {
                int[] layers = new int[] { 784, 15, 10 };
                FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers, 1.0F);
                Console.WriteLine("Neural network constructed.");

                MNISTTestNetwork(fnn);

                FileStream labelsStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\train-labels.idx1-ubyte", FileMode.Open);
                FileStream imagesStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\train-images.idx3-ubyte", FileMode.Open);
                Console.WriteLine("Datasets found.");
                BinaryReader labelsReader = new BinaryReader(labelsStream);
                BinaryReader imagesReader = new BinaryReader(imagesStream);
                int magic1 = imagesReader.ReadInt32();
                int numImages = (imagesReader.ReadByte() << 24) | (imagesReader.ReadByte() << 16) | (imagesReader.ReadByte() << 8) | (imagesReader.ReadByte());
                int numRows = imagesReader.ReadInt32();
                int numColumns = imagesReader.ReadInt32();
                int magic2 = labelsReader.ReadInt32();
                int numLabels = labelsReader.ReadInt32();
                Console.WriteLine("Image databases prepared.");

                

                Console.WriteLine(numImages + " total images detected.  Beginning training system.");

                for (int q = 0; q < 1; ++q) {
                    for (int r = 0; r < numImages; ++r) {
                        Matrix input = new Matrix(784, 1);
                        Matrix expectedOutput = new Matrix(10, 1);
                        for (int i = 1; i <= 784; ++i) {
                            input[i, 1] = imagesReader.ReadByte() / 256;
                        }
                        int expectedNum = labelsReader.ReadByte();
                        expectedOutput[expectedNum + 1, 1] = 1;

                        fnn.TrainIteration(input, expectedOutput);

                        if ((r + 1) % 100 == 0) {
                            Console.WriteLine((r + 1) / 100 + "00 images trained.");
                        }
                    }
                    
                }

                MNISTTestNetwork(fnn);

            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
            }
        }

        public static void MNISTTestNetwork(FeedforwardNeuralNetwork fnn) {
            try {
                Console.WriteLine("Beginning neural network test procedure.");
                FileStream labelsStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\t10k-labels.idx1-ubyte", FileMode.Open);
                FileStream imagesStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\t10k-images.idx3-ubyte", FileMode.Open);
                Console.WriteLine("Test datasets found.");
                BinaryReader labelsReader = new BinaryReader(labelsStream);
                BinaryReader imagesReader = new BinaryReader(imagesStream);
                int magic1 = imagesReader.ReadInt32();
                int numImages = (imagesReader.ReadByte() << 24) | (imagesReader.ReadByte() << 16) | (imagesReader.ReadByte() << 8) | (imagesReader.ReadByte());
                int numRows = imagesReader.ReadInt32();
                int numColumns = imagesReader.ReadInt32();
                int magic2 = labelsReader.ReadInt32();
                int numLabels = labelsReader.ReadInt32();
                Console.WriteLine("Test image databases prepared.");

                int numSuccesses = 0;
                for (int r = 0; r < numImages; ++r) {
                    Matrix input = new Matrix(784, 1);
                    Matrix expectedOutput = new Matrix(10, 1);
                    for (int i = 1; i <= 784; ++i) {
                        input[i, 1] = imagesReader.ReadByte() / 256;
                    }
                    int expectedNum = labelsReader.ReadByte();
                    expectedOutput[expectedNum + 1, 1] = 1;
                    Matrix actualOutput = fnn.Evaluate(input);
                    bool successfulRun = true;
                    float largestOutputValue = 0;
                    int index = 0;
                    for (int j = 1; j <= 10; ++j) {
                        if (actualOutput[j, 1] > largestOutputValue) {
                            largestOutputValue = actualOutput[j, 1];
                            index = j;
                        }
                    }
                    if (index - 1 == expectedNum) {
                        ++numSuccesses;
                    }

                    Console.WriteLine("Test value: " + expectedNum);
                    Console.WriteLine("Output vector: " + actualOutput[1, 1] + actualOutput[2, 1] + actualOutput[3, 1] + actualOutput[4, 1] + actualOutput[5, 1] + actualOutput[6, 1] + actualOutput[7, 1] + actualOutput[8, 1] + actualOutput[9, 1] + actualOutput[10, 1]);


                    if ((r + 1) % 100 == 0) {
                        Console.WriteLine((r + 1) / 100 + "00 test images processed.  Current success percentage: " + (float)numSuccesses/(r+1) * 100 + "%");
                    }
                }
                Console.WriteLine("Test regimen completed.  Network was correct for " + (float)numSuccesses / numImages * 100 + "% of " + numImages + " images.");

            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
            }
        }

        public static void DecimalBinaryExample() {
            int[] layers = new int[] { 2, 3, 5, 4 };
            FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers, 1.0F);
            DecimalBinaryTestNetwork(fnn);
            Matrix[] expectedOutputs = { new Matrix(new float[4,1]{ {1},
                                                                    {0},
                                                                    {0},
                                                                    {0}}), 
                                         new Matrix(new float[4,1]{ {0},
                                                                    {1},
                                                                    {0},
                                                                    {0}}), 
                                         new Matrix(new float[4,1]{ {0},
                                                                    {0},
                                                                    {1},
                                                                    {0}}), 
                                         new Matrix(new float[4,1]{ {0},
                                                                    {0},
                                                                    {0},
                                                                    {1}})};
            Matrix[] inputs =          { new Matrix(new float[2,1]{ {0},
                                                                    {0}}),
                                         new Matrix(new float[2,1]{ {0},
                                                                    {1}}),
                                         new Matrix(new float[2,1]{ {1},
                                                                    {0}}),
                                         new Matrix(new float[2,1]{ {1},
                                                                    {1}})};
            //fnn.TrainEpochs(inputs, expectedOutputs, 1000);
            DecimalBinaryTestNetwork(fnn);
        }

        public static void DecimalBinaryTestNetwork(FeedforwardNeuralNetwork fnn) {
            Matrix inputs = new Matrix(2, 1);

            Matrix outputs = fnn.Evaluate(inputs);   //test for input = 0
            if (outputs[1, 1] > 0.9 && outputs[2, 1] < 0.1 && outputs[3, 1] < 0.1 && outputs[4, 1] < 0.1) {
                Console.WriteLine("0 success.");
            } else {
                Console.WriteLine("0 fail." + outputs[1, 1] + " " + outputs[2, 1]);
            }
            inputs[1, 1] = 0;
            inputs[2, 1] = 1;
            outputs = fnn.Evaluate(inputs);  //test for input = 1
            if (outputs[1, 1] < 0.1 && outputs[2, 1] > 0.9 && outputs[3, 1] < 0.1 && outputs[4, 1] < 0.1) {
                Console.WriteLine("1 success.");
            } else {
                Console.WriteLine("1 fail." + outputs[1, 1] + " " + outputs[2, 1]);
            }
            inputs[1, 1] = 1;
            inputs[2, 1] = 0;
            outputs = fnn.Evaluate(inputs);  //test for input = 2
            if (outputs[1, 1] < 0.1 && outputs[2, 1] < 0.1 && outputs[3, 1] > 0.9 && outputs[4, 1] < 0.1) {
                Console.WriteLine("2 success.");
            } else {
                Console.WriteLine("2 fail." + outputs[1, 1] + " " + outputs[2, 1]);
            }
            inputs[1, 1] = 0;
            inputs[2, 1] = 1;
            outputs = fnn.Evaluate(inputs);  //test for input = 3
            if (outputs[1, 1] < 0.1 && outputs[2, 1] < 0.1 && outputs[3, 1] < 0.1 && outputs[4, 1] > 0.9) {
                Console.WriteLine("3 success.");
            } else {
                Console.WriteLine("3 fail." + outputs[1, 1] + " " + outputs[2, 1]);
            }
        }

        public static void LogicGatesExample() {
            //First layer is input layer
            //Initialize a 2-layer ANN
            //Two inputs, one output
            int[] layers = new int[] { 2, 3, 1 };
            FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers, 1.0F);
            //Train this many cycles
            int numTrainingEpochs = 10000;
            TrainingExample ex1 = new TrainingExample(new Matrix(new float[,]{ { 0 },
                                                                               { 0 } }),
                                                      new Matrix(new float[,]{ { 0 } }));

            TrainingExample ex2 = new TrainingExample(new Matrix(new float[,]{ { 0 },
                                                                               { 1 } }),
                                                      new Matrix(new float[,]{ { 1 } }));

            TrainingExample ex3 = new TrainingExample(new Matrix(new float[,]{ { 1 },
                                                                               { 0 } }),
                                                      new Matrix(new float[,]{ { 1 } }));

            TrainingExample ex4 = new TrainingExample(new Matrix(new float[,]{ { 1 },
                                                                               { 1 } }),
                                                      new Matrix(new float[,]{ { 0 } }));

            TrainingExample[] trainingExamples = { ex1, ex2, ex3, ex4 };
            for (int i = 0; i < numTrainingEpochs; ++i) {
                //Sets input matrices and output matrices and trains the network accordingly for all combinations
                Random rand = new Random();
                trainingExamples = trainingExamples.OrderBy(x => rand.Next()).ToArray();
                for (int j = 0; j < 4; ++j) {
                    fnn.TrainIteration(trainingExamples[j].input, trainingExamples[j].expectedOutput);
                }
            }
            //After training, evaluates the network with respect to all possible inputs
            //Prints their outputs to the console to see the result of training
            float[,] inputArray = new float[,] { { 0 },
                                                 { 0 } };
            Matrix input = new Matrix(inputArray);
            Matrix output = fnn.Evaluate(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 0 },
                                        { 1 } };
            input = new Matrix(inputArray);
            output = fnn.Evaluate(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 1 },
                                        { 0 } };
            input = new Matrix(inputArray);
            output = fnn.Evaluate(input);
            Console.WriteLine(output[1, 1]);
            inputArray = new float[,] { { 1 },
                                        { 1 } };
            input = new Matrix(inputArray);
            output = fnn.Evaluate(input);
            Console.WriteLine(output[1, 1]);
        }

        public static void SineExample() {
            int[] layers = { 1, 3, 1 };
            FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers, 1.0F);

        }
    }
}
