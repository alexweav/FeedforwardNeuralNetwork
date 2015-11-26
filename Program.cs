using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Matrices;

namespace NeuralNetwork {
    class Program {
        //This is simply a main function to demonstrate the use of the network
        //Not to be included in a program which employs the network
        static void Main(string[] args) {
            //LogicGatesExample();
            //DecimalBinaryExample();
            MNISTExample();
        }

        public static void MNISTExample() {
            try {
                TrainingExample[] testExamples = GetTestExamples();
                TrainingExample[] trainingExamples = GetTrainingExamples();
                int[] layers = new int[] { 784, 400, 10 };
                FeedforwardNeuralNetwork fnn = new FeedforwardNeuralNetwork(layers, 0.5F);
                Console.WriteLine("Neural network constructed.");
                
                MNISTTestNetwork(fnn, testExamples);

                MNISTTrainNetwork(fnn, trainingExamples, testExamples);

                MNISTTestNetwork(fnn, testExamples);

            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
            }
        }

        public static void MNISTTestNetwork(FeedforwardNeuralNetwork fnn, TrainingExample[] testExamples) {
            try {
                Console.WriteLine("\nBeginning neural network test procedure.\n");
                int numSuccesses = 0;
                int numImages = testExamples.Length;
                int expectedNum = 0;
                for (int r = 0; r < numImages; ++r) {
                    for (int i = 1; i <= 10; ++i) {
                        if (testExamples[r].expectedOutput[i, 1] == 1) {
                            expectedNum = i % 10;
                        }
                    }
                    Matrix actualOutput = fnn.Evaluate(testExamples[r].input);
                    float largestOutputValue = 0;
                    int index = 0;
                    for (int j = 1; j <= 10; ++j) {
                        if (actualOutput[j, 1] > largestOutputValue) {
                            largestOutputValue = actualOutput[j, 1];
                            index = j;
                        }
                    }
                    index %= 10;
                    if (index == expectedNum) {
                        ++numSuccesses;
                    }

                    //Console.WriteLine("Test value: " + expectedNum);
                    //Console.WriteLine("Network detected: " + (index));


                    if ((r + 1) % 1000 == 0) {
                        Console.WriteLine((r + 1) / 100 + "00 test images processed.  Current success percentage: " + (float)numSuccesses/(r+1) * 100 + "%");
                    }
                }
                Console.WriteLine("\nTest regimen completed.  Network was correct for " + (float)numSuccesses / numImages * 100 + "% of " + numImages + " images.\n");

            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
            }
        }

        private static void MNISTTrainNetwork(FeedforwardNeuralNetwork fnn, TrainingExample[] trainingExamples, TrainingExample[] testExamples) {
            Console.WriteLine("\nBeginning neural network training procedure.\n");
            Console.WriteLine("Enter desired number of epochs.");
            string epochsStr = Console.ReadLine();
            int epochs = Convert.ToInt32(epochsStr);
            do {
                Console.WriteLine("Please wait for training cycle to complete...");
                var timer = Stopwatch.StartNew();
                fnn.TrainEpochs(trainingExamples, epochs);
                Console.WriteLine("Training cycle completed in " + timer.Elapsed + ".");
                Console.WriteLine("Test network? Enter Y for yes or anything else for no.");
                string inputStr = Console.ReadLine();
                char input = Convert.ToChar(inputStr);
                if (input == 'Y') {
                    MNISTTestNetwork(fnn, testExamples);
                }
                Console.WriteLine("Continue training? Enter 0 for no or a valid number of epochs for yes.");
                epochsStr = Console.ReadLine();
                epochs = Convert.ToInt32(epochsStr);
            } while (epochs > 0);
        }

        private static TrainingExample[] GetTestExamples() {
            try {
                Console.WriteLine("Searching for test datasets.");
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
                Console.WriteLine("Populating test examples.");
                TrainingExample[] testExamples = new TrainingExample[numImages];
                //List<TrainingExample> testExamples = new List<TrainingExample>();
                for (int r = 0; r < numImages; ++r) {
                    Matrix input = new Matrix(784, 1);
                    Matrix expectedOutput = new Matrix(10, 1);
                    for (int i = 1; i <= 784; ++i) {
                        byte b = imagesReader.ReadByte();
                        input[i, 1] = (float)b / (float)256;
                    }
                    int expectedNum = labelsReader.ReadByte();
                    if (expectedNum == 0) {
                        expectedOutput[10, 1] = 1;
                    } else {
                        expectedOutput[expectedNum, 1] = 1;
                    }
                    testExamples[r] = new TrainingExample(input, expectedOutput);
                    /*if ((expectedNum == 2 || expectedNum == 6)) {
                        testExamples.Add(new TrainingExample(input, expectedOutput));
                    }*/
                }
                Console.WriteLine("Test examples populated.");
                return testExamples.ToArray();

            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
                return null;
            }
        }

        private static TrainingExample[] GetTrainingExamples() {
            try {
                Console.WriteLine("Searching for training datasets.");
                FileStream labelsStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\train-labels.idx1-ubyte", FileMode.Open);
                FileStream imagesStream = new FileStream(@"E:\Users\Alexander Weaver\My Documents\Programs\MNIST\train-images.idx3-ubyte", FileMode.Open);
                Console.WriteLine("Training datasets found.");
                BinaryReader labelsReader = new BinaryReader(labelsStream);
                BinaryReader imagesReader = new BinaryReader(imagesStream);
                int magic1 = imagesReader.ReadInt32();
                int numImages = (imagesReader.ReadByte() << 24) | (imagesReader.ReadByte() << 16) | (imagesReader.ReadByte() << 8) | (imagesReader.ReadByte());
                int numRows = imagesReader.ReadInt32();
                int numColumns = imagesReader.ReadInt32();
                int magic2 = labelsReader.ReadInt32();
                int numLabels = labelsReader.ReadInt32();
                Console.WriteLine("Popluating training examples.");
                TrainingExample[] trainingExamples = new TrainingExample[numImages];
                //List<TrainingExample> trainingExamples = new List<TrainingExample>();
                for (int r = 0; r < numImages; ++r) {
                    Matrix input = new Matrix(784, 1);
                    Matrix expectedOutput = new Matrix(10, 1);
                    for (int i = 1; i <= 784; ++i) {
                        byte b = imagesReader.ReadByte();
                        input[i, 1] = (float)b / (float)256;
                    }
                    int expectedNum = labelsReader.ReadByte();
                    if (expectedNum == 0) {
                        expectedOutput[10, 1] = 1;
                    } else {
                        expectedOutput[expectedNum, 1] = 1;
                    }
                    trainingExamples[r] = new TrainingExample(input, expectedOutput);
                    /*if ((expectedNum == 2 || expectedNum == 6)) {
                        trainingExamples.Add(new TrainingExample(input, expectedOutput));
                    }*/
                }
                Console.WriteLine("Training examples populated.");
                return trainingExamples.ToArray();
            } catch (Exception ex) {
                Console.WriteLine(ex.ToString());
                return null;
            }
        }

        private static void displayImage(TrainingExample example) {
            for (int i = 1; i <= 784; ++i) {
                if (i - 1 % 28 == 0) {
                    Console.Write("\n");
                }
                Console.Write(example.input[i, 1] + ", ");
            }
            Console.Write("\n");
            string s = "";
            for (int i = 1; i <= 784; ++i) {
                
                
                if (example.input[i, 1] == 0) {
                    s += " ";
                } else if (example.input[i, 1] < 0.5F) {
                    s += ".";
                } else if (example.input[i, 1] <= 1.0F) {
                    s += "O";
                }
                if (i % 28 == 0) {
                    s += "\n";
                }
            }
            Console.WriteLine(s);
            Console.WriteLine("\n\n" + example.expectedOutput[1, 1] + " " + example.expectedOutput[2, 1] + " " + example.expectedOutput[3, 1] + " " + example.expectedOutput[4, 1] + " " + example.expectedOutput[5, 1] + " " + example.expectedOutput[6, 1] + " " + example.expectedOutput[7, 1] + " " + example.expectedOutput[8, 1] + " " + example.expectedOutput[9, 1] + " " + example.expectedOutput[10, 1] + " ");
        }

        public static void DecimalBinaryExample() {
            int[] layers = new int[] { 2, 5, 4 };
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
            TrainingExample[] examples = new TrainingExample[4];
            for (int i = 0; i < 4; ++i) {
                examples[i] = new TrainingExample(inputs[i], expectedOutputs[i]);
            }
            fnn.TrainEpochs(examples, 1000);
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
            inputs[1, 1] = 1;
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
