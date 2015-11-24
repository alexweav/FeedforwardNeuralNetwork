using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Matrices;

namespace NeuralNetwork {
    class TrainingExample {

        public readonly Matrix input;
        public readonly Matrix expectedOutput;

        public TrainingExample(Matrix input, Matrix expectedOutput) {
            this.input = input;
            this.expectedOutput = expectedOutput;
        }
    }
}
