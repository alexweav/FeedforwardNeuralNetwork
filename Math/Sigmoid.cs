using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Matrices;

namespace NeuralNetwork {
    public class Sigmoid {

        //For a real number n, the sigmoid of n is defined as 1/(1+e^(-n))
        public static float Sigma(float n) {
            return 1/(1 + (float)Math.Exp(-n));
        }

        //For a matrix a, the sigmoid of a is defined as the matrix for which each element sigmoid(a)[i, j] = sigmoid(a[i, j])
        //Applies the sigmoid function to every value in a matrix and returns the new matrix
        public static Matrix Sigma(Matrix a) {
            if (a == null) {
                throw new NullReferenceException("Cannot take the sigmoid of a null matrix.");
            }
            Matrix output = new Matrix(a.NumRows, a.NumColumns);
            for (int i = 1; i <= output.NumRows; ++i) {
                for (int j = 1; j <= output.NumColumns; ++j) {
                    output[i, j] = Sigma(a[i, j]);
                }
            }
            return output;
        }

        //For a real number n, returns the derivative at n of the sigmoid function
        //Derivative of Sigma(n) at the point n
        public static float SigmaPrime(float n) {
            float sigmoid = Sigma(n);
            return sigmoid * (1 - sigmoid);
        }

        //For a matrix a, the sigmoid prime of a is defined as the matrix for which each element SigmoidPrime(a)[i, j] = SigmoidPrime(a[i, j]);
        //Applies the sigmoid prime function to every value in a matrix and returns the new matrix
        public static Matrix SigmaPrime(Matrix a) {
            if (a == null) {
                throw new NullReferenceException("Cannot take the sigmoid of a null matrix.");
            }
            Matrix output = new Matrix(a.NumRows, a.NumColumns);
            for (int i = 1; i <= output.NumRows; ++i) {
                for (int j = 1; j <= output.NumColumns; ++j) {
                    float sigmoid = Sigma(a[i, j]);
                    output[i, j] = sigmoid * (1 - sigmoid);
                }
            }
            return output;
        }
    }
}
