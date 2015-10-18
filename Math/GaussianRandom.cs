using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    public class GaussianRandom {

        private Random random;
        private double mean;
        private double stdDev;

        //Creates the Gaussian random value generator set to the given mean and standard deviation
        public GaussianRandom(double mean, double stdDev) {
            random = new Random();
            this.mean = mean;
            this.stdDev = stdDev;
        }
        
        //Creates the Gaussian random value generator set to the given mean and standard deviation with a specified seed
        public GaussianRandom(int seed, double mean, double stdDev) {
            random = new Random(seed);
            this.mean = mean;
            this.stdDev = stdDev;
        }

        public double Mean {
            get {
                return mean;
            }
            set {
                mean = value;
            }
        }

        public double StdDev {
            get {
                return stdDev;
            }
            set {
                stdDev = value;
            }
        }
        
        //Returns a random double that follows the set Gaussian distribution
        public double NextDouble() {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
        
    }
}
