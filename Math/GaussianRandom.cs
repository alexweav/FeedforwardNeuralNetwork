using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork {
    class GaussianRandom {

        Random random;
        double mean;
        double stdDev;

        //Creates the Gaussian random value generator set to the given mean and standard deviation
        public void GaussianRandom(double mean, double stdDev) {
            random = new Random();
            this.mean = mean;
            this.stdDev = stdDev;
        }
        
        //Creates the Gaussian random value generator set to the given mean and standard deviation with a specified seed
        public void GaussianRandom(int seed, double mean, double stdDev) {
            random = new Random(seed);
            this.mean = mean;
            this.stdDev = stdDev;
        }
        
        //Returns a random double that follows the set Gaussian distribution
        private double NextDouble() {
            double u1 = random.NextDouble();
            double u2 = random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
        
    }
}
