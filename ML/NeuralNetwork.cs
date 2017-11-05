using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ML
{
    class NeuralNetwork
    {
        int InputSize;
        List<Matrix<double>> Weights;
        List<Vector<double>> Biases;

        public NeuralNetwork(int inputSize, IEnumerable<int> layers)
        {
            InputSize = inputSize;
            Weights = new List<Matrix<double>>();
            Biases = new List<Vector<double>>();

            int prevColumns = inputSize;
            foreach (int neuronsCount in layers)
            {
                Weights.Add(Matrix<double>.Build.Random(prevColumns, neuronsCount));
                Biases.Add(Vector<double>.Build.Random(neuronsCount));

                prevColumns = neuronsCount;
            }
        }

        public double[] GetValue(IEnumerable<double> input)
        {
            Vector<double> result = Vector<double>.Build.DenseOfEnumerable(input);

            return GetValue(result).ToArray();
        }

        public Vector<double> GetValue(Vector<double> input)
        {
            Vector<double> result = input;
            for (int i = 0; i < Weights.Count; i++)
                result = result * Weights[i] + Biases[i];

            return result;
        }
        
        private Matrix<double> Sigmoid(Matrix<double> z)
        {
            Matrix<double> one = Matrix<double>.Build.Dense(z.RowCount, z.ColumnCount, 1);
            z = -z;
            return one.PointwiseDivide(one + z.PointwiseExp());
        }

        public void Learn(IEnumerable<double> input, IEnumerable<double> expectedOutput)
        {

        }

        private double Cost(Vector<double> expected, Vector<double> actual)
        {
            return 0.5 * (expected - actual).PointwisePower(2).Sum();
        }
    }
}
