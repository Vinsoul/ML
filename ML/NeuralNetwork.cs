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
        public List<Matrix<double>> Weights;
        public List<Vector<double>> Biases;

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
                result = Sigmoid(result * Weights[i] + Biases[i]);

            return result;
        }
        
        private Vector<double> Sigmoid(Vector<double> z)
        {
            Vector<double> one = Vector<double>.Build.Dense(z.Count, 1);
            z = -z;
            return one.PointwiseDivide(one + z.PointwiseExp());
        }

        public void Learn(List<List<double>> inputs, List<List<double>> expectedOutputs, double learningRate)
        {
            double prevCost = AverageCost(inputs, expectedOutputs);
            double currentCost = 0;
            while (Math.Abs(prevCost - currentCost) >= 0.1 * learningRate * learningRate)
            {
                prevCost = currentCost;
                Shuffle(inputs, expectedOutputs);
                for (int inputIndex = 0; inputIndex < inputs.Count; inputIndex++)
                {
                    Vector<double> input = Vector<double>.Build.DenseOfEnumerable(inputs[inputIndex]);
                    Vector<double> expected = Vector<double>.Build.DenseOfEnumerable(expectedOutputs[inputIndex]);

                    List<Vector<double>> neuronsValues = new List<Vector<double>>(Weights.Count + 1);

                    for (int i = 0; i < Weights.Count; i++)
                    {
                        input = Sigmoid(input * Weights[i] + Biases[i]);
                        neuronsValues.Add(input);
                    }

                    for (int L = neuronsValues.Count - 1; L >= 1; L--)
                    {
                        // Adjusting weights
                        for (int j = 0; j < Weights[L].RowCount; j++)
                        {
                            for (int k = 0; k < Weights[L].ColumnCount; k++)
                            {
                                double dC_dWjk = neuronsValues[L - 1][j] * neuronsValues[L][k] * (1 - neuronsValues[L][k]) * (neuronsValues[L][k] - expected[k]);
                                Weights[L][j, k] = Weights[L][j, k] - learningRate * dC_dWjk;
                            }
                        }

                        // Adjusting biases
                        for (int k = 0; k < Biases[L].Count; k++)
                        {
                            double dC_dBj = neuronsValues[L][k] * (1 - neuronsValues[L][k]) * (neuronsValues[L][k] - expected[k]);
                            Biases[L][k] = Biases[L][k] - learningRate * dC_dBj;
                        }

                        // Adjusting neuron values
                        if (L != 0)
                        {
                            expected = Vector<double>.Build.Dense(neuronsValues[L - 1].Count);
                            for (int k = 0; k < neuronsValues[L - 1].Count; k++)
                            {
                                double dC_dAk = 0;
                                for (int j = 0; j < neuronsValues[L].Count; j++)
                                {
                                    dC_dAk += Weights[L][k, j] * neuronsValues[L][j] * (1 - neuronsValues[L][j]) * (neuronsValues[L][j] - expected[j]);
                                }

                                expected[k] = neuronsValues[L - 1][k] - learningRate * dC_dAk;
                            }
                        }
                    }
                }
                currentCost = AverageCost(inputs, expectedOutputs);
            }
        }

        private double Cost(Vector<double> expected, Vector<double> actual)
        {
            return 0.5 * (expected - actual).PointwisePower(2).Sum();
        }

        private double AverageCost(List<List<double>> inputs, List<List<double>> expected)
        {
            double sum = 0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += Cost(Vector<double>.Build.DenseOfEnumerable(expected[i]), Vector<double>.Build.DenseOfArray(GetValue(inputs[i])));
            }

            return sum / expected.Count;
        }

        private void Shuffle<T>(List<T> inputs, List<T> expected)
        {
            Random rand = new Random();
            for (int i = 0; i < inputs.Count; i++)
            {
                int j = rand.Next(i);
                T buffer = inputs[j];
                inputs[j] = inputs[i];
                inputs[i] = buffer;

                buffer = expected[j];
                expected[j] = expected[i];
                expected[i] = buffer;
            }
        }
    }
}
