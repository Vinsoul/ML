using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.CompilerServices;

namespace ML
{
    class Network
    {
        public int Layers { set; get; }
        public List<int> Sizes { set; get; }
        public List<Vector<double>> Biases { set; get; }
        public List<Matrix<double>> Weights { set; get; }

        /// <summary>
        /// Initializes NN with sizes.Count layers, each layer with sizes[i] neurons
        /// </summary>
        /// <param name="sizes"></param>
        public Network(List<int> sizes)
        {
            Layers = sizes.Count;
            Sizes = sizes;

            Biases = new List<Vector<double>>(Layers);
            Weights = new List<Matrix<double>>(Layers);

            for (int i = 1; i < Layers; i++)
            {
                Biases.Add(Vector<double>.Build.Random(sizes[i]));
                Weights.Add(Matrix<double>.Build.Random(sizes[i], sizes[i - 1]));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector<double> NextLayer(Vector<double> input, Matrix<double> weights, Vector<double> biases)
        {
            return Sigmoid(weights.Multiply(input).Add(biases));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector<double> Sigmoid(Vector<double> x)
        {
            Vector<double> one = Vector<double>.Build.Dense(x.Count, 1);
            x = x.PointwiseExp().PointwisePower(-1);
            x = x.Add(1);
            one = one.PointwiseDivide(x);

            return one;
        }

        public Vector<double> FeedForward(Vector<double> input)
        {
            for (int i = 1; i < Layers; i++)
                input = NextLayer(input, Weights[i-1], Biases[i-1]);

            return input;
        }

        public void StohasticGradientDescent(List<Vector<double>> inputs, List<Vector<double>> expected, int epochs, double learningRate)
        {
            if (inputs.Count != expected.Count)
                throw new ArgumentException("inputs and expected have different number of vectors");

            while (epochs-- > 0)
            {
                inputs.ShuffleTrainingSet(expected);
                for (int i = 0; i < inputs.Count; i++)
                {

                }
            }
        }

    }
}
