using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(1, new[] { 3, 2 });
            for (int i = 0; i < 10000; i++)
            {
                for (double x = 0; x <= 1; x += 0.01)
                {
                    double[] expected = Math.Round(x) == (int)x ? new[] { 1.0, 0.0 } : new[] { 0.0, 1.0 };
                    nn.Learn(new double[] { x - (int)x }, expected, 0.001);
                }
            }
            double[] test = new[] { 0.95 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.15 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.25 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.35 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.45 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.55 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
            test = new[] { 0.85 };
            foreach (double result in nn.GetValue(test))
                Console.WriteLine(result);
        }
    }
}
