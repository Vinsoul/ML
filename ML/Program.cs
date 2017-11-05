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
            NeuralNetwork nn = new NeuralNetwork(2, new[] { 3, 1 });
            foreach (double result in nn.GetValue(new double[] { 1, 1 }))
                Console.WriteLine(result);
        }
    }
}
