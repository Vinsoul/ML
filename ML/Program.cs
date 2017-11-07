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
            nn.Learn(inputs, expected, 0.001);

            int total = 0;
            int success = 0;
            for (double i = 0; i <= 1; i += 0.001)
            {

                double[] result = nn.GetValue(new double[] { i });
                if (i <= 0.5)
                {
                    if (result[0] > result[1])
                        success++;
                    else
                        Console.WriteLine($"{i} {result[0]} {result[1]}");
                }
                else
                {
                    if (result[0] < result[1])
                        success++;
                    else
                        Console.WriteLine($"{i} {result[0]} {result[1]}");
                }
                total++;
            }

            Console.WriteLine(success + " " + total);
        }

        static List<List<double>> inputs = new List<List<double>>()
        {
            { new List<double>() { 0.1 } },
            { new List<double>() { 0.2 } },
            { new List<double>() { 0.3 } },
            { new List<double>() { 0.4 } },
            { new List<double>() { 0.5 } },
            { new List<double>() { 0.6 } },
            { new List<double>() { 0.7 } },
            { new List<double>() { 0.8 } },
            { new List<double>() { 0.9 } }
        };

        static List<List<double>> expected = new List<List<double>>()
        {
            { new List<double>() { 1, 0 } },
            { new List<double>() { 1, 0 } },
            { new List<double>() { 1, 0 } },
            { new List<double>() { 1, 0 } },
            { new List<double>() { 0, 1 } },
            { new List<double>() { 0, 1 } },
            { new List<double>() { 0, 1 } },
            { new List<double>() { 0, 1 } },
            { new List<double>() { 0, 1 } }
        };
    }
}
