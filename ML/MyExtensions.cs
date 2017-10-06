using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Security.Cryptography;

namespace ML
{
    static class MyExtensions
    {
        public static void ShuffleTrainingSet<T>(this IList<T> input, IList<T> output)
        {
            RNGCryptoServiceProvider random = new RNGCryptoServiceProvider();

            int n = input.Count;
            while (n-- > 0)
            {
                byte[] bytes = new byte[4];
                random.GetBytes(bytes);
                int j = BitConverter.ToInt32(bytes, 0) % n;

                T buffer = input[n];
                input[n] = input[j];
                input[j] = buffer;

                buffer = output[n];
                output[n] = output[j];
                output[j] = buffer;
            }
        }
    }
}
