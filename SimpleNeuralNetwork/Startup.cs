using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork.Elements;
using System.Threading;

namespace SimpleNeuralNetwork
{
	static class Startup
	{
		static void Main()
		{
			double[][] inputs = new double[][]{
				new double[] { 0.01, 0.01 },
				new double[] { 0.99, 0.01 },
				new double[] { 0.01, 0.99 },
				new double[] { 0.99, 0.99 }
			};

			double[] targets = { 0.01, 0.99, 0.99, 0.01 };

			Network neuralNetwork = new Network(2, 3, 1, 0.3);

			for (int j = 0; j < 50000; j++)
			{
				for (int i = 0; i < inputs.Length; i++)
				{
					neuralNetwork.Train(inputs[i], targets[i]);
				}
			}
			Console.WriteLine("True: {0}", neuralNetwork.Probe(new double[] { 0.01, 0.99 }));
			Console.WriteLine("False: {0}", neuralNetwork.Probe(new double[] { 0.99, 0.99 }));
			Console.Read();
		}
	}
}
