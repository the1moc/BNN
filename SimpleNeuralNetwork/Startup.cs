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
				new double[] { 0.0, 0.0 },
				new double[] { 1.0, 0.0 },
				new double[] { 0.0, 1.0 },
				new double[] { 1.0, 1.0 }
			};

			double[] targets = { 0.0, 1.0, 1.0, 0.0 };

			Network neuralNetwork = new Network(2, 2, 1, 0.7);

			for (int j = 0; j < 5000; j++)
			{
				for (int i = 0; i < inputs.Length; i++)
				{
					neuralNetwork.Train(inputs[i], targets[i]);
				}
			}

			Console.WriteLine(neuralNetwork.Probe(new double[] { 0.1, 0.9 }));
			Console.WriteLine(neuralNetwork.Probe(new double[] { 0.1, 0.1 }));
		}
	}
}
