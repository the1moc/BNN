using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork.Nodes;

namespace SimpleNeuralNetwork
{
	static class Startup
	{
		static void Main()
		{
			double[][] inputs = new double[][]{
				new double[] { 0, 0 },
				new double[] { 1, 0 },
				new double[] { 0, 1 },
				new double[] { 1, 1 }
			};

			double[] targets = { 0, 1, 1, 0 };

			Network neuralNetwork = new Network(2, 2, 1, 0.5);

			for (int i = 0; i < inputs.Length; i++)
			{
				neuralNetwork.Train(inputs[i], targets[i]);
			}
		}
	}
}
