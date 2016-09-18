using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork.Elements;

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

			double[][] targets = new double[][]{
				new double[] { 0.01 },
				new double[] { 0.99 },
				new double[] { 0.99 },
				new double[] { 0.01 }
			};

			Network neuralNetwork = new Network(2, 2, 1, 0.3);

			for (int j = 0; j < 400; j++)
			{
				for (int i = 0; i < inputs.Length; i++)
				{
					neuralNetwork.Train(inputs[i], targets[i]);
				}
			}
			
		}
	}
}
