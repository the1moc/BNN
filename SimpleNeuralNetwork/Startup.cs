using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork.Elements;
using System.Threading;
using System.IO;
using SimpleNeuralNetwork.FileProcessing;

namespace SimpleNeuralNetwork
{
	static class Startup
	{
		static void Main()
		{

			StreamReader reader = new StreamReader(@"C:\Users\malcolm-campbell\Documents\Final year stuff\mnist_train.csv");

			Network neuralNetwork = new Network(784, 200, 10, 0.4);

			int counter = 0;

			while (!reader.EndOfStream && counter < 500)
			{
				string line = reader.ReadLine();

				double[] targets          = DataHelper.GenerateTargets(line);
				double[] normalizedInputs = DataHelper.NormaliseData(line);

				neuralNetwork.Train(normalizedInputs, targets);

				counter++;
			}

			reader = new StreamReader(@"C:\Users\malcolm-campbell\Documents\Final year stuff\mnist_test.csv");

			counter = 0;
			while (!reader.EndOfStream && counter < 20)
			{
				string line = reader.ReadLine();

				double[] normalizedInputs = DataHelper.NormaliseData(line);

				Console.WriteLine("Target: {0}", line[0]);

				double[] test = neuralNetwork.Probe(normalizedInputs);

				for (int i = 0; i < test.Length; i++)
				{
					Console.WriteLine("Number {0}: {1}", i, test[i]);
				}
				counter++;
			}

			reader.Dispose();

			Console.Read();
		}
	}
}
