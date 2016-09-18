using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork;

namespace SimpleNeuralNetwork.Elements
{
	class Node
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="Node"/> class.
		/// </summary>
		public Node()
		{
		}

		/// <summary>
		/// Gets or sets the Node output.
		/// </summary>
		/// <value>
		/// The Node output.
		/// </value>
		public double NodeOutput { get; set; }

		/// <summary>
		/// Gets or sets the Node input.
		/// </summary>
		/// <value>
		/// The Node input.
		/// </value>
		public double NodeInput { get; set; }

		/// <summary>
		/// Gets or sets the weights coming from this Node.
		/// </summary>
		/// <value>
		/// The weights.
		/// </value>
		public double[] Weights { get; set; }

		/// <summary>
		/// Gets or sets the bias.
		/// </summary>
		/// <value>
		/// The bias.
		/// </value>
		public double Bias { get; set; }

		/// <summary>
		/// Initialises the random weights for this Node.
		/// </summary>
		/// <param name="weightCount">The number of weights this Node will have.</param>
		public void InitialisRandomeWeights(int weightCount)
		{
			Weights    = new double[weightCount];
			Random rng = new Random();

			Bias = 0.1;

			for(int i = 0; i < weightCount; i++)
			{
				// On large input sets, stop the result becoming too high and nullyfing the sigmoid function.
				Weights[i] = rng.NextDouble() - (1.0/2.0);
			}
		}

		/// <summary>
		/// Applies the sigmoid function to the input of this Node.
		/// </summary>
		public void Sigmoid()
		{
			this.NodeOutput = SimpleNeuralNetwork.Sigmoid.SigmoidFunction(this.NodeInput);
		}

		/// <summary>
		/// Applies the derivative of the sigmoid function to the output of this Node.
		/// </summary>
		public double SigmoidDerivative()
		{
			return SimpleNeuralNetwork.Sigmoid.SigmoidFunction(this.NodeOutput);
		}
	}
}
