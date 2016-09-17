using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Nodes
{
	class Neuron
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="Neuron"/> class.
		/// </summary>
		public Neuron()
		{
		}

		/// <summary>
		/// Gets or sets the neuron output.
		/// </summary>
		/// <value>
		/// The neuron output.
		/// </value>
		public double NeuronOutput { get; set; }

		/// <summary>
		/// Gets or sets the neuron input.
		/// </summary>
		/// <value>
		/// The neuron input.
		/// </value>
		public double NeuronInput { get; set; }

		/// <summary>
		/// Gets or sets the weights coming from this neuron.
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
		/// Initialises the random weights for this neuron.
		/// </summary>
		/// <param name="weightCount">The number of weights this neuron will have.</param>
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
		/// Applies the sigmoid function to the input of this neuron.
		/// </summary>
		public void Sigmoid()
		{
			this.NeuronOutput = SimpleNeuralNetwork.Sigmoid.SigmoidFunction(this.NeuronInput);
		}

		/// <summary>
		/// Applies the derivative of the sigmoid function to the output of this neuron.
		/// </summary>
		public double SigmoidDerivative()
		{
			return SimpleNeuralNetwork.Sigmoid.SigmoidFunction(this.NeuronOutput);
		}
	}
}
