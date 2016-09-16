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
		/// Initialises the weights for this neuron.
		/// </summary>
		/// <param name="weightCount">The number of weights this neuron will have.</param>
		public void InitialiseWeights(int weightCount)
		{
			Weights    = new double[weightCount];
			Random rng = new Random();

			for(int i = 0; i < weightCount; i++)
			{
				Weights[i] = rng.NextDouble() - 0.5;
			}
		}
	}
}
