using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Nodes
{
	class Neurons : List<Neuron>
	{
		/// <summary>
		/// Gets or sets the network layer.
		/// </summary>
		/// <value>
		/// The network layer.
		/// </value>
		public Layer NetworkLayer { get; set; }

		/// <summary>
		/// Initializes a new instance of the <see cref="Neurons"/> class.
		/// </summary>
		/// <param name="nodeCount">The number of nodes in this layer.</param>
		public Neurons(int nodeCount, Layer layer)
		{
			NetworkLayer = layer;

			for (int i = 0; i < nodeCount; i ++)
			{
				this.Add(new Neuron());
			}
		}

		/// <summary>
		/// Initialises the weights for the neurons in this layer.
		/// </summary>
		/// <param name="weightCount">The number of weights for each neuron.</param>
		public void InitialiseWeights(int weightCount)
		{
			foreach(Neuron neuron in this)
			{
				neuron.InitialiseWeights(weightCount);
			}
		}

		/// <summary>
		/// Apply the sigmoid function to all the neurons in this layer.
		/// </summary>
		public void ApplySigmoid()
		{
			foreach(Neuron neuron in this)
			{
				neuron.Sigmoid();
			}
		}
	}
}
