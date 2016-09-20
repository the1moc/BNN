using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Elements
{
	class Nodes : List<Node>
	{
		/// <summary>
		/// Gets or sets the network layer.
		/// </summary>
		/// <value>
		/// The network layer.
		/// </value>
		public LayerType NetworkLayer { get; set; }

		/// <summary>
		/// Initializes a new instance of the <see cref="Nodes"/> class.
		/// </summary>
		/// <param name="nodeCount">The number of nodes in this layer.</param>
		public Nodes(int nodeCount, LayerType layer)
		{
			NetworkLayer = layer;

			for (int i = 0; i < nodeCount; i ++)
			{
				this.Add(new Node());
			}
		}

		/// <summary>
		/// Initialises the weights for the Nodes in this layer.
		/// </summary>
		/// <param name="weightCount">The number of weights for each Node.</param>
		public void InitialiseWeights(int weightCount)
		{
			this.ForEach(node => node.InitialisRandomeWeights(weightCount));
		}

		/// <summary>
		/// Apply the sigmoid function to all the Nodes in this layer.
		/// </summary>
		public void ApplySigmoid()
		{
			this.ForEach(node => node.Sigmoid());
		}

		public void ClearInputs()
		{
			this.ForEach(node => node.ClearInputs());
		}
	}
}
