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
			this.ForEach(node => node.InitialiseRandomWeights(weightCount));
		}

		/// <summary>
		/// Apply the sigmoid function to all the Nodes in this layer.
		/// </summary>
		public void ApplySigmoidToNodes()
		{
			this.ForEach(node => node.Sigmoid());
		}

        /// <summary>
        /// Clears the input for all of the nodes contained in this collection.
        /// </summary>
        public void ClearInputs()
		{
			this.ForEach(node => node.ClearInput());
		}

        /// <summary>
        /// Clears the output for all of the nodes contained in this collection.
        /// </summary>
        public void ClearOutputs()
        {
            this.ForEach(node => node.ClearOutput());
        }

        /// <summary>
        /// Multiplies each nodes weights by all outgoing weights for each of the nodes in this collection.
        /// </summary>
        public void MultiplyWeightsByInput()
        {
            this.ForEach(node => node.MultiplyByWeights());
        }
	}
}
