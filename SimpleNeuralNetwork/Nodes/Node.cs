using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SimpleNeuralNetwork.Sigmoid;

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
		/// Gets or sets the node output.
		/// </summary>
		/// <value>
		/// The node output.
		/// </value>
		public double NodeOutput { get; set; }

		/// <summary>
		/// Gets or sets the node input.
		/// </summary>
		/// <value>
		/// The node input.
		/// </value>
		public double NodeInput { get; set; }

		/// <summary>
		/// Gets or sets the weights coming from this node.
		/// </summary>
		/// <value>
		/// The weights.
		/// </value>
		public double[] Weights { get; set; }

		/// <summary>
		/// Gets or sets each of the weights multiplied by the nodes input.
		/// </summary>
		/// <value>
		/// The multiplied input by weights.
		/// </value>
		public double[] MultipliedInputByWeights { get; set; }

		/// <summary>
		/// Gets or sets the bias.
		/// </summary>
		/// <value>
		/// The bias.
		/// </value>
		public double Bias { get; set; }

		/// <summary>
		/// Gets or sets the error on this node.
		/// </summary>
		/// <value>
		/// The error.
		/// </value>
		public double Error { get; set; }

		/// <summary>
		/// Initialises the random weights for this node.
		/// </summary>
		/// <param name="weightCount">The number of weights this node will have.</param>
		public void InitialiseRandomWeights(int weightCount)
		{
			Weights    = new double[weightCount];

			for(int i = 0; i < weightCount; i++)
			{
				// On large input sets, stop the result becoming too high and nullyfing the sigmoid function.
				Weights[i] = Util.NumberUtils.GetRandomDouble();
			}
		}

		/// <summary>
		/// Applies the sigmoid function to the input of this node.
		/// </summary>
		public void Sigmoid()
		{
			this.NodeOutput = SigmoidFunctions.Sigmoid(this.NodeInput);
		}

		/// <summary>
		/// Applies the derivative of the sigmoid function to the output of this node.
		/// </summary>
		public double SigmoidDerivative()
		{
			return SigmoidFunctions.SigmoidDerivative(this.NodeOutput);
		}

		/// <summary>
		/// Clears the input for this node (set it back to 0).
		/// </summary>
		public void ClearInput()
		{
			NodeInput = 0;
		}

		/// <summary>
		/// Clears the input for this node (set it back to 0).
		/// </summary>
		public void ClearOutput()
		{
			NodeOutput = 0;
		}

		/// <summary>
		/// Multiply the input of this node by all the outgoing weights.
		/// </summary> 
		/// <param name="weights">The weights.</param>
		public void MultiplyByWeights()
		{
			MultipliedInputByWeights = Weights.Select(weight => weight * (NodeInput == 0 ? NodeOutput : NodeInput)).ToArray();
		}
	}
}
