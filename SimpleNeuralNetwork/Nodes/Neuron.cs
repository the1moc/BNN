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
	}
}
