using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Network
{
	class Network
	{
		/// <summary>
		/// The input count for the network.
		/// </summary>
		private int _inputCount;

		/// <summary>
		/// The number of hidden neurons.
		/// </summary>
		private int _hiddenNeuronCount;

		/// <summary>
		/// The number of output neurons.
		/// </summary>
		private int _outputNeuronCount;

		/// <summary>
		/// The learning rate of the network.
		/// </summary>
		private double _learningRate;

		/// <summary>
		/// Initializes a new instance of the <see cref="Network"/> class.
		/// </summary>
		/// <param name="inputCount">The input count.</param>
		/// <param name="hiddenNeuronCount">The hidden neuron count.</param>
		/// <param name="outputNeuronCount">The output neuron count.</param>
		/// <param name="learningRate">The learning rate.</param>
		public Network(int inputCount, int hiddenNeuronCount, int outputNeuronCount, double learningRate)
		{
			_inputCount        = inputCount;
			_hiddenNeuronCount = hiddenNeuronCount;
			_outputNeuronCount = outputNeuronCount;
			_learningRate      = learningRate;
		}

	}
}
