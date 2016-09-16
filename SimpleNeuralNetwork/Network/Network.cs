using SimpleNeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork
{
	class Network
	{
		/// <summary>
		/// The input count for the network.
		/// </summary>
		private readonly int _inputNeuronCount;

		/// <summary>
		/// The number of hidden neurons.
		/// </summary>
		private readonly int _hiddenNeuronCount;

		/// <summary>
		/// The number of output neurons.
		/// </summary>
		private readonly int _outputNeuronCount;

		/// <summary>
		/// The learning rate of the network.
		/// </summary>
		private readonly double _learningRate;

		/// <summary>
		/// The input neurons.
		/// </summary>
		private Neurons _inputNeurons;

		/// <summary>
		/// The hidden neurons.
		/// </summary>
		private Neurons _hiddenNeurons;

		/// <summary>
		/// The output neurons.
		/// </summary>
		private Neurons _outputNeurons;

		/// <summary>
		/// Initializes a new instance of the <see cref="Network"/> class.
		/// </summary>
		/// <param name="inputCount">The input count.</param>
		/// <param name="hiddenNeuronCount">The hidden neuron count.</param>
		/// <param name="outputNeuronCount">The output neuron count.</param>
		/// <param name="learningRate">The learning rate.</param>
		public Network(int inputCount, int hiddenNeuronCount, int outputNeuronCount, double learningRate)
		{
			_inputNeuronCount  = inputCount;
			_hiddenNeuronCount = hiddenNeuronCount;
			_outputNeuronCount = outputNeuronCount;
			_learningRate      = learningRate;

			CreateNeuronLayers();
		}

		/// <summary>
		/// Query the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="targetOutput">The target output.</param>
		public double[] Probe(double[] inputs, double targetOutput)
		{
			// Set the input neurons output to be the data inputs.
			// No need for the input to be set
			_inputNeurons.Select((neuron, index) => neuron.NeuronOutput = inputs[index]);

			Console.WriteLine(_inputNeurons);

			// Input layer -> Hidden layer
			foreach (Neuron neuron in _inputNeurons)
			{
				double[] multipliedInputs = MultiplyByWeights(neuron.Weights, neuron.NeuronOutput);

				for (int i = 0; i < multipliedInputs.Length; i++)
				{
					_hiddenNeurons.ElementAt(i).NeuronInput += multipliedInputs[i];
				}
			}

			// Apply the sigmoid function to each of the hidden layer neurons
			_hiddenNeurons.Select(neuron => neuron.NeuronOutput = Sigmoid.SigmoidFunction(neuron.NeuronInput));

			// Hidden layer -> Output layer
			foreach (Neuron neuron in _hiddenNeurons)
			{
				double[] multipliedInputs = MultiplyByWeights(neuron.Weights, neuron.NeuronOutput);

				for (int i = 0; i < multipliedInputs.Length; i++)
				{
					_outputNeurons.ElementAt(i).NeuronInput += multipliedInputs[i];
				}
			}

			// Apply the sigmoid function to each of the hidden layer neurons
			_outputNeurons.Select(neuron => neuron.NeuronOutput = Sigmoid.SigmoidFunction(neuron.NeuronInput));

			return _outputNeurons.Select(neuron => neuron.NeuronOutput).ToArray();
		}

		/// <summary>
		/// Feed the inputs through the network.
		/// </summary>
		/// <param name="weights">The weights.</param>
		public double[] MultiplyByWeights(double[] weights, double input)
		{
			return weights.Select(weight => weight * input).ToArray();
		}

		/// <summary>
		/// Creates the neurons and initialise their weights.
		/// </summary>
		private void CreateNeuronLayers()
		{
			// Create the neuron layers and give them some random initial weights
			_inputNeurons  = new Neurons(_inputNeuronCount, Layer.Input);
			_inputNeurons.InitialiseWeights(_hiddenNeuronCount);

			_hiddenNeurons = new Neurons(_hiddenNeuronCount, Layer.Hidden);
			_hiddenNeurons.InitialiseWeights(_outputNeuronCount);

			// No weights are needed for the output neurons
			_outputNeurons = new Neurons(_outputNeuronCount, Layer.Output);
		}

	}
}
