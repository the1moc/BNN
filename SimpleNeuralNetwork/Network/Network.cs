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
		/// A list of all the layers in this network.
		/// </summary>
		private Layers _layerList;

		/// <summary>
		/// Have the weights been generated already?
		/// </summary>
		private bool _areWeightsInitialised;

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
		public string Probe(double[] inputs)
		{
			SetInitialInputs(inputs);
			FeedForward();

			// Get the last layer (output) and return a string of the output values.
			return String.Join(",", _layerList.ElementAt(_layerList.Count - 1).Select(neuron => neuron.NeuronOutput.ToString()));
		}

		/// <summary>
		/// Train the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="targetOutput">The target output.</param>
		public void Train(double[] inputs, double target)
		{
			SetInitialInputs(inputs);
			FeedForward();

			BackPropogate(target);
		}

		/// <summary>
		/// Feeds the input forward through the network.
		/// </summary>
		/// <returns></returns>
		public void FeedForward()
		{
			// Input layer -> Hidden layer
			FeedForwardRoot(Layer.Input, Layer.Hidden);

			// Hidden layer -> Output layer
			FeedForwardRoot(Layer.Hidden, Layer.Output);
		}

		/// <summary>
		/// Perform all calculations when passing values from one layer to another.
		/// </summary>
		/// <param name="sending">The sending.</param>
		/// <param name="receiving">The receiving.</param>
		public void FeedForwardRoot(Layer sending, Layer receiving)
		{
			foreach (Neuron sendingNeuron in _layerList.GetLayer(sending))
			{
				// Multiply the neuron output by each of its weights.
				// The first element matches the first node position in the receiving layer, etc etc.
				double[] multipliedInputs = MultiplyByWeights(sendingNeuron.Weights, sendingNeuron.NeuronOutput);

				// Index that matches the receiving layers node position
				int receivingNeuronIndex = 0;

				foreach (Neuron receivingNeuron in _layerList.GetLayer(receiving))
				{
					receivingNeuron.NeuronInput += multipliedInputs[receivingNeuronIndex];
					receivingNeuronIndex++;
				}

				// Apply the sigmoid function to each of neurons in this layer.
				_layerList.GetLayer(receiving).ApplySigmoid();
			}
		}

		/// <summary>
		/// Backpropogate the error through the network.
		/// </summary>
		/// <param name="target">The target output.</param>
		private void BackPropogate(double target)
		{
			// The amount each weight will be changing
			double[,] hiddenToOutputWeightChanges = new double[_hiddenNeuronCount, _outputNeuronCount];
			double[,] inputToHiddenWeightChanges  = new double[_inputNeuronCount, _hiddenNeuronCount];

			int outputNeuronIndex = 0;
			int hiddenNeuronIndex = 0;

			// Populate the weight changes array with the amount the weights will be effected
			foreach (Neuron outputNeuron in _layerList.GetLayer(Layer.Output))
			{
				Console.WriteLine(outputNeuron.NeuronOutput);

				double errorAgainstNetInput = -(target - outputNeuron.NeuronOutput) * outputNeuron.SigmoidDerivative();

				foreach(Neuron hiddenNeuron in _layerList.GetLayer(Layer.Hidden))
				{
					double netInputAgainstWeight                                      = hiddenNeuron.NeuronOutput;
					hiddenToOutputWeightChanges[hiddenNeuronIndex, outputNeuronIndex] = errorAgainstNetInput * netInputAgainstWeight;

					hiddenNeuronIndex++;
				}

				outputNeuronIndex++;
			}

			hiddenNeuronIndex = 0;
			outputNeuronIndex = 0;

			// Populate the weight changes array with the amount the weights will be effected
			foreach (Neuron hiddenNeuron in _layerList.GetLayer(Layer.Hidden))
			{
				double totalErrorAgainstHiddenNeuron = 0.0;
				outputNeuronIndex                    = 0;
				
				foreach (Neuron outputNeuron in _layerList.GetLayer(Layer.Output))
				{
					double errorAgainstNetInput = -(target - outputNeuron.NeuronOutput) * outputNeuron.SigmoidDerivative();

					totalErrorAgainstHiddenNeuron += errorAgainstNetInput * hiddenNeuron.Weights[outputNeuronIndex];

					outputNeuronIndex++;
				}

				double outputAgainstInput = hiddenNeuron.SigmoidDerivative();

				int inputNeuronIndex = 0;

				foreach(Neuron inputNeuron in _layerList.GetLayer(Layer.Input))
				{
					double inputAgainstWeight = inputNeuron.NeuronOutput;
					inputToHiddenWeightChanges[inputNeuronIndex, hiddenNeuronIndex] = totalErrorAgainstHiddenNeuron * outputAgainstInput * inputAgainstWeight;

					inputNeuronIndex++;
				}

				hiddenNeuronIndex++;
			}

			UpdateWeights(inputToHiddenWeightChanges, hiddenToOutputWeightChanges);
		}

		/// <summary>
		/// Update the network weights..
		/// </summary>
		/// <returns></returns>
		private void UpdateWeights(double[,] inputWeights, double[,] hiddenWeights)
		{
			int inputNeuronIndex = 0;
			foreach(Neuron inputNeuron in _layerList.GetLayer(Layer.Input))
			{
				for(int i = 0; i < inputNeuron.Weights.Length; i++)
				{
					inputNeuron.Weights[i] -= _learningRate * inputWeights[inputNeuronIndex,i];
				}
			}

			int hiddenNeuronIndex = 0;
			foreach (Neuron hiddenNeuron in _layerList.GetLayer(Layer.Hidden))
			{
				for (int i = 0; i < hiddenNeuron.Weights.Length; i++)
				{
					hiddenNeuron.Weights[i] -= _learningRate * hiddenWeights[hiddenNeuronIndex, i];
				}
			}
		}

		/// <summary>
		/// Give the input layer of neurons their output data.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		public void SetInitialInputs(double[] inputs)
		{
			// Set the input neurons output to be the data inputs.
			int inputIndex = 0;

			foreach (Neuron inputNeuron in _layerList.ElementAt(0))
			{
				inputNeuron.NeuronOutput = inputs[inputIndex];
				inputIndex++;
			}
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
			Neurons _inputNeurons  = new Neurons(_inputNeuronCount, Layer.Input);
			_inputNeurons.InitialiseWeights(_hiddenNeuronCount);

			Neurons _hiddenNeurons = new Neurons(_hiddenNeuronCount, Layer.Hidden);
			_hiddenNeurons.InitialiseWeights(_outputNeuronCount);

			// No weights are needed for the output neurons
			Neurons _outputNeurons = new Neurons(_outputNeuronCount, Layer.Output);

			// Add each layer to a list.
			_layerList = new Layers();

			_layerList.Add(_inputNeurons);
			_layerList.Add(_hiddenNeurons);
			_layerList.Add(_outputNeurons);
		}

	}
}
