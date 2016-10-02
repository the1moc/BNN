using SimpleNeuralNetwork.Elements;
using System;
using System.Linq;

namespace SimpleNeuralNetwork
{
	public class Network
	{
		/// <summary>
		/// The input count for the network.
		/// </summary>
		private readonly int _inputNodeCount;

		/// <summary>
		/// The number of hidden nodes.
		/// </summary>
		private readonly int _hiddenNodeCount;

		/// <summary>
		/// The number of output nodes.
		/// </summary>
		private readonly int _outputNodeCount;

		/// <summary>
		/// The learning rate of the nodes.
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
		/// <param name="hiddenNodeCount">The hidden node count.</param>
		/// <param name="outputNodeCount">The output node count.</param>
		/// <param name="learningRate">The learning rate.</param>
		public Network(int inputCount, int hiddenNodeCount, int outputNodeCount, double learningRate)
		{
			_inputNodeCount  = inputCount;
			_hiddenNodeCount = hiddenNodeCount;
			_outputNodeCount = outputNodeCount;
			_learningRate    = learningRate;

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
			return String.Join(",", _layerList.ElementAt(_layerList.Count - 1).Select(neuron => neuron.NodeOutput.ToString()));
		}

		/// <summary>
		/// Train the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="targetOutput">The target output.</param>
		public void Train(double[] inputs, double target)
		{
			ClearExistingInputs();
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
			FeedForwardRoot(LayerType.Input, LayerType.Hidden);

			// Hidden layer -> Output layer
			FeedForwardRoot(LayerType.Hidden, LayerType.Output);
		}

		/// <summary>
		/// Perform all calculations when passing values from one layer to another.
		/// </summary>
		/// <param name="sending">The sending.</param>
		/// <param name="receiving">The receiving.</param>
		public void FeedForwardRoot(LayerType sending, LayerType receiving)
		{
			foreach (Node sendingNeuron in _layerList.GetLayer(sending))
			{
				// Multiply the neuron output by each of its weights.
				// The first element matches the first node position in the receiving layer, etc etc.
				double[] multipliedInputs = MultiplyByWeights(sendingNeuron.Weights, sendingNeuron.NodeOutput);

				// Index that matches the receiving layers node position
				int receivingNeuronIndex = 0;

				foreach (Node receivingNeuron in _layerList.GetLayer(receiving))
				{
					receivingNeuron.NodeInput += multipliedInputs[receivingNeuronIndex];
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
			double[,] changeOutputWeights = new double[_hiddenNodeCount, _outputNodeCount];
			double[,] changeHiddenWeights = new double[_inputNodeCount, _hiddenNodeCount];

			int hiddenNodeIndex;
			int outputNodeIndex;
			int inputNodeIndex;

			// Calculate the error for each output node
			foreach (Node outputNeuron in _layerList.GetLayer(LayerType.Output))
			{
				outputNeuron.Error = (target - outputNeuron.NodeOutput) * outputNeuron.SigmoidDerivative();
			}

			outputNodeIndex = 0;

			// Calculate the error for each hidden node
			foreach (Node hiddenNeuron in _layerList.GetLayer(LayerType.Hidden))
			{
				double totalErrorAgainstHiddenNeuron = 0.0;

				foreach (Node outputNeuron in _layerList.GetLayer(LayerType.Output))
				{
					totalErrorAgainstHiddenNeuron += outputNeuron.Error * hiddenNeuron.Weights[outputNodeIndex];

					outputNodeIndex++;
				}

				hiddenNeuron.Error = totalErrorAgainstHiddenNeuron * hiddenNeuron.SigmoidDerivative();
				outputNodeIndex    = 0;
			}

			hiddenNodeIndex = 0;

			foreach (Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
			{
				foreach (Node outputNode in _layerList.GetLayer(LayerType.Output))
				{
					for (int i = 0; i < _outputNodeCount; i++)
					{
						changeOutputWeights[hiddenNodeIndex, i] = hiddenNode.NodeOutput * outputNode.Error;
					}
				}

				hiddenNodeIndex++;
			}

			inputNodeIndex = 0;

			foreach (Node inputNode in _layerList.GetLayer(LayerType.Input))
			{
				foreach (Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
				{
					for (int i = 0; i < _hiddenNodeCount; i++)
					{
						changeHiddenWeights[inputNodeIndex, i] = inputNode.NodeOutput * hiddenNode.Error;
					}
				}
				inputNodeIndex++;
			}

			UpdateWeights(changeHiddenWeights, changeOutputWeights);
		}

		/// <summary>
		/// Update the network weights..
		/// </summary>
		/// <returns></returns>
		private void UpdateWeights(double[,] changeInputWeights, double[,] changeHiddenWeights)
		{
			int hiddenNeuronIndex = 0;
			foreach (Node hiddenNeuron in _layerList.GetLayer(LayerType.Hidden))
			{
				for (int i = 0; i < hiddenNeuron.Weights.Length; i++)
				{
					hiddenNeuron.Weights[i] += _learningRate * changeHiddenWeights[hiddenNeuronIndex, i];
				}
				hiddenNeuronIndex++;
			}

			int inputNeuronIndex = 0;
			foreach(Node inputNeuron in _layerList.GetLayer(LayerType.Input))
			{
				for(int i = 0; i < inputNeuron.Weights.Length; i++)
				{
					inputNeuron.Weights[i] += _learningRate * changeInputWeights[inputNeuronIndex,i];
				}
				inputNeuronIndex++;
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

			foreach (Node inputNeuron in _layerList.ElementAt(0))
			{
				inputNeuron.NodeOutput = inputs[inputIndex];
				inputIndex++;
			}
		}

		/// <summary>
		/// Reset all of the existing node inputs to 0.
		/// </summary>
		/// <returns></returns>
		private void ClearExistingInputs()
		{
			_layerList.ClearInputs();
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
			Nodes _inputNeurons  = new Nodes(_inputNodeCount, LayerType.Input);
			_inputNeurons.InitialiseWeights(_hiddenNodeCount);

			Nodes _hiddenNeurons = new Nodes(_hiddenNodeCount, LayerType.Hidden);
			_hiddenNeurons.InitialiseWeights(_outputNodeCount);

			// No weights are needed for the output neurons
			Nodes _outputNeurons = new Nodes(_outputNodeCount, LayerType.Output);

			// Add each layer to a list.
			_layerList = new Layers();

			_layerList.Add(_inputNeurons);
			_layerList.Add(_hiddenNeurons);
			_layerList.Add(_outputNeurons);
		}

	}
}

