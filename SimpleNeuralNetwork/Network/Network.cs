using SimpleNeuralNetwork.Elements;
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
		private readonly int _inputNodeCount;

		/// <summary>
		/// The number of hidden Nodes.
		/// </summary>
		private readonly int _hiddenNodeCount;

		/// <summary>
		/// The number of output Nodes.
		/// </summary>
		private readonly int _outputNodeCount;

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
		/// <param name="hiddenNodeCount">The hidden Node count.</param>
		/// <param name="outputNodeCount">The output Node count.</param>
		/// <param name="learningRate">The learning rate.</param>
		public Network(int inputCount, int hiddenNodeCount, int outputNodeCount, double learningRate)
		{
			_inputNodeCount  = inputCount;
			_hiddenNodeCount = hiddenNodeCount;
			_outputNodeCount = outputNodeCount;
			_learningRate      = learningRate;

			CreateNodeLayers();
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
			return String.Join(",", _layerList.ElementAt(_layerList.Count - 1).Select(Node => Node.NodeOutput.ToString()));
		}

		/// <summary>
		/// Train the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="targetOutput">The target output.</param>
		public void Train(double[] inputs, double[] target)
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
			foreach (Node sendingNode in _layerList.GetLayer(sending))
			{
				// Multiply the Node output by each of its weights.
				// The first element matches the first node position in the receiving layer, etc etc.
				double[] multipliedInputs = MultiplyByWeights(sendingNode.Weights, sendingNode.NodeOutput);

				// Index that matches the receiving layers node position
				int receivingNodeIndex = 0;

				foreach (Node receivingNode in _layerList.GetLayer(receiving))
				{
					receivingNode.NodeInput += multipliedInputs[receivingNodeIndex];
					receivingNodeIndex++;
				}

				// Apply the sigmoid function to each of Nodes in this layer.
				_layerList.GetLayer(receiving).ApplySigmoid();
			}
		}

		/// <summary>
		/// Backpropogate the error through the network.
		/// </summary>
		/// <param name="target">The target output.</param>
		private void BackPropogate(double[] target)
		{
			// The amount each weight will be changing
			double[,] hiddenToOutputWeightChanges  = new double[_hiddenNodeCount, _outputNodeCount];
			double[,] inputToHiddenWeightChanges   = new double[_inputNodeCount, _hiddenNodeCount];
			double[] outputNodeErrors              = new double[_outputNodeCount];
			double[] hiddenNodeErrors              = new double[_hiddenNodeCount];

			// Index to cycle through the nodes in each layer
			int outputNodeIndex = 0;
			int hiddenNodeIndex = 0;
			int inputNodeIndex  = 0;

			// Calculate the error for each output node
			foreach (Node outputNode in _layerList.GetLayer(LayerType.Output))
			{
				Console.WriteLine(Math.Pow(target[0] - outputNode.NodeOutput, 2) * 0.5);

				outputNodeErrors[outputNodeIndex] = (target[outputNodeIndex] - outputNode.NodeOutput) * outputNode.SigmoidDerivative();

				outputNodeIndex++;
			}

			// Input the hidden to output weight changes
			foreach(Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
			{
				for(outputNodeIndex = 0; outputNodeIndex < _outputNodeCount; outputNodeIndex++)
				{
					hiddenToOutputWeightChanges[hiddenNodeIndex, outputNodeIndex] = hiddenNode.NodeOutput * outputNodeErrors[outputNodeIndex];
				}

				hiddenNodeIndex++;
			}

			// Reset the indexes
			hiddenNodeIndex = 0;

			// Calculate the error for each of the hidden nodes
			foreach(Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
			{
				outputNodeIndex = 0;

				for(int weightIndex = 0; weightIndex < hiddenNode.Weights.Length; weightIndex++)
				{
					hiddenNodeErrors[hiddenNodeIndex] += outputNodeErrors[outputNodeIndex] * hiddenNode.Weights[weightIndex];
					outputNodeIndex++;
				}

				hiddenNodeErrors[hiddenNodeIndex] = hiddenNodeErrors[hiddenNodeIndex] * hiddenNode.NodeOutput * hiddenNode.SigmoidDerivative();
				hiddenNodeIndex++;
			}

			// Input the hidden to output weight changes
			foreach (Node inputNode in _layerList.GetLayer(LayerType.Input))
			{
				for (hiddenNodeIndex = 0; hiddenNodeIndex < _hiddenNodeCount; hiddenNodeIndex++)
				{
					inputToHiddenWeightChanges[inputNodeIndex, hiddenNodeIndex] = inputNode.NodeOutput * hiddenNodeErrors[hiddenNodeIndex];
				}

				inputNodeIndex++;
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
			foreach (Node inputNeuron in _layerList.GetLayer(LayerType.Input))
			{
				for (int i = 0; i < inputNeuron.Weights.Length; i++)
				{
					inputNeuron.Weights[i] += _learningRate * inputWeights[inputNeuronIndex, i];
				}
			}

			int hiddenNeuronIndex = 0;
			foreach (Node hiddenNeuron in _layerList.GetLayer(LayerType.Hidden))
			{
				for (int i = 0; i < hiddenNeuron.Weights.Length; i++)
				{
					hiddenNeuron.Weights[i] += _learningRate * hiddenWeights[hiddenNeuronIndex, i];
				}
			}
		}

		/// <summary>
		/// Give the input layer of Nodes their output data.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		public void SetInitialInputs(double[] inputs)
		{
			// Set the input Nodes output to be the data inputs.
			int inputIndex = 0;

			foreach (Node inputNode in _layerList.ElementAt(0))
			{
				inputNode.NodeOutput = inputs[inputIndex];
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
		/// Creates the Nodes and initialise their weights.
		/// </summary>
		private void CreateNodeLayers()
		{
			// Create the Node layers and give them some random initial weights
			Nodes _inputNodes  = new Nodes(_inputNodeCount, LayerType.Input);
			_inputNodes.InitialiseWeights(_hiddenNodeCount);

			Nodes _hiddenNodes = new Nodes(_hiddenNodeCount, LayerType.Hidden);
			_hiddenNodes.InitialiseWeights(_outputNodeCount);

			// No weights are needed for the output Nodes
			Nodes _outputNodes = new Nodes(_outputNodeCount, LayerType.Output);

			// Add each layer to a list.
			_layerList = new Layers();

			_layerList.Add(_inputNodes);
			_layerList.Add(_hiddenNodes);
			_layerList.Add(_outputNodes);
		}

	}
}
