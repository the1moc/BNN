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

			CreateNodeLayers();
		}

		/// <summary>
		/// Query the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		public double Probe(double[] inputs)
		{
			ClearExistingInputs();
			ClearExistingOutputs();
			SetInitialInputs(inputs);
			FeedForward();

			// Get the last layer (output) and return a string of the output values.
			return _layerList.GetLayer(LayerType.Output).ElementAt(0).NodeOutput;
		}

		/// <summary>
		/// Train the network.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		/// <param name="targetOutput">The target output.</param>
		public void Train(double[] inputs, double target)
		{
			ClearExistingInputs();
			ClearExistingOutputs();
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
		/// <param name="sending">The sending layer type.</param>
		/// <param name="receiving">The receiving layer type.</param>
		public void FeedForwardRoot(LayerType sending, LayerType receiving)
		{
			// For all ths nodes in this layer, multiply their input by their outgoing weights and set this value.
			_layerList.GetLayer(sending).MultiplyWeightsByInput();

			// Index that matches the receiving layers node position
			int receivingNodeIndex = 0;

			foreach (Node receivingNode in _layerList.GetLayer(receiving))
			{
				_layerList.GetLayer(sending).ForEach(sendingNode => receivingNode.NodeInput += sendingNode.MultipliedInputByWeights[receivingNodeIndex]);
				receivingNodeIndex++;
			}

			// Apply the sigmoid function to each of nodes in this layer.
			_layerList.GetLayer(receiving).ApplySigmoidToNodes();
		}

		/// <summary>
		/// Backpropogate the error through the network.
		/// </summary>
		/// <param name="target">The target output.</param>
		private void BackPropogate(double target)
		{
			int outputNodeIndex;

			// Calculate the error for each output node
			foreach (Node outputNode in _layerList.GetLayer(LayerType.Output))
			{
				outputNode.Error = (target - outputNode.NodeOutput) * outputNode.SigmoidDerivative();
			}

			outputNodeIndex = 0;

			// Calculate the error for each hidden node
			foreach (Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
			{
				double totalErrorAgainstHiddenNode = 0.0;

				foreach (Node outputNode in _layerList.GetLayer(LayerType.Output))
				{
					totalErrorAgainstHiddenNode += outputNode.Error * hiddenNode.Weights[outputNodeIndex];

					outputNodeIndex++;
				}

				// Mulitply the summed total error against the hidden node derivative.
				hiddenNode.Error = totalErrorAgainstHiddenNode * hiddenNode.SigmoidDerivative();
				outputNodeIndex  = 0;
			}

			UpdateWeights();
		}

		/// <summary>
		/// Update the network weights.
		/// </summary>
		/// <returns></returns>
		private void UpdateWeights()
		{
			int outputNodeIndex = 0;
		
			// Update the hidden weights.
			foreach (Node outputNode in _layerList.GetLayer(LayerType.Output))
			{
				foreach (Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
				{
					hiddenNode.Weights[outputNodeIndex] += _learningRate * outputNode.Error * hiddenNode.NodeOutput;
				}
				outputNodeIndex++;
			}

			int hiddenNodeIndex = 0;

			// Update the input weights.
			foreach (Node hiddenNode in _layerList.GetLayer(LayerType.Hidden))
			{
				foreach (Node inputNode in _layerList.GetLayer(LayerType.Input))
				{
					inputNode.Weights[hiddenNodeIndex] += _learningRate * hiddenNode.Error * inputNode.NodeOutput;
				}
				hiddenNodeIndex++;
			}
		}

		/// <summary>
		/// Give the input layer of nodes their output data.
		/// </summary>
		/// <param name="inputs">The inputs.</param>
		public void SetInitialInputs(double[] inputs)
		{
			// Set the input nodes output to be the data inputs.
			int inputIndex = 0;

			foreach (Node inputNode in _layerList.ElementAt(0))
			{
				inputNode.NodeOutput = inputs[inputIndex];
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
		/// Reset all of the existing node outputs to 0.
		/// </summary>
		/// <returns></returns>
		private void ClearExistingOutputs()
		{
			_layerList.ClearOutputs();
		}

		/// <summary>
		/// Creates the nodes and initialise their weights.
		/// </summary>
		private void CreateNodeLayers()
		{
			// Create the node layers and give them some random initial weights
			Nodes _inputNodes  = new Nodes(_inputNodeCount, LayerType.Input);
			_inputNodes.InitialiseWeights(_hiddenNodeCount);

			Nodes _hiddenNodes = new Nodes(_hiddenNodeCount, LayerType.Hidden);
			_hiddenNodes.InitialiseWeights(_outputNodeCount);

			// No weights are needed for the output nodes
			Nodes _outputNodes = new Nodes(_outputNodeCount, LayerType.Output);

			// Add each layer to a list.
			_layerList = new Layers();

			_layerList.Add(_inputNodes);
			_layerList.Add(_hiddenNodes);
			_layerList.Add(_outputNodes);
		}

	}
}

