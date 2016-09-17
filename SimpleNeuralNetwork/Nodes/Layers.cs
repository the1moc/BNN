using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Nodes
{
	class Layers: List<Neurons>
	{
		/// <summary>
		/// Gets the requested layer.
		/// </summary>
		/// <param name="layer">The layer.</param>
		/// <returns></returns>
		public Neurons GetLayer(Layer layer)
		{
			return this.Find(neuronLayer => neuronLayer.NetworkLayer == layer);
		}
	}
}
