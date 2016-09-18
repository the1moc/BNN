using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Elements
{
	class Layers: List<Nodes>
	{
		/// <summary>
		/// Gets the requested layer.
		/// </summary>
		/// <param name="layer">The layer.</param>
		/// <returns></returns>
		public Nodes GetLayer(LayerType layer)
		{
			return this.Find(NodeLayer => NodeLayer.NetworkLayer == layer);
		}
	}
}
