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
			return this.Find(nodeLayer => nodeLayer.NetworkLayer == layer);
		}

		/// <summary>
		/// Clears the inputs for each list of nodes.
		/// </summary>
		public void ClearInputs()
		{
			this.ForEach(nodes => nodes.ClearInputs());
		}

        /// Clears the outputs for each list of nodes.
        /// </summary>
        public void ClearOutputs()
        {
            this.ForEach(nodes => nodes.ClearOutputs());
        }
    }
}
