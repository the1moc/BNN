using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleNeuralNetwork.Util
{
	public static class NumberUtils
	{
		private static Random rng = new Random();

		/// <summary>
		/// Generate a random double number 
		/// </summary>
		/// <returns></returns>
		public static double GetRandomDouble()
		{
			return 2 * rng.NextDouble() - 0.5;
		}
	}
}
