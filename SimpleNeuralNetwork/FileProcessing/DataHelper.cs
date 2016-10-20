using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace SimpleNeuralNetwork.FileProcessing
{
	public static class DataHelper
	{
		/// <summary>
		/// Gets the target number from this line.
		/// </summary>
		/// <param name="line">The line.</param>
		/// <returns></returns>
		public static int GetTargetNumber(string line)
		{
			return (int)Char.GetNumericValue(line[0]);
		}

		/// <summary>
		/// Generates the targets array.
		/// </summary>
		/// <param name="line">The line.</param>
		/// <returns></returns>
		public static double[] GenerateTargets(string line)
		{
			double[] targets = Enumerable.Repeat(0.01, 10).ToArray();
			targets[GetTargetNumber(line)] = 0.99;
			return targets;
		}

		/// <summary>
		/// Normalises the data.
		/// </summary>
		/// <param name="line">The line.</param>
		/// <returns></returns>
		public static double[] NormaliseData(string line)
		{
			return line.Split(',').Skip(1).Select(c => (Int32.Parse(c) / 255.0 * 0.99) + 0.01).ToArray();
		}
	}
}
