using System;

namespace SimpleNeuralNetwork
{
	static class Sigmoid
	{
		/// <summary>
		/// Apply the sigmoid function.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <returns></returns>
		public static double SigmoidFunction(double value)
		{
			return 1.0 / (1.0 + Math.Exp(-value));
		}

		/// <summary>
		/// The derivative of the sigmoid function.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <returns></returns>
		public static double DerivativeSigmoid(double value)
		{
			return value * (1 - value);
		}
	}
}
