using System;

namespace SimpleNeuralNetwork.Sigmoid
{
	static class SigmoidFunctions
	{
		/// <summary>
		/// The sigmoid function.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <returns></returns>
		public static double Sigmoid(double value)
		{
			return 1.0 / (1.0 + Math.Exp(-value));
		}

		/// <summary>
		/// The derivative of the sigmoid function.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <returns></returns>
		public static double SigmoidDerivative(double value)
		{
			return value * (1.0 - value);
		}
	}
}
