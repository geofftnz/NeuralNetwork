using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes.Activations
{
    public class Sigmoid : IActivationFunction
    {
        public float Activation(float input) => 1f / (1f + (float)Math.Exp(-input));

        public float Derivative(float output) => output * (1f - output);
    }
}
