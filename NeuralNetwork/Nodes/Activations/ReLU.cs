using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes.Activations
{
    public class ReLU : IActivationFunction
    {
        public float Activation(float input) => Math.Max(0.0f, input);

        public float Derivative(float output) => 1f / (1f + (float)Math.Pow(Math.E, -output));
    }
}
