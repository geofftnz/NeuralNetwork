using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes.Activations
{
    public class Ramp : IActivationFunction
    {
        public float Activation(float input) => Math.Clamp(input,-1f,1f);

        public float Derivative(float output) => 1f;
    }
}
