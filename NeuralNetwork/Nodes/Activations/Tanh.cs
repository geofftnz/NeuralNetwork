using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes.Activations
{
    public class Tanh : IActivationFunction
    {
        public float Activation(float input) => (float)Math.Tanh(input);

        public float Derivative(float output) => 1f - output * output;
                
    }
}
