using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes.Activations
{
    public class TanhFast : IActivationFunction
    {
        public float Activation(float input)
        {
            float in2 = input * input;
            return (input / (1f + in2 / (3f + in2 / (7f + in2 / (9f + in2 / 11f))))) * 1.075f;
        }

        public float Derivative(float output) => 1f - output * output;
                
    }
}
