using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public class NetworkRunContext : INetworkRunContext
    {
        private float[] activation;
        private float[] error;
        private float[] delta;
        private float[] target;


        public IList<float> Activation => activation;
        public IList<float> Error => error;
        public IList<float> Delta => delta;
        public IList<float> Target => target;

        public NetworkRunContext(int length, int inputs, int outputs)
        {
            activation = new float[length];
            error = new float[length];
            delta = new float[length];

            target = new float[outputs];

            InputCount = inputs;
            OutputCount = outputs;
            OutputStart = length - outputs;

            if (OutputStart < 0 || OutputStart >= length)
                throw new ArgumentOutOfRangeException($"Length & Outputs are out of range.");

        }

        public int Length => activation.Length;

        public int InputCount { get; private set; }

        public int OutputCount { get; private set; }

        public int OutputStart { get; private set; }

    }
}
