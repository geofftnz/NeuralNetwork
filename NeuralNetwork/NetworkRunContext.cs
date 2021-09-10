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


        public IIndexable<float, int> Activation { get; private set; }
        public IIndexable<float, int> Error { get; private set; }
        public IIndexable<float, int> Delta { get; private set; }
        public IIndexable<float, int> Target { get; private set; }

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

            Activation = Indexer<float, int>.For(activation);
            Error = Indexer<float, int>.For(error);
            Delta = Indexer<float, int>.For(delta);
            Target = Indexer<float, int>.For(target);
        }

        public int Length => activation.Length;

        public int InputCount { get; private set; }

        public int OutputCount { get; private set; }

        public int OutputStart { get; private set; }

    }
}
