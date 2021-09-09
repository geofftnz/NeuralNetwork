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

        private float[] target;


        public IIndexer<float> Activation { get; private set; }

        public NetworkRunContext(int length, int inputs, int outputs)
        {
            activation = new float[length];
            error = new float[length];

            target = new float[outputs];

            InputCount = inputs;
            OutputCount = outputs;
            OutputStart = length - outputs;

            if (OutputStart < 0 || OutputStart >= length)
                throw new ArgumentOutOfRangeException($"Length & Outputs are out of range.");

            Activation = new Indexer<float>(activation);
        }

        public int Length => activation.Length;

        public int InputCount { get; private set; }

        public int OutputCount { get; private set; }

        public int OutputStart { get; private set; }


        public float GetActivation(int index) => activation[index];
        public void SetActivation(int index, float value) => activation[index] = value;

        public float GetError(int index) => error[index];
        public void SetError(int index, float value) => error[index] = value;
        public void AddError(int index, float value) => error[index] += value;


        public float GetOutputTarget(int outputIndex) => target[outputIndex];
        public void SetOutputTarget(int outputIndex, float value) => target[outputIndex] = value;

        public float GetOutputError(int index)
        {
            if (index < OutputStart)
                throw new InvalidOperationException($"GetOutputError: not an output node ({index})");

            return target[index - OutputStart] - activation[index];
        }
    }
}
