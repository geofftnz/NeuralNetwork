using NeuralNetwork.Extensions;
using NeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    /*
    Simple, oldskool backprop Neural Network.

    What we need:
    - Node activations (big array of floats) - inputs given to network initially, remainder updated when network runs. Probably separate this out.
    
    
    - Nodes
     */
    public class Network
    {
        private Random rand = new Random();

        public int InputCount { get; private set; } = 0;

        public List<NetworkLayer> Layers { get; private set; } = new List<NetworkLayer>();

        public int OutputStart => Layers.Any() ? Layers.Last().OutputOffset : 0;
        public int OutputCount => Layers.Any() ? Layers.Last().NodeCount : 0;
        public int ContextSize => InputCount + Layers.Sum(l => l.NodeCount);

        // learning rates
        public float LearningRate { get; set; } = 0.05f;
        public float Momentum { get; set; } = 0.05f;

        public Network()
        {

        }

        public Network SetInputs(int count)
        {
            InputCount = count;
            return this;
        }

        public Network AddLayer(NetworkLayer layer)
        {
            Layers.Add(layer);
            return this;
        }

        public Network AddLayer(IActivationFunction func, int count)
        {
            int outputOffset = InputCount + Layers.Sum(l => l.NodeCount);

            // default inputs for first layer
            int inputOffset = 0;
            int inputCount = InputCount;

            // otherwise take our inputs from the previous layer
            if (Layers.Any())
            {
                inputOffset = Layers.Last().OutputOffset;
                inputCount = Layers.Last().NodeCount;
            }

            Layers.Add(new NetworkLayer(func, count, outputOffset, inputOffset, inputCount, rand));
            return this;
        }

        public INetworkRunContext GetNewContext()
        {
            return new NetworkRunContext(ContextSize, InputCount, OutputCount);
        }

        public void Run(INetworkRunContext context)
        {
            // feed-forward through layers
            foreach (var layer in Layers)
            {
                layer.Run(context);
            }
        }

        public void Train(INetworkRunContext context)
        {
            // feed-forward through layers
            foreach (var layer in Layers)
            {
                layer.Run(context);
            }

            // Clear all errors
            context.ClearError();

            // set the error of the output layer
            Layers.Last().SetOutputLayerErrorFromTarget(context);

            // back-prop errors
            foreach (var layer in Layers.FastReverse())
            {
                layer.BackPropagateError(context);
            }

            // update weight error derivatives of all nodes
            foreach (var node in Layers.SelectMany(l => l.Nodes))
            {
                node.UpdateWeightErrorDerivatives(context);
            }
        }

        public float Test(INetworkRunContext context)
        {
            // feed-forward through layers
            foreach (var layer in Layers)
            {
                layer.Run(context);
            }

            // Clear all errors
            context.ClearError();

            // set the error of the output layer
            Layers.Last().SetOutputLayerErrorFromTarget(context);

            return context.TotalError;
        }

            public void Update()
        {
            foreach (var node in Layers.SelectMany(l => l.Nodes))
            {
                node.UpdateWeights(LearningRate, Momentum);
            }
        }

        public void AddNoise(float amount)
        {
            foreach (var node in Layers.SelectMany(l => l.Nodes))
            {
                node.AddNoise(rand, amount);
            }
        }

        public void Reset()
        {
            foreach (var node in Layers.SelectMany(l => l.Nodes))
            {
                node.Reset(rand);
            }
        }

        public override string ToString()
        {
            return $"in:({InputCount})->" + string.Join("->", Layers.Select(layer => layer.ToString()));
        }
    }
}

