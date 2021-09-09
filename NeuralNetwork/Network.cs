using NeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;

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
            Layers.Add(new NetworkLayer(func, count, outputOffset, rand));
            return this;
        }

        public INetworkRunContext GetNewContext()
        {
            return new NetworkRunContext(ContextSize, InputCount, OutputCount);
        }

        public void Run(INetworkRunContext context)
        {
            // feed-forward through layers
            foreach(var layer in Layers)
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

            // back-prop errors
        }
    }
}


