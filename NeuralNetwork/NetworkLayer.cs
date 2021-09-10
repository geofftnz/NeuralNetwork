using NeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NetworkLayer
    {
        public List<Node> Nodes { get; } = new List<Node>();
        public int NodeCount => Nodes.Count;
        public int OutputOffset => Nodes.FirstOrDefault()?.OutputIndex ?? 0;

        public NetworkLayer()
        {

        }

        public NetworkLayer(IActivationFunction func, int count, int outputOffset, int inputIndex, int inputCount, Random rand = null)
        {
            Nodes.AddRange(
                Enumerable.Range(outputOffset, count)
                .Select(i => new Node(func, i, inputIndex, inputCount, rand))
                );
        }

        public void Run(INetworkRunContext context)
        {
            // feed-forward.
            foreach (var node in Nodes)
            {
                node.RunInto(context);
            }
        }

        public void SetOutputLayerErrorFromTarget(INetworkRunContext context)
        {
            if (OutputOffset != context.OutputStart || NodeCount != context.OutputCount)
            {
                throw new InvalidOperationException("SetOutputLayerErrorFromTarget called for an internal layer");
            }
            foreach (var node in Nodes)
            {
                node.CalculateError(context);
            }
        }

        public void BackPropagateError(INetworkRunContext context)
        {
            foreach (var node in Nodes)
            {
                node.BackPropagateError(context);
            }
        }

    }
}
