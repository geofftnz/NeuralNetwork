using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    public interface INeuralNetworkNode
    {
        void RunInto(INetworkRunContext context);
    }
}
