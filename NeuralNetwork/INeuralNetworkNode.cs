using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public interface INeuralNetworkNode
    {
        void RunInto(INetworkRunContext context);
    }
}
