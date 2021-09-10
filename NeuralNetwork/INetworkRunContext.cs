using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public interface INetworkRunContext
    {
        /// <summary>
        /// Total number of entries in the activation array.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Total number of network inputs
        /// </summary>
        int InputCount { get; }

        /// <summary>
        /// Total number of network outputs
        /// </summary>
        int OutputCount { get; }

        /// <summary>
        /// Start index of network outputs
        /// </summary>
        int OutputStart { get; }

        IIndexable<float, int> Activation { get; }
        IIndexable<float, int> Target { get; }
        IIndexable<float, int> Error { get; }
        IIndexable<float, int> Delta { get; }

    }
}
