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

        /// <summary>
        /// Sets the activation value of an element in the context.
        /// </summary>
        /// <param name="index"></param>
        /// <param name="value"></param>
        void SetActivation(int index, float value);

        /// <summary>
        /// returns the activation value of an element in the context.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        float GetActivation(int index);

        IIndexer<float> Activation { get; }


        void SetOutputTarget(int outputIndex, float value);
        float GetOutputTarget(int outputIndex);
        float GetOutputError(int index);

        void SetError(int index, float value);
        void AddError(int index, float value);
        float GetError(int index);




    }
}
