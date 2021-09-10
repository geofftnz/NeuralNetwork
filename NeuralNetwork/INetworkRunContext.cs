using NeuralNetwork.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;
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

        float TotalError
        {
            get
            {
                float total = 0f;
                for (int i = OutputStart; i < OutputStart + OutputCount; i++)
                {
                    total += Error[i] * Error[i];
                }
                return total / OutputCount;
            }
        }

        void Set(IEnumerable<float> inputs)
        {
            int i = 0;
            foreach (var x in inputs.Take(InputCount))
            {
                Activation[i++] = x;
            }
        }

        void Set(IEnumerable<float> inputs, IEnumerable<float> targets)
        {
            Set(inputs);
            int i = 0;
            foreach (var x in targets.Take(OutputCount))
            {
                Target[i++] = x;
            }
        }

        void SetTraining(float[] inputsAndTargets)
        {
            if (inputsAndTargets.Length < InputCount + OutputCount)
            {
                throw new InvalidOperationException("SetCombined: Not enough data supplied for inputs+targets");
            }

            int i = 0;

            for (int a = 0; a < InputCount; a++)
            {
                Activation[a] = inputsAndTargets[i++];
            }

            for (int t = 0; t < OutputCount; t++)
            {
                Target[t] = inputsAndTargets[i++];
            }
        }

        IEnumerable<float> Inputs
        {
            get
            {
                for (int i = 0; i < InputCount; i++)
                {
                    yield return Activation[i];
                }
            }
        }
        IEnumerable<float> Outputs
        {
            get
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    yield return Activation[OutputStart + i];
                }
            }
        }
        IEnumerable<float> Targets
        {
            get
            {
                for (int i = 0; i < OutputCount; i++)
                {
                    yield return Target[i];
                }
            }
        }

    }
}
