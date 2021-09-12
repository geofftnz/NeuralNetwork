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

        IList<float> Activation { get; }
        IList<float> Target { get; }
        IList<float> Error { get; }
        IList<float> Delta { get; }

        float TotalError
        {
            get
            {
                float total = 0f;
                for (int i = OutputStart; i < OutputStart + OutputCount; i++)
                {
                    total += Error[i]*Error[i];
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

            // reset all errors to zero
            for (int e = 0; e < Error.Count; e++)
            {
                Error[e] = 0f;
                Delta[e] = 0f;
            }
        }

        void ClearError()
        {
            // assume that error and delta are the same length.
            for(int i = 0; i < Error.Count; i++)
            {
                Error[i] = 0f;
                Delta[i] = 0f;
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
