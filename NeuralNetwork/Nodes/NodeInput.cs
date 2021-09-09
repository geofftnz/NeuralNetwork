using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    public struct NodeInput
    {
        /// <summary>
        /// Index into the context to get this input
        /// </summary>
        public int Index;

        /// <summary>
        /// Weight of this input
        /// </summary>
        public float Weight;

        /// <summary>
        /// Current movement of this weight in parameter space.
        /// Used to introduce momentum during training.
        /// </summary>
        public float DWeight;

        /// <summary>
        /// Direction we have to move the weight to minimise error.
        /// </summary>
        public float WeightErrorDerivative;

        public NodeInput(int index, float weight)
        {
            Index = index;
            Weight = weight;
            DWeight = 0f;
            WeightErrorDerivative = 0f;
        }
    }
}
