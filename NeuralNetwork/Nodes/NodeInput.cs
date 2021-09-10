using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    public class NodeInput
    {
        /// <summary>
        /// Index into the context to get this input
        /// </summary>
        public int Index { get; set; }
        
        /// <summary>
        /// Weight of this input
        /// </summary>
        public float Weight { get; set; }

        /// <summary>
        /// Current movement of this weight in parameter space.
        /// Used to introduce momentum during training.
        /// </summary>
        public float DWeight { get; set; }

        /// <summary>
        /// Direction we have to move the weight to minimise error.
        /// </summary>
        public float WeightErrorDerivative { get; set; }

        public NodeInput(int index, float weight)
        {
            Index = index;
            Weight = weight;
            DWeight = 0f;
            WeightErrorDerivative = 0f;
        }
    }
}
