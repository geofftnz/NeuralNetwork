using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.Nodes
{
    /// <summary>
    /// Standard neural network node.
    /// 
    /// 
    /// </summary>
    public class Node : INeuralNetworkNode
    {
        public IActivationFunction ActivationFunc { get; set; }
        public int OutputIndex { get; set; }

        public float Bias { get; set; } = 0f;

        public List<NodeInput> Inputs { get; } = new List<NodeInput>();

        public Node(IActivationFunction activationFunction, int outputIndex, Random rand = null)
        {
            ActivationFunc = activationFunction;
            OutputIndex = outputIndex;

            if (rand != null)
            {
                Bias = (float)(rand.NextDouble() - 0.5) * 0.01f;
            }
        }

        public Node(IActivationFunction activationFunction, int outputIndex, int inputIndex, int inputCount, Random rand = null)
            : this(activationFunction, outputIndex, rand)
        {
            Inputs.AddRange(
                Enumerable.Range(inputIndex, inputCount)
                .Select(i => new NodeInput(i, (float)(rand.NextDouble() - 0.5) * 0.01f))
                );
        }

        public void RunInto(INetworkRunContext context)
        {
            context.SetActivation(OutputIndex, ActivationFunc.Activation(Bias + Inputs.Select(i => context.GetActivation(i.Index) * i.Weight).Sum()));
        }
        public void CalculateError(INetworkRunContext context)
        {
            if (OutputIndex < context.OutputStart)
            {
                throw new InvalidOperationException("CalculateError: not an output node.");
            }

            context.SetError(OutputIndex, context.GetOutputError(OutputIndex));
        }

        public void BackPropagateError(INetworkRunContext context)
        {
            // compute delta (error multiplied by the derivative of our activation function) as e * a * (1-a);
            //float activation = context.GetActivation(OutputIndex);
            float activation = context.Activation[OutputIndex];
            float delta = context.GetError(OutputIndex) * ActivationFunc.Derivative(activation);

            // propagate this error to our inputs
            foreach (var input in Inputs)
            {
                context.AddError(input.Index, delta * input.Weight);
            }

        }

    }
}
