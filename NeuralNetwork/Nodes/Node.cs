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
        public float DBias { get; set; } = 0f;
        public float BiasErrorDerivative { get; set; } = 0f;


        public List<NodeInput> Inputs { get; } = new List<NodeInput>();

        public Node(IActivationFunction activationFunction, int outputIndex, Random rand = null)
        {
            ActivationFunc = activationFunction;
            OutputIndex = outputIndex;

            if (rand != null)
            {
                Bias = (float)(rand.NextDouble() - 0.5) * 0.1f;
            }
        }

        public Node(IActivationFunction activationFunction, int outputIndex, int inputIndex, int inputCount, Random rand = null)
            : this(activationFunction, outputIndex, rand)
        {
            Inputs.AddRange(
                Enumerable.Range(inputIndex, inputCount)
                .Select(i => new NodeInput(i, (float)(rand.NextDouble() - 0.5) * 0.1f))
                );
        }

        public void ResetWeights(Random rand)
        {
            foreach (var input in Inputs)
            {
                input.Weight = (float)(rand.NextDouble() - 0.5) * 0.1f;
                input.DWeight = 0f;
                input.WeightErrorDerivative = 0f;
            }
        }

        public void Reset(Random rand)
        {
            ResetWeights(rand);

            Bias = 0f;
            DBias = 0f;
            BiasErrorDerivative = 0f;
        }

        public void RunInto(INetworkRunContext context)
        {
            context.Activation[OutputIndex] = ActivationFunc.Activation(Bias + Inputs.Select(i => context.Activation[i.Index] * i.Weight).Sum());
        }
        public void CalculateError(INetworkRunContext context)
        {
            if (OutputIndex < context.OutputStart)
            {
                throw new InvalidOperationException("CalculateError: not an output node.");
            }

            context.Error[OutputIndex] = context.Target[OutputIndex - context.OutputStart] - context.Activation[OutputIndex];
        }

        public void BackPropagateError(INetworkRunContext context)
        {
            // compute delta (error multiplied by the derivative of our activation function) as e * a * (1-a);
            context.Delta[OutputIndex] = context.Error[OutputIndex] * ActivationFunc.Derivative(context.Activation[OutputIndex]);

            // propagate this error to our inputs
            foreach (var input in Inputs)
            {
                context.Error[input.Index] += context.Delta[OutputIndex] * input.Weight;
            }
        }

        public void UpdateWeightErrorDerivatives(INetworkRunContext context)
        {
            BiasErrorDerivative += context.Delta[OutputIndex];
            foreach (var input in Inputs)
            {
                input.WeightErrorDerivative += context.Delta[OutputIndex] * context.Activation[input.Index];
            }
        }

        public void ClearWeightErrorDerivatives()
        {
            BiasErrorDerivative = 0f;
            foreach (var input in Inputs)
            {
                input.WeightErrorDerivative = 0f;
            }
        }

        public void UpdateWeights(float learningRate, float momentum)
        {
            DBias = BiasErrorDerivative * learningRate + DBias * momentum;
            Bias += DBias;
            BiasErrorDerivative = 0f;

            foreach (var input in Inputs)
            {
                input.DWeight = input.WeightErrorDerivative * learningRate + input.DWeight * momentum;
                input.Weight += input.DWeight;
                input.WeightErrorDerivative = 0f;
            }
        }

        public void AddNoise(Random rand, float amount)
        {
            foreach (var input in Inputs)
            {
                input.Weight += (float)(rand.NextDouble() - 0.5) * 2f * amount;
            }
        }

        public override string ToString()
        {
            return $"({Bias}+"+string.Join(',', Inputs.Select(input => input.Index.ToString() + ":" + input.Weight.ToString()))+")";
        }

    }
}
