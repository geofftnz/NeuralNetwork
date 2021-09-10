using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork;
using NeuralNetwork.Nodes;
using NeuralNetwork.Nodes.Activations;

namespace NeuralNetConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("NeuralNetwork testbench");

            IActivationFunction activation = new Sigmoid();

            var net = new Network();
            net.SetInputs(2);
            net.AddLayer(activation, 2);
            net.AddLayer(activation, 1);

            net.LearningRate = 0.5f;
            net.Momentum = 0.1f;

            var context = net.GetNewContext();

            // training data
            var trainingruns = new List<float[]>();
            trainingruns.Add(new float[] { 0f, 0f, 0f });
            trainingruns.Add(new float[] { 1f, 0f, 1f });
            trainingruns.Add(new float[] { 0f, 1f, 1f });
            trainingruns.Add(new float[] { 1f, 1f, 0f });

            Random rand = new Random();

            // do some training
            for (int i = 0; i < 10000; i++)
            {
                context.SetTraining(trainingruns[rand.Next(trainingruns.Count)]);
                net.Train(context);
                net.Update();
                Console.WriteLine($"Run {i}: {string.Join(',',context.Inputs.Select(x=>x.ToString()))} -> {string.Join(',', context.Outputs.Select(x => x.ToString("0.000")))}  ({string.Join(',', context.Targets.Select(x => x.ToString()))}) {context.TotalError}");

                //Console.WriteLine($"{i}: {string.Join(',',context.Activation.Select(x=>x.ToString()))}");
            }

        }
    }
}
