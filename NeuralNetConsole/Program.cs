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
            net.Momentum = 0.5f;


            var context = net.GetNewContext();

            // training data
            var trainingruns = new List<float[]>();
            trainingruns.Add(new float[] { 0f, 0f, 0f });
            trainingruns.Add(new float[] { 1f, 0f, 1f });
            trainingruns.Add(new float[] { 0f, 1f, 1f });
            trainingruns.Add(new float[] { 1f, 1f, 0f });

            Random rand = new Random();

            // do some training
            int totalConvergence = 0;
            int samples = 10;
            float errorThreshold = 0.05f;
            int trainingRunCount = 10000000;

            Console.WriteLine($"Converging {samples} times to error<{errorThreshold}...");

            for (int r = 0; r < samples; r++)
            {
                net.Reset();
                float error = 0.5f;
                for (int i = 0; i < trainingRunCount; i++)
                {
                    context.SetTraining(trainingruns[rand.Next(trainingruns.Count)]);
                    net.Train(context);
                    net.Update();

                    error = error * 0.9f + 0.1f * context.TotalError;

                    // tweak learning rate and momentum based on error
                    // to reduce bouncing around the true solution
                    //if (i % 1000 == 0)
                    //{
                    //    net.LearningRate = error * 0.5f;
                    //    net.Momentum = error * 0.1f;
                    //    //Console.WriteLine($"->{net.LearningRate} {net.Momentum}");
                    //}
                    
                    if (error > 0.2f && i%100 == 0)
                    {
                        net.AddNoise((error - 0.2f) * 0.05f);
                    }

                    /*
                    if (i % 1000 == 0)
                    {
                        Console.WriteLine($"Run {i}: error {error}");
                    }*/

                    if (error < errorThreshold)
                    {
                        totalConvergence += i;
                        //Console.WriteLine($"Run {i}: error {error}");
                        Console.WriteLine($"Converged in {i} iterations.");
                        break;
                    }
                    //Console.WriteLine($"Run {i}: {string.Join(',',context.Inputs.Select(x=>x.ToString()))} -> {string.Join(',', context.Outputs.Select(x => x.ToString("0.000")))}  ({string.Join(',', context.Targets.Select(x => x.ToString()))}) {context.TotalError}");
                    //Console.WriteLine($"{i}: {string.Join(',',context.Activation.Select(x=>x.ToString()))}");
                }
                if (error >= errorThreshold)
                {
                    totalConvergence += trainingRunCount;
                    Console.WriteLine($"Did not converge in {trainingRunCount} iterations.");
                }
            }

            Console.WriteLine($"Average convergence iterations {totalConvergence / samples}");
        }
    }
}
