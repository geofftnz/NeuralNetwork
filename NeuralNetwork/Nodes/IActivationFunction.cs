namespace NeuralNetwork.Nodes
{
    public interface IActivationFunction
    {
        float Activation(float input);
        float Derivative(float output);
    }
}