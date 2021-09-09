namespace NeuralNetwork.Extensions
{
    public interface IIndexer<T>
    {
        T this[int i] { get; set; }
    }
}