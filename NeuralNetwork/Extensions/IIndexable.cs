namespace NeuralNetwork.Extensions
{
    public interface IIndexable<T,I>
    {
        T this[I i] { get; set; }
    }
}