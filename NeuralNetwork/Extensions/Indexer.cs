using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Extensions
{
    public class Indexer<T, I> : IIndexable<T, I>
    {
        private Action<I, T> SetFunc;
        private Func<I, T> GetFunc;

        public T this[I i]
        {
            get => GetFunc(i);
            set => SetFunc(i, value);
        }

        public Indexer(Action<I, T> setFunc, Func<I, T> getFunc)
        {
            SetFunc = setFunc;
            GetFunc = getFunc;
        }

        public static Indexer<T, int> For(T[] array) => new Indexer<T, int>((i, x) => array[i] = x, (i) => array[i]);
        public static Indexer<T, K> For<K>(IDictionary<K,T> dict) => new Indexer<T, K>((i, x) => dict[i] = x, (i) => dict[i]);

    }
}
