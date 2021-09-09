using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Extensions
{
    public class Indexer<T> : IIndexer<T>
    {
        private Action<int, T> SetFunc;
        private Func<int, T> GetFunc;

        public T this[int i]
        {
            get => GetFunc(i);
            set => SetFunc(i, value);
        }

        public Indexer(Action<int, T> setFunc, Func<int, T> getFunc)
        {
            SetFunc = setFunc;
            GetFunc = getFunc;
        }

        public Indexer(T[] arr)
        {
            SetFunc = (i, x) => arr[i] = x;
            GetFunc = (i) => arr[i];
        }

    }
}
