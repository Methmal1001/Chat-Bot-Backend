using System;
using System.Collections.Generic;
using System.Linq;

namespace Helpers
{
    public static class TextSimilarity
    {
        // Tokenize → lowercase + split
        public static List<string> Tokenize(string text)
        {
            return text.ToLower()
                       .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                       .Select(w => w.Trim())
                       .ToList();
        }

        // Term Frequency
        public static Dictionary<string, double> TermFrequency(List<string> words)
        {
            var tf = new Dictionary<string, double>();
            int total = words.Count;

            foreach (var w in words)
            {
                if (!tf.ContainsKey(w))
                    tf[w] = 0;

                tf[w] += 1.0 / total;
            }
            return tf;
        }

        // IDF
        public static Dictionary<string, double> InverseDocumentFrequency(List<List<string>> docs)
        {
            var idf = new Dictionary<string, double>();
            int totalDocs = docs.Count;

            var allWords = docs.SelectMany(d => d).Distinct();

            foreach (var word in allWords)
            {
                int docsWithWord = docs.Count(d => d.Contains(word));
                idf[word] = Math.Log((double)totalDocs / (1 + docsWithWord));
            }

            return idf;
        }

        // TF-IDF Vector
        public static Dictionary<string, double> TfIdfVector(List<string> words, Dictionary<string, double> idf)
        {
            var tf = TermFrequency(words);
            return tf.ToDictionary(k => k.Key, v => v.Value * idf.GetValueOrDefault(v.Key, 0));
        }

        // Cosine Similarity
        public static double CosineSimilarity(Dictionary<string, double> v1, Dictionary<string, double> v2)
        {
            double dot = 0, a = 0, b = 0;

            foreach (var key in v1.Keys)
            {
                if (v2.ContainsKey(key)) dot += v1[key] * v2[key];
                a += Math.Pow(v1[key], 2);
            }
            foreach (var x in v2.Values) b += Math.Pow(x, 2);

            if (a == 0 || b == 0) return 0;
            return dot / (Math.Sqrt(a) * Math.Sqrt(b));
        }
    }
}
