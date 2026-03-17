import numpy as np
import pandas as pd
from collections import Counter
import math

class CustomTFIDF:
    def __init__(self, max_features=1000):

        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def _tokenize(self, text):

        import re
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def fit(self, texts):

        df = Counter() # Document Frequency
        N = len(texts)
        
        for text in texts:
            words = set(self._tokenize(text)) # set -> remover dups
            for w in words:
                df[w] += 1
                
        top_words = df.most_common(self.max_features)
        self.vocab = {word: i for i, (word, count) in enumerate(top_words)}
        
            # log(N / (1 + docs com palavra))
        self.idf = {word: math.log(N / (1 + df[word])) for word in self.vocab}
        print(f"Vocabulário criado com {len(self.vocab)} palavras")

    def transform(self, texts):

        X = np.zeros((len(texts), len(self.vocab)))
        
        for i, text in enumerate(texts):
            words = self._tokenize(text)
            word_counts = Counter(words)
            total_words = len(words)
            
            if total_words == 0:
                continue
                
            for word, count in word_counts.items():
                if word in self.vocab:
                        
                    tf = count / float(total_words)                 # Term frequency
                    col_idx = self.vocab[word]
                    X[i, col_idx] = tf * self.idf[word]             # TF * IDF
                    
        return X
    

def encode_labels(labels):

    unique_labels = sorted(list(set(labels)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    Y = np.zeros((len(labels), len(unique_labels)))
    
    for i, label in enumerate(labels):
        col_idx = label_to_idx[label]
        Y[i, col_idx] = 1
        
    print(f"Classes detetadas: {unique_labels}")
    return Y, label_to_idx, idx_to_label