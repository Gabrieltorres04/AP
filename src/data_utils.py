import re
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.text_processing import CustomTFIDF

def stratified_split(df, text_col, label_col, test_size=0.2, seed=2026):
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[label_col]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def build_label_maps(class_order):
    label_to_idx = {c: i for i, c in enumerate(class_order)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

def encode_labels(labels, label_to_idx):
    return np.array([label_to_idx[l] for l in labels], dtype=np.int64)

def create_tfidf_features(text_train, text_val, max_features=1000):
    tfidf = CustomTFIDF(max_features=max_features)
    tfidf.fit(text_train)
    X_train = tfidf.transform(text_train).astype(np.float32)
    X_val = tfidf.transform(text_val).astype(np.float32)
    return tfidf, X_train, X_val

def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()

def build_vocab(texts, max_vocab=20000, min_freq=3):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if len(vocab) >= max_vocab:
            break
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_texts(texts, vocab, max_len=120):
    out = []
    for text in texts:
        ids = [vocab.get(t, 1) for t in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return np.array(out, dtype=np.int64)

def load_pretrained_embeddings(vocab, source='glove', embed_dim=100):
    """
    Descarrega embeddings pré-treinados e constrói uma matriz alinhada com o vocab.

    source:
        'glove'    — GloVe Wikipedia+Gigaword; embed_dim: 50, 100, 200, 300
        'fasttext' — FastText Wikipedia News subwords; embed_dim: 300 (único disponível)

    Palavras não encontradas são inicializadas com N(0, 0.6).
    <pad> (índice 0) é sempre zeros.
    """
    import gensim.downloader as api

    if source == 'glove':
        model_name = f"glove-wiki-gigaword-{embed_dim}"
    elif source == 'fasttext':
        model_name = "fasttext-wiki-news-subwords-300"
        embed_dim = 300  # fasttext só existe em 300d
    else:
        raise ValueError(f"source deve ser 'glove' ou 'fasttext', não '{source}'")

    print(f"A carregar {model_name}...")
    vectors = api.load(model_name)

    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embed_dim)).astype(np.float32)
    embedding_matrix[0] = 0  # <pad> = zeros

    found = 0
    for word, i in vocab.items():
        if word in vectors:
            embedding_matrix[i] = vectors[word]
            found += 1

    print(f"{source} {embed_dim}d: {found}/{vocab_size} palavras encontradas ({100*found/vocab_size:.1f}%)")
    return torch.tensor(embedding_matrix), embed_dim

def make_dataloaders_from_arrays(X_train, y_train, X_val, y_val, batch_size=64):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader