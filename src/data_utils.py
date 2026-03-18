import numpy as np
import pandas as pd
import torch
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

def make_dataloaders_from_arrays(X_train, y_train, X_val, y_val, batch_size=64):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader