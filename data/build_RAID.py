import os
import pandas as pd

from datasets import load_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "raw", "RAID_filtered.csv")

if os.path.exists(save_path):
    print("File already exists:", save_path)
    df = pd.read_csv(save_path)
else:
    # chatgpt = GPT-3.5-turbo (add later if we dont have enough gpt4 data)
    KEEP_MODELS = {"gpt4", "human"}
    print(f"Streaming RAID, collecting: {KEEP_MODELS} (no attacks)...")
    rows = []
    dataset = load_dataset("liamdugan/raid", "raid", streaming=True)
    for row in dataset["train"]:
        if row["model"] in KEEP_MODELS and row["attack"] == "none":
            rows.append({"text": row["generation"], "model": row["model"], "domain": row["domain"]})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

# Inspect
print("\nShape:", df.shape)
print("\nSamples per model:")
print(df["model"].value_counts())
print("\nSamples per domain:")
print(df["domain"].value_counts())
word_counts = df["text"].apply(lambda x: len(str(x).split()))
print(f"\nWord count stats:\n{word_counts.describe()}")
print("\nSample gpt4 text (first 300 chars):")
print(df[df["model"] == "gpt4"]["text"].iloc[0][:300])
print("\nSample human text (first 300 chars):")
print(df[df["model"] == "human"]["text"].iloc[0][:300])

# Chunk into sentence-aware ~100-word segments

import re

def chunk_by_sentences(text: str, min_words: int = 80, max_words: int = 120) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_words and current_len >= min_words:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len
    if current_len >= min_words:
        chunks.append(" ".join(current))
    return chunks

label_map = {"gpt4": "OpenAI", "human": "Human"}

rows_chunked = []
for _, row in df.iterrows():
    for chunk in chunk_by_sentences(str(row["text"])):
        rows_chunked.append({"Text": chunk, "Label": label_map[row["model"]]})

df_chunked = pd.DataFrame(rows_chunked)

print(f"\nChunked dataset size: {len(df_chunked)}")
print(df_chunked["Label"].value_counts())
word_counts = df_chunked["Text"].apply(lambda x: len(x.split()))
print(f"\nWord count stats after chunking:\n{word_counts.describe()}")

out_path = os.path.join(script_dir, "dataset_RAID.csv")
df_chunked.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
