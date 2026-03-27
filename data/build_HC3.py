import os
import pandas as pd
from huggingface_hub import hf_hub_download


script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "raw", "HC3_raw.parquet")

if os.path.exists(save_path):
    print("File already exists:", save_path)
    df = pd.read_parquet(save_path)
else:
    print("Downloading HC3 raw JSONL files...")
    os.makedirs(os.path.join(script_dir, "raw"), exist_ok=True)

    # HC3 has one JSONL file per domain — download all English subsets
    subsets = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    dfs = []
    for subset in subsets:
        path = hf_hub_download(
            repo_id="Hello-SimpleAI/HC3",
            filename=f"{subset}.jsonl",
            repo_type="dataset",
        )
        dfs.append(pd.read_json(path, lines=True).assign(source=subset))
        print(f"  {subset}: {len(dfs[-1])} rows")

    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")

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

rows = []
for answers in df["human_answers"]:
    for ans in answers:
        for chunk in chunk_by_sentences(str(ans)):
            rows.append({"Text": chunk, "Label": "Human"})

df_chunked = pd.DataFrame(rows)

print(f"\nChunked dataset size: {len(df_chunked)}")
wc_chunked = df_chunked["Text"].apply(lambda x: len(x.split()))
print(f"Word count stats after chunking:\n{wc_chunked.describe().round(1)}")

out_path = os.path.join(script_dir, "clean_datasets", "dataset_HC3_human.csv")
df_chunked.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
