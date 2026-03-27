import os
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, "raw", "HC3_raw.parquet")

# Available subsets: finance, medicine, open_qa, reddit_eli5, wiki_csai
# Academic/scientific subsets only:
KEEP_SUBSETS = {"finance", "medicine", "wiki_csai"}

if os.path.exists(raw_path):
    print("Raw parquet found, loading...")
    df = pd.read_parquet(raw_path)
else:
    from huggingface_hub import hf_hub_download
    print("Downloading HC3 raw JSONL files...")
    os.makedirs(os.path.join(script_dir, "raw"), exist_ok=True)
    all_subsets = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    dfs = []
    for subset in all_subsets:
        path = hf_hub_download(
            repo_id="Hello-SimpleAI/HC3",
            filename=f"{subset}.jsonl",
            repo_type="dataset",
        )
        dfs.append(pd.read_json(path, lines=True).assign(source=subset))
        print(f"  {subset}: {len(dfs[-1])} rows")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(raw_path, index=False)
    print(f"Saved {len(df)} rows to {raw_path}")

df = df[df["source"].isin(KEEP_SUBSETS)]

print(f"Rows after domain filter: {len(df)}")
print(df["source"].value_counts())


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
for _, row in df.iterrows():
    for ans in row["human_answers"]:
        for chunk in chunk_by_sentences(str(ans)):
            rows.append({"Text": chunk, "Label": "Human"})

df_chunked = pd.DataFrame(rows)

print(f"\nChunked dataset size: {len(df_chunked)}")
wc = df_chunked["Text"].apply(lambda x: len(x.split()))
print(f"Word count stats:\n{wc.describe().round(1)}")

out_path = os.path.join(script_dir, "domain_treated_datasets", "dataset_HC3_domains.csv")
df_chunked.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
