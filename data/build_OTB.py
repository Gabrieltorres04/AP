import os
import re
import pandas as pd

from datasets import load_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "raw", "OTB_raw.csv")
if os.path.exists(save_path):
    print("File already exists:", save_path)
    df = pd.read_csv(save_path)
else:
    print("File does not exist, will download/save.")
    dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain")
    print(dataset.keys())
    df = dataset["train"].to_pandas()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


print("\nAll unique models:")
print(df["model"].unique())
print(f"\nTotal unique models: {df['model'].nunique()}")
print("\nSamples per model:")
print(df["model"].value_counts())

# Map models to the 5 target classes
family_keywords = {
    "Meta":      ["llama"],
    "Google":    ["gemma", "gemini"],
    "OpenAI":    ["gpt"],
    "Anthropic": ["claude"],
    "Human":     ["human"],
}

def assign_family(model_name: str) -> str:
    model_lower = model_name.lower()
    for family, keywords in family_keywords.items():
        if any(kw in model_lower for kw in keywords):
            return family
    return "Unknown"

df["ai_family"] = df["model"].apply(assign_family)

keep_families = ["Human", "OpenAI", "Anthropic", "Google", "Meta"]
df = df[df["ai_family"].isin(keep_families)]

print(f"\nRows after filtering: {len(df)}")
print(df["ai_family"].value_counts())

df = df.drop(columns=["model"])
df = df.rename(columns={"ai_family": "Label"})

# Chunk texts into sentence groups that average ~100-words


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

text_col = "content"
rows = []
for _, row in df.iterrows():
    for chunk in chunk_by_sentences(str(row[text_col])):
        rows.append({"Text": chunk, "Label": row["Label"]})

df_chunked = pd.DataFrame(rows)

print(f"\nChunked dataset size: {len(df_chunked)}")
print(df_chunked["Label"].value_counts())


out_path = os.path.join(script_dir, "clean_datasets", "dataset_OTB.csv")
df_chunked.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")