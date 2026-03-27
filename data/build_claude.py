import os
import re
import pandas as pd

from datasets import load_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "raw", "claude_raw.parquet")

if os.path.exists(save_path):
    print("File already exists:", save_path)
    df = pd.read_parquet(save_path)
else:
    print("File does not exist, will download/save.")
    dataset = load_dataset("Norquinal/claude_multiround_chat_30k")
    print("Splits:", dataset.keys())
    df = dataset["train"].to_pandas()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_parquet(save_path, index=False)

# Inspect structure
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)


# Extract all Claude (gpt) responses per conversation
def get_claude_turns(convo) -> list[str]:
    return [turn["value"] for turn in convo if turn["from"] == "gpt"]

df["claude_texts"] = df["conversations"].apply(get_claude_turns)

# Analyse turn counts
turn_counts = df["claude_texts"].apply(len)
print(f"\nClaude turns per conversation — min: {turn_counts.min()}, max: {turn_counts.max()}, mean: {turn_counts.mean():.1f}")

# Code detection: flag conversations where any Claude turn contains code
CODE_PATTERNS = re.compile(
    r'```|'                          # markdown code block
    r'\bdef \w+\s*\(|'              # Python function
    r'\bclass \w+[\s:(]|'           # Python/Java class
    r'\bimport \w+|'                # import statement
    r'\bfunction\s+\w+\s*\(|'      # JS function
    r'#include\s*<|'                # C/C++ include
    r'\bpublic\s+static\s+|'       # Java
    r'=>\s*\{',                     # arrow function
    re.MULTILINE
)

MATH_PATTERNS = re.compile(
    r'[±√∑∫∂∞≤≥≠×÷]|'             # math unicode symbols
    r'\b\d*[a-z]\d*\s*[=<>]\s*[-\d(√]|'  # equations like x = 3, ax2 = ...
    r'\(\s*[-\w]+\s*[±+\-]\s*\w+\s*\)\s*/\s*\d|'  # fractions like (-b ± ...) / 2
    r'(?<!\w)\d+[a-z]\d+(?!\w)',   # math notation like ax2, b2
    re.MULTILINE
)

def has_code(claude_turns: list[str]) -> bool:
    return any(CODE_PATTERNS.search(t) for t in claude_turns)

def has_math(claude_turns: list[str]) -> bool:
    return any(MATH_PATTERNS.search(t) for t in claude_turns)

df["has_code"] = df["claude_texts"].apply(has_code)
df["has_math"] = df["claude_texts"].apply(has_math)

total = len(df)
code_rows = df["has_code"].sum()
math_rows = df["has_math"].sum()
both = (df["has_code"] | df["has_math"]).sum()
clean = (~df["has_code"] & ~df["has_math"]).sum()
print(f"\nConversations with code:      {code_rows} / {total} ({100*code_rows/total:.1f}%)")
print(f"Conversations with math:      {math_rows} / {total} ({100*math_rows/total:.1f}%)")
print(f"Conversations with code+math: {both} / {total} ({100*both/total:.1f}%)")
print(f"Clean conversations:          {clean} / {total} ({100*clean/total:.1f}%)")


# Chunk each Claude turn into sentence-aware ~100-word segments

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

df_clean = df[~df["has_code"] & ~df["has_math"]]

rows = []
for _, row in df_clean.iterrows():
    for turn_text in row["claude_texts"]:
        for chunk in chunk_by_sentences(turn_text):
            rows.append({"Text": chunk, "Label": "Anthropic"})

df_chunked = pd.DataFrame(rows)

print(f"\nChunked dataset size: {len(df_chunked)}")
word_counts = df_chunked["Text"].apply(lambda x: len(x.split()))
print(f"Word count stats:\n{word_counts.describe()}")

out_path = os.path.join(script_dir, "clean_datasets", "dataset_claude.csv")
df_chunked.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
