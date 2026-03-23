import os
import pandas as pd

SAMPLES_PER_CLASS = 60000
RANDOM_SEED = 2026

script_dir = os.path.dirname(os.path.abspath(__file__))

# All source datasets with their file and the column names they use
# Each entry: (filepath, text_col, label_col)
SOURCES = [
    # New large datasets
    ("dataset_OTB.csv",          "Text",  "Label"),
    ("dataset_claude.csv",       "Text",  "Label"),
    ("dataset_RAID.csv",         "Text",  "Label"),
    ("dataset_HC3_human.csv",    "Text",  "Label"),
    ("dataset_human_filtrado.csv", "text", "label"),
    # Original small datasets
    ("dataset_anthropic.csv",    "text",  "label"),
    ("dataset_google.csv",       "text",  "label"),
    ("dataset_meta.csv",         "text",  "label"),
    ("dataset_openai.csv",       "text",  "label"),
]

dfs = []
missing = []
for filename, text_col, label_col in SOURCES:
    path = os.path.join(script_dir, filename)
    if not os.path.exists(path):
        print(f"  [MISSING] {filename} — run the corresponding build script first")
        missing.append(filename)
        continue
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        print(f"  [ERROR]   {filename} — expected columns '{text_col}' and '{label_col}', got {df.columns.tolist()}")
        missing.append(filename)
        continue
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])
    dfs.append(df)
    print(f"  [OK]      {filename:35s}: {len(df):>7,} rows  labels: {df['label'].unique().tolist()}")

if missing:
    print(f"\n{len(missing)} file(s) missing or invalid — fix them before continuing.")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal before sampling: {len(df_all):,}")
print(df_all["label"].value_counts())

# Sample at max SAMPLES_PER_CLASS per label
sampled = (
    df_all
    .groupby("label", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), SAMPLES_PER_CLASS), random_state=RANDOM_SEED))
)
sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nFinal dataset: {len(sampled):,} rows")
print(sampled["label"].value_counts())

out_path = os.path.join(script_dir, "dataset_final.csv")
sampled.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
