import os
import pandas as pd

SAMPLES_PER_CLASS = 14000
RANDOM_SEED = 2026

script_dir = os.path.dirname(os.path.abspath(__file__))
CLEAN  = os.path.join(script_dir, "clean_datasets")
DOMAIN = os.path.join(script_dir, "domain_treated_datasets")

# Human and OpenAI use domain-filtered datasets; others use clean datasets
SOURCES = [
    (f"{CLEAN}/dataset_OTB.csv",             "Text",  "Label"),   # Google + Meta
    (f"{CLEAN}/dataset_claude.csv",           "Text",  "Label"),   # Anthropic
    (f"{DOMAIN}/dataset_RAID_domains.csv",    "Text",  "Label"),   # OpenAI (academic)
    (f"{DOMAIN}/dataset_HC3_domains.csv",     "Text",  "Label"),   # Human (academic)
    (f"{CLEAN}/dataset_human_filtrado.csv",   "text",  "label"),   # Human (extra)
    # Original small datasets
    (f"{CLEAN}/dataset_anthropic.csv",        "text",  "label"),
    (f"{CLEAN}/dataset_google.csv",           "text",  "label"),
    (f"{CLEAN}/dataset_meta.csv",             "text",  "label"),
    (f"{CLEAN}/dataset_openai.csv",           "text",  "label"),
]

dfs = []
missing = []
for filename, text_col, label_col in SOURCES:
    path = filename
    if not os.path.exists(path):
        print(f"  [MISSING] {os.path.basename(filename)} — run the corresponding build script first")
        missing.append(filename)
        continue
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        print(f"  [ERROR]   {os.path.basename(filename)} — expected columns '{text_col}' and '{label_col}', got {df.columns.tolist()}")
        missing.append(filename)
        continue
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"])
    dfs.append(df)
    print(f"  [OK]      {os.path.basename(filename):40s}: {len(df):>7,} rows  labels: {df['label'].unique().tolist()}")

if missing:
    print(f"\n{len(missing)} file(s) missing or invalid — fix them before continuing.")
    exit(1)

df_all = pd.concat(dfs, ignore_index=True)
print(f"\nTotal before sampling: {len(df_all):,}")
print(df_all["label"].value_counts())

# Sample at max SAMPLES_PER_CLASS per label
sampled = df_all.groupby("label").sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
sampled = sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nFinal dataset: {len(sampled):,} rows")
print(sampled["label"].value_counts())

out_path = os.path.join(script_dir, "domain_treated_datasets", "dataset_final_domains.csv")
sampled.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
