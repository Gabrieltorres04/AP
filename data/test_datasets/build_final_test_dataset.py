import pandas as pd

files = {
    'exemplos': 'dataset-exemplos.csv',
    'subm1':    'subm1_labels_revealed.csv',
    'subm2':    'subm2_labels_revealed.csv',
}

dfs = []
for name, fname in files.items():
    df = pd.read_csv(fname, sep=';', on_bad_lines='skip')
    df['_source'] = name
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"Total rows before dedup: {len(combined)}")

# Drop duplicates by Text, keeping the first occurrence (exemplos > subm1 > subm2)
deduped = combined.drop_duplicates(subset='Text', keep='first')
print(f"Total rows after dedup:  {len(deduped)}")
print(f"Removed: {len(combined) - len(deduped)} rows")
print()
print("Label distribution:")
print(deduped['Label'].value_counts())

deduped = deduped.drop(columns='_source').reset_index(drop=True)
deduped.to_csv('final_test_dataset.csv', sep=';', index=False)
print("\nSaved to final_test_dataset.csv")