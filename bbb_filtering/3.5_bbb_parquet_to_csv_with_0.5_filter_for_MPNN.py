import pandas as pd

# Load merged parquet file
merged_file = r"C:\Users\ayamin\Downloads\data_parquet_bbb\merged_final_passed.parquet"
df = pd.read_parquet(merged_file)

threshold = 0.998510  # 50% reduction to ~1 million molecules

# Filter molecules above the threshold
df_filtered = df[df["bbb_probability"] >= threshold]

# Save filtered dataset
output_filtered = r"C:\Users\ayamin\Downloads\data_parquet_bbb\filtered_molecules_for_GCN.parquet"
df_filtered.to_parquet(output_filtered)

print(f"\nFiltered dataset saved to {output_filtered}")
