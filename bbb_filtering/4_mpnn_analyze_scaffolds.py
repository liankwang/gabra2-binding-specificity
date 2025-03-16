import pandas as pd

file_path = r"C:\Users\ayamin\OneDrive\Desktop\post_all\analyzed_molecules_with_clusters_and_scaffolds.csv"

df = pd.read_csv(file_path)

# Identify top 5 most common scaffolds
top_scaffolds = df["Scaffold"].value_counts().head(5).index

# Find the highest probability molecule for each of the top 5 scaffolds
top_molecules = []
for scaffold in top_scaffolds:
    subset = df[df["Scaffold"] == scaffold]
    highest_prob_molecule = subset.loc[subset["PCA1"].idxmax()]
    top_molecules.append(highest_prob_molecule)

# Convert to DataFrame
top_molecules_df = pd.DataFrame(top_molecules)

# Save results to CSV
output_path = r"C:\Users\ayamin\OneDrive\Desktop\post_all\top_5_scaffolds_highest_probability.csv"
top_molecules_df.to_csv(output_path, index=False)

print(f"Done")