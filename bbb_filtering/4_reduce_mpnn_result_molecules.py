import pandas as pd

top_n = 50

# Load the dataset
df = pd.read_csv('/Users/asadyamin/Documents/cs229/MPNN/predictions.csv')

# Filter for positive predictions
df_positive = df[df["Prediction"] == 1]

# Sort by probability in descending order and select top 50
df_top_positive = df_positive.sort_values(by="Probability", ascending=False).head(top_n)

#Save to CSV
df_top_positive.to_csv('/Users/asadyamin/Documents/cs229/MPNN/predictions_filtered_50.csv', index=False)

print(f"Predictions saved")