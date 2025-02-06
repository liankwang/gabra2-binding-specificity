''' Perform some simple data processing, including 
- Computing pChEMBL values from standard values for those missing
'''

import numpy as np
import pandas as pd

# Load the datasets
file_path = "../data/raw/ChEMBL-alpha2-bioactivities-274.csv"
data = pd.read_csv(file_path, delimiter=';')

# Check which columns have missing values
print(data.shape)
print(data.columns)
print(data.isnull().sum())

# Only select certain rows
data = data[data['Assay Organism'] == 'Homo sapiens']
data = data[data['Assay Type'] == 'B']
data = data[data['Standard Type'].isin(['K', 'Ki', 'IC50'])]
data = data.dropna(subset=['Standard Value', 'pChEMBL Value'], how='all')

# Select relevant columns
relevant_columns = [
    "Molecule ChEMBL ID", "Molecular Weight", "AlogP", "Smiles", "Standard Type", "Standard Relation",
    "Standard Value", "Standard Units", "pChEMBL Value", 'Ligand Efficiency BEI', 'Ligand Efficiency LE',
       'Ligand Efficiency LLE', 'Ligand Efficiency SEI'
]
data_cleaned = data[relevant_columns]
data_cleaned

# Fill in missing pChEMBL values
for index, row in data_cleaned[data_cleaned['pChEMBL Value'].isnull()].iterrows():
    print(f"Row {index} has a missing pChEMBL Value.")

    # Convert Standard Value from uM to nM
    if row['Standard Units'] == 'uM':
        M = float(row['Standard Value']) / float(10**6)
    elif row['Standard Units'] == 'nM':
        M = float(row['Standard Value']) / float(10**9)
    
    # Calculate pChEMBL Value
    pChEMBL = -np.log10(M)
    data_cleaned.at[index, 'pChEMBL Value'] = pChEMBL

# Save the cleaned data
print("Cleaned data shape: ", data_cleaned.shape)
data_cleaned.to_csv("../data/processed/ChEMBL-alpha2-bioactivities-274.csv", index=False)
