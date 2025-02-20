'''
Performs several linear regressions of pChEMBL Value with features:
- Molecular Weight and AlogP
- All numerical features, including all predictors computed from SMILES strings 
- Features selected via univariate feature selection for several k's
'''

import pandas as pd
import numpy as np
import utils
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from mlxtend.feature_selection import SequentialFeatureSelector


# Load the data
data = pd.read_csv("data/processed/ChEMBL-alpha2-bioactivities-274-bulked.csv")
data = data.drop(columns=['Molecule ChEMBL ID', 'Smiles', 'Standard Type', 'Standard Relation', 'Standard Value', 'Standard Units', 'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI'])
X_simple = data[['Molecular Weight', 'AlogP']]
X = data.drop(columns=['pChEMBL Value'])
Y = data['pChEMBL Value']

# Initialize model
model = LinearRegression()

# With only simple features
print("Running LOOCV with only Mol Weight and AlogP")
mse = utils.perform_loocv(X_simple, Y, model)
print(f"MSE: {round(mse,4)}")

#  With all features -- massive overfitting
print("Running LOOCV with all features -- overfitting!!")
mse = utils.perform_loocv(X, Y, model)
print(f"MSE: {round(mse,4)}")

# With feature selection
import matplotlib.pyplot as plt

print("Running LOOCV with feature selection (univariate)")
k_values = [3, 10, 15, 17, 20, 25, 30, 50, 100]
rmse_values = []

for k in k_values:
    print(f"Univariate selection with k={k}")
    selector = SelectKBest(f_regression, k=k)
    X_new = pd.DataFrame(selector.fit_transform(X, Y))
    selected_features = selector.get_support(indices=True)
    print(f"Selected features: {X.columns[selected_features]}")

    rmse = utils.perform_loocv(X_new, Y, model)
    rmse_values.append(rmse)
    print(f"RMSE: {round(rmse,4)}")

# Plotting k against RMSE
plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o')
plt.xlabel('Number of Features (k)')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Features (k)')
plt.grid(True)
plt.savefig('output/rmse_vs_k.png')
plt.show()
