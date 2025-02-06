'''
This script performs leave-one-out cross-validation (LOOCV) for different models:
- Random Forest
- Support Vector Machine
- Linear Regression (naive without features selection)
'''


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import utils

class Model:
    def __init__ (self, file_path, model_type):
        self.file_path = file_path
        self.data = None
        self.model_type = model_type
        self.X = None
        self.y = None
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.dropna(subset=['Smiles', 'pChEMBL Value'])
        
        # Only keep numerical columns and get rid of ligand efficiency (uses binding affinity to calculate)
        self.data = self.data.drop(columns=['Molecule ChEMBL ID', 'Smiles', 'Standard Type', 'Standard Relation', 'Standard Value', 'Standard Units', 'Ligand Efficiency BEI', 'Ligand Efficiency LE', 'Ligand Efficiency LLE', 'Ligand Efficiency SEI'])
        
        # Check for NaN values
        nan_rows = self.data[self.data.isna().any(axis=1)]
        if not nan_rows.empty:
            print("Rows with NaN values:")
            print(nan_rows)
        
        # Get X, y
        self.X = self.data.drop(columns=['pChEMBL Value'])
        self.y = self.data['pChEMBL Value']
    
    def initialize_model(self):
        if self.model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "svm":
            self.model = SVR(kernel='rbf') # LOOK INTO WHICH KERNEL TO USE
        elif self.model_type == "lr":
            self.model = LinearRegression()
        else:
            raise ValueError("Please enter a valid model type ('rf', 'svm', or 'lr')")
    
    def rf_feature_importance(self):
        # Get feature importances from the Random Forest model
        importances = self.model.feature_importances_

        # Create a DataFrame for better visualization
        feature_importances = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': importances
        })

        # Sort by importance
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        # Display the top 10 features
        print("Ten most important features:")
        print(feature_importances.head(10))
    
    def loocv(self):
        # Perform Leave-One-Out Cross-Validation
        mse = utils.perform_loocv(self.X, self.y, self.model)
        print(f"MSE: {round(mse,4)}")

    def run(self):
        self.load_data()
        self.initialize_model()

        # Run LOOCV for given model
        print(f"Performing LOOCV for {self.model_type} model...")
        self.loocv()

        # Run feature importance for random forests
        if self.model_type == "rf":
            self.rf_feature_importance()
    
def main():
    file_path = "../data/processed/ChEMBL-alpha2-bioactivities-274-bulked.csv"

    # Linear regression
    lr_model = Model(file_path, "lr")
    lr_model.run()
    
    # Random forests
    rf_model = Model(file_path, "rf")
    rf_model.run()

    # SVM
    svm_model = Model(file_path, "svm")
    svm_model.run()

if __name__ == "__main__":
    main()