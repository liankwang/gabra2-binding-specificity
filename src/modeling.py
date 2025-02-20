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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import utils
from sklearn.model_selection import train_test_split

class Model:
    def __init__ (self, file_path, model_type):
        self.file_path = file_path
        self.data = None
        self.model_type = model_type
        self.X = None
        self.y = None
        self.model = None
    
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
            self.model = RandomForestRegressor(n_estimators=100, 
                                               max_depth=10, 
                                               max_features='sqrt', 
                                               min_samples_leaf=1,
                                               min_samples_split=2,
                                               random_state=42)
        elif self.model_type == "svm":
            self.model = SVR(kernel='rbf') # LOOK INTO WHICH KERNEL TO USE
        elif self.model_type == "lr":
            self.model = LinearRegression()
        else:
            raise ValueError("Please enter a valid model type ('rf', 'svm', or 'lr')")
    
    def rf_feature_importance(self):
        # Fit the model
        self.model.fit(self.X, self.y)

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
        rmse = utils.perform_loocv(self.X, self.y, self.model)
        print(f"RMSE: {round(rmse,4)}")

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
    file_path = "data/processed/ChEMBL-alpha2-bioactivities-274-bulked.csv"

    # Linear regression
    lr_model = Model(file_path, "lr")
    lr_model.run()
    
    # Random forests
    rf_model = Model(file_path, "rf")
    rf_model.run()

    # SVM
    svm_model = Model(file_path, "svm")
    svm_model.run()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(svm_model.X, svm_model.y, test_size=0.2, random_state=42)

    # Hyperparameter tuning for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=2, cv=5)
    grid.fit(X_train, y_train)

    print("Best parameters found for SVM:")
    print(grid.best_params_)

    # Evaluate the best model on the test set
    best_svm_model = grid.best_estimator_
    y_pred = best_svm_model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Test RMSE: {round(test_rmse, 4)}")

    # Evaluate the vanilla SVR (rbf kernel) model on the test set
    vanilla_svr_model = SVR(kernel='rbf')
    vanilla_svr_model.fit(X_train, y_train)
    y_pred_vanilla = vanilla_svr_model.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_test - y_pred_vanilla) ** 2))
    print(f"Base RMSE: {round(test_rmse, 4)}")



if __name__ == "__main__":
    main()