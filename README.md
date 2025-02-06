```
data/  
|── raw/               # Raw data downloaded from ChEMBL  
|── processed/         # Includes processed data  

src/
|── modeling.py        # Implements random forest and SVM models  
|── regression.py      # Implements multiple linear regression (with feature selection)  
|── data_processing.py # Basic data processing  
|── compute_features.py # Computes additional structural features based on SMILES strings  
|── utils.py           # Utility functions  

```