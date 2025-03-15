'''' Compute new features from SMILES strings '''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    def __init__(self, data):
        self.data = data
    
    # def load_data(self):
    #     self.data = pd.read_csv(self.file_path)
    #     #self.data = self.data.dropna(subset=['Smiles', 'pChEMBL Value'])
    #     print("Data shape: ", self.data.shape)
        
    #     nan_rows = self.data[self.data.isna().any(axis=1)]
    #     if not nan_rows.empty:
    #         print("Rows with NaN values:")
    #         print(nan_rows)

    def calculate_rdkit_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = []
            for name, func in Descriptors.descList:
                descriptors.append(func(mol))

            return descriptors
        else:
            print('Calculating RDKit descriptors: NaN molecule!')
            return [np.nan] * len(Descriptors.descList)

    def calculate_fingerprints(self, smiles, radius=2, bit_length=1024):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_length)
            return np.array(fingerprint)
        else:
            print('Calculating fingerprints: NaN molecule!')
            return np.zeros(bit_length)

    def combine_features(self):
        # Calculate RDKit descriptors
        descriptors = self.data['Smiles'].apply(self.calculate_rdkit_descriptors)
        descriptors_df = pd.DataFrame(descriptors.tolist(), columns=[name for name, _ in Descriptors.descList])
        print("Number of RDKIT descriptors: ", descriptors_df.shape[1])

        # Calculate Morgan fingerprints
        fingerprints = self.data['Smiles'].apply(self.calculate_fingerprints)
        fingerprints_df = pd.DataFrame(fingerprints.tolist(), columns=[f'FP_{i}' for i in range(fingerprints[0].shape[0])])
        print("Number of fingerprint features: ", fingerprints_df.shape[1])

        # Combine descriptors and fingerprints with the original data
        new_data = pd.concat([self.data, descriptors_df, fingerprints_df], axis=1)
        return new_data

    def run(self):
        return self.combine_features()

if __name__ == "__main__":
    file_path = "../data/processed/ChEMBL-alpha2-bioactivities-274.csv"
    data = pd.read_csv(file_path)
    extractor = FeatureExtractor(data)
    new_data = extractor.run()
    print(new_data.shape)
    new_data.to_csv("../data/processed/ChEMBL-alpha2-bioactivities-274-bulked.csv", index=False)
