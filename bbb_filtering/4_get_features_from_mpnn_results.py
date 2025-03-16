from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\ayamin\OneDrive\Desktop\post_all\predictions_filtered.csv")

smiles_list = df["SMILES"].dropna().tolist()
results = []

for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Compute 2D descriptors
        properties = {
            "SMILES": smiles,
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumRings": rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
            "Scaffold": Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)),
        }
        
        # Generate 3D Conformer
        mol_3d = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_3d) == 0:
            AllChem.UFFOptimizeMolecule(mol_3d)
            properties.update({
                "3D_GyrationRadius": rdMolDescriptors.CalcRadiusOfGyration(mol_3d),
                "3D_Asphericity": rdMolDescriptors.CalcAsphericity(mol_3d),
                "3D_Eccentricity": rdMolDescriptors.CalcEccentricity(mol_3d),
                "3D_InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol_3d),
                "3D_SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol_3d)
            })
        else:
            properties.update({
                "3D_GyrationRadius": np.nan,
                "3D_Asphericity": np.nan,
                "3D_Eccentricity": np.nan,
                "3D_InertialShapeFactor": np.nan,
                "3D_SpherocityIndex": np.nan
            })
        
        results.append(properties)

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results.to_csv(r"C:\Users\ayamin\OneDrive\Desktop\post_all\predictions_filtered_3d_analysis.csv", index=False)
print(f"Analysis complete. Results saved")