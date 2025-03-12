import numpy as np
import pandas as pd
import argparse

from rdkit import Chem
import torch
from torch_geometric.data import Data, InMemoryDataset

"""
This script creates a PyTorch Geometric dataset from a list of SMILES strings and (optionally) labels.
Arguments:
--data_path: Path to a CSV file with columns 'Smiles' and 'Interaction' (if labels are available)
--output_path: Path to save the PyTorch Geometric dataset. The file will be saved as "{output_path}.pt"
--no_labels (optional): If this flag is set, the dataset will be created without labels (even if it is available in the CSV file)
"""

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def get_atom_features(atom):
    """
    Get atom features for a single atom, with features based on the MolGraphConv featurizer in deepchem:
    - Atom type: One-hot vector of length 10 (C, N, O, F, P, S, Cl, Br, I, other)
    - Formal charge
    - Hybridization: one-hot vector of length 4 (SP, SP2, SP3, other)
    - Aromatic: 1 if atom is aromatic, 0 otherwise
    - Degree: one-hot vector of length 7 (0, 1, 2, 3, 4, 5, other)
    - Num hydrogens: one-hot vector of length 6 (0, 1, 2, 3, 4, other)
    """
    # Atom type: One-hot vector of length 10
    type_map = {
        'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'Br': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'I': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    }
    symb = atom.GetSymbol()
    if symb in ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']:
        type = type_map[symb]
    else:
        type = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


    # Formal charge
    charge = [atom.GetFormalCharge()]
    
    # Hybridization: one-hot vector of length 4
    hybridization = atom.GetHybridization()
    if hybridization == Chem.rdchem.HybridizationType.SP:
        hybrid = [1, 0, 0, 0, 0]
    elif hybridization == Chem.rdchem.HybridizationType.SP2:
        hybrid = [0, 1, 0, 0, 0]
    elif hybridization == Chem.rdchem.HybridizationType.SP3:
        hybrid = [0, 0, 1, 0, 0]
    elif hybridization == Chem.rdchem.HybridizationType.S:
        hybrid = [0, 0, 0, 1, 0]
    else:
        hybrid = [0, 0, 0, 0, 0]
        print('Hybridization not SP, SP2, SP3, or S. Instead:', hybridization)

    # Hydro gen bonding

    # Aromatic
    aromatic = [atom.GetIsAromatic()]

    # Degree
    deg_map = {
        0: [1, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0], 
        3: [0, 0, 0, 1, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0]
    }
    degree = deg_map.get(atom.GetDegree(), [0, 0, 0, 0, 0, 0, 1])

    # Num hydrogens:
    num_h_map = {0: [1, 0, 0, 0, 0, 0],
                 1: [0, 1, 0, 0, 0, 0],
                 2: [0, 0, 1, 0, 0, 0],
                 3: [0, 0, 0, 1, 0, 0],
                 4: [0, 0, 0, 0, 1, 0]}
    num_h = num_h_map.get(atom.GetTotalNumHs(), [0, 0, 0, 0, 0, 1])
    
    return np.concatenate([type, charge, hybrid, aromatic, degree, num_h])

def get_bond_features(bond):
    """Get bond features for a single bond, with features based on the MolGraphConv featurizer in deepchem:
    - Bond type: one-hot vector of length 5 (single, double, triple, aromatic, other)
    - Same ring: 1 if bond is in a ring, 0 otherwise
    - Conjugated: 1 if bond is conjugated, 0 otherwise
    - Stereo: one-hot vector of length 5 (STEREONONE, STEREOANY, STEREOZ, STEREOE, other)
    """

    # Bond type: one-hot vector of length 5
    bond_type = bond.GetBondType()
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1, 0],
    }
    bond_type_one_hot = bond_type_map.get(bond_type, [0, 0, 0, 0, 1])
    bond_type_one_hot = bond_type_map[bond_type]

    # Same ring
    same_ring = bond.IsInRing()

    # Conjugated?
    conj = bond.GetIsConjugated()

    # Stereo
    stereo = bond.GetStereo()
    bond_stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: [1, 0, 0, 0, 0],
        Chem.rdchem.BondStereo.STEREOANY: [0, 1, 0, 0, 0],
        Chem.rdchem.BondStereo.STEREOZ: [0, 0, 1, 0, 0],
        Chem.rdchem.BondStereo.STEREOE: [0, 0, 0, 1, 0],
    }
    bond_stereo_one_hot = bond_stereo_map.get(stereo, [0, 0, 0, 0, 1])

    return np.concatenate([bond_type_one_hot, [same_ring], [conj], bond_stereo_one_hot])
    
def get_distance(conf, i, j):
    if conf is None:
        return 0
    pos_i = np.array(conf.GetAtomPosition(i))
    pos_j = np.array(conf.GetAtomPosition(j))
    return np.linalg.norm(pos_i - pos_j)

def get_bond_angle(conf, mol, i, j):
    
    # Get neighbors of j
    neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors()]
    neighbors.remove(i)

    if len(neighbors) == 0 or conf is None:
        return -1
    
    # Get positions of neighbors
    pos_i = np.array(conf.GetAtomPosition(i))
    pos_j = np.array(conf.GetAtomPosition(j))
    pos_k = np.array(conf.GetAtomPosition(neighbors[0]))

    # Create vectors from atom j to atoms i and k
    v1 = pos_i - pos_j
    v2 = pos_k - pos_j

    # Compute angle between vectors
    cos_theta = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta) * (180.0 / np.pi)
    return theta

def mol_to_graph(mol, label=None):
    """Convert an RDKit molecule into a PyTorch Geometric graph object. 
    Calls atom and bond featurizers.
    """
    #print(f"Working on mol {Chem.MolToSmiles(mol)} with label {label}")

    # Add hydrogens
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    
    # Get 3D coordinates
    # num_fails = 0
    # conf = None
    # if mol.GetNumConformers() == 0:
    #     success = AllChem.EmbedMolecule(mol, randomSeed=42)
    #     if success != 0:
    #         print(f"Failed to generate 3D coordinates: Mol {Chem.MolToSmiles(mol)}.")
    #         num_fails += 1
    #     else:
    #         #print(f"Generated 3D coordinates for Mol {Chem.MolToSmiles(mol)}.")
    #         conf = mol.GetConformer()

    # Compute atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(np.array(atom_features), dtype=torch.float)

    # Compute edge index (adjacency matrix) and edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))  # Undirected Graph

        # Add distance feature
        # dist = get_distance(conf, i, j)

        # Add bond angle feature
        # bond_angle = get_bond_angle(conf, mol, i, j)

        # Add other features
        all_feats = np.concatenate([get_bond_features(bond)])
        edge_attr.append(all_feats)

    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    if label is not None:
        y = torch.tensor([label], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    

class MolGraphDataset(InMemoryDataset):
    """ Creates a PyTorch Geometric dataset from a list of RDKit molecules and labels. """
    def __init__(self, mols, smiles_list, labels=None, transform=None, pre_transform=None):
        self.mols = mols
        self.smiles = smiles_list
        super(MolGraphDataset, self).__init__('.', transform, pre_transform)

        if labels is not None:
            self.labels = labels
            data_list = [mol_to_graph(mol, label) for mol, label in zip(mols, labels)]
        else:
            data_list = [mol_to_graph(mol) for mol in mols]
            
        
        self.data, self.slices = self.collate(data_list)


    def __len__(self):
        return len(self.mols)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--no_labels', action='store_true')
    args = parser.parse_args()
    
    df = pd.read_csv(args.data_path)
    smiles_list = df['Smiles'].tolist()

    # Create Mol objects from SMILES strings
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    if 'Interaction' in df.columns and not args.no_labels:
        labels = torch.tensor(df['Interaction'].tolist(), dtype=torch.long)
        dataset = MolGraphDataset(mols, smiles_list, labels)
    else:
        print("Creating dataset with no true labels.")
        dataset = MolGraphDataset(mols, smiles_list)

    torch.save(dataset, f"{args.output_path}.pt")
    print(f"Saved dataset to {args.output_path}.pt")


if __name__ == '__main__':
    main()