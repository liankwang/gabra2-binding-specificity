#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Blood-Brain Barrier (BBB) Penetration Prediction Cascade Pipeline

This file implements a three-stage cascade model for predicting BBB penetration:
1. Stage 1: Fast screening with Random Forest on simple descriptors
2. Stage 2: Medium-detail screening with XGBoost on MACCS fingerprints
3. Stage 3: High-precision screening with Neural Network on comprehensive fingerprints

Many functions were deprecated + duplicated with changes, and heavy addition of argument parsing was added in production and retroactively
"""

import sys
import os
import logging

import rdkit
import torch
import xgboost

import gc
import pickle
import random
import json
import time
import csv
import psutil
import traceback

from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, AllChem, Lipinski, GraphDescriptors, MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# Suppress RDKit logs
RDLogger.DisableLog('rdApp.*')

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, fbeta_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, VarianceThreshold

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Command line
import argparse

#-----------------------------------------------------------------------------------------------------
#           Logging Setup + Config for Files & Models
#----------------------------------------------------------------------------------

# Set up logging to both console and a file
def setup_logging(output_dir: str) -> logging.Logger:

    os.makedirs(output_dir, exist_ok=True)

    # Organize by date (testing multiple times)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"bbb_cascade_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # For console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                       datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Configuration settings for the BBB pipeline
class Config:
    def __init__(self, output_dir: str = "output"):
        # File paths
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.model_dir = os.path.join(output_dir, "models")
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.cache_dir = os.path.join(output_dir, "cache")
        self.log_dir = os.path.join(output_dir, "logs")
        self.input_dir = os.path.join(output_dir, "input")

        # Create directories
        for directory in [self.data_dir, self.model_dir, self.results_dir, 
                          self.plots_dir, self.cache_dir, self.log_dir, self.input_dir]:
            os.makedirs(directory, exist_ok=True)

        # Model files 
        self.stage1_model = os.path.join(self.model_dir, "stage1_model.pkl")
        self.stage2_model = os.path.join(self.model_dir, "stage2_model.pkl")
        self.stage3_model = os.path.join(self.model_dir, "stage3_model.pt") # .pt for tensors
        self.stage3_scaler = os.path.join(self.model_dir, "stage3_scaler.pkl")
        self.thresholds_json = os.path.join(self.results_dir, "thresholds.json")

        # Replicability and efficiency
        self.random_state = 42
        self.max_processes = max(1, cpu_count() - 1)
        # Old (ignore) -----------------------
        self.chunk_size_stage1 = 1000000 
        self.chunk_size_stage2 = 100000  
        self.chunk_size_stage3 = 50000  
        
        # Memory management optimization
        self.max_batch_size_stage3 = 8192  # Maximum batch size for stage 3 NN model
        self.memory_check_interval = 5     # Check memory usage every few batches
        self.min_batch_size = 512          # Minimum batch size
        self.memory_usage_threshold = 0.8  # Fraction of available memory

        # GPU (running on VM and PC)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Logging to see discarded molecules per stage
        self.discarded_log = {
            "stage1": os.path.join(self.log_dir, "stage1_discarded.csv"),
            "stage2": os.path.join(self.log_dir, "stage2_discarded.csv"),
            "stage3": os.path.join(self.log_dir, "stage3_discarded.csv")
        }
        
        # Default fixed thresholds (alterable)
        self.threshold = {
            "stage1": 0.763, # Empirically determined to pass ~50% of molecules through stage 1
            "stage2": 0.956, # Empirically determined to pass ~20% of ~50% of molecules through stage 2
            "stage3": 0.108 
        }
        
# Get current memory usage of the process
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

# Input: CSV substructural patterns
# Output: returns a list of tuples: (bit, description, compiled SMARTS as an RDKit Mol)
# Description: Load substructure patterns per CSV file, with CSV's having columns "Bit", "Description", "SMARTS"
def load_substructure_patterns(filepath: str) -> List[Tuple[int, str, Chem.Mol]]:
    df = pd.read_csv(filepath)
    patterns = []
    for idx, row in df.iterrows():
        try:
            smarts = row["SMARTS"]
            # Compile the SMARTS pattern
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                bit = row["Bit"] 
                description = row["Description"]
                patterns.append((bit, description, pattern))
        except Exception as e:
            logging.warning(f"Failed to load pattern on row {idx}: {e}")
    return patterns

# Input: An RDKit molecule and a substructural pattern
# Output: 1 if the molecule has a match, 0 otherwise in a a binary numpy array
# Description: Computes whether the molecule has the relevant pattern
def compute_substructure_fingerprint(mol: Chem.Mol, patterns: List[Tuple[int, str, Chem.Mol]]) -> np.ndarray:
    fp = np.array([1 if mol.HasSubstructMatch(pattern) else 0 for (_, _, pattern) in patterns], dtype=np.int8)
    return fp

# Load substructure patterns from the standard location off VM
pattern_path = os.path.join("input", "SubStructureFingerprinter.csv")
if not os.path.exists(pattern_path):
    pattern_path = "/home/AYamin/cascading_bbb/input/SubStructureFingerprinter.csv"
    logging.warning(f"Using fallback path for substructure patterns: {pattern_path}")
global_substructure_patterns = load_substructure_patterns(pattern_path)
logging.info(f"Loaded {len(global_substructure_patterns)} substructure patterns from {pattern_path}")


# Global variables for SMARTS patterns
SUB_SMARTS_PATTERNS = {}
KR_SMARTS_PATTERNS = {}
ESTATE_SMARTS_PATTERNS = {}

# List of RDKit 2D descriptors for Stage 3
RDKIT_2D_DESCRIPTORS = [
    # Basic physical properties
    ('MolWt', Descriptors.MolWt),
    ('ExactMolWt', rdMolDescriptors.CalcExactMolWt),
    ('NumHeavyAtoms', Descriptors.HeavyAtomCount),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('TPSA', Descriptors.TPSA),
    ('MolLogP', Descriptors.MolLogP),
    ('MolMR', Descriptors.MolMR),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('NumHeteroatoms', Descriptors.NumHeteroatoms),
    ('NumBridgeheadAtoms', rdMolDescriptors.CalcNumBridgeheadAtoms),
    ('NumSpiroAtoms', rdMolDescriptors.CalcNumSpiroAtoms),
    
    # Charge and ionization-related
    ('MaxAbsPartialCharge', Descriptors.MaxAbsPartialCharge),
    ('MaxPartialCharge', Descriptors.MaxPartialCharge),
    ('MinPartialCharge', Descriptors.MinPartialCharge),
    ('NumRadicalElectrons', Descriptors.NumRadicalElectrons),
    ('FormalCharge', lambda m: Chem.GetFormalCharge(m)),

    # Rings
    ('RingCount', Descriptors.RingCount),
    ('NumAromaticRings', Lipinski.NumAromaticRings),
    ('NumAliphaticRings', Lipinski.NumAliphaticRings),
    ('NumSaturatedRings', Lipinski.NumSaturatedRings),
    
    # Graph-theoretic 
    ('BalabanJ', Descriptors.BalabanJ),
    ('BertzCT', GraphDescriptors.BertzCT),
    ('Chi0v', GraphDescriptors.Chi0v),
    ('Chi1v', GraphDescriptors.Chi1v),
    ('Chi2v', GraphDescriptors.Chi2v),
    ('Chi3v', GraphDescriptors.Chi3v),
    ('Chi4v', GraphDescriptors.Chi4v),
    ('HallKierAlpha', Descriptors.HallKierAlpha),
    ('Kappa1', Descriptors.Kappa1),
    ('Kappa2', Descriptors.Kappa2),
    ('Kappa3', Descriptors.Kappa3),
    
    # Pharmacophore features
    ('LabuteASA', rdMolDescriptors.CalcLabuteASA),
    ('PEOE_VSA1', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[0]),
    ('PEOE_VSA2', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[1]),
    ('PEOE_VSA3', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[2]),
    ('PEOE_VSA4', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[3]),
    ('PEOE_VSA5', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[4]),
    ('PEOE_VSA6', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[5]),
    ('PEOE_VSA7', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[6]),
    ('PEOE_VSA8', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[7]),
    ('PEOE_VSA9', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[8]),
    ('PEOE_VSA10', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[9]),
    ('PEOE_VSA11', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[10]),
    ('PEOE_VSA12', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[11]),
    ('PEOE_VSA13', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[12]),
    ('PEOE_VSA14', lambda mol: rdMolDescriptors.PEOE_VSA_(mol)[13]),
    
    # SlogP-derived descriptors
    ('SlogP_VSA1', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[0]),
    ('SlogP_VSA2', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[1]),
    ('SlogP_VSA3', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[2]),
    ('SlogP_VSA4', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[3]),
    ('SlogP_VSA5', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[4]),
    ('SlogP_VSA6', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[5]),
    ('SlogP_VSA7', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[6]),
    ('SlogP_VSA8', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[7]),
    ('SlogP_VSA9', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[8]),
    ('SlogP_VSA10', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[9]),
    ('SlogP_VSA11', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[10]),
    ('SlogP_VSA12', lambda mol: rdMolDescriptors.SlogP_VSA_(mol)[11]),
    
    # QED (drug-likeness) descriptor
    ('QED', QED.qed),
    
    # Atom counts for common elements
    ('NumCarbon', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'C')),
    ('NumNitrogen', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'N')),
    ('NumOxygen', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'O')),
    ('NumSulfur', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'S')),
    ('NumHalogen', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I'])),
    ('NumPhosphorus', lambda m: sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'P')),
    
    # BBB-specific descriptors
    ('NumPositiveAtoms', lambda m: sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0)),
    ('NumNegativeAtoms', lambda m: sum(1 for a in m.GetAtoms() if a.GetFormalCharge() < 0)),
    ('NumAcidicGroups', lambda m: len(Chem.MolFromSmarts('C(=O)[OH]').GetSubstructMatches(m)))
]

# Input: An RDKit molecule
# Output: The max length of consecutive polar atoms
# Description: Part of BBB-specific additional descriptors
def max_consecutive_polar_atoms(mol):
    """Calculate maximum consecutive polar atoms (O, N, S, P)"""
    try:
        polar_atoms = [i for i, atom in enumerate(mol.GetAtoms()) 
                     if atom.GetSymbol() in ['O', 'N', 'S', 'P']]
        
        if not polar_atoms:
            return 0

        max_length = 0
        for start_atom in polar_atoms:
            visited = set()
            queue = [(start_atom, 1)]  # (atom_idx, path_length)

            while queue:
                current, length = queue.pop(0)
                visited.add(current)
                max_length = max(max_length, length)

                for neighbor in [b.GetOtherAtomIdx(current) for b in mol.GetAtomWithIdx(current).GetBonds()]:
                    if neighbor in polar_atoms and neighbor not in visited:
                        queue.append((neighbor, length + 1))

        return max_length
    except:
        return 0

# Input: An RDKit molecule
# Output: The number of basic nitrogen centers
# Description: Part of BBB-specific additional descriptors
def count_basic_centers(mol):
    try:
        basic_n_pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(NS=O)]")
        if basic_n_pattern:
            matches = mol.GetSubstructMatches(basic_n_pattern)
            return len(matches)
        return 0
    except:
        return 0

# Input: An RDKit molecule
# Output: The number of terminal atoms that can form H-bonds
# Description: Part of BBB-specific additional descriptors
def terminal_h_bond_atoms(mol):
    try:
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['O', 'N']:
                if sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != 'H') <= 1:
                    count += 1
        return count
    except:
        return 0

# Input: An RDKit molecule
# Output: The number of proximal H-bond donor-acceptor pairs
# Description: Part of BBB-specific additional descriptors
def count_proximal_hbond_pairs(mol):
    try:
        donors = [atom.GetIdx() for atom in mol.GetAtoms() 
                if atom.GetSymbol() in ['O', 'N'] and 
                any(nbr.GetSymbol() == 'H' for nbr in atom.GetNeighbors())]

        acceptors = [atom.GetIdx() for atom in mol.GetAtoms() 
                   if atom.GetSymbol() in ['O', 'N'] and 
                   atom.GetIdx() not in donors]

        count = 0
        max_path_len = 7

        for d in donors:
            for a in acceptors:
                if d == a:
                    continue
                try:
                    path = Chem.GetShortestPath(mol, d, a)
                    if path is None:
                        continue
                    path_len = len(path)
                    if 4 <= path_len <= max_path_len:
                        count += 1
                except:
                    continue

        return min(count, 5)  # Cap at 5 to avoid extreme values
    except:
        return 0

# Input: An RDKit molecule
# Output: The H-bond density (H-bond donors and acceptors per molecular weight)
# Description: Part of BBB-specific additional descriptors
def hydrogen_bond_density(mol):
    donors = Descriptors.NumHDonors(mol)
    acceptors = Descriptors.NumHAcceptors(mol)
    mw = Descriptors.MolWt(mol)
    return (donors + acceptors) / mw if mw > 0 else 0

# Input: An RDKit molecule
# Output: The number of rotatable bonds per heavy atom
# Description: Part of BBB-specific additional descriptors
def rotatable_bond_density(mol):
    rb = Descriptors.NumRotatableBonds(mol)
    ha = mol.GetNumHeavyAtoms()
    return rb / ha if ha > 0 else 0

# Add BBB-specific descriptors to list
BBB_DESCRIPTORS = [
    ('MaxConsecutivePolarAtoms', max_consecutive_polar_atoms),
    ('BasicCenters', count_basic_centers),
    ('TerminalHBondAtoms', terminal_h_bond_atoms),
    ('ProximalHBondPairs', count_proximal_hbond_pairs),
    ('LogP_MW_Ratio', lambda m: Descriptors.MolLogP(m) / Descriptors.MolWt(m) if Descriptors.MolWt(m) > 0 else 0),
    ('TPSA_MW_Ratio', lambda m: Descriptors.TPSA(m) / Descriptors.MolWt(m) if Descriptors.MolWt(m) > 0 else 0),
    ('AromaticProportion', lambda m: sum(1 for a in m.GetAtoms() if a.GetIsAromatic()) / m.GetNumAtoms() if m.GetNumAtoms() > 0 else 0),
    ('HBD_HBA_Ratio', lambda m: Descriptors.NumHDonors(m) / Descriptors.NumHAcceptors(m) if Descriptors.NumHAcceptors(m) > 0 else 0)
]

# Combine all descriptor functions
ALL_DESCRIPTORS = RDKIT_2D_DESCRIPTORS + BBB_DESCRIPTORS

# Input: A SMILES string
# Output: All features for SMILES string
# Description: Process a single SMILES string to compute features
def process_single_smiles(smiles):
    return compute_all_features(smiles)

#-----------------------------------------------------------------------------------------
#           Stage 1: Feature Computation (Fast / BBB-Specific) for RF
#------------------------------------------------------------------------------------------

# Input: A SMILES string
# Output: Either a list of features or a dictionary with quick_reject information
# Description: Computed features for the stage 1 RF fast-screening model
def compute_stage1_features(smiles: str) -> Optional[Union[List[float], Dict[str, Any]]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'quick_reject': True, 'reason': 'invalid_smiles'}

        # Quick pre-filter for problematic molecules
        pattern_sulfonic_acid = Chem.MolFromSmarts("[$([#16X4](=[OX1])(=[OX1])([#6])-[OX2H,OX1H0-])]") 
        pattern_phosphonic_acid = Chem.MolFromSmarts("[$([#15X4](=[OX1])(=[OX1])([#6])-[OX2H,OX1H0-])]") 
        pattern_quaternary_n = Chem.MolFromSmarts("[NX4+]")
        pattern_charged = Chem.MolFromSmarts("[+1,+2,+3,+4,-1,-2,-3,-4]")
        
        if pattern_sulfonic_acid and mol.HasSubstructMatch(pattern_sulfonic_acid):
            return {'quick_reject': True, 'reason': 'sulfonic_acid_present'}
        if pattern_phosphonic_acid and mol.HasSubstructMatch(pattern_phosphonic_acid):
            return {'quick_reject': True, 'reason': 'phosphonic_acid_present'}
        if pattern_quaternary_n and mol.HasSubstructMatch(pattern_quaternary_n):
            return {'quick_reject': True, 'reason': 'quaternary_nitrogen_present'}
        if pattern_charged and len(mol.GetSubstructMatches(pattern_charged)) > 1:
            return {'quick_reject': True, 'reason': 'multiple_charged_atoms'}

        # Original 12 features -- from numerical filtering
        mw = Descriptors.MolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        basic_centers = count_basic_centers(mol)
        consecutive_polar = max_consecutive_polar_atoms(mol)
        h_bond_terminals = terminal_h_bond_atoms(mol)
        chi_v3 = GraphDescriptors.Chi3v(mol) if mol.GetNumAtoms() > 4 else 0
        hbond_pairs = count_proximal_hbond_pairs(mol)
        logp = Descriptors.MolLogP(mol)
        rotbonds = Descriptors.NumRotatableBonds(mol)
        original_features = [mw, hba, hbd, tpsa, aromatic_atoms, basic_centers,
                             consecutive_polar, h_bond_terminals, chi_v3, hbond_pairs,
                             logp, rotbonds]

        # Additional 5 features previously used -- from numerical filtering
        fsp3 = Descriptors.FractionCSP3(mol)
        ring_count = Lipinski.RingCount(mol)
        ring_atom_ratio = ring_count / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
        psa_mw_ratio = tpsa / mw if mw > 0 else 0
        complexity = GraphDescriptors.BertzCT(mol)
        additional_features = [fsp3, ring_count, ring_atom_ratio, psa_mw_ratio, complexity]

        # Extra descriptors to help filter molecules:
        chi0v = GraphDescriptors.Chi0v(mol)
        chi1v = GraphDescriptors.Chi1v(mol)
        kappa1 = Descriptors.Kappa1(mol)
        kappa2 = Descriptors.Kappa2(mol)
        kappa3 = Descriptors.Kappa3(mol)
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        extra_features = [chi0v, chi1v, kappa1, kappa2, kappa3, num_heavy_atoms]

        # Combine all features into one vector
        all_features = original_features + additional_features + extra_features
        return all_features

    except Exception as e:
        logging.debug(f"Error in compute_stage1_features for SMILES {smiles}: {e}")
        return {'quick_reject': True, 'reason': f'computation_error: {str(e)}'}

#------------------------------------------------------------------------
#           Stage 2 Feature Computation (MACCS + BBB Descriptors) for XGBoost
#------------------------------------------------------------------------

# Input: SMILES String
# Output: Computed Stage 2 (XGBoost) features as Dict
# Description: Computes existing descriptors, functional group counts, substructure fingerprints, and MACCS and ECFP4 fingeprints
def compute_stage2_features(smiles: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Compute existing descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rotbonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Lipinski.NumAromaticRings(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        mr = Descriptors.MolMR(mol)
        heavy_halogen = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in ['Br', 'I'])
        aromatic_proportion = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / max(1, mol.GetNumAtoms())

        # Compute MACCS fingerprint
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)

        # Compute ECFP4 fingerprint
        try:
            mol_h = Chem.AddHs(mol)
            from rdkit.Chem import rdFingerprintGenerator
            fp_size = 1024
            radius = 2
            ao = rdFingerprintGenerator.AdditionalOutput()
            ao.AllocateBitInfoMap()
            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
            ecfp4 = morgan_gen.GetFingerprint(mol_h, additionalOutput=ao)
            ecfp4_array = np.zeros((fp_size,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(ecfp4, ecfp4_array)
        except Exception as e:
            logging.warning(f"ECFP4 fingerprint failed for {smiles}: {e}")
            ecfp4_array = np.zeros((1024,), dtype=np.int8)

        # Functional group counts
        carboxylic_pattern = Chem.MolFromSmarts("C(=O)[OH]")
        n_carboxylic = len(mol.GetSubstructMatches(carboxylic_pattern)) if carboxylic_pattern else 0
        amine_pattern = Chem.MolFromSmarts("[NX3;!$(NC=O)]")
        n_amine = len(mol.GetSubstructMatches(amine_pattern)) if amine_pattern else 0

        descriptors = np.array([
            mw, logp, tpsa, hba, hbd, rotbonds, aromatic_rings, formal_charge, mr,
            heavy_halogen, aromatic_proportion, n_carboxylic, n_amine
        ], dtype=np.float32)

        # Compute the substructure fingerprint with loaded CSV
        sub_fp = compute_substructure_fingerprint(mol, global_substructure_patterns)
        return {
            'descriptors': descriptors,
            'maccs': maccs_array,
            'ecfp4': ecfp4_array,
            'substructure': sub_fp
        }
    except Exception as e:
        logging.debug(f"Error in compute_stage2_features for SMILES {smiles}: {e}")
        return None

#-------------------------------------------------------------------------------
#           Stage 3 Feature Computation: Many Features for NN
#------------------------------------------------------------------------------

# Initializes worker processes with SMARTS patterns
def pool_initializer(sub_patterns, kr_patterns, estate_patterns):
    
    global SUB_SMARTS_PATTERNS, KR_SMARTS_PATTERNS, ESTATE_SMARTS_PATTERNS
    
    # Set patterns in each worker process
    SUB_SMARTS_PATTERNS = sub_patterns
    KR_SMARTS_PATTERNS = kr_patterns
    ESTATE_SMARTS_PATTERNS = estate_patterns

# Loads SMARTS patterns from CSV files for all imported fingerprints
def load_smarts_patterns(sub_filepath, kr_filepath, estate_filepath=None):
    global SUB_SMARTS_PATTERNS, KR_SMARTS_PATTERNS, ESTATE_SMARTS_PATTERNS
    
    # Load Substructure SMARTS
    logging.info(f"Loading Substructure SMARTS patterns from {sub_filepath}")
    if os.path.exists(sub_filepath):
        sub_df = pd.read_csv(sub_filepath)
        for i, row in sub_df.iterrows():
            try:
                bit = row["Bit"]
                smarts = row["SMARTS"]
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    SUB_SMARTS_PATTERNS[bit] = pattern
            except Exception as e:
                logging.warning(f"Error loading Substructure pattern at row {i}: {e}")
    else:
        logging.warning(f"Substructure SMARTS file not found: {sub_filepath}")
    
    # Load Klekota-Roth SMARTS
    logging.info(f"Loading Klekota-Roth SMARTS patterns from {kr_filepath}")
    if os.path.exists(kr_filepath):
        kr_df = pd.read_csv(kr_filepath)
        for i, row in kr_df.iterrows():
            try:
                bit = row["Bit"]
                smarts = row["SMARTS"]
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    KR_SMARTS_PATTERNS[bit] = pattern
            except Exception as e:
                logging.warning(f"Error loading Klekota-Roth pattern at row {i}: {e}")
    else:
        logging.warning(f"Klekota-Roth SMARTS file not found: {kr_filepath}")
    
    logging.info(f"Loaded {len(SUB_SMARTS_PATTERNS)} Substructure and {len(KR_SMARTS_PATTERNS)} Klekota-Roth patterns")

    # Load EState SMARTS
    if estate_filepath and os.path.exists(estate_filepath):
        logging.info(f"Loading EState SMARTS patterns from {estate_filepath}")
        estate_df = pd.read_csv(estate_filepath)
        for i, row in estate_df.iterrows():
            try:
                bit = row["Bit"] if "Bit" in row else i
                name = row["Name"] if "Name" in row else f"EState_{bit}"
                smarts = row["SMARTS"]
                pattern = Chem.MolFromSmarts(smarts)
                if pattern:
                    ESTATE_SMARTS_PATTERNS[name] = (bit, pattern)
            except Exception as e:
                logging.warning(f"Error loading EState pattern at row {i}: {e}")
        logging.info(f"Loaded {len(ESTATE_SMARTS_PATTERNS)} EState patterns")
    else:
        logging.warning(f"EState SMARTS file not found or not provided")

# Input: RDKit Molecule
# Output: RDKit 2D descriptors
# Description: Computes all RDKit 2D descriptors
def compute_all_descriptors(mol):
    results = {}
    
    for name, func in ALL_DESCRIPTORS:
        try:
            results[name] = func(mol)
        except:
            # Handle errors gracefully
            results[name] = np.nan
    
    return results

# Input: RDKit Molecule
# Output: MACCS fingerprint as array or empty
# Description: Computes MACCS fingerprint for molecule
def compute_maccs_fingerprint(mol):
    """Compute MACCS fingerprint"""
    try:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)
        return maccs_array
    except:
        return np.zeros((167,), dtype=np.int8)

# Input: RDKit Molecule
# Output: ECFP6 (r = 3) fingerprint as array or empty
# Description: Computes ECFP6 fingerprint for molecule
def compute_ecfp6_fingerprint(mol):
    try:
        from rdkit.Chem import rdFingerprintGenerator
        fp_size = 2048
        radius = 3  # ECFP6 with radius 3
        
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        ecfp6 = morgan_gen.GetFingerprint(mol, additionalOutput=ao)
        bit_info = ao.GetBitInfoMap()
        
        # Convert to numpy array
        ecfp6_array = np.zeros((fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(ecfp6, ecfp6_array)
        
        return ecfp6_array
    except Exception as e:
        logging.warning(f"Error computing ECFP6 fingerprint: {e}")
        return np.zeros((2048,), dtype=np.int8)

# Input: RDKit Molecule
# Output: Substructure fingerprint as array or empty
# Description: Computes Substructural CSV fingerprint for molecule
def compute_substructure_fps(mol):
    results = {}
    
    # Check if pattern exists for running without
    if not SUB_SMARTS_PATTERNS:
        logging.warning("SUB_SMARTS_PATTERNS is empty in worker process")
        return results
    
    for bit_id, pattern in SUB_SMARTS_PATTERNS.items():
        try:
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            results[f'Sub_{bit_id}'] = len(matches)
        except Exception as e:
            logging.debug(f"Error in substructure match for bit {bit_id}: {e}")
            results[f'Sub_{bit_id}'] = 0
    
    return results

# Input: RDKit Molecule
# Output: Klekota-Roth 2D fingerprint as array or empty
# Description: Computes Klekota-Roth 2D CSV fingerprint for molecule
def compute_klekotaroth_fps(mol):
    results = {}
    
    # Check if pattern exists for running without
    if not KR_SMARTS_PATTERNS:
        logging.warning("KR_SMARTS_PATTERNS is empty in worker process")
        return results
    
    for bit_id, pattern in KR_SMARTS_PATTERNS.items():
        try:
            if pattern is None:
                continue
            matches = mol.GetSubstructMatches(pattern)
            results[f'KR_{bit_id}'] = len(matches)
        except Exception as e:
            logging.debug(f"Error in KR match for bit {bit_id}: {e}")
            results[f'KR_{bit_id}'] = 0
    
    return results

# Input: RDKit Molecule
# Output: EState (Electronic) fingerprint as array or empty
# Description: Computes EState CSV fingerprint for molecule
def compute_estate_fps(mol):
    results = {}
    
    for name, (bit, pattern) in ESTATE_SMARTS_PATTERNS.items():
        try:
            # Check if pattern exists for running without
            if not pattern or not isinstance(pattern, Chem.Mol):
                logging.debug(f"Invalid SMARTS pattern for EState_{name}")
                results[f'EState_{name}'] = 0
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            results[f'EState_{name}'] = len(matches) if matches else 0
        except Exception as e:
            logging.debug(f"Error computing EState fingerprint {name}: {e}")
            results[f'EState_{name}'] = 0
    
    return results

# Input: SMILES code
# Output: All molecular features as array or none
# Description: Computes all features for molecule
def compute_all_features(smiles):
    """Compute all molecular features (descriptors and fingerprints)"""
    try:
        # Create RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Could not parse SMILES: {smiles}")
            return None
        
        # Compute descriptors
        descriptors = compute_all_descriptors(mol)
        
        # Compute MACCS fingerprints
        maccs_fp = compute_maccs_fingerprint(mol)
        for i, bit in enumerate(maccs_fp):
            descriptors[f'MACCS_{i}'] = bit
        
        # Compute ECFP6 fingerprints
        ecfp6_fp = compute_ecfp6_fingerprint(mol)
        for i, bit in enumerate(ecfp6_fp):
            descriptors[f'ECFP6_{i}'] = bit
        
        # Compute substructure fingerprints - check if patterns available
        if SUB_SMARTS_PATTERNS:
            sub_fps = compute_substructure_fps(mol)
            descriptors.update(sub_fps)
        else:
            logging.debug("No SUB_SMARTS_PATTERNS available, skipping substructure fingerprints")
        
        # Compute Klekota-Roth fingerprints - check if patterns available
        if KR_SMARTS_PATTERNS:
            kr_fps = compute_klekotaroth_fps(mol)
            descriptors.update(kr_fps)
        else:
            logging.debug("No KR_SMARTS_PATTERNS available, skipping Klekota-Roth fingerprints")

        # Compute EState fingerprints - check if patterns available
        if ESTATE_SMARTS_PATTERNS:
            estate_fps = compute_estate_fps(mol)
            descriptors.update(estate_fps)
        else:
            logging.debug("No ESTATE_SMARTS_PATTERNS available, skipping EState fingerprints")
        
        return descriptors
    except Exception as e:
        logging.warning(f"Error computing features for SMILES {smiles}: {e}")
        return None

#---------------------------------------------------------------------------
#           Model Evaluators
#-------------------------------------------------------------------------

# Input: y (true, scores), stage, threshold, and output directory
# Output: Figures and stats (dict)
# Description: Saves model evaluations as files and returns a stats dict
def evaluate_and_visualize(y_true: np.ndarray, y_scores: np.ndarray, 
                          stage_name: str, threshold: float, 
                          output_dir: str = None) -> Dict[str, float]:
    # Flatten arrays to 1D
    y_true_flat = np.asarray(y_true).flatten()
    y_scores_flat = np.asarray(y_scores).flatten()
    
    y_pred = (y_scores_flat >= threshold).astype(int)

    # Evaluate model
    precision = precision_score(y_true_flat, y_pred)
    recall = recall_score(y_true_flat, y_pred)
    f1 = f1_score(y_true_flat, y_pred)
    f2 = fbeta_score(y_true_flat, y_pred, beta=2.0)
    accuracy = accuracy_score(y_true_flat, y_pred)
    fpr, tpr, _ = roc_curve(y_true_flat, y_scores_flat)
    roc_auc = roc_auc_score(y_true_flat, y_scores_flat)
    precisions, recalls, _ = precision_recall_curve(y_true_flat, y_scores_flat)
    avg_precision = average_precision_score(y_true_flat, y_scores_flat)
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred).ravel()

    logging.info(f"\n- - {stage_name.upper()} Model Evaluation - -")
    logging.info(f"  Threshold: {threshold:.4f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  F2 Score: {f2:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  Average Precision: {avg_precision:.4f}")
    logging.info(f"  Confusion Matrix:")
    logging.info(f"    True Positives: {tp}")
    logging.info(f"    False Positives: {fp}")
    logging.info(f"    True Negatives: {tn}")
    logging.info(f"    False Negatives: {fn}")

    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{stage_name} - Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(plots_dir, f'{stage_name}_roc_curve.png'), dpi=300)
        plt.close()

        # PR curve
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{stage_name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(plots_dir, f'{stage_name}_pr_curve.png'), dpi=300)
        plt.close()

        # Score distribution
        plt.figure(figsize=(10, 8))
        pos_indices = np.where(y_true_flat == 1)[0]
        neg_indices = np.where(y_true_flat == 0)[0]
        
        # Create separate arrays for histogram
        pos_scores = y_scores_flat[pos_indices] if len(pos_indices) > 0 else np.array([])
        neg_scores = y_scores_flat[neg_indices] if len(neg_indices) > 0 else np.array([])
        
        plt.hist([neg_scores, pos_scores], bins=50, label=['Negative', 'Positive'], alpha=0.5)
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        plt.xlabel('Prediction Score')
        plt.ylabel('Count')
        plt.title(f'{stage_name} - Score Distribution')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'{stage_name}_score_dist.png'), dpi=300)
        plt.close()

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true_flat, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{stage_name} - Confusion Matrix')
        plt.savefig(os.path.join(plots_dir, f'{stage_name}_confusion_matrix.png'), dpi=300)
        plt.close()

    stats = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'auroc': roc_auc,
        'avg_precision': avg_precision,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
    return stats

#----------------------------
#           Stage 1 Training
#-----------------------------

# Input: Molecular DF
# Output: Tuple of model, threshold, and statistics
# Descriptions: Trains a RF Classifier on 23 features
def train_stage1_model(df: pd.DataFrame, config: Config) -> Tuple[Any, float, Dict[str, float]]:
    logging.info("\n- - Training Stage 1 Model - -")
    # Use all 23 features
    feature_cols = [
        'mw', 'hba', 'hbd', 'tpsa', 'aromatic_atoms', 'basic_centers',
        'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
        'logp', 'rotbonds', 'fsp3', 'ring_count', 'ring_atom_ratio', 
        'psa_mw_ratio', 'complexity', 'chi0v', 'chi1v', 'kappa1', 
        'kappa2', 'kappa3', 'num_heavy_atoms'
    ]
    X = df[feature_cols]
    y = df['bbb_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=config.random_state,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    y_scores = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    # Find threshold that maximizes F1 score
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # Exclude the last element that doesn't have a threshold
    threshold = thresholds[best_idx]

    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    logging.info("Feature importances:")
    for _, row in importance_df.iterrows():
        logging.info(f"  {row['feature']}: {row['importance']:.4f}")

    test_scores = model.predict_proba(X_test)[:, 1]
    stats = evaluate_and_visualize(y_test, test_scores, 'stage1', threshold, config.output_dir)

    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importances)[::-1]
    plt.bar(range(len(feature_importances)), feature_importances[indices])
    plt.xticks(range(len(feature_importances)), [feature_cols[i] for i in indices], rotation=90)
    plt.title('Stage 1 - Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(config.plots_dir, 'stage1_feature_importances.png'), dpi=300)
    plt.close()

    # Store model with feature names and threshold
    model_package = {
        'model': model,
        'feature_names': feature_cols,
        'threshold': threshold,
        'test_stats': stats
    }

    with open(config.stage1_model, 'wb') as f:
        pickle.dump(model_package, f)

    logging.info(f"Stage 1 model saved to {config.stage1_model}")
    logging.info(f"Optimal threshold: {threshold:.4f}")
    return model, threshold, stats

#----------------------------------------------------------
#           Stage 2 Training
#----------------------------------------------------------

# Class for training Stage 2
class Stage2ModelWithFeatureSelection:
    def __init__(self, var_selector, feature_selector, xgb_model, feature_cols):
        self.var_selector = var_selector
        self.feature_selector = feature_selector
        self.xgb_model = xgb_model
        self.feature_cols = feature_cols
        
    def predict_proba(self, X):
        # Convert X to DataFrame with proper feature names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_cols[:X.shape[1]])
        
        # Ensure all required columns are present
        missing_cols = [col for col in self.feature_cols if col not in X.columns]
        if missing_cols:
            for col in missing_cols:
                X[col] = 0  # Add missing columns with default value
        
        # Apply feature selection pipeline
        X_var = self.var_selector.transform(X[self.feature_cols]) 
        X_feat = self.feature_selector.transform(X_var)
        
        # Make predictions
        dmatrix = xgb.DMatrix(X_feat)
        preds = self.xgb_model.predict(dmatrix)
        
        # Format as 2D array with both class probabilities
        proba = np.zeros((len(preds), 2))
        proba[:, 1] = preds
        proba[:, 0] = 1 - preds
        
        return proba

# Input: Molecular DF
# Output: Tuple of the final model, threshold, and statistics
# Description: Trains stage 2 (XGBoost) model
def train_stage2_model(df: pd.DataFrame, config: Config) -> Tuple[Any, float, Dict[str, float]]:
    """
    Enhanced Stage 2 model that focuses on maintaining high recall while 
    providing improved precision over Stage 1.
    """
    logging.info("\n===== Training Enhanced Stage 2 Model =====")

    # Combine relevant descriptors with fingerprints
    descriptor_cols = [
        'mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotbonds',
        'aromatic_rings', 'formal_charge', 'mr', 
        'heavy_halogen', 'aromatic_proportion',
        'n_carboxylic', 'n_amine'
    ]
    
    # Add important Stage 1 features 
    stage1_specific = [
        'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
        'fsp3', 'ring_count', 'ring_atom_ratio', 'psa_mw_ratio', 'complexity'
    ]
    
    # Fingerprints
    maccs_cols = [f'maccs_{i}' for i in range(167)]
    ecfp4_cols = [f'ecfp4_{i}' for i in range(1024)]  # Include ECFP4
    substructure_cols = [f'sub_fp_{i}' for i in range(len(global_substructure_patterns))]
    
    # Combine all features
    feature_cols = descriptor_cols + stage1_specific + maccs_cols + ecfp4_cols + substructure_cols
    X = df[feature_cols]
    y = df['bbb_label']
    
    # Apply feature selection to reduce dimensionality
    from sklearn.feature_selection import SelectFromModel, VarianceThreshold
    
    # Remove low variance features (especially in fingerprints)
    var_selector = VarianceThreshold(threshold=0.01)  # Remove features with < 1% variance
    X_var_filtered = var_selector.fit_transform(X)
    selected_var_indices = var_selector.get_support(indices=True)
    selected_var_features = [feature_cols[i] for i in selected_var_indices]
    
    logging.info(f"Removed {len(feature_cols) - len(selected_var_features)} low variance features")
    
    # Feature importance selection
    from xgboost import XGBClassifier
    selector_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.random_state
    )
    
    # Fit selector on variance-filtered features
    X_train_var, X_val_var, y_train, y_val = train_test_split(
        X_var_filtered, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    
    selector_model.fit(X_train_var, y_train)
    feature_selector = SelectFromModel(selector_model, threshold='median')
    feature_selector.fit(X_train_var, y_train)
    
    # Transform data with selected features
    X_train_selected = feature_selector.transform(X_train_var)
    X_val_selected = feature_selector.transform(X_val_var)
    
    logging.info(f"Selected {X_train_selected.shape[1]} out of {X_train_var.shape[1]} features after importance filtering")
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    
    # Apply the same feature selection pipeline to train and test data
    X_train = var_selector.transform(X_train)
    X_train = feature_selector.transform(X_train)
    
    X_test = var_selector.transform(X_test)
    X_test = feature_selector.transform(X_test)
    
    # Define model parameters, prevent recall degradation
    model_params = {
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 2,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'objective': 'binary:logistic',
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),  # Balance classes
        'random_state': config.random_state,
        'verbosity': 0
    }
    
    # Check for XGBoost GPU (GCloud VM v my PC)
    gpu_available = False
    try:
        model_params['device'] = 'cuda'
        test_model = xgb.train(model_params, xgb.DMatrix(X_train[:5], label=y_train[:5]), num_boost_round=1)
        gpu_available = True
        logging.info("Using GPU for XGBoost training")
    except Exception as e:
        gpu_available = False
        model_params.pop('device', None)
        model_params['tree_method'] = 'hist'
        logging.info(f"Using CPU with histogram optimization: {e}")
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train with early stopping with separate validation set
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.random_state, stratify=y_train
    )
    
    dtrain_main = xgb.DMatrix(X_train_main, label=y_train_main)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        model_params,
        dtrain_main,
        num_boost_round=1000,
        early_stopping_rounds=50,
        evals=[(dtrain_main, 'train'), (dval, 'val')],
        verbose_eval=100
    )
    
    # Save best iteration and feature selectors
    best_iteration = model.best_iteration
    logging.info(f"Best iteration: {best_iteration}")
 
    # Create model
    final_model = Stage2ModelWithFeatureSelection(
        var_selector=var_selector,
        feature_selector=feature_selector,
        xgb_model=model,
        feature_cols=feature_cols
    )
    
    # Evaluate on test set
    test_scores = final_model.predict_proba(pd.DataFrame(df.iloc[y_test.index][feature_cols]))[:, 1]
    
    # Find optimal threshold from PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    threshold = thresholds[best_idx]
    
    logging.info(f"Selected threshold: {threshold:.4f}")
    
    # Evaluate with the selected threshold
    stats = evaluate_and_visualize(y_test, test_scores, 'stage2', threshold, config.output_dir)
    
    # Create a picklable model package with all necessary components
    model_package = {
        'model': final_model,
        'feature_names': feature_cols,
        'threshold': threshold,
        'var_selector': var_selector,
        'feature_selector': feature_selector,
        'xgb_model': model,
        'test_stats': stats
    }
    
    with open(config.stage2_model, 'wb') as f:
        pickle.dump(model_package, f)
    
    logging.info(f"Stage 2 model saved to {config.stage2_model}")
    
    return final_model, threshold, stats

#----------------------------------------------------------
#           Stage 3: PyTorch NN (from PC on B3DB)
#----------------------------------------------------------

# Simple feedforward neural network for binary classification
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5, activation='relu'):
        super(SimpleNN, self).__init__()
        
        # Store input dimension for reference
        self.input_dim = input_dim
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Chosen activation function
        act_fn = nn.ReLU()
        
        # Create hidden layers
        for i, dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, dim))
            self.layers.append(act_fn)
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input shape validation
        if x.dim() > 1 and x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: model expects {self.input_dim} features but got {x.shape[1]}")
    
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return self.sigmoid(x)

# Input: Config
# Output: model, threshold, stats, scaler
# Description: Train Stage 3 model WITHOUT data augmentation
def train_stage3_model(config: Config):
    logging.info("\n- - Training Stage 3 Model (Enhanced Feature Set) - -")
    # Prepare B3DB dataset
    stage3_input = os.path.join(config.input_dir, "B3DB_classification.csv")
    
    # Find SMARTS pattern files
    sub_filepath = os.path.join(config.input_dir, "SubStructureFingerprinter.csv")
    kr_filepath = os.path.join(config.input_dir, "KlekotaRothFingerprinter.csv") 
    estate_filepath = os.path.join(config.input_dir, "EStateFingerprinter.csv")
    
    if not os.path.exists(stage3_input):
        logging.error(f"Stage 3 input file not found: {stage3_input}")
        raise FileNotFoundError(f"Stage 3 input file not found: {stage3_input}")
            
    # Compute features for all molecules with comprehensive fingerprints
    X, y = prepare_stage3_data(stage3_input, sub_filepath, kr_filepath, estate_filepath)
    
    # Remove SMILES column before scaling
    smiles = X['SMILES']
    X_data = X.drop(columns=['SMILES'])
    
    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Save the scaler for inference
    with open(config.stage3_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(X_train)
    train_labels = torch.FloatTensor(y_train.values).view(-1, 1)
    val_tensor = torch.FloatTensor(X_val)
    val_labels = torch.FloatTensor(y_val.values).view(-1, 1)
    test_tensor = torch.FloatTensor(X_test)
    test_labels = torch.FloatTensor(y_test.values).view(-1, 1)
    
    # Define model with PC specifications
    input_dim = X_train.shape[1]  # Dimensionality from all features
    hidden_dims = [292, 188]  # From PC model architecture
    dropout_rate = 0.296 
    
    logging.info(f"Creating SimpleNN model with input_dim={input_dim}, hidden_dims={hidden_dims}")
    
    model = SimpleNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        activation='relu'
    ).to(config.device)
    
    # Define loss function and optimizer with proper weight decay
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=2.3164e-05, 
        weight_decay=1.5e-08 
    )
    
    # Training settings
    batch_size = 8  # Original batch size
    n_epochs = 139  # Original epochs
    patience = 10   # For early stopping
    
    # Create data loaders with correct batch size
    train_dataset = TensorDataset(train_tensor, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_tensor, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Collect batches for AUC calculation
        batch_outputs = []
        batch_targets = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Store for AUC calculation (detach to prevent gradient tracking)
            batch_outputs.append(outputs.detach().cpu().numpy())
            batch_targets.append(targets.cpu().numpy())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Calculate training AUC
        train_outputs = np.concatenate(batch_outputs)
        train_targets = np.concatenate(batch_targets)
        train_auc = roc_auc_score(train_targets, train_outputs)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_outputs.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate validation AUC
        val_outputs = np.concatenate(val_outputs)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_outputs)
        
        logging.info(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            logging.info(f"New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_tensor.to(config.device)).cpu().numpy().flatten()
    
    threshold = 0.108  # PC model threshold
    
    # Calculate final metrics
    test_auc = roc_auc_score(test_labels, test_outputs)
    test_preds = (test_outputs >= threshold).astype(int)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    
    logging.info(f"Test Results: AUC: {test_auc:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), config.stage3_model)
    
    # Save model package with metadata
    model_package = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'threshold': threshold,
        'scaler': scaler,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1
    }
    
    with open(os.path.join(config.model_dir, 'stage3_model_package.pkl'), 'wb') as f:
        pickle.dump(model_package, f)
    
    # Evaluate and visualize results
    stats = evaluate_and_visualize(
        test_labels.numpy(),
        test_outputs,
        'stage3',
        threshold,
        config.output_dir
    )
    
    return model, threshold, stats, scaler

#----------------------------------------------------------
#           Discarded Logging & Large Dataset Pipeline
#----------------------------------------------------------

# Input: SMILES list, reason for discarding, filename
# Output: -
# Description: Logs discarded molecules with reasons, used for debugging
def log_discarded_molecules(smiles_list: List[str], reason: str, filename: str) -> None:
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['smiles', 'reason', 'timestamp'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for s in smiles_list:
            writer.writerow([s, reason, timestamp])

# Input: DYNAMIC threshold calculator for large SMILES inference set
# Output: Calculatd threshold
# Description: Calculate threshold that would result in the target pass ratio with diagnostic logging and maximum threshold caps
"""
Args:
    scores: List of prediction scores
    target_ratio: Target pass ratio
    stage_name: Name of stage for logging
    max_threshold: Maximum allowed threshold value
"""
def calculate_threshold_for_ratio(scores, target_ratio, stage_name=None, max_threshold=None):
    if not scores or len(scores) == 0:
        logging.warning(f"No scores available for threshold calculation")
        return None
    
    # Diagnostic logging for score distribution
    score_array = np.array(scores)
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    dist = np.percentile(score_array, percentiles)
    
    if stage_name:
        logging.info(f"{stage_name} score distribution for threshold adaptation:")
        logging.info(f"  Percentiles {percentiles}: {dist}")
        logging.info(f"  Mean: {np.mean(score_array):.4f}, Std: {np.std(score_array):.4f}")
        logging.info(f"  Target ratio: {target_ratio:.4f} (keeping top {target_ratio*100:.1f}%)")
    
    # Sort scores in descending order
    sorted_scores = sorted(scores, reverse=True)
    
    # Calculate index based on target ratio
    target_index = int(len(sorted_scores) * target_ratio)
    
    # Ensure valid index
    if target_index >= len(sorted_scores):
        target_index = len(sorted_scores) - 1
    elif target_index < 0:
        target_index = 0
    
    # Get threshold at target index
    calculated_threshold = sorted_scores[target_index]
    
    # Apply maximum threshold cap
    if max_threshold is not None and calculated_threshold > max_threshold:
        if stage_name:
            logging.warning(f"{stage_name} calculated threshold {calculated_threshold:.4f} exceeds " +
                          f"maximum {max_threshold:.4f}, capping at maximum")
        return max_threshold
    
    return calculated_threshold

# Input: Chunk, RF model, config, and other parameters
# Output: A Dict of pass and fail indices
# Description: For inference, processes the first stage RF Model (WITHOUT SCORES - OLD)
def process_stage1_chunk(chunk: pd.DataFrame, model: Any, threshold: float, config: Config, chunk_id: int, use_cache: bool = False) -> Dict[str, List[int]]:
    cache_file = os.path.join(config.cache_dir, f"stage1_features_chunk{chunk_id}.parquet")

    failed_indices = []
    quick_rejects = []
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 1 features for chunk {chunk_id}")
        try:
            features_df = pd.read_parquet(cache_file)
        except Exception as e:
            logging.error(f"Failed to load cached features: {e}")
            features_df = None
    else:
        features_df = None

    if features_df is None:
        logging.info(f"Computing Stage 1 features for chunk {chunk_id}")
        quick_reject_counts = {}
        
        with Pool(processes=config.max_processes) as pool:
            features_list = list(tqdm(
                pool.imap(compute_stage1_features, chunk['smiles']),
                total=len(chunk),
                desc="Computing Stage 1 features"
            ))

        valid_features = []
        valid_indices = []
        quick_reject_indices = []
        quick_reject_reasons = []

        for i, result in enumerate(features_list):
            if result is None:
                failed_indices.append(i)
            elif isinstance(result, dict) and result.get('quick_reject', False):
                reason = result.get('reason', 'unknown')
                quick_reject_counts[reason] = quick_reject_counts.get(reason, 0) + 1
                quick_reject_indices.append(i)
                quick_reject_reasons.append(reason)
            else:
                valid_features.append(result)
                valid_indices.append(i)
        
        # Log quick rejection stats
        if quick_reject_counts:
            logging.info(f"Chunk {chunk_id} quick rejection statistics:")
            for reason, count in quick_reject_counts.items():
                logging.info(f"  {reason}: {count} molecules")

        if valid_features:
            # Build DF with all 23 features
            feature_names = [
                'mw', 'hba', 'hbd', 'tpsa', 'aromatic_atoms', 'basic_centers',
                'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
                'logp', 'rotbonds',     # original 12
                'fsp3', 'ring_count', 'ring_atom_ratio', 'psa_mw_ratio', 'complexity',   # additional 5
                'chi0v', 'chi1v', 'kappa1', 'kappa2', 'kappa3', 'num_heavy_atoms'         # extra 6
            ]
            
            features_df = pd.DataFrame(valid_features, columns=feature_names)
            features_df['smiles_idx'] = valid_indices
            os.makedirs(config.cache_dir, exist_ok=True)
            features_df.to_parquet(cache_file)
        else:
            features_df = pd.DataFrame()
        
        # Log quick reject smiles
        if quick_reject_indices:
            quick_reject_smiles = chunk.iloc[quick_reject_indices]['smiles'].tolist()
            log_file = os.path.join(config.log_dir, f"stage1_quick_rejected_chunk{chunk_id}.csv")
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles', 'reason'])
                for i, smiles in enumerate(quick_reject_smiles):
                    writer.writerow([smiles, quick_reject_reasons[i]])
            
            log_discarded_molecules(quick_reject_smiles, 'quick_rejected_stage1', config.discarded_log['stage1'])
            logging.info(f"Logged {len(quick_reject_indices)} quick rejected SMILES in Stage 1 chunk {chunk_id}")

        if failed_indices:
            failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
            log_discarded_molecules(failed_smiles, 'parsing_error_stage1', config.discarded_log['stage1'])
            logging.info(f"Logged {len(failed_indices)} failed SMILES in Stage 1 chunk {chunk_id}")

    if len(features_df) == 0:
        return {'pass': [], 'fail': list(range(len(chunk)))}

    # Extract feature names from the model
    feature_cols = None
    if hasattr(model, 'feature_names'):
        feature_cols = model.feature_names
    
    if feature_cols is not None:
        missing_cols = [col for col in feature_cols if col not in features_df.columns and col != 'smiles_idx']
        if missing_cols:
            logging.warning(f"Missing columns in Stage 1 input: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0  # Add missing columns with default values

        X = features_df[[col for col in feature_cols if col in features_df.columns]]
    else:
        X = features_df.drop(columns=['smiles_idx'])
    
    # Make predictions
    y_scores = model.predict_proba(X)[:, 1]
    
    score_percentiles = np.percentile(y_scores, [0, 25, 50, 75, 90, 95, 99, 100])
    logging.info(f"Stage 1 score distribution: {score_percentiles}")
    pass_count = sum(y_scores >= threshold)
    logging.info(f"Molecules passing threshold ({threshold:.4f}): {pass_count} out of {len(y_scores)} ({pass_count/len(y_scores)*100:.2f}%)")
    
    pass_mask = y_scores >= threshold
    pass_indices = features_df.loc[pass_mask, 'smiles_idx'].astype(int).tolist()
    fail_indices = features_df.loc[~pass_mask, 'smiles_idx'].astype(int).tolist()
    
    logging.info(f"Pass mask has {sum(pass_mask)} True values, resulting in {len(pass_indices)} passed indices")
    if len(pass_indices) == 0 and sum(pass_mask) > 0:
        logging.warning("WARNING: Molecules passed threshold but no indices were collected")
        logging.info(f"First 10 mask values: {pass_mask[:10]}")
        if 'smiles_idx' in features_df.columns:
            logging.info(f"First 10 indices: {features_df['smiles_idx'].iloc[:10].tolist()}")

    # Add quick reject indices to fail_indices
    if 'quick_reject_indices' in locals() and quick_reject_indices:
        fail_indices.extend(quick_reject_indices)
        
    # Log molecules that failed the score threshold
    failed_stage_smiles = chunk.iloc[fail_indices]['smiles'].tolist()
    log_discarded_molecules(failed_stage_smiles, 'failed_stage1', config.discarded_log['stage1'])

    if failed_indices:
        fail_indices.extend(failed_indices)

    return {'pass': pass_indices, 'fail': fail_indices}

# Input: Chunk, RF model, config, and other parameters
# Output: A Dict of pass and fail indices
# Description: Modified Stage 1 Model (same) but includes scores for thresholding
def process_stage1_chunk_with_scores(chunk: pd.DataFrame, model: Any, threshold: float, 
                        config: Config, chunk_id: int, use_cache: bool = False) -> Dict[str, List[int]]:
    cache_file = os.path.join(config.cache_dir, f"stage1_features_chunk{chunk_id}.parquet")

    failed_indices = []
    quick_rejects = []
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 1 features for chunk {chunk_id}")
        try:
            features_df = pd.read_parquet(cache_file)
        except Exception as e:
            logging.error(f"Failed to load cached features: {e}")
            features_df = None
    else:
        features_df = None

    if features_df is None:
        logging.info(f"Computing Stage 1 features for chunk {chunk_id}")
        quick_reject_counts = {}
        
        with Pool(processes=config.max_processes) as pool:
            features_list = list(tqdm(
                pool.imap(compute_stage1_features, chunk['smiles']),
                total=len(chunk),
                desc="Computing Stage 1 features"
            ))

        valid_features = []
        valid_indices = []
        quick_reject_indices = []
        quick_reject_reasons = []

        for i, result in enumerate(features_list):
            if result is None:
                failed_indices.append(i)
            elif isinstance(result, dict) and result.get('quick_reject', False):
                reason = result.get('reason', 'unknown')
                quick_reject_counts[reason] = quick_reject_counts.get(reason, 0) + 1
                quick_reject_indices.append(i)
                quick_reject_reasons.append(reason)
            else:
                valid_features.append(result)
                valid_indices.append(i)
        
        # Log quick rejection statistics
        if quick_reject_counts:
            logging.info(f"Chunk {chunk_id} quick rejection statistics:")
            for reason, count in quick_reject_counts.items():
                logging.info(f"  {reason}: {count} molecules")

        if valid_features:
            # Build DF with all features
            feature_names = [
                'mw', 'hba', 'hbd', 'tpsa', 'aromatic_atoms', 'basic_centers',
                'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
                'logp', 'rotbonds',     # original 12
                'fsp3', 'ring_count', 'ring_atom_ratio', 'psa_mw_ratio', 'complexity',   # additional 5
                'chi0v', 'chi1v', 'kappa1', 'kappa2', 'kappa3', 'num_heavy_atoms'         # extra 6
            ]
            
            features_df = pd.DataFrame(valid_features, columns=feature_names)
            features_df['smiles_idx'] = valid_indices
            os.makedirs(config.cache_dir, exist_ok=True)
            features_df.to_parquet(cache_file)
        else:
            features_df = pd.DataFrame()
        
        # Log quick reject smiles
        if quick_reject_indices:
            quick_reject_smiles = chunk.iloc[quick_reject_indices]['smiles'].tolist()
            log_file = os.path.join(config.log_dir, f"stage1_quick_rejected_chunk{chunk_id}.csv")
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles', 'reason'])
                for i, smiles in enumerate(quick_reject_smiles):
                    writer.writerow([smiles, quick_reject_reasons[i]])
            
            log_discarded_molecules(quick_reject_smiles, 'quick_rejected_stage1', config.discarded_log['stage1'])
            logging.info(f"Logged {len(quick_reject_indices)} quick rejected SMILES in Stage 1 chunk {chunk_id}")

        if failed_indices:
            failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
            log_discarded_molecules(failed_smiles, 'parsing_error_stage1', config.discarded_log['stage1'])
            logging.info(f"Logged {len(failed_indices)} failed SMILES in Stage 1 chunk {chunk_id}")

    if len(features_df) == 0:
        return {'pass': [], 'fail': list(range(len(chunk))), 'scores': []}

    feature_cols = None
    if hasattr(model, 'feature_names'):
        feature_cols = model.feature_names
    
    if feature_cols is not None:
        missing_cols = [col for col in feature_cols if col not in features_df.columns and col != 'smiles_idx']
        if missing_cols:
            logging.warning(f"Missing columns in Stage 1 input: {missing_cols}")
            for col in missing_cols:
                features_df[col] = 0  # Add missing columns with default values
        
        X = features_df[[col for col in feature_cols if col in features_df.columns]]
    else:
        X = features_df.drop(columns=['smiles_idx'])
    
    # Make predictions
    y_scores = model.predict_proba(X)[:, 1]
    
    # Log score distribution statistics
    score_percentiles = np.percentile(y_scores, [0, 25, 50, 75, 90, 95, 99, 100])
    logging.info(f"Stage 1 score distribution: {score_percentiles}")
    pass_count = sum(y_scores >= threshold)
    logging.info(f"Molecules passing threshold ({threshold:.4f}): {pass_count} out of {len(y_scores)} ({pass_count/len(y_scores)*100:.2f}%)")
    
    pass_mask = y_scores >= threshold
    pass_indices = features_df.loc[pass_mask, 'smiles_idx'].astype(int).tolist()
    fail_indices = features_df.loc[~pass_mask, 'smiles_idx'].astype(int).tolist()
    
    # Diagnostic logging
    logging.info(f"Pass mask has {sum(pass_mask)} True values, resulting in {len(pass_indices)} passed indices")
    if len(pass_indices) == 0 and sum(pass_mask) > 0:
        logging.warning("WARNING: Molecules passed threshold but no indices were collected")
        logging.info(f"First 10 mask values: {pass_mask[:10]}")
        if 'smiles_idx' in features_df.columns:
            logging.info(f"First 10 indices: {features_df['smiles_idx'].iloc[:10].tolist()}")

    # Add quick reject indices to fail_indices
    if 'quick_reject_indices' in locals() and quick_reject_indices:
        fail_indices.extend(quick_reject_indices)
        
    # Log molecules that failed the score threshold
    failed_stage_smiles = chunk.iloc[fail_indices]['smiles'].tolist()
    log_discarded_molecules(failed_stage_smiles, 'failed_stage1', config.discarded_log['stage1'])

    if failed_indices:
        fail_indices.extend(failed_indices)

    # Store the scores for threshold adaptation
    scores_list = y_scores.tolist()

    return {'pass': pass_indices, 'fail': fail_indices, 'scores': scores_list}

# Input: Chunk, XGBoost model, config, and other parameters
# Output: A Dict of pass and fail indices
# Description: For inference, processes the second stage XGBoost Model with scores for thresholding
def process_stage2_chunk_with_scores(chunk: pd.DataFrame, model: Any, threshold: float, 
                       config: Config, chunk_id: int, use_cache: bool = False) -> Dict[str, List[int]]:
    cache_file = os.path.join(config.cache_dir, f"stage2_features_chunk{chunk_id}.parquet")
    failed_indices = []
    
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 2 features for chunk {chunk_id}")
        try:
            features_df = pd.read_parquet(cache_file)
        except Exception as e:
            logging.error(f"Failed to load cached features: {e}")
            features_df = None
    else:
        features_df = None

    if features_df is None:
        logging.info(f"Computing Stage 2 features for chunk {chunk_id}")
        with Pool(processes=config.max_processes) as pool:
            features_list = list(tqdm(
                pool.imap(compute_stage2_features, chunk['smiles']),
                total=len(chunk),
                desc="Computing Stage 2 features"
            ))
        valid_indices = [i for i, x in enumerate(features_list) if x is not None]
        valid_features = [features_list[i] for i in valid_indices]
        failed_indices = [i for i, x in enumerate(features_list) if x is None]

        if not valid_features:
            if failed_indices:
                failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
                log_discarded_molecules(failed_smiles, 'parsing_error_stage2', config.discarded_log['stage2'])
            return {'pass': [], 'fail': list(range(len(chunk))), 'scores': []}

        descriptors = [x['descriptors'] for x in valid_features]
        maccs_fps = [x['maccs'] for x in valid_features]
        ecfp4_fps = [x['ecfp4'] for x in valid_features]
        substructure_fps = [x['substructure'] for x in valid_features]

        descriptor_names = [
            'mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotbonds',
            'aromatic_rings', 'formal_charge', 'mr', 
            'heavy_halogen', 'aromatic_proportion', 'n_carboxylic', 'n_amine'
        ]
        maccs_names = [f'maccs_{i}' for i in range(167)]
        ecfp4_names = [f'ecfp4_{i}' for i in range(1024)]
        substructure_names = [f'sub_fp_{i}' for i in range(len(global_substructure_patterns))]

        descriptors_df = pd.DataFrame(descriptors, columns=descriptor_names)
        maccs_df = pd.DataFrame(maccs_fps, columns=maccs_names)
        ecfp4_df = pd.DataFrame(ecfp4_fps, columns=ecfp4_names)
        sub_df = pd.DataFrame(substructure_fps, columns=substructure_names)

        # Concatenate all features
        features_df = pd.concat([descriptors_df, maccs_df, ecfp4_df, sub_df], axis=1)
        features_df['smiles_idx'] = valid_indices
        os.makedirs(config.cache_dir, exist_ok=True)
        features_df.to_parquet(cache_file)

        if failed_indices:
            failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
            log_discarded_molecules(failed_smiles, 'parsing_error_stage2', config.discarded_log['stage2'])
            logging.info(f"Logged {len(failed_indices)} failed SMILES in Stage 2 chunk {chunk_id}")

    if len(features_df) == 0:
        return {'pass': [], 'fail': list(range(len(chunk))), 'scores': []}

    # Get expected feature columns from the model
    expected_features = None
    if hasattr(model, 'feature_cols'):
        expected_features = model.feature_cols
    
    available_features = [col for col in features_df.columns if col != 'smiles_idx']
    
    if expected_features is not None:
        missing_features = [f for f in expected_features if f not in available_features]
        if missing_features:
            logging.warning(f"Adding {len(missing_features)} missing features for Stage 2")
            for feat in missing_features:
                features_df[feat] = 0
        
        # Create DataFrame for VarianceThreshold
        X = pd.DataFrame(features_df[expected_features].values, 
                         columns=expected_features)
    else:
        X = features_df.drop(columns=['smiles_idx'])
    
    # Make predictions using DataFrame
    try:
        y_scores = model.predict_proba(X)[:, 1]
        
        # Log score distribution statistics
        score_percentiles = np.percentile(y_scores, [0, 25, 50, 75, 90, 95, 99, 100])
        logging.info(f"Stage 2 score distribution: {score_percentiles}")
        pass_count = sum(y_scores >= threshold)
        logging.info(f"Molecules passing threshold ({threshold:.4f}): {pass_count} out of {len(y_scores)} ({pass_count/len(y_scores)*100:.2f}%)")
        
        pass_mask = y_scores >= threshold
        pass_indices = features_df.loc[pass_mask, 'smiles_idx'].astype(int).tolist()
        fail_indices = features_df.loc[~pass_mask, 'smiles_idx'].astype(int).tolist()
        
        # Store scores for threshold adaptation
        scores_list = y_scores.tolist()
        
        failed_stage_smiles = chunk.iloc[fail_indices]['smiles'].tolist() 
        log_discarded_molecules(failed_stage_smiles, 'failed_stage2', config.discarded_log['stage2'])
        
        if failed_indices:
            fail_indices.extend(failed_indices)
            
    except Exception as e:
        logging.error(f"Error predicting with Stage 2 model: {e}")
        logging.error("Passing all molecules to next stage as fallback")
        # Pass all molecules to the next stage as fallback
        pass_indices = features_df['smiles_idx'].astype(int).tolist()
        fail_indices = []
        scores_list = []

    return {'pass': pass_indices, 'fail': fail_indices, 'scores': scores_list}

# Input: Chunk, XGBoost model, config, and other parameters
# Output: A Dict of pass and fail indices
# Description: For inference, processes the second stage XGBoost Model (WITHOUT SCORES - OLD)
def process_stage2_chunk(chunk: pd.DataFrame, model: Any, threshold: float, 
                       config: Config, chunk_id: int, use_cache: bool = False) -> Dict[str, List[int]]:
    cache_file = os.path.join(config.cache_dir, f"stage2_features_chunk{chunk_id}.parquet")
    failed_indices = []
    
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 2 features for chunk {chunk_id}")
        try:
            features_df = pd.read_parquet(cache_file)
        except Exception as e:
            logging.error(f"Failed to load cached features: {e}")
            features_df = None
    else:
        features_df = None

    if features_df is None:
        logging.info(f"Computing Stage 2 features for chunk {chunk_id}")
        with Pool(processes=config.max_processes) as pool:
            features_list = list(tqdm(
                pool.imap(compute_stage2_features, chunk['smiles']),
                total=len(chunk),
                desc="Computing Stage 2 features"
            ))
        valid_indices = [i for i, x in enumerate(features_list) if x is not None]
        valid_features = [features_list[i] for i in valid_indices]
        failed_indices = [i for i, x in enumerate(features_list) if x is None]

        if not valid_features:
            if failed_indices:
                failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
                log_discarded_molecules(failed_smiles, 'parsing_error_stage2', config.discarded_log['stage2'])
            return {'pass': [], 'fail': list(range(len(chunk)))}

        descriptors = [x['descriptors'] for x in valid_features]
        maccs_fps = [x['maccs'] for x in valid_features]
        ecfp4_fps = [x['ecfp4'] for x in valid_features]
        substructure_fps = [x['substructure'] for x in valid_features]

        descriptor_names = [
            'mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotbonds',
            'aromatic_rings', 'formal_charge', 'mr', 
            'heavy_halogen', 'aromatic_proportion', 'n_carboxylic', 'n_amine'
        ]
        maccs_names = [f'maccs_{i}' for i in range(167)]
        ecfp4_names = [f'ecfp4_{i}' for i in range(1024)]
        substructure_names = [f'sub_fp_{i}' for i in range(len(global_substructure_patterns))]

        descriptors_df = pd.DataFrame(descriptors, columns=descriptor_names)
        maccs_df = pd.DataFrame(maccs_fps, columns=maccs_names)
        ecfp4_df = pd.DataFrame(ecfp4_fps, columns=ecfp4_names)
        sub_df = pd.DataFrame(substructure_fps, columns=substructure_names)

        # Concatenate all features
        features_df = pd.concat([descriptors_df, maccs_df, ecfp4_df, sub_df], axis=1)
        features_df['smiles_idx'] = valid_indices
        os.makedirs(config.cache_dir, exist_ok=True)
        features_df.to_parquet(cache_file)

        if failed_indices:
            failed_smiles = chunk.iloc[failed_indices]['smiles'].tolist()
            log_discarded_molecules(failed_smiles, 'parsing_error_stage2', config.discarded_log['stage2'])
            logging.info(f"Logged {len(failed_indices)} failed SMILES in Stage 2 chunk {chunk_id}")

    if len(features_df) == 0:
        return {'pass': [], 'fail': list(range(len(chunk)))}

    # Get expected feature columns from the model
    expected_features = None
    if hasattr(model, 'feature_cols'):
        expected_features = model.feature_cols
    
    available_features = [col for col in features_df.columns if col != 'smiles_idx']
    
    if expected_features is not None:
        missing_features = [f for f in expected_features if f not in available_features]
        if missing_features:
            logging.warning(f"Adding {len(missing_features)} missing features for Stage 2")
            for feat in missing_features:
                features_df[feat] = 0
        
        # Create DataFrame for VarianceThreshold
        X = pd.DataFrame(features_df[expected_features].values, 
                         columns=expected_features)
    else:
        # Without explicit feature list, use all available features
        X = features_df.drop(columns=['smiles_idx'])
    
    # Make predictions using DataFrame
    try:
        y_scores = model.predict_proba(X)[:, 1]
        
        pass_mask = y_scores >= threshold
        pass_indices = features_df.loc[pass_mask, 'smiles_idx'].astype(int).tolist()
        fail_indices = features_df.loc[~pass_mask, 'smiles_idx'].astype(int).tolist()
        
        failed_stage_smiles = chunk.iloc[fail_indices]['smiles'].tolist() 
        log_discarded_molecules(failed_stage_smiles, 'failed_stage2', config.discarded_log['stage2'])
        
        if failed_indices:
            fail_indices.extend(failed_indices)
            
    except Exception as e:
        logging.error(f"Error predicting with Stage 2 model: {e}")
        logging.error("Passing all molecules to next stage as fallback")
        # Pass all molecules to the next stage as fallback
        pass_indices = features_df['smiles_idx'].astype(int).tolist()
        fail_indices = []

    return {'pass': pass_indices, 'fail': fail_indices}

# Input: Chunk, NN model, config, and other parameters
# Output: A Dict of pass and fail indices
# Description: For inference, processes the third stage NN Model
def process_stage3_chunk(chunk: pd.DataFrame, model, threshold: float, scaler: StandardScaler, 
                        config: Config, chunk_id: int, use_cache: bool = False) -> Dict[str, Any]:
    global SUB_SMARTS_PATTERNS, KR_SMARTS_PATTERNS, ESTATE_SMARTS_PATTERNS
    
    # Setup SMARTS patterns and file paths for Stage 3
    sub_filepath = os.path.join(config.input_dir, "SubStructureFingerprinter.csv")
    kr_filepath = os.path.join(config.input_dir, "KlekotaRothFingerprinter.csv") 
    estate_filepath = os.path.join(config.input_dir, "EStateFingerprinter.csv")
    
    SUB_SMARTS_PATTERNS.clear()
    KR_SMARTS_PATTERNS.clear()
    ESTATE_SMARTS_PATTERNS.clear()
    
    load_smarts_patterns(sub_filepath, kr_filepath, estate_filepath)
    logging.info(f"Loaded patterns for Stage 3: {len(SUB_SMARTS_PATTERNS)} substructure, {len(KR_SMARTS_PATTERNS)} Klekota-Roth, {len(ESTATE_SMARTS_PATTERNS)} EState")
    
    # Cache file paths for feature storage
    cache_file_features = os.path.join(config.cache_dir, f"stage3_features_chunk{chunk_id}.parquet")
    cache_file_indices = os.path.join(config.cache_dir, f"stage3_indices_chunk{chunk_id}.npy")
    
    # Check cache first
    features_df = None
    valid_indices = None
    
    if use_cache and os.path.exists(cache_file_features) and os.path.exists(cache_file_indices):
        try:
            features_df = pd.read_parquet(cache_file_features)
            valid_indices = np.load(cache_file_indices)
            logging.info(f"Loaded {len(features_df)} cached features for Stage 3 chunk {chunk_id}")
        except Exception as e:
            logging.error(f"Failed to load cached Stage 3 features: {e}")
            features_df = None
            valid_indices = None
    
    # Compute features if not in cache
    if features_df is None or valid_indices is None:
        # Prepare temporary CSV for chunk processing
        temp_csv = os.path.join(config.cache_dir, f"stage3_chunk_{chunk_id}.csv")
        with open(temp_csv, 'w') as f:
            f.write("SMILES,BBB_LABEL\n")
            for _, row in chunk.iterrows():
                f.write(f"{row['smiles']},0\n")
        
        try:
            # Make local copies of patterns to pass to worker processes
            sub_patterns_copy = SUB_SMARTS_PATTERNS.copy()
            kr_patterns_copy = KR_SMARTS_PATTERNS.copy()
            estate_patterns_copy = ESTATE_SMARTS_PATTERNS.copy()
            
            all_smiles = chunk['smiles'].tolist()
            features_list = []
            
            # Initialize the pool once with patterns
            with Pool(
                processes=config.max_processes,
                initializer=pool_initializer,
                initargs=(sub_patterns_copy, kr_patterns_copy, estate_patterns_copy)
            ) as pool:
                features_list = list(tqdm(
                    pool.imap(process_single_smiles, all_smiles),
                    total=len(all_smiles),
                    desc=f"Computing features for chunk {chunk_id}"
                ))
            
            # Collect valid results
            valid_features = [f for f in features_list if f is not None]
            valid_indices = np.array([i for i, f in enumerate(features_list) if f is not None])
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(valid_features)
            
            # Add SMILES to the feature DataFrame
            if len(valid_indices) > 0:
                feature_df['SMILES'] = [chunk.iloc[i]['smiles'] for i in valid_indices]
            else:
                # Handle empty case
                feature_df['SMILES'] = []
            
            # Cache computed features
            os.makedirs(config.cache_dir, exist_ok=True)
            feature_df.to_parquet(cache_file_features)
            np.save(cache_file_indices, valid_indices)
            
            features_df = feature_df
            
            # Clean-up
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
                
        except Exception as e:
            logging.error(f"Error computing Stage 3 features for chunk {chunk_id}: {e}")
            return {
                'pass': [],
                'fail': list(range(len(chunk))),
                'probabilities': np.array([]),
                'chunk_with_probs': chunk.copy(),
                'valid_indices': np.array([]),
                'predictions': np.array([])
            }
    
    if len(features_df) == 0 or len(valid_indices) == 0:
        logging.warning(f"No valid molecules for Stage 3 in chunk {chunk_id}")
        # Return empty results
        return {
            'pass': [],
            'fail': list(range(len(chunk))),
            'probabilities': np.array([]),
            'chunk_with_probs': chunk.copy(),
            'valid_indices': np.array([]),
            'predictions': np.array([])
        }
    
    # Remove SMILES column before scaling
    X_data = features_df.drop(columns=['SMILES'])
    
    # Scale features
    try:
        X_scaled = scaler.transform(X_data)
    except Exception as e:
        logging.error(f"Error scaling features for Stage 3: {e}")
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != X_data.shape[1]:
            logging.warning(f"Feature dimension mismatch: scaler expects {scaler.n_features_in_} features, got {X_data.shape[1]}")
            if scaler.n_features_in_ > X_data.shape[1]:
                padding = np.zeros((X_data.shape[0], scaler.n_features_in_ - X_data.shape[1]))
                X_padded = np.hstack((X_data, padding))
                X_scaled = scaler.transform(X_padded)
            else:
                X_scaled = scaler.transform(X_data.iloc[:, :scaler.n_features_in_])
        else:
            # Fallback to unscaled data
            X_scaled = X_data.values
            
    # Convert to tensor for model prediction
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(config.device)
    
    # Run inference in batches
    batch_size = min(config.max_batch_size_stage3, 1024)
    model.to(config.device)
    model.eval()
    
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            end = min(i + batch_size, len(X_tensor))
            batch = X_tensor[i:end]
            preds = model(batch).cpu().numpy().flatten()
            all_predictions.append(preds)
    
    # Combine predictions
    predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
    
    # Initialize arrays with the right size
    final_indices = np.arange(len(chunk))
    final_probs = np.zeros(len(chunk))
    
    # Set the probabilities for valid molecules
    for i, idx in enumerate(valid_indices):
        if i < len(predictions) and idx < len(final_probs):
            final_probs[idx] = predictions[i]
    
    # Add probabilities to the chunk dataframe
    chunk_with_probs = chunk.copy()
    chunk_with_probs['bbb_probability'] = final_probs
    
    # Log stage 3 processing completion
    logging.info(f"Stage 3 processed {len(valid_indices)} valid molecules in chunk {chunk_id}")
    
    # No molecules are failed at this stage - all go forward with their probabilities
    pass_indices = list(valid_indices)
    fail_indices = []

    # Return results with explicit array conversion to avoid ambiguity
    return {
        'pass': pass_indices, 
        'fail': fail_indices,
        'probabilities': final_probs,
        'chunk_with_probs': chunk_with_probs,
        'valid_indices': valid_indices,
        'predictions': predictions if len(predictions) > 0 else np.array([])
    }

# Saves model thresholds
def save_model_thresholds(thresholds: Dict[str, float], config: Config) -> None:
    json_thresholds = {k: float(v) for k, v in thresholds.items()}
    with open(config.thresholds_json, 'w') as f:
        json.dump(json_thresholds, f, indent=4)
    logging.info(f"Saved model thresholds to {config.thresholds_json}")

# Input: Post-Numerically filtered input parquet for inference
# Output: Dict with stats, threshold
# Descriptions: Process massive dataset through cascade for inference - with dynamic thresholds
def process_large_dataset(input_parquet: str, config: Config, 
                        stage1_model: Any, stage1_threshold: float,
                        stage2_model: Any, stage2_threshold: float,
                        stage3_model: nn.Module, stage3_threshold: float,
                        stage3_scaler: StandardScaler,
                        start_chunk: int = 1) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(input_parquet)
        total_dataset_size = metadata.num_rows
        logging.info(f"Detected total dataset size from metadata: {total_dataset_size:,} molecules")
    except Exception as e:
        logging.warning(f"Could not determine total dataset size from metadata: {e}")
        total_dataset_size = None
        
    # Initialize dataframes to store results
    stage1_passed_df = pd.DataFrame()
    stage2_passed_df = pd.DataFrame()
    final_passed_df = pd.DataFrame()
    
    # Set up stats tracking
    chunk_stats_csv_path = os.path.join(config.results_dir, "chunk_processing_stats.csv")
    with open(chunk_stats_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "total_molecules", "stage1_pass", "stage1_fail", 
                         "stage2_pass", "stage2_fail", "stage3_processed", "stage3_mean_prob", 
                         "stage1_threshold", "stage2_threshold", "stage3_threshold", "memory_usage_mb"])
    
    # Track all scores for threshold calculation
    stage1_score_storage = []
    stage2_score_storage = []
    stage3_score_storage = []
    
    # Set target filtering ratios - to replace config thresholds
    TARGET_STAGE1_PASS_RATIO = 0.5  # 50% pass rate (1M -> 500K)
    TARGET_STAGE2_PASS_RATIO = 0.2  # 20% pass rate (500K -> 100K)
    TARGET_STAGE3_PASS_RATIO = 0.1  # 10% pass rate (100K -> 10K)
    
    # Track cumulative counts
    total_molecules = 0
    stage1_pass_count = 0
    stage2_pass_count = 0
    stage3_processed_count = 0
    
    # Minimum number of molecules to process before adapting thresholds
    MIN_MOLECULES_FOR_ADAPTATION = 50000
    
    # Flags to indicate threshold adaptation status
    thresholds_adapted = False
    force_adapted = False
    
    # Restart at any chunk (pipeline crashed with memory error previously)
    chunk_id = start_chunk - 1
    
    # Function to calculate the threshold for a target pass ratio
    def calculate_threshold_for_ratio(scores, target_ratio, stage_name=None, max_threshold=None):
        if not scores or len(scores) == 0:
            return None
            
        # Add logging for score distribution
        if stage_name:
            score_array = np.array(scores)
            percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
            dist = np.percentile(score_array, percentiles)
            
            logging.info(f"{stage_name} score distribution for threshold adaptation:")
            logging.info(f"  Percentiles {percentiles}: {dist}")
            logging.info(f"  Mean: {np.mean(score_array):.4f}, Std: {np.std(score_array):.4f}")
            logging.info(f"  Target ratio: {target_ratio:.4f} (keeping top {target_ratio*100:.1f}%)")
        
        # Sort scores in descending order
        sorted_scores = sorted(scores, reverse=True)
        
        # Calculate index based on target ratio
        target_index = int(len(sorted_scores) * target_ratio)

        if target_index >= len(sorted_scores):
            target_index = len(sorted_scores) - 1
        elif target_index < 0:
            target_index = 0
            
        # Return the threshold at the target index
        calculated_threshold = sorted_scores[target_index]
        
        # Apply maximum threshold cap
        if max_threshold is not None and calculated_threshold > max_threshold:
            if stage_name:
                logging.warning(f"{stage_name} calculated threshold {calculated_threshold:.4f} exceeds " +
                            f"maximum {max_threshold:.4f}, capping at maximum")
            return max_threshold
        
        return calculated_threshold

    # Helper function to process one chunk
    def process_chunk(chunk, chunk_id):
        nonlocal total_molecules, stage1_pass_count, stage2_pass_count, stage3_processed_count
        nonlocal stage1_passed_df, stage2_passed_df, final_passed_df
        nonlocal stage1_threshold, stage2_threshold, stage3_threshold, thresholds_adapted, force_adapted
        
        try:
            chunk_start = time.time()
            chunk_size = len(chunk)
            total_molecules += chunk_size
            
            # Monitor memory usage
            initial_memory = get_memory_usage()
            
            # Stage 1 processing with score collection
            logging.info(f"Processing Stage 1 with threshold {stage1_threshold:.4f}")
            try:
                stage1_results = process_stage1_chunk_with_scores(chunk, stage1_model, stage1_threshold, config, chunk_id)
                stage1_pass = len(stage1_results['pass'])
                stage1_fail = chunk_size - stage1_pass
                
                # Store Stage 1 scores for threshold adaptation
                if 'scores' in stage1_results and isinstance(stage1_results['scores'], list):
                    stage1_score_storage.extend(stage1_results['scores'])
            except Exception as e:
                logging.error(f"Error in Stage 1 processing for chunk {chunk_id}: {e}")
                # Record failed chunk stats
                with open(chunk_stats_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        chunk_id, chunk_size, 0, chunk_size, 0, 0, 0, 0.0,
                        stage1_threshold, stage2_threshold, stage3_threshold, initial_memory
                    ])
                return
            
            if stage1_pass == 0:
                logging.info(f"No molecules passed Stage 1 in chunk {chunk_id}")
                with open(chunk_stats_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        chunk_id, chunk_size, 0, chunk_size, 0, 0, 0, 0.0,
                        stage1_threshold, stage2_threshold, stage3_threshold, initial_memory
                    ])
                return
            
            stage1_passed_chunk = chunk.iloc[stage1_results['pass']]
            stage1_passed_df = pd.concat([stage1_passed_df, stage1_passed_chunk])
            stage1_pass_count += stage1_pass
            
            # Stage 2 processing with score collection
            logging.info(f"Processing Stage 2 with threshold {stage2_threshold:.4f}")
            try:
                stage2_results = process_stage2_chunk_with_scores(stage1_passed_chunk, stage2_model, 
                                                stage2_threshold, config, chunk_id)
                stage2_pass = len(stage2_results['pass'])
                stage2_fail = stage1_pass - stage2_pass
                
                # Store Stage 2 scores for threshold adaptation
                if 'scores' in stage2_results and isinstance(stage2_results['scores'], list):
                    stage2_score_storage.extend(stage2_results['scores'])
            except Exception as e:
                logging.error(f"Error in Stage 2 processing for chunk {chunk_id}: {e}")
                # Record failed chunk stats
                with open(chunk_stats_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        chunk_id, chunk_size, stage1_pass, stage1_fail, 0, stage1_pass, 0, 0.0,
                        stage1_threshold, stage2_threshold, stage3_threshold, initial_memory
                    ])
                return
            
            if stage2_pass == 0:
                logging.info(f"No molecules passed Stage 2 in chunk {chunk_id}")
                with open(chunk_stats_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        chunk_id, chunk_size, stage1_pass, stage1_fail,
                        0, stage1_pass, 0, 0.0,
                        stage1_threshold, stage2_threshold, stage3_threshold, initial_memory
                    ])
                return
            
            stage2_passed_chunk = stage1_passed_chunk.iloc[stage2_results['pass']]
            stage2_passed_df = pd.concat([stage2_passed_df, stage2_passed_chunk])
            stage2_pass_count += stage2_pass

            # Stage 3 processing with predictions
            logging.info(f"Processing Stage 3 with threshold {stage3_threshold:.4f}")
            try:
                stage3_results = process_stage3_chunk(stage2_passed_chunk, stage3_model, 
                                                    stage3_threshold, stage3_scaler, 
                                                    config, chunk_id)
                
                # Extract predictions and valid indices
                valid_indices = stage3_results.get('valid_indices', np.array([]))
                predictions = stage3_results.get('predictions', np.array([]))
                
                # Log explicitly for debugging
                logging.info(f"Stage 3 returned {len(valid_indices)} valid indices and {len(predictions)} predictions")
                
                # Store Stage 3 scores for threshold adaptation
                if (predictions is not None and 
                    isinstance(predictions, (list, np.ndarray)) and 
                    len(predictions) > 0):
                    # Convert to list if numpy array
                    if isinstance(predictions, np.ndarray):
                        stage3_score_storage.extend(predictions.tolist())
                    else:
                        stage3_score_storage.extend(predictions)
                    
                # Add probability scores to original data
                final_passed_chunk = stage2_passed_chunk.copy()
                
                has_valid_indices = (valid_indices is not None and 
                                    isinstance(valid_indices, (list, np.ndarray)) and 
                                    len(valid_indices) > 0)
                has_predictions = (predictions is not None and 
                                isinstance(predictions, (list, np.ndarray)) and 
                                len(predictions) > 0)
                
                if has_valid_indices and has_predictions:
                    # Initialize probability column with zeros
                    final_passed_chunk['bbb_probability'] = 0.0
                    
                    # Update probabilities for valid molecules
                    for i, idx in enumerate(valid_indices):
                        if i < len(predictions) and idx < len(final_passed_chunk):
                            final_passed_chunk.iloc[idx, final_passed_chunk.columns.get_loc('bbb_probability')] = predictions[i]
                else:
                    logging.warning(f"No valid predictions returned from Stage 3 for chunk {chunk_id}")
                    final_passed_chunk['bbb_probability'] = 0.5
                
                # Apply Stage 3 threshold
                if hasattr(config, 'stage3_all_pass') and config.stage3_all_pass:
                    logging.info(f"Stage 3 'All Pass': Passing all {len(final_passed_chunk)} molecules with probabilities")
                    final_passing = final_passed_chunk.copy()
                else:
                    final_passing = final_passed_chunk[final_passed_chunk['bbb_probability'] >= stage3_threshold]
                    
            except Exception as e:
                logging.error(f"Error in Stage 3 processing for chunk {chunk_id}: {e}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                # Continue with all Stage 2 molecules as a fallback
                final_passed_chunk = stage2_passed_chunk.copy()
                final_passed_chunk['bbb_probability'] = 0.5  # Assign default probability
                final_passing = final_passed_chunk.copy()  # Keep all molecules
            
            # Add to final output
            final_passed_df = pd.concat([final_passed_df, final_passing])
            stage3_processed = len(final_passed_chunk)
            stage3_pass = len(final_passing)
            stage3_processed_count += stage3_processed
            
            # Calculate mean probability for logging
            mean_prob = final_passed_chunk['bbb_probability'].mean() if 'bbb_probability' in final_passed_chunk.columns else 0.0
            
            # Dynamic threshold logic
            if total_molecules >= MIN_MOLECULES_FOR_ADAPTATION and not thresholds_adapted:
                # Attempt to calculate new thresholds with appropriate caps
                new_stage1_threshold = calculate_threshold_for_ratio(
                    stage1_score_storage, TARGET_STAGE1_PASS_RATIO, 
                    stage_name="Stage 1", max_threshold=0.95
                )
                new_stage2_threshold = calculate_threshold_for_ratio(
                    stage2_score_storage, TARGET_STAGE2_PASS_RATIO, 
                    stage_name="Stage 2", max_threshold=0.98
                )
                new_stage3_threshold = calculate_threshold_for_ratio(
                    stage3_score_storage, TARGET_STAGE3_PASS_RATIO, 
                    stage_name="Stage 3", max_threshold=0.5
                )
                
                if new_stage1_threshold is not None:
                    logging.info(f"Adapting Stage 1 threshold: {stage1_threshold:.4f} -> {new_stage1_threshold:.4f}")
                    stage1_threshold = new_stage1_threshold
                
                if new_stage2_threshold is not None:
                    logging.info(f"Adapting Stage 2 threshold: {stage2_threshold:.4f} -> {new_stage2_threshold:.4f}")
                    stage2_threshold = new_stage2_threshold
                    
                if new_stage3_threshold is not None:
                    logging.info(f"Adapting Stage 3 threshold: {stage3_threshold:.4f} -> {new_stage3_threshold:.4f}")
                    stage3_threshold = new_stage3_threshold
                
                # Save adapted thresholds
                save_model_thresholds({
                    'stage1': stage1_threshold,
                    'stage2': stage2_threshold,
                    'stage3': stage3_threshold
                }, config)
                            
                thresholds_adapted = True
                
                # Clear score storage to save memory
                stage1_score_storage.clear()
                stage2_score_storage.clear()
                stage3_score_storage.clear()

            # Force adaptation after first chunk
            if chunk_id == 1 and not force_adapted and len(stage1_score_storage) > 1000:
                try:
                    logging.info(f"[FORCE] Attempting forced threshold adaptation after first chunk")
                    
                    # Force calculate thresholds for all three stages with caps
                    forced_stage1_threshold = calculate_threshold_for_ratio(
                        stage1_score_storage, TARGET_STAGE1_PASS_RATIO, 
                        stage_name="Stage 1 [FORCE]", max_threshold=0.95
                    )
                    if forced_stage1_threshold is not None:
                        logging.info(f"[FORCE] Adapting Stage 1 threshold: {stage1_threshold:.4f} -> {forced_stage1_threshold:.4f}")
                        stage1_threshold = forced_stage1_threshold
                    
                    if len(stage2_score_storage) > 100:
                        forced_stage2_threshold = calculate_threshold_for_ratio(
                            stage2_score_storage, TARGET_STAGE2_PASS_RATIO, 
                            stage_name="Stage 2 [FORCE]", max_threshold=0.98
                        )
                        if forced_stage2_threshold is not None:
                            logging.info(f"[FORCE] Adapting Stage 2 threshold: {stage2_threshold:.4f} -> {forced_stage2_threshold:.4f}")
                            stage2_threshold = forced_stage2_threshold
                    
                    if len(stage3_score_storage) > 10:
                        forced_stage3_threshold = calculate_threshold_for_ratio(
                            stage3_score_storage, TARGET_STAGE3_PASS_RATIO, 
                            stage_name="Stage 3 [FORCE]", max_threshold=0.5
                        )
                        if forced_stage3_threshold is not None:
                            logging.info(f"[FORCE] Adapting Stage 3 threshold: {stage3_threshold:.4f} -> {forced_stage3_threshold:.4f}")
                            stage3_threshold = forced_stage3_threshold
                    
                    # Save the forced thresholds
                    save_model_thresholds({
                        'stage1': stage1_threshold,
                        'stage2': stage2_threshold,
                        'stage3': stage3_threshold
                    }, config)
                    
                    logging.info(f"[FORCE] Adapted thresholds after first chunk processing")
                    force_adapted = True
                    thresholds_adapted = True
                except Exception as e:
                    logging.error(f"Error during forced threshold adaptation: {e}")
            
            # Save results every 100k molecules
            if len(final_passed_df) >= 100000:
                output_path = os.path.join(config.data_dir, f"final_passed_interim_{chunk_id}.parquet")
                final_passed_df.to_parquet(output_path)
                logging.info(f"Saved {len(final_passed_df):,} interim results to {output_path}")
                final_passed_df = pd.DataFrame()
            
            # Get current memory usage
            current_memory = get_memory_usage()
            
            chunk_time = time.time() - chunk_start
            logging.info(f"Chunk {chunk_id} processed in {chunk_time:.2f} seconds")
            logging.info(f"Molecules passing Stage 1: {stage1_pass:,} (threshold: {stage1_threshold:.4f})")
            logging.info(f"Molecules passing Stage 2: {stage2_pass:,} (threshold: {stage2_threshold:.4f})")
            logging.info(f"Molecules passing Stage 3: {stage3_pass:,} (threshold: {stage3_threshold:.4f})")
            logging.info(f"Memory usage: {current_memory:.1f} MB (change: {current_memory - initial_memory:.1f} MB)")
            
            # Record statistics
            with open(chunk_stats_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    chunk_id, chunk_size,
                    stage1_pass, stage1_fail,
                    stage2_pass, stage2_fail,
                    stage3_processed, mean_prob,
                    stage1_threshold, stage2_threshold, stage3_threshold,
                    current_memory
                ])
                
            # Print progress towards target reduction
            current_reduction_ratio = stage3_pass / chunk_size
            target_reduction_ratio = TARGET_STAGE1_PASS_RATIO * TARGET_STAGE2_PASS_RATIO * TARGET_STAGE3_PASS_RATIO
            logging.info(f"Current reduction ratio: {current_reduction_ratio:.4f} (target: {target_reduction_ratio:.4f})")
            logging.info(f"Estimated final count from 1M: {int(1000000 * current_reduction_ratio):,} molecules")
            
            # Force garbage collection to free memory
            gc.collect()
            
        except Exception as e:
            # Prevent the whole pipeline from crashing
            logging.error(f"Unexpected error processing chunk {chunk_id}: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            logging.warning(f"Continuing to next chunk...")
            
            try:
                with open(chunk_stats_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        chunk_id, chunk_size if 'chunk_size' in locals() else 0,
                        0, 0, 0, 0, 0, 0.0,
                        stage1_threshold, stage2_threshold, stage3_threshold,
                        get_memory_usage()
                    ])
            except:
                logging.error("Failed to record chunk failure stats")

    try:
        import pyarrow.parquet as pq
        
        pf = pq.ParquetFile(input_parquet)
        
        # Process row groups as chunks
        for i in range(pf.num_row_groups):
            chunk_id += 1
            
            # Skip chunks before the start_chunk
            if chunk_id < start_chunk:
                logging.info(f"Skipping chunk {chunk_id} (starting from {start_chunk})")
                continue
                
            try:
                # Read a row group
                row_group = pf.read_row_group(i)
                chunk = row_group.to_pandas()
                
                # Check if chunk is empty with explicit length check
                if len(chunk) == 0:
                    logging.info(f"Empty chunk {chunk_id}, skipping")
                    continue
                    
                logging.info(f"Processing chunk {chunk_id}, size: {len(chunk):,}")
                process_chunk(chunk, chunk_id)
                
                # Clear memory after each chunk
                del chunk
                del row_group
                gc.collect()
                
            except Exception as chunk_error:
                logging.error(f"Error processing row group {i} (chunk {chunk_id}): {chunk_error}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                logging.info(f"Skipping to next row group")
                continue

    except Exception as e:
        logging.warning(f"Failed to process parquet file by row groups: {e}")
        logging.warning(f"Traceback: {traceback.format_exc()}")
        logging.info("Falling back to reading in chunks")
        
        try:
            import pyarrow.parquet as pq
            
            chunk_size = config.chunk_size_stage1
            
            # Read the file in chunks instead of all at once
            with pq.ParquetFile(input_parquet) as pf:
                total_rows = pf.metadata.num_rows
                for i in range(0, total_rows, chunk_size):
                # Read a subset of rows
                    try:
                        table = pf.read(skip=i, num_rows=min(chunk_size, total_rows - i))
                        sub_chunk = table.to_pandas()
                        
                        chunk_id += 1
                        
                        # Skip chunks before the start_chunk
                        if chunk_id < start_chunk:
                            logging.info(f"Skipping fallback chunk {chunk_id} (starting from {start_chunk})")
                            continue
                            
                        logging.info(f"Processing fallback chunk {chunk_id}, size: {len(sub_chunk):,}")
                    
                        process_chunk(sub_chunk, chunk_id)
                        
                        # Clear memory
                        del sub_chunk
                        del table
                        gc.collect()
                        
                    except Exception as chunk_error:
                        logging.error(f"Error processing fallback chunk at position {i}: {chunk_error}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        logging.info(f"Skipping to next chunk")
                        continue
                        
        except Exception as inner_e:
            logging.warning(f"Failed to process parquet in chunks: {inner_e}")
            logging.warning(f"Traceback: {traceback.format_exc()}")
            logging.info("Reading entire file as last resort")
            
            try:
                full_df = pd.read_parquet(input_parquet)
                
                # Process in chunks of config.chunk_size_stage1
                for i in range(0, len(full_df), config.chunk_size_stage1):
                    try:
                        chunk_id += 1
                        
                        # Skip chunks before the start_chunk
                        if chunk_id < start_chunk:
                            logging.info(f"Skipping final fallback chunk {chunk_id} (starting from {start_chunk})")
                            continue
                            
                        # Get a chunk of the data
                        chunk_end = min(i + config.chunk_size_stage1, len(full_df))
                        chunk = full_df.iloc[i:chunk_end].copy()
                        
                        logging.info(f"Processing final fallback chunk {chunk_id}, size: {len(chunk):,}")
                        
                        process_chunk(chunk, chunk_id)
                        
                        # Clear memory
                        del chunk
                        gc.collect()
                        
                    except Exception as chunk_error:
                        logging.error(f"Error processing final fallback chunk at position {i}: {chunk_error}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        logging.info(f"Skipping to next chunk")
                        continue
                        
                # Clean up
                del full_df
                gc.collect()
                
            except Exception as last_e:
                logging.error(f"Fatal error: Unable to process the Parquet file: {last_e}")
                logging.error(f"Traceback: {traceback.format_exc()}")

        logging.info("\nSaving final results...")

        # Save interim files
        interim_files = [os.path.join(config.data_dir, f) 
                        for f in os.listdir(config.data_dir) 
                        if f.startswith("final_passed_interim_") and f.endswith(".parquet")]

        for file in interim_files:
            try:
                interim_df = pd.read_parquet(file)
                final_passed_df = pd.concat([final_passed_df, interim_df])
                os.remove(file)
            except Exception as e:
                logging.error(f"Error loading interim file {file}: {e}")

        # Save final results
        try:
            if not stage1_passed_df.empty:
                stage1_passed_df.to_parquet(os.path.join(config.data_dir, "stage1_passed.parquet"))
                logging.info(f"Saved {len(stage1_passed_df):,} molecules that passed Stage 1")
        except Exception as e:
            logging.error(f"Error saving Stage 1 results: {e}")

        try:
            if not stage2_passed_df.empty:
                stage2_passed_df.to_parquet(os.path.join(config.data_dir, "stage2_passed.parquet"))
                logging.info(f"Saved {len(stage2_passed_df):,} molecules that passed Stage 2")
        except Exception as e:
            logging.error(f"Error saving Stage 2 results: {e}")

        try:
            if not final_passed_df.empty:
                final_passed_df.to_parquet(os.path.join(config.data_dir, "final_with_probabilities.parquet"))
                logging.info(f"Saved {len(final_passed_df):,} molecules that passed all stages with probabilities")
                
                # Also save high confidence molecules for reference
                if 'bbb_probability' in final_passed_df.columns:
                    high_prob_df = final_passed_df[final_passed_df['bbb_probability'] >= 0.5]
                    high_prob_df.to_parquet(os.path.join(config.data_dir, "high_probability_bbb.parquet"))
                    logging.info(f"Saved {len(high_prob_df):,} high probability (≥0.5) molecules")
            else:
                logging.warning("No molecules passed all stages")
        except Exception as e:
            logging.error(f"Error saving final results: {e}")

        # Final statistics
        final_pass_count = len(final_passed_df)
        overall_reduction_ratio = final_pass_count / total_molecules if total_molecules > 0 else 0

        logging.info("\nOverall Pipeline Results:")
        logging.info(f"Total input molecules processed: {total_molecules:,}")
        logging.info(f"Molecules passing Stage 1 (threshold {stage1_threshold:.4f}): {stage1_pass_count:,}")
        logging.info(f"Molecules passing Stage 2 (threshold {stage2_threshold:.4f}): {stage2_pass_count:,}")
        logging.info(f"Molecules passing Stage 3 (threshold {stage3_threshold:.4f}): {final_pass_count:,}")
        logging.info(f"Overall reduction ratio: {overall_reduction_ratio:.6f}")

        if total_molecules >= 1000000:
            per_million = int(1000000 * overall_reduction_ratio)
            logging.info(f"Per million molecules: {per_million:,} pass all stages")

        # Save summary data
        try:
            summary = {
                'stage1_threshold': float(stage1_threshold),
                'stage2_threshold': float(stage2_threshold),
                'stage3_threshold': float(stage3_threshold),
                'total_processed': int(total_molecules),
                'stage1_pass': int(stage1_pass_count),
                'stage2_pass': int(stage2_pass_count),
                'final_pass': int(final_pass_count),
                'overall_reduction_ratio': float(overall_reduction_ratio),
                'per_million_estimate': int(1000000 * overall_reduction_ratio) if total_molecules > 0 else 0
            }
            with open(os.path.join(config.results_dir, "cascade_summary.json"), 'w') as f:
                json.dump(summary, f, indent=4)
            logging.info(f"Saved summary data to {os.path.join(config.results_dir, 'cascade_summary.json')}")
        except Exception as e:
            logging.error(f"Error saving summary data: {e}")

    return {
        'total': total_molecules,
        'stage1_pass': stage1_pass_count,
        'stage2_pass': stage2_pass_count,
        'final_pass': final_pass_count,
        'stage1_threshold': stage1_threshold,
        'stage2_threshold': stage2_threshold,
        'stage3_threshold': stage3_threshold
    }

# Input: Molecular DF, config, cache
# Output: Stage 1 data as DF
# Description: Prepares stage 1 data
def prepare_stage1_data(df: pd.DataFrame, config: Config, use_cache: bool = True) -> pd.DataFrame:
    logging.info("Preparing stage 1 data...")
    cache_file = os.path.join(config.cache_dir, "stage1_features.parquet")
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 1 features from {cache_file}")
        try:
            features_df = pd.read_parquet(cache_file)
            result_df = df.reset_index(drop=True).merge(
                features_df, left_index=True, right_on='smiles_idx', how='inner'
            )
            result_df.drop(columns=['smiles_idx'], inplace=True)
            logging.info(f"Stage 1 data prepared from cache: {len(result_df)} valid molecules")
            return result_df
        except Exception as e:
            logging.warning(f"Failed to load cache, computing features: {e}")

    logging.info("Computing stage 1 descriptors...")
    features = []
    valid_indices = []
    quick_reject_counts = {}

    with Pool(processes=config.max_processes) as pool:
        results = list(tqdm(
             pool.imap(compute_stage1_features, df['smiles']),
             total=len(df),
             desc="Computing Stage 1 features"
        ))

    for i, result in enumerate(results):
        if result is None:
            continue
        elif isinstance(result, dict) and result.get('quick_reject', False):
            reason = result.get('reason', 'unknown')
            quick_reject_counts[reason] = quick_reject_counts.get(reason, 0) + 1
        else:
            features.append(result)
            valid_indices.append(i)

    # Log quick rejections
    if quick_reject_counts:
        logging.info("Quick rejection statistics:")
        for reason, count in quick_reject_counts.items():
            logging.info(f"  {reason}: {count} molecules")

    feature_names = [
        'mw', 'hba', 'hbd', 'tpsa', 'aromatic_atoms', 'basic_centers',
        'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
        'logp', 'rotbonds',     # original 12
        'fsp3', 'ring_count', 'ring_atom_ratio', 'psa_mw_ratio', 'complexity',   # additional 5
        'chi0v', 'chi1v', 'kappa1', 'kappa2', 'kappa3', 'num_heavy_atoms'         # extra 6
    ]

    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['smiles_idx'] = valid_indices

    os.makedirs(config.cache_dir, exist_ok=True)
    features_df.to_parquet(cache_file)

    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    result_df = pd.concat([df_valid, features_df.drop(columns=['smiles_idx'])], axis=1)

    failed_indices = [i for i in range(len(df)) if i not in valid_indices]
    if failed_indices:
        failed_smiles = df.iloc[failed_indices]['smiles'].tolist()
        with open(os.path.join(config.log_dir, "failed_smiles_stage1.txt"), 'w') as f:
            f.write("\n".join(failed_smiles))
        logging.info(f"Logged {len(failed_indices)} failed SMILES in Stage 1")

    logging.info(f"Stage 1 data prepared: {len(result_df)} valid molecules")
    return result_df

# Input: Molecular DF, config, cache
# Output: DF of stage 2 data
# Description: Compute Stage 2 features: MACCS, ECFP4 fingerprints, and basic descriptors, ensure Stage 1 specific features are included
def prepare_stage2_data(df: pd.DataFrame, config: Config, use_cache: bool = True) -> pd.DataFrame:
    logging.info("Preparing stage 2 data...")

    # Stage 1 specific features that need to be included
    stage1_specific = [
        'consecutive_polar', 'h_bond_terminals', 'chi_v3', 'hbond_pairs',
        'fsp3', 'ring_count', 'ring_atom_ratio', 'psa_mw_ratio', 'complexity'
    ]
    
    # Check if Stage 1 features are already present
    missing_stage1_features = [f for f in stage1_specific if f not in df.columns]
    
    # Compute missing Stage 1 features if needed
    if missing_stage1_features:
        logging.info(f"Computing missing Stage 1 features: {missing_stage1_features}")
        # Use a simplified approach to compute the missing features
        for i, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    if 'fsp3' in missing_stage1_features:
                        df.at[i, 'fsp3'] = Descriptors.FractionCSP3(mol)
                    if 'ring_count' in missing_stage1_features:
                        df.at[i, 'ring_count'] = Lipinski.RingCount(mol)
                    if 'ring_atom_ratio' in missing_stage1_features:
                        ring_count = Lipinski.RingCount(mol)
                        df.at[i, 'ring_atom_ratio'] = ring_count / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0
                    if 'psa_mw_ratio' in missing_stage1_features:
                        tpsa = Descriptors.TPSA(mol)
                        mw = Descriptors.MolWt(mol)
                        df.at[i, 'psa_mw_ratio'] = tpsa / mw if mw > 0 else 0
                    if 'complexity' in missing_stage1_features:
                        df.at[i, 'complexity'] = GraphDescriptors.BertzCT(mol)
                    if 'consecutive_polar' in missing_stage1_features:
                        df.at[i, 'consecutive_polar'] = max_consecutive_polar_atoms(mol)
                    if 'h_bond_terminals' in missing_stage1_features:
                        df.at[i, 'h_bond_terminals'] = terminal_h_bond_atoms(mol)
                    if 'chi_v3' in missing_stage1_features:
                        df.at[i, 'chi_v3'] = GraphDescriptors.Chi3v(mol) if mol.GetNumAtoms() > 4 else 0
                    if 'hbond_pairs' in missing_stage1_features:
                        df.at[i, 'hbond_pairs'] = count_proximal_hbond_pairs(mol)
            except Exception as e:
                logging.warning(f"Error computing Stage 1 features for {row['smiles']}: {e}")
                # Fill with default values
                for feat in missing_stage1_features:
                    if feat not in df.columns:
                        df[feat] = 0
                    df.at[i, feat] = 0

    cache_file = os.path.join(config.cache_dir, "stage2_features.parquet")
    if use_cache and os.path.exists(cache_file):
        logging.info(f"Loading cached Stage 2 features from {cache_file}")
        try:
            features_df = pd.read_parquet(cache_file)
            if 'smiles_idx' not in features_df.columns:
                raise KeyError("Missing 'smiles_idx' in cached features")
            result_df = df.reset_index(drop=True).merge(
                features_df, left_index=True, right_on='smiles_idx', how='inner'
            )
            result_df.drop(columns=['smiles_idx'], inplace=True)
            logging.info(f"Stage 2 data prepared from cache: {len(result_df)} valid molecules")
            return result_df
        except Exception as e:
            logging.warning(f"Failed to load cache, recomputing features: {e}")

    logging.info("Computing stage 2 features...")
    descriptor_list, maccs_list, ecfp4_list, substructure_list, valid_indices = [], [], [], [], []

    with Pool(
        processes=config.max_processes,
        initializer=pool_initializer,
        initargs=(SUB_SMARTS_PATTERNS, KR_SMARTS_PATTERNS, ESTATE_SMARTS_PATTERNS)
    ) as pool:
        results = list(tqdm(
            pool.imap(compute_stage2_features, df['smiles']),
            total=len(df),
            desc="Computing Stage 2 features"
        ))

    failed_indices, failed_smiles = [], []

    for i, result in enumerate(results):
        if result is not None:
            descriptor_list.append(result['descriptors'])
            maccs_list.append(result['maccs'])
            ecfp4_list.append(result['ecfp4'])
            substructure_list.append(result['substructure'])
            valid_indices.append(i)
        else:
            failed_indices.append(i)
            failed_smiles.append(df.iloc[i]['smiles'])

    if not valid_indices:
        logging.error("No valid molecules found for Stage 2 features")
        raise ValueError("No valid molecules found for Stage 2 features")

    logging.info(f"Valid results: {len(valid_indices)} out of {len(df)} molecules")

    if not substructure_list:
        logging.error("Substructure fingerprints are missing")
        raise ValueError("Substructure fingerprints list is empty")
    
    expected_fp_length = len(substructure_list[0])
    for fp in substructure_list:
        if len(fp) != expected_fp_length:
            logging.error(f"Inconsistent SubFP length: Expected {expected_fp_length}, got {len(fp)}")
            raise ValueError("Inconsistent substructure fingerprint lengths")

    descriptor_names = [
        'mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotbonds',
        'aromatic_rings', 'formal_charge', 'mr',
        'heavy_halogen', 'aromatic_proportion', 'n_carboxylic', 'n_amine'
    ]

    maccs_names = [f'maccs_{i}' for i in range(167)]
    ecfp4_names = [f'ecfp4_{i}' for i in range(1024)]
    substructure_names = [f'sub_fp_{i}' for i in range(expected_fp_length)]

    descriptors_df = pd.DataFrame(descriptor_list, columns=descriptor_names)
    maccs_df = pd.DataFrame(maccs_list, columns=maccs_names)
    ecfp4_df = pd.DataFrame(ecfp4_list, columns=ecfp4_names)
    sub_df = pd.DataFrame(substructure_list, columns=substructure_names)

    # Concatenate all features
    features_df = pd.concat([descriptors_df, maccs_df, ecfp4_df, sub_df], axis=1)
    features_df['smiles_idx'] = valid_indices

    os.makedirs(config.cache_dir, exist_ok=True)
    features_df.to_parquet(cache_file)

    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    result_df = pd.concat([df_valid, features_df.drop(columns=['smiles_idx'])], axis=1)

    if failed_indices:
        os.makedirs(config.log_dir, exist_ok=True)
        failed_log_path = os.path.join(config.log_dir, "failed_smiles_stage2.txt")
        with open(failed_log_path, 'w') as f:
            f.write("\n".join(failed_smiles))
        logging.info(f"Logged {len(failed_indices)} failed SMILES in {failed_log_path}")

    logging.info(f"Stage 2 data prepared: {len(result_df)} valid molecules")
    return result_df

# Input: input D3DB Database with Substructure CSVs for unaugmented calculation, config, cache
# Output: DF of stage 3 data
# Description: Prepare dataset with all features for model training
def prepare_stage3_data(input_file, smarts_sub=None, smarts_kr=None, smarts_estate=None, n_jobs=None):
    global SUB_SMARTS_PATTERNS, KR_SMARTS_PATTERNS, ESTATE_SMARTS_PATTERNS
    
    # Load SMARTS patterns
    if smarts_sub and smarts_kr:
        SUB_SMARTS_PATTERNS.clear()
        KR_SMARTS_PATTERNS.clear()
        ESTATE_SMARTS_PATTERNS.clear()
        
        load_smarts_patterns(smarts_sub, smarts_kr, smarts_estate)

        logging.info(f"Loaded patterns: {len(SUB_SMARTS_PATTERNS)} substructure, {len(KR_SMARTS_PATTERNS)} Klekota-Roth, {len(ESTATE_SMARTS_PATTERNS)} EState")
    
    # Load dataset
    df = pd.read_csv(input_file)
    logging.info(f"Loaded dataset with {len(df)} molecules")
    
    for col in df.columns:
        if col.lower() == 'smiles':
            df['SMILES'] = df[col]
        if col.lower() == 'bbb_label':
            df['BBB_LABEL'] = df[col]
    
    if 'SMILES' not in df.columns:
        raise ValueError("Input file must contain a 'SMILES' column")
    if 'BBB_LABEL' not in df.columns:
        raise ValueError("Input file must contain a 'BBB_LABEL' column")
    
    # Convert BBB_LABEL to binary (from old use)
    df['BBB_LABEL'] = df['BBB_LABEL'].apply(
        lambda x: 1 if str(x).strip().lower() in ['1', '+'] else 0
    )
    
    # Report class distribution
    class_counts = df['BBB_LABEL'].value_counts()
    logging.info(f"Class distribution: BBB+ = {class_counts.get(1, 0)}, BBB- = {class_counts.get(0, 0)}")
    
    # Compute all features in parallel
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    logging.info(f"Computing molecular features using {n_jobs} parallel jobs")
    
    # Make local copies of the patterns to pass to the initializer
    sub_patterns_copy = SUB_SMARTS_PATTERNS.copy()
    kr_patterns_copy = KR_SMARTS_PATTERNS.copy()
    estate_patterns_copy = ESTATE_SMARTS_PATTERNS.copy()
    
    # Using initializer and initargs to pass copies of the patterns to each worker
    with Pool(
        processes=n_jobs,
        initializer=pool_initializer,
        initargs=(sub_patterns_copy, kr_patterns_copy, estate_patterns_copy)
    ) as pool:
        features_list = list(tqdm(
            pool.imap(compute_all_features, df['SMILES']),
            total=len(df),
            desc="Computing features"
        ))
    
    # Identify valid molecules and their indices
    valid_indices = [i for i, f in enumerate(features_list) if f is not None]
    valid_features = [features_list[i] for i in valid_indices]
    
    # Report success rate
    logging.info(f"Successfully processed {len(valid_features)}/{len(df)} molecules ({len(valid_features)/len(df)*100:.1f}%)")
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(valid_features)
    
    # Handle NaN values
    na_count = feature_df.isna().sum().sum()
    if na_count > 0:
        logging.warning(f"Found {na_count} NaN values in features, filling with medians")
        feature_df = feature_df.fillna(feature_df.median())
    
    # Extract labels
    labels = df.iloc[valid_indices]['BBB_LABEL']
    
    # Keep SMILES for reference
    smiles = df.iloc[valid_indices]['SMILES']
    
    # Save SMILES to feature_df for model evaluation
    feature_df['SMILES'] = smiles.values
    
    return feature_df, labels

# Input: Molecular DF
# Output: Augmented Molecular DF
# Descriptions: Augments Molecular DF due to misbalance in B3BD DF [-] for Stage 1 and 2 (Stage 3 model was optimized without augmentation on PC by testing several values)
"""
    Args:
        df: DataFrame with 'smiles' and 'bbb_label' columns
        logger: Logger instance
        augment_ratio: How many augmented molecules to create relative to original count
        random_state: Random seed for reproducibility
"""
def augment_dataset(df: pd.DataFrame, logger: logging.Logger, 
                    augment_ratio: float = 0.5,
                    random_state: int = 42) -> pd.DataFrame:
    # Only augment BBB- molecules (label=0) for Stage 1 and 2
    # Stage 3 will use the original dataset without augmentation
    logger.info("Performing data augmentation for Stages 1 and 2...")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Get original class distribution
    class_counts = df['bbb_label'].value_counts()
    logger.info(f"Original class distribution: BBB+ = {class_counts.get(1, 0)}, BBB- = {class_counts.get(0, 0)}")
    
    # Only augment the negative class (BBB-)
    df_neg = df[df['bbb_label'] == 0].copy()
    
    # Calculate number of molecules to augment
    n_aug = int(len(df_neg) * augment_ratio)
    logger.info(f"Will create {n_aug} augmented negative samples")
    
    if n_aug == 0:
        logger.info("No augmentation needed")
        return df
    
    # Sample molecules to augment
    if n_aug > len(df_neg):
        # Sample with replacement if more augmented molecules than originals are needed
        sample_idx = np.random.choice(len(df_neg), n_aug, replace=True)
    else:
        # Sample without replacement
        sample_idx = np.random.choice(len(df_neg), n_aug, replace=False)
    
    sample_df = df_neg.iloc[sample_idx]
    
    # Simple augmentation by adding/moving functional groups
    augmented_smiles = []
    for _, row in sample_df.iterrows():
        try:
            smiles = row['smiles']
            mol = Chem.MolFromSmiles(smiles)
            
            # Skip if invalid molecule
            if mol is None:
                continue
                
            # Apply different transformations
            aug_type = random.randint(1, 5)
            
            if aug_type == 1:
                # Add methyl group to aromatic carbon
                patt = Chem.MolFromSmarts("c[H]")
                if mol.HasSubstructMatch(patt):
                    rxn = AllChem.ReactionFromSmarts("c[H]>>cC")
                    products = rxn.RunReactants((mol,))
                    if products and products[0]:
                        aug_mol = products[0][0]
                        Chem.SanitizeMol(aug_mol)
                        aug_smiles = Chem.MolToSmiles(aug_mol)
                        augmented_smiles.append(aug_smiles)
                        continue
            
            elif aug_type == 2:
                # Add hydroxyl group to an aliphatic carbon
                patt = Chem.MolFromSmarts("C[H]")
                if mol.HasSubstructMatch(patt):
                    rxn = AllChem.ReactionFromSmarts("C[H]>>CO")
                    products = rxn.RunReactants((mol,))
                    if products and products[0]:
                        aug_mol = products[0][0]
                        Chem.SanitizeMol(aug_mol)
                        aug_smiles = Chem.MolToSmiles(aug_mol)
                        augmented_smiles.append(aug_smiles)
                        continue
            
            elif aug_type == 3:
                # Methylate a hydroxyl group
                patt = Chem.MolFromSmarts("O[H]")
                if mol.HasSubstructMatch(patt):
                    rxn = AllChem.ReactionFromSmarts("O[H]>>OC")
                    products = rxn.RunReactants((mol,))
                    if products and products[0]:
                        aug_mol = products[0][0]
                        Chem.SanitizeMol(aug_mol)
                        aug_smiles = Chem.MolToSmiles(aug_mol)
                        augmented_smiles.append(aug_smiles)
                        continue
            
            elif aug_type == 4:
                # Randomize SMILES (different atom order)
                aug_smiles = Chem.MolToSmiles(mol, doRandom=True)
                augmented_smiles.append(aug_smiles)
                continue
                
            else:
                # Add ethyl instead of methyl
                patt = Chem.MolFromSmarts("c[H]")
                if mol.HasSubstructMatch(patt):
                    rxn = AllChem.ReactionFromSmarts("c[H]>>cCC")
                    products = rxn.RunReactants((mol,))
                    if products and products[0]:
                        aug_mol = products[0][0]
                        Chem.SanitizeMol(aug_mol)
                        aug_smiles = Chem.MolToSmiles(aug_mol)
                        augmented_smiles.append(aug_smiles)
                        continue
            
            # Fallback: Use original SMILES if all transformations failed
            augmented_smiles.append(smiles)
            
        except Exception as e:
            logger.warning(f"Error during augmentation: {e}")
            # Skip molecule if augmentation fails
            continue
    
    # Create DataFrame with augmented molecules
    aug_df = pd.DataFrame({
        'smiles': augmented_smiles,
        'bbb_label': 0 
    })
    
    # Combine original and augmented DataFrames
    result_df = pd.concat([df, aug_df], ignore_index=True)
    
    # Get new class distribution
    new_class_counts = result_df['bbb_label'].value_counts()
    logger.info(f"Augmented class distribution: BBB+ = {new_class_counts.get(1, 0)}, BBB- = {new_class_counts.get(0, 0)}")
    logger.info(f"Added {len(aug_df)} augmented molecules")
    
    return result_df

# Input: Input D3DB Database of BBB+/- molecules for training
# Output: Dictionary with DataFrames for each stage
# Description: Prepare data for training all three stages of the model.
"""
Args:
input_csv: Path to input CSV file with SMILES and BBB_LABEL columns
config: Configuration object
augment_for_stage1_stage2: Whether to perform data augmentation for Stage 1 and 2
"""
def prepare_data_for_training(input_csv: str, config: Config, augment_for_stage1_stage2: bool = True) -> Dict[str, pd.DataFrame]:
    logging.info(f"Loading training data from {input_csv}")
    
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    # Standardize column names - OLD (testing database sets)
    for col in df.columns:
        if col.lower() == 'smiles':
            df['smiles'] = df[col]
        if col.lower() in ['bbb_label', 'bbb', 'label']:
            df['bbb_label'] = df[col]
    
    # Convert BBB label to binary
    df['bbb_label'] = df['bbb_label'].apply(
        lambda x: 1 if str(x).strip().lower() in ['1', 'yes', 'true', 'positive', '+'] else 0
    )
    
    # Report class distribution
    class_counts = df['bbb_label'].value_counts()
    logging.info(f"Class distribution: BBB+ = {class_counts.get(1, 0)}, BBB- = {class_counts.get(0, 0)}")
    
    # Create a copy of the original dataset for Stage 3 (no augmentation)
    original_df = df.copy()
    
    # Perform data augmentation for Stage 1 and 2
    if augment_for_stage1_stage2:
        augmented_df = augment_dataset(df, logging)
    else:
        augmented_df = df
    
    # Prepare Stage 1 data (with augmentation)
    logging.info("Preparing Stage 1 data...")
    stage1_df = prepare_stage1_data(augmented_df, config)
    
    # Prepare Stage 2 data (with augmentation)
    logging.info("Preparing Stage 2 data...")
    stage2_df = prepare_stage2_data(augmented_df, config)
    
    # Create a separate directory for the original non-augmented dataset for Stage 3
    os.makedirs(config.input_dir, exist_ok=True)
    stage3_input = os.path.join(config.input_dir, "B3DB_classification.csv")
    original_df.to_csv(stage3_input, index=False)
    logging.info(f"Saved original non-augmented dataset for Stage 3 to {stage3_input}")
    
    return {
        'stage1': stage1_df,
        'stage2': stage2_df,
        'stage3_input': stage3_input
    }

def main():
    parser = argparse.ArgumentParser(
        description="Blood-Brain Barrier (BBB) Penetration Prediction Cascade Pipeline"
    )

    parser.add_argument("--train", action="store_true", 
                        help="Train the cascade models with B3DB labeled dataset")
    parser.add_argument("--train-and-infer", action="store_true",
                        help="Train models and run inference together")
    parser.add_argument("--input", type=str, required=False, 
                        help="Input Parquet file with molecules to process (for inference)")
    parser.add_argument("--train-data", type=str, 
                        help="CSV file with labeled data for training")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output directory for results")
    parser.add_argument("--use-cache", action="store_true", 
                        help="Use cached features if possible")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation for Stage 1 and 2, by default augments")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Only run Stage 1 model")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU")
    parser.add_argument("--stage3-all-pass", action="store_true",
                        help="Stage 3 passes all molecules with probabilities attached (for post-processing)")

    parser.add_argument("--stage1-threshold", type=float, 
                        help="Override Stage 1 threshold")
    parser.add_argument("--stage2-threshold", type=float, 
                        help="Override Stage 2 threshold")
    parser.add_argument("--stage3-threshold", type=float, 
                        help="Override Stage 3 threshold")
    parser.add_argument("--start-chunk", type=int, default=1,
                    help="Start processing from this chunk number")
    
    args = parser.parse_args()

    config = Config(output_dir=args.output)
    
    # Set GPU device (VM, PC)
    if args.gpu and torch.cuda.is_available():
        config.device = torch.device("cuda:0")
        logging.info(f"Using GPU device: {config.device}")
    else:
        config.device = torch.device("cpu")
        logging.info(f"Using CPU device")
        
    if args.stage3_all_pass:
        logging.info("Stage 3 'All Pass' mode activated - all molecules will pass with probabilities attached")
        config.stage3_all_pass = True
    else:
        config.stage3_all_pass = False
    
    logger = setup_logging(config.log_dir)

    logger.info("BBB Penetration Prediction Cascade Pipeline")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total, " +
               f"{psutil.virtual_memory().available / (1024**3):.1f} GB available")
    logger.info(f"CPU count: {psutil.cpu_count()} (using {config.max_processes} processes)")
    logger.info(f"Using device: {config.device}")

    # Variables to store models for combined training and inference
    stage1_model = None
    stage1_threshold = None
    stage2_model = None
    stage2_threshold = None
    stage3_model = None
    stage3_threshold = None
    stage3_scaler = None

    # Training or combined mode
    if args.train or args.train_and_infer:
        logger.info("Running in training mode...")
        logger.info(f"Using labeled data: {args.train_data}")
        
        # Prepare data for all three stages
        train_data = prepare_data_for_training(
            args.train_data, 
            config, 
            augment_for_stage1_stage2=not args.no_augment
        )
        
        # Train Stage 1 model with augmented data
        logger.info("Training Stage 1 model...")
        stage1_model, stage1_threshold, stage1_stats = train_stage1_model(train_data['stage1'], config)

        # Train Stage 2 model with augmented data
        logger.info("Training Stage 2 model...")
        stage2_model, stage2_threshold, stage2_stats = train_stage2_model(train_data['stage2'], config)

        # Train Stage 3 model with non-augmented original data
        logger.info("Training Stage 3 model...")
        stage3_model, stage3_threshold, stage3_stats, stage3_scaler = train_stage3_model(config)

        # Save thresholds for all stages
        thresholds = {
            "stage1": stage1_threshold,
            "stage2": stage2_threshold,
            "stage3": stage3_threshold
        }
        save_model_thresholds(thresholds, config)
        
        # Apply command-line threshold overrides
        if args.train_and_infer:
            if args.stage1_threshold is not None:
                logger.info(f"Overriding Stage 1 threshold: {stage1_threshold:.4f} -> {args.stage1_threshold:.4f}")
                stage1_threshold = args.stage1_threshold
                
            if args.stage2_threshold is not None:
                logger.info(f"Overriding Stage 2 threshold: {stage2_threshold:.4f} -> {args.stage2_threshold:.4f}")
                stage2_threshold = args.stage2_threshold
                
            if args.stage3_threshold is not None:
                logger.info(f"Overriding Stage 3 threshold: {stage3_threshold:.4f} -> {args.stage3_threshold:.4f}")
                stage3_threshold = args.stage3_threshold
                
            # If thresholds were overridden, save new values
            if args.stage1_threshold is not None or args.stage2_threshold is not None or args.stage3_threshold is not None:
                thresholds = {
                    "stage1": stage1_threshold,
                    "stage2": stage2_threshold,
                    "stage3": stage3_threshold
                }
                save_model_thresholds(thresholds, config)
                logger.info("Saved updated thresholds to thresholds.json")
        
        if not args.train_and_infer:
            logger.info("Model training complete")
            logger.info(f"Models saved to {config.model_dir}")
            return
    
    # Inference mode (either standalone or after training)
    if not args.train or args.train_and_infer:
        # Only load models if not in train-and-infer mode (where models are already in memory)
        if not args.train_and_infer:
            logger.info("Running in inference mode...")
            if not os.path.exists(config.stage1_model):
                logger.error(f"Stage 1 model file not found: {config.stage1_model}")
                logger.error("Please train the models first using --train option")
                sys.exit(1)
                
            # Load Stage 1 model
            logger.info(f"Loading Stage 1 model from {config.stage1_model}")
            try:
                with open(config.stage1_model, 'rb') as f:
                    stage1_data = pickle.load(f)
                    
                if isinstance(stage1_data, tuple):
                    # Old format: (model, threshold)
                    stage1_model, stage1_threshold = stage1_data
                else:
                    # New format: dictionary with model, feature_names, threshold
                    stage1_model = stage1_data['model']
                    stage1_threshold = stage1_data['threshold']
            except Exception as e:
                logger.error(f"Failed to load Stage 1 model: {e}")
                sys.exit(1)
            
            # If only running Stage 1, don't need to load other models
            if not args.stage1_only:
                # Load Stage 2 model
                if not os.path.exists(config.stage2_model):
                    logger.error(f"Stage 2 model file not found: {config.stage2_model}")
                    logger.error("Please train the models first using --train option")
                    sys.exit(1)
                    
                logger.info(f"Loading Stage 2 model from {config.stage2_model}")
                try:
                    with open(config.stage2_model, 'rb') as f:
                        stage2_data = pickle.load(f)
                        
                    if isinstance(stage2_data, tuple):
                        # Old format: (model, threshold)
                        stage2_model, stage2_threshold = stage2_data
                    else:
                        # New format: dictionary with model and metadata
                        stage2_model = stage2_data['model']
                        stage2_threshold = stage2_data['threshold']
                except Exception as e:
                    logger.error(f"Failed to load Stage 2 model: {e}")
                    sys.exit(1)
                
                # Load Stage 3 model
                if not os.path.exists(config.stage3_model):
                    logger.error(f"Stage 3 model file not found: {config.stage3_model}")
                    logger.error("Please train the models first using --train option")
                    sys.exit(1)
                    
                if not os.path.exists(config.stage3_scaler):
                    logger.error(f"Stage 3 scaler file not found: {config.stage3_scaler}")
                    logger.error("Please train the models first using --train option")
                    sys.exit(1)
                    
                logger.info(f"Loading Stage 3 model from {config.stage3_model}")
                try:
                    # Load the scaler
                    with open(config.stage3_scaler, 'rb') as f:
                        stage3_scaler = pickle.load(f)
                    
                    # Try to load model package first
                    model_package_path = os.path.join(config.model_dir, 'stage3_model_package.pkl')
                    if os.path.exists(model_package_path):
                        with open(model_package_path, 'rb') as f:
                            model_package = pickle.load(f)
                            
                        # Create model from package
                        stage3_model = SimpleNN(
                            input_dim=model_package['input_dim'],
                            hidden_dims=model_package['hidden_dims'],
                            dropout_rate=0.296,
                            activation='relu'
                        )
                        stage3_model.load_state_dict(model_package['model_state_dict'])
                        stage3_threshold = model_package['threshold']
                    else:
                        # Fallback to loading state dict directly
                        state_dict = torch.load(config.stage3_model, map_location=config.device)
                        
                        # Infer input_dim from first layer weight
                        first_layer_key = [k for k in state_dict.keys() if 'layers.0.weight' in k][0]
                        input_dim = state_dict[first_layer_key].shape[1]
                        
                        # Use the architecture from PC
                        stage3_model = SimpleNN(
                            input_dim=input_dim,
                            hidden_dims=[292, 188],
                            dropout_rate=0.296,
                            activation='relu'
                        )
                        stage3_model.load_state_dict(state_dict)
                        stage3_threshold = 0.108  # Default PC model threshold
                except Exception as e:
                    logger.error(f"Failed to load Stage 3 model: {e}")
                    sys.exit(1)
                    
                # Move model to device
                stage3_model = stage3_model.to(config.device)
            
            # Load thresholds from JSON
            if os.path.exists(config.thresholds_json):
                try:
                    with open(config.thresholds_json, 'r') as f:
                        thresholds = json.load(f)
                        
                    stage1_threshold = thresholds.get('stage1', stage1_threshold)
                    
                    if not args.stage1_only:
                        stage2_threshold = thresholds.get('stage2', stage2_threshold)
                        stage3_threshold = thresholds.get('stage3', stage3_threshold)
                    
                    logger.info(f"Loaded thresholds from {config.thresholds_json}")
                except Exception as e:
                    logger.warning(f"Failed to load thresholds from JSON: {e}")
        
        # Apply command-line threshold overrides
        if args.stage1_threshold is not None:
            logger.info(f"Overriding Stage 1 threshold to {args.stage1_threshold:.4f}")
            stage1_threshold = args.stage1_threshold
            
        if not args.stage1_only and args.stage2_threshold is not None:
            logger.info(f"Overriding Stage 2 threshold to {args.stage2_threshold:.4f}")
            stage2_threshold = args.stage2_threshold
            
        if not args.stage1_only and args.stage3_threshold is not None:
            logger.info(f"Overriding Stage 3 threshold to {args.stage3_threshold:.4f}")
            stage3_threshold = args.stage3_threshold
            
        # If any thresholds were overridden, save them
        if args.stage1_threshold is not None or args.stage2_threshold is not None or args.stage3_threshold is not None:
            thresholds = {
                "stage1": stage1_threshold,
                "stage2": stage2_threshold,
                "stage3": stage3_threshold
            }
            save_model_thresholds(thresholds, config)
            logger.info("Saved overridden thresholds to thresholds.json")
        
        # Apply config default thresholds if available and not set already
        if stage1_threshold is None and hasattr(config, "threshold") and "stage1" in config.threshold:
            logger.info(f"Using config Stage 1 threshold: {config.threshold['stage1']:.4f}")
            stage1_threshold = config.threshold["stage1"]
            
        if stage2_threshold is None and hasattr(config, "threshold") and "stage2" in config.threshold:
            logger.info(f"Using config Stage 2 threshold: {config.threshold['stage2']:.4f}")
            stage2_threshold = config.threshold["stage2"]
            
        if stage3_threshold is None and hasattr(config, "threshold") and "stage3" in config.threshold:
            logger.info(f"Using config Stage 3 threshold: {config.threshold['stage3']:.4f}")
            stage3_threshold = config.threshold["stage3"]
        
        # Log final thresholds for inference
        logger.info(f"Final Stage 1 threshold: {stage1_threshold:.4f}")
        if not args.stage1_only:
            logger.info(f"Final Stage 2 threshold: {stage2_threshold:.4f}")
            logger.info(f"Final Stage 3 threshold: {stage3_threshold:.4f}")
        
        # Run inference on the input dataset
        if args.stage1_only:
            logger.info(f"Processing dataset with Stage 1 model only: {args.input}")
            # Implement Stage 1 only processing here
            with open(os.path.join(config.log_dir, "stage1_only_mode.txt"), 'w') as f:
                f.write(f"Stage 1 only mode used for {args.input}\n")
                f.write(f"Threshold: {stage1_threshold}")
            
            logger.info("Stage 1 only processing not fully implemented - please use the full pipeline")
        else:
            logger.info(f"Processing dataset through cascade pipeline: {args.input}")
            
            # Run cascade pipeline
            results = process_large_dataset(
                args.input, 
                config,
                stage1_model, stage1_threshold,
                stage2_model, stage2_threshold,
                stage3_model, stage3_threshold,
                stage3_scaler,
                start_chunk=args.start_chunk 
            )
            
            logger.info("Processing complete")
            logger.info(f"Results saved to {config.data_dir}")
            logger.info(f"Final output: {results['final_pass']:,} molecules")

if __name__ == "__main__":
    main()