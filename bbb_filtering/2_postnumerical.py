#!/usr/bin/env python3
import pandas as pd
import cupy as cp
import numpy as np
import os
import sys
import time
import logging
import gc
import re
import traceback
import multiprocessing as mp
from numba import cuda
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pyarrow as pa
import pyarrow.parquet as pq
import lz4.frame
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import signal
import math
from tqdm import tqdm
import io
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(processName)s] %(levelname)s: %(message)s',
    datefmt='%F %T'
)

# Configuration
TARGET_KEEP_RATIO = 0.15  # Goal to keep ~15% of molecules

# Batch processing configuration
BATCH_SIZE = 500000  # Process 250K molecules per batch
MAX_WORKERS = max(1, min(os.cpu_count() - 2, 24))
ROBUST_MODE = True 

# Filtering criteria from last run
MW_min, MW_max = 180, 400
sLogP_min, sLogP_max = 1.8, 3.6
TPSA_min, TPSA_max = 20, 60
HBA_max = 6
HBD_max = 2
RotBonds_max = 6
Fsp3_min, Fsp3_max = 0.3, 0.6

# New filtering criteria
QED_min = 0.45  # Minimum druglikeness score
TPSA_sLogP_ratio_max = 20  # Maximum ratio of polar surface area to lipophilicity

# Downsampling rates for different categories
DOWNSAMPLE_RATES = {
    'unwanted_functional_groups': 0.05,  # Keep 5% of molecules with unwanted functional groups
    'highly_conjugated': 0.10,           # Keep 10% of molecules with >3 aromatic rings
    'high_flexibility_low_mw': 0.03,     # Keep 3% of molecules with high flex/low MW ratio
    'poor_tpsa_logp_ratio': 0.02,        # Keep 2% of molecules with poor TPSA/logP ratio
    'low_qed': 0.02                      # Keep 2% of molecules with poor drug-likeness
}

# Columns to keep in the parquet file (only SMILES and numeric properties)
PARQUET_COLUMNS = ['smiles', 'mw', 'slogp', 'hba', 'hbd', 'rotbonds', 'fsp3', 'tpsa', 'qed']

# Expected columns
EXPECTED_COLS = [
    'smiles', 'id', 'MW', 'HAC', 'sLogP', 'HBA', 'HBD', 'RotBonds',
    'FSP3', 'TPSA', 'QED', 'lead-like', '350/3_lead-like', 'fragments',
    'strict_fragments', 'natural_product-like', 'Type', 'InChiKey'
]
EXPECTED_COLS_LOWER = [col.lower() for col in EXPECTED_COLS]

OPTIMIZE_FOR_SPEED = True   
MAX_RUNTIME_HOURS = 3.0    
USE_PATTERN_CUTOFF = True   

# Global stats counters
total_processed = mp.Value('i', 0)
total_kept = mp.Value('i', 0)
total_errors = mp.Value('i', 0)

# Pre-compiled SMARTS patterns for performance
_UNWANTED_PATTERNS = None

# Utility Functions
# Initialize SMARTS patterns for unwanted functional groups
def initialize_smarts_patterns():
    global _UNWANTED_PATTERNS
    
    if _UNWANTED_PATTERNS is not None:
        return _UNWANTED_PATTERNS
        
    # Define SMARTS patterns categorized by severity
    pattern_groups = {
        'high_severity': [  # These are most problematic for BBB
            '[NX4+]',  # Quaternary ammonium
            '[CX3]=[NX3+]=[NX3]',  # Guanidinium
            '[SX3+]',  # Sulfonium
            '[PX4](=O)([OX2H1,OX1H0])[OX2H1,OX1H0]',  # Phosphate
            '[SX4](=O)(=O)[OX2H1,OX1H0]',  # Sulfate/Sulfonic acid
        ],
        'medium_severity': [  # These are moderately problematic
            '[CX3](=O)[OX1H1,OX2H0]',  # Carboxyl
            '[C]1[O]1',  # Epoxide
            '[C]1[N]1',  # Aziridine 
        ],
        'low_severity': [  # These are less problematic
            '[CX3H1](=O)',  # Aldehyde
            '[CX3](=O)[NX3]',  # Amide
        ]
    }
    
    # Compile all patterns once
    _UNWANTED_PATTERNS = {
        severity: [Chem.MolFromSmarts(pattern) for pattern in patterns]
        for severity, patterns in pattern_groups.items()
    }
    
    return _UNWANTED_PATTERNS

# Fast heuristic check for problematic substructures
# Output: severity estimate (0-3) or None if needs deeper check
def quick_substructure_check(smiles):
    # Check for obviously problematic groups
    if '+' in smiles:  # Charged groups
        return 3
    
    # Check for carboxylic acids and esters
    if 'C(=O)O' in smiles or 'COOH' in smiles:
        return 2
    
    # Check for phosphates
    if 'OP(=O)' in smiles:
        return 3
        
    # Check for sulfates/sulfonates
    if 'OS(=O)' in smiles or 'S(=O)(=O)O' in smiles:
        return 3

    return None

# Check for unwanted functional groups with severity levels
# Output: 0 (no problems), 1 (low severity), 2 (medium severity), 3 (high severity)
def check_unwanted_functional_groups(smiles):
    # First try the fast heuristic check
    if OPTIMIZE_FOR_SPEED:
        quick_result = quick_substructure_check(smiles)
        if quick_result is not None:
            return quick_result
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 3  # Highest severity if can't parse
            
        # Initialize patterns if needed
        patterns = initialize_smarts_patterns()
        
        # Check high severity patterns first (fail fast)
        for pattern in patterns['high_severity']:
            if mol.HasSubstructMatch(pattern):
                return 3
                
        # Then medium severity
        for pattern in patterns['medium_severity']:
            if mol.HasSubstructMatch(pattern):
                return 2
                
        # Then low severity
        for pattern in patterns['low_severity']:
            if mol.HasSubstructMatch(pattern):
                return 1
                
        return 0  # No unwanted groups found
    except:
        return 3  # If exception occurs, highest severity

# Ring checking function (CPU-based using RDKit)
def has_ring(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return mol.GetRingInfo().NumRings() > 0
    except:
        return False

# Function to count aromatic rings (for conjugated systems filter)
def count_aromatic_rings(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
            
        # Get ring information
        ri = mol.GetRingInfo()
        
        # Count aromatic rings
        aromatic_rings = 0
        for ring in ri.AtomRings():
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic_rings += 1
                
        return aromatic_rings
    except:
        return 0

# Parallel processing of SMILES for structural features
def process_smiles_batch(smiles_list, features_to_check):
    results = {feature: [] for feature in features_to_check}
    
    for smiles in smiles_list:
        try:
            if 'has_ring' in features_to_check:
                results['has_ring'].append(has_ring(smiles))
            if 'unwanted_groups' in features_to_check:
                results['unwanted_groups'].append(check_unwanted_functional_groups(smiles))
            if 'aromatic_rings' in features_to_check:
                results['aromatic_rings'].append(count_aromatic_rings(smiles))
        except Exception as e:
            # Add default values for error cases
            for feature in features_to_check:
                if feature == 'has_ring':
                    results[feature].append(False)
                elif feature == 'unwanted_groups':
                    results[feature].append(3)  # Highest severity
                elif feature == 'aromatic_rings':
                    results[feature].append(0)
    
    return results

# GPU Processing Functions
# CUDA kernel for primary filtering
@cuda.jit
def filter_kernel(MW, sLogP, TPSA, HBA, HBD, RotBonds, FSP3, QED, fragments_flag, 
                 is_fragment, is_strict_fragment, mask, TPSA_sLogP_ratio, category_flags,
                 MW_min, MW_max, sLogP_min, sLogP_max, TPSA_min, TPSA_max,
                 HBA_max, HBD_max, RotBonds_max, Fsp3_min, Fsp3_max, QED_min,
                 TPSA_sLogP_ratio_max):
    i = cuda.grid(1)
    if i < MW.size:
        # Initialize category flags (for stratified sampling)
        # 0 = perfect, 1 = high_flex_low_mw, 2 = poor_tpsa_logp, 3 = low_qed, 4 = fragment, 15 = multiple issues
        category_flags[i] = 0
        
        # Reject molecules marked as fragments or strict_fragments
        if is_fragment[i] == 1 or is_strict_fragment[i] == 1:
            category_flags[i] = 4
            mask[i] = 0
            return
            
        # Apply primary physicochemical filters (double check b/c of GPU precision)
        if not (MW[i] >= MW_min and MW[i] <= MW_max and
                sLogP[i] >= sLogP_min and sLogP[i] <= sLogP_max and
                TPSA[i] >= TPSA_min and TPSA[i] <= TPSA_max and
                HBA[i] <= HBA_max and
                HBD[i] <= HBD_max and
                RotBonds[i] <= RotBonds_max and
                FSP3[i] >= Fsp3_min and FSP3[i] <= Fsp3_max):
            mask[i] = 0
            return
            
        # Track issues for stratified sampling rather than immediate rejection
        issue_count = 0
        
        # 1. QED threshold check
        if QED[i] < QED_min:
            category_flags[i] = 3  # low_qed
            issue_count += 1
        
        # 2. Check flexibility to MW ratio
        if RotBonds[i] > 5 and MW[i] < 350:
            # If already has an issue, mark as multiple issues
            if issue_count > 0:
                category_flags[i] = 15
            else:
                category_flags[i] = 1  # high_flex_low_mw
            issue_count += 1
            
        # 3. Check TPSA to sLogP ratio
        if TPSA_sLogP_ratio[i] > TPSA_sLogP_ratio_max:
            # If already has an issue, mark as multiple issues
            if issue_count > 0:
                category_flags[i] = 15
            else:
                category_flags[i] = 2  # poor_tpsa_logp
            issue_count += 1
            
        # For GPU pass, keep everything that meets primary criteria for stratified sampling in CPU later
        mask[i] = 1

# Function to prepare GPU memory by clearing it
def prepare_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    mem_pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(mem_pool.malloc)

# Batch Processing Functions
# Function to validate only the numeric columns using vectorized conversion, 
    # drop rows with conversion errors, and batch log error messages.
def clean_chunk(chunk, numeric_cols, dropped_log_filename):
    # Convert numeric columns in one go
    chunk[numeric_cols] = chunk[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Identify rows with conversion errors
    invalid_rows = chunk[numeric_cols].isna().any(axis=1)
    if invalid_rows.any():
        error_messages = [
            f"Row {idx}: Conversion error in one or more numeric columns."
            for idx in chunk[invalid_rows].index
        ]
        try:
            with open(dropped_log_filename, "a") as log_file:
                log_file.write("\n".join(error_messages) + "\n")
        except Exception as e:
            logging.error(f"Error writing dropped rows log to {dropped_log_filename}: {e}")
    
    return chunk[~invalid_rows]

# Function to filter a batch of molecules using GPU
def filter_molecules_batch(df, current_keep_ratio=None):
    # Fix for the TARGET_KEEP_RATIO reference issue
    if current_keep_ratio is None:
        current_keep_ratio = TARGET_KEEP_RATIO
        
    batch_start_time = time.time()
    batch_size = len(df)
    
    with total_processed.get_lock():
        total_processed.value += batch_size
    
    # 1. Parse fragments and strict_fragments flags
    fragment_filter = df["fragments"].fillna("").str.upper() == "TRUE"
    strict_fragment_filter = df["strict_fragments"].fillna("").str.upper() == "TRUE"
    is_fragment_np = fragment_filter.astype(np.int8).to_numpy()
    is_strict_fragment_np = strict_fragment_filter.astype(np.int8).to_numpy()
    
    # Prepare data for GPU
    try:
        MW_np = pd.to_numeric(df['mw'], errors='coerce').fillna(0).to_numpy()
        sLogP_np = pd.to_numeric(df['slogp'], errors='coerce').fillna(0).to_numpy()
        TPSA_np = pd.to_numeric(df['tpsa'], errors='coerce').fillna(0).to_numpy()
        HBA_np = pd.to_numeric(df['hba'], errors='coerce').fillna(0).to_numpy()
        HBD_np = pd.to_numeric(df['hbd'], errors='coerce').fillna(0).to_numpy()
        RotBonds_np = pd.to_numeric(df['rotbonds'], errors='coerce').fillna(0).to_numpy()
        FSP3_np = pd.to_numeric(df['fsp3'], errors='coerce').fillna(0).to_numpy()
        QED_np = pd.to_numeric(df['qed'], errors='coerce').fillna(0).to_numpy()
        
        # Calculate TPSA/sLogP ratio (with safety for division by zero)
        TPSA_sLogP_ratio_np = np.zeros_like(TPSA_np)
        mask = sLogP_np > 0
        TPSA_sLogP_ratio_np[mask] = TPSA_np[mask] / sLogP_np[mask]
        TPSA_sLogP_ratio_np[~mask] = 100  # Assign high value to flag for rejection
        
    except Exception as e:
        logging.error(f"Error converting filtering columns: {e}")
        return pd.DataFrame()

    # Transfer data to GPU
    prepare_gpu_memory()
    d_MW = cp.asarray(MW_np)
    d_sLogP = cp.asarray(sLogP_np)
    d_TPSA = cp.asarray(TPSA_np)
    d_HBA = cp.asarray(HBA_np)
    d_HBD = cp.asarray(HBD_np)
    d_RotBonds = cp.asarray(RotBonds_np)
    d_FSP3 = cp.asarray(FSP3_np)
    d_QED = cp.asarray(QED_np)
    d_fragments_flag = cp.asarray(np.zeros_like(MW_np, dtype=np.int8))
    d_is_fragment = cp.asarray(is_fragment_np)
    d_is_strict_fragment = cp.asarray(is_strict_fragment_np)
    d_TPSA_sLogP_ratio = cp.asarray(TPSA_sLogP_ratio_np)

    # Allocate output mask and category flags on GPU
    d_mask = cp.empty(d_MW.shape, dtype=cp.int8)
    d_category_flags = cp.zeros(d_MW.shape, dtype=cp.int8)
    
    # Launch kernel
    threadsperblock = 256
    blockspergrid = (d_MW.size + threadsperblock - 1) // threadsperblock
    
    filter_kernel[blockspergrid, threadsperblock](
        d_MW, d_sLogP, d_TPSA, d_HBA, d_HBD, d_RotBonds, d_FSP3, d_QED, d_fragments_flag,
        d_is_fragment, d_is_strict_fragment, d_mask, d_TPSA_sLogP_ratio, d_category_flags,
        MW_min, MW_max, sLogP_min, sLogP_max, TPSA_min, TPSA_max,
        HBA_max, HBD_max, RotBonds_max, Fsp3_min, Fsp3_max, QED_min,
        TPSA_sLogP_ratio_max
    )
    
    # Get results back from GPU
    mask = cp.asnumpy(d_mask) > 0
    category_flags = cp.asnumpy(d_category_flags)
    
    # Clean up GPU memory after use
    del d_MW, d_sLogP, d_TPSA, d_HBA, d_HBD, d_RotBonds, d_FSP3, d_QED
    del d_fragments_flag, d_is_fragment, d_is_strict_fragment, d_mask
    del d_TPSA_sLogP_ratio, d_category_flags
    prepare_gpu_memory()
    
    # Use GPU classification for stratified sampling
    df['category'] = category_flags
    
    # Keep all molecules for initial check
    df_filtered = df.copy()
    gpu_time = time.time() - batch_start_time
    logging.debug(f"GPU processing completed in {gpu_time:.2f}s")
    
    # Group molecules by category for smart sampling
    perfect = df_filtered[df_filtered['category'] == 0]
    high_flex_low_mw = df_filtered[df_filtered['category'] == 1]
    poor_tpsa_logp = df_filtered[df_filtered['category'] == 2]
    low_qed = df_filtered[df_filtered['category'] == 3]
    fragments = df_filtered[df_filtered['category'] == 4]
    multiple_issues = df_filtered[df_filtered['category'] == 15]
    
    # Get filtered set for additional CPU-based processing
    # Exclude fragments completely
    df_filtered = df_filtered[df_filtered['category'] != 4].copy()
    logging.debug(f"After removing fragments, {len(df_filtered)} molecules remain.")
    
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Apply CPU-based filters to the reduced set
    cpu_start_time = time.time()
    
    # 1. Check for at least one ring (required)
    logging.debug("Checking for ring structures...")
    
    # Use multiprocessing for SMILES checking
    smiles_batch_size = min(10000, len(df_filtered))
    smiles_batches = [df_filtered['smiles'].iloc[i:i+smiles_batch_size].tolist() 
                     for i in range(0, len(df_filtered), smiles_batch_size)]
    
    features_to_check = ['has_ring']
    
    # Use a smaller thread pool for SMILES processing to avoid memory issues
    with ThreadPoolExecutor(max_workers=min(8, MAX_WORKERS)) as executor:
        results = list(executor.map(
            lambda batch: process_smiles_batch(batch, features_to_check),
            smiles_batches
        ))
    
    # Combine results from all batches
    has_ring_results = []
    for result in results:
        has_ring_results.extend(result['has_ring'])
    
    # Apply ring filter
    has_ring_mask = pd.Series(has_ring_results, index=df_filtered.index)
    df_filtered = df_filtered[has_ring_mask]
    logging.debug(f"After ring filter, {len(df_filtered)} molecules remain.")
    
    if df_filtered.empty:
        return pd.DataFrame()
        
    # 2. Stratified sampling based on molecule categories
    # Calculate how many to keep based on target ratio
    target_keep_count = int(batch_size * current_keep_ratio)
    
    # Initialize result parts list
    result_parts = []
    
    # Adjust keep count to maintain overall ratio
    with total_kept.get_lock():
        current_overall_ratio = total_kept.value / max(total_processed.value, 1)
        ratio_factor = TARGET_KEEP_RATIO / max(current_overall_ratio, 0.001)
        target_keep_count = int(target_keep_count * ratio_factor)
        target_keep_count = max(1, min(target_keep_count, len(df_filtered)))
    
    # 1. Add molecules with rings and good properties
    perfect_count = len(perfect)
    if perfect_count <= target_keep_count:
        # Keep all perfect molecules
        result_parts.append(perfect)
        target_keep_count -= perfect_count
    else:
        # Sample from perfect molecules
        if 'qed' in perfect.columns:
            weights = perfect['qed'].fillna(0.5)
            sampled = perfect.sample(n=target_keep_count, weights=weights)
        else:
            sampled = perfect.sample(n=target_keep_count)
        result_parts.append(sampled)
        target_keep_count = 0
    
    # 2. If more molecules needed, sample from the other categories
    if target_keep_count > 0:
        # Sample from high_flex_low_mw
        flex_count = len(high_flex_low_mw)
        if flex_count > 0:
            sample_count = min(flex_count, int(target_keep_count * DOWNSAMPLE_RATES['high_flexibility_low_mw']))
            if sample_count > 0:
                sampled = high_flex_low_mw.sample(n=sample_count)
                result_parts.append(sampled)
                target_keep_count -= sample_count
        
        # Sample from poor_tpsa_logp
        tpsa_count = len(poor_tpsa_logp)
        if tpsa_count > 0 and target_keep_count > 0:
            sample_count = min(tpsa_count, int(target_keep_count * DOWNSAMPLE_RATES['poor_tpsa_logp_ratio']))
            if sample_count > 0:
                sampled = poor_tpsa_logp.sample(n=sample_count)
                result_parts.append(sampled)
                target_keep_count -= sample_count
        
        # Sample from low_qed
        qed_count = len(low_qed)
        if qed_count > 0 and target_keep_count > 0:
            sample_count = min(qed_count, int(target_keep_count * DOWNSAMPLE_RATES['low_qed']))
            if sample_count > 0:
                sampled = low_qed.sample(n=sample_count)
                result_parts.append(sampled)
                target_keep_count -= sample_count
        
        # Sample from multiple_issues if we still need more
        multi_count = len(multiple_issues)
        if multi_count > 0 and target_keep_count > 0:
            sample_count = min(multi_count, int(target_keep_count * 0.01))  # Very low rate
            if sample_count > 0:
                sampled = multiple_issues.sample(n=sample_count)
                result_parts.append(sampled)
    
    # Combine all result parts
    result = pd.concat(result_parts) if result_parts else pd.DataFrame(columns=df.columns)
    
    # Update total kept counter
    with total_kept.get_lock():
        total_kept.value += len(result)
    
    cpu_time = time.time() - cpu_start_time
    total_time = time.time() - batch_start_time
    
    logging.debug(f"CPU processing: {cpu_time:.2f}s, Total batch time: {total_time:.2f}s")
    logging.debug(f"Kept {len(result)} molecules from batch of {batch_size} ({len(result)/batch_size*100:.2f}%)")
    
    return result

# Batch processing function for a chunk of molecules and saving results
def process_batch(batch_idx, df_chunk, output_dir):
    if df_chunk is None or df_chunk.empty:
        logging.warning(f"Batch {batch_idx}: Empty chunk, skipping")
        return 0
        
    try:
        logging.info(f"Batch {batch_idx}: Processing {len(df_chunk)} molecules")
        
        # Handle numeric columns
        numeric_cols = ['mw', 'slogp', 'tpsa', 'hba', 'hbd', 'rotbonds', 'fsp3', 'qed']
        dropped_log_filename = os.path.join(output_dir, f"dropped_batch_{batch_idx}.log")
        
        # Clean numeric data
        df_cleaned = clean_chunk(df_chunk, numeric_cols, dropped_log_filename)
        if df_cleaned.empty:
            logging.warning(f"Batch {batch_idx}: No valid molecules after cleaning")
            return 0
            
        # Apply filters
        with total_processed.get_lock(), total_kept.get_lock():
            current_ratio = total_kept.value / max(total_processed.value, 1)
            current_keep_ratio = TARGET_KEEP_RATIO
            
            # Adjust ratio if falling behind or ahead
            if total_processed.value > 1000000:  # Only adjust after processing enough data
                if current_ratio < TARGET_KEEP_RATIO:
                    # Keeping too few, increase ratio
                    current_keep_ratio = min(1.0, TARGET_KEEP_RATIO * 1.2)
                elif current_ratio > TARGET_KEEP_RATIO:
                    # Keeping too many, decrease ratio
                    current_keep_ratio = max(0.01, TARGET_KEEP_RATIO * 0.8)
        
        # Apply filtering
        filtered_df = filter_molecules_batch(df_cleaned, current_keep_ratio)
        
        if filtered_df.empty:
            logging.info(f"Batch {batch_idx}: No molecules passed filtering")
            return 0
            
        # Save filtered molecules
        batch_csv_lz4 = os.path.join(output_dir, f"batch_{batch_idx}.csv.lz4")
        batch_parquet = os.path.join(output_dir, f"batch_{batch_idx}.parquet")
        
        # Save full data to CSV.LZ4
        with lz4.frame.open(batch_csv_lz4, 'wb') as f:
            csv_bytes = filtered_df.to_csv(
                index=False,
                sep="\t",
                header=True
            ).encode('utf-8')
            f.write(csv_bytes)
            
        # Save subset of columns to parquet with snappy compression
        parquet_df = filtered_df[PARQUET_COLUMNS].copy()
        pq.write_table(
            pa.Table.from_pandas(parquet_df),
            batch_parquet,
            compression='snappy'
        )
        
        logging.info(f"Batch {batch_idx}: Kept {len(filtered_df)} of {len(df_chunk)} molecules")
        logging.info(f"Batch {batch_idx}: Written to {batch_csv_lz4} and {batch_parquet}")
        
        return len(filtered_df)
        
    except Exception as e:
        logging.error(f"Batch {batch_idx}: Error processing batch: {e}")
        logging.error(traceback.format_exc())
        with total_errors.get_lock():
            total_errors.value += 1
        return 0

# Function to read a chunk of cxsmiles data robustly, handling malformed rows
def robust_read_cxsmiles_chunk(file_handle, chunksize, separator="\t", header_line=None):
    lines = []
    corrupt_lines = []
    line_count = 0
    expected_fields = len(EXPECTED_COLS)
    
    # Buffer for reading
    chunk_buffer = io.StringIO()
    current_size = 0
    
    # Read lines until have enough or end of file
    while current_size < chunksize:
        line = file_handle.readline()
        if not line:  
            break
            
        # Check if the line appears valid
        fields = line.count(separator) + 1
        
        if fields == expected_fields:
            # Line appears valid
            chunk_buffer.write(line)
            current_size += 1
        else:
            # Line appears corrupt
            corrupt_line_info = f"Line with {fields} fields instead of {expected_fields}: {line[:100]}..."
            corrupt_lines.append(corrupt_line_info)
            with total_errors.get_lock():
                total_errors.value += 1
    
    if current_size == 0:
        # No valid lines
        return None, corrupt_lines
    
    # Reset buffer to beginning
    chunk_buffer.seek(0)
    
    # Read into pandas
    try:
        if header_line is not None:
            # Prepend header line
            full_buffer = io.StringIO()
            full_buffer.write(header_line)
            full_buffer.write(chunk_buffer.getvalue())
            full_buffer.seek(0)
            df = pd.read_csv(full_buffer, sep=separator, header=0, 
                             names=EXPECTED_COLS_LOWER, low_memory=False)
        else:
            # No header needed
            df = pd.read_csv(chunk_buffer, sep=separator, header=None, 
                             names=EXPECTED_COLS_LOWER, low_memory=False)
    except Exception as e:
        logging.error(f"Error parsing chunk: {e}")
        logging.error(traceback.format_exc())
        return None, corrupt_lines
        
    return df, corrupt_lines

# Function to wait for at least one future to complete, then return the completed ones
def wait_for_first_completion(futures):
    # Initialize sets
    done = set()
    pending = set(futures)
    
    # Wait for at least one task to complete (timeout to avoid infinite wait)
    wait_time = 0.1
    while not done and pending:
        done_now, pending = wait(
            pending, timeout=wait_time, 
            return_when='FIRST_COMPLETED'
        )
        done.update(done_now)
        
        # Increase wait time slowly if nothing completes
        if not done:
            wait_time = min(wait_time * 1.2, 5.0)
    
    # Return the done and pending futures as lists
    return list(done), list(pending)

# Function to merge batch outputs into single files
def merge_outputs(output_dir):
    logging.info("Merging batch outputs")
    
    # Create merged directory
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Merge CSV.LZ4 files
    csv_files = sorted(glob.glob(os.path.join(output_dir, "batch_*.csv.lz4")))
    if csv_files:
        merged_csv = os.path.join(merged_dir, "filtered_molecules.csv.lz4")
        logging.info(f"Merging {len(csv_files)} CSV.LZ4 files into {merged_csv}")
        
        with lz4.frame.open(merged_csv, 'wb') as out_file:
            # Write header from first file
            with lz4.frame.open(csv_files[0], 'rb') as first_file:
                header = first_file.readline()
                out_file.write(header)
            
            # Write data from all files (skipping headers)
            for i, file_path in enumerate(csv_files):
                logging.info(f"Adding file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
                with lz4.frame.open(file_path, 'rb') as in_file:
                    # Skip header
                    in_file.readline()
                    # Copy rest of file
                    while True:
                        chunk = in_file.read(1024*1024)  # 1MB chunks
                        if not chunk:
                            break
                        out_file.write(chunk)
        
    # Merge Parquet files
    parquet_files = sorted(glob.glob(os.path.join(output_dir, "batch_*.parquet")))
    if parquet_files:
        merged_parquet = os.path.join(merged_dir, "filtered_molecules.parquet")
        logging.info(f"Merging {len(parquet_files)} Parquet files into {merged_parquet}")
        
        # Use pyarrow to merge parquet files
        tables = []
        for file_path in parquet_files:
            try:
                table = pq.read_table(file_path)
                tables.append(table)
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
        
        if tables:
            merged_table = pa.concat_tables(tables)
            pq.write_table(merged_table, merged_parquet, compression='snappy')
            logging.info(f"Merged {len(tables)} tables with {len(merged_table)} total rows")
        else:
            logging.warning("No valid parquet tables to merge")
    
    logging.info("Merging complete")

# Main file processing function
def process_file(input_file, output_dir, batch_size=BATCH_SIZE):
    global total_processed, total_kept, total_errors
    
    start_time = time.time()
    logging.info(f"Starting processing of {input_file}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for error logs
    error_dir = os.path.join(output_dir, "errors")
    os.makedirs(error_dir, exist_ok=True)
    error_log = os.path.join(error_dir, "corrupt_lines.log")
    
    # Get base info of the file
    file_size = os.path.getsize(input_file)
    estimated_molecules = int(file_size / 500)  # Rough estimate
    logging.info(f"File size: {file_size/1024/1024:.2f} MB, estimated molecules: {estimated_molecules:,}")
    
    # Reset counters
    with total_processed.get_lock():
        total_processed.value = 0
    with total_kept.get_lock():
        total_kept.value = 0
    with total_errors.get_lock():
        total_errors.value = 0
    
    # Process the file in batches
    try:
        # Determine if file is compressed
        if input_file.endswith('.lz4'):
            open_func = lz4.frame.open
        else:
            open_func = open
        
        batch_idx = 0
        with open_func(input_file, 'rt') as f:
            # Read the header line
            header_line = f.readline().strip()
            header_fields = header_line.split("\t")
            
            if len(header_fields) != len(EXPECTED_COLS):
                logging.warning(f"Header has {len(header_fields)} fields, expected {len(EXPECTED_COLS)}")
                logging.warning(f"Header: {header_line[:100]}...")
                
                # Infer if this is actually data and not a header
                if len(header_fields) > 3 and header_fields[0].startswith('C'):
                    logging.warning("First line appears to be data, not header. Using expected column names.")
                    # Reset file pointer
                    f.seek(0)
                    header_line = "\t".join(EXPECTED_COLS)
                
            # Create a process pool for batch processing
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                batch_futures = []
                
                # Process file in chunks
                while True:
                    logging.info(f"Reading batch {batch_idx}")
                    df_chunk, corrupt_lines = robust_read_cxsmiles_chunk(f, batch_size, header_line=None)
                    
                    # Log corrupt lines
                    if corrupt_lines:
                        with open(error_log, 'a') as error_file:
                            error_file.write(f"Batch {batch_idx} corrupt lines:\n")
                            for line in corrupt_lines:
                                error_file.write(f"{line}\n")
                        logging.warning(f"Batch {batch_idx}: Found {len(corrupt_lines)} corrupt lines")
                    
                    if df_chunk is None or df_chunk.empty:
                        if not corrupt_lines:
                            # EOF with no corrupt lines
                            break
                        else:
                            # Had corrupt lines, but no valid data - continue to next batch
                            batch_idx += 1
                            continue
                    
                    # Submit batch for processing
                    future = executor.submit(process_batch, batch_idx, df_chunk, output_dir)
                    batch_futures.append(future)
                    batch_idx += 1
                    
                    # If too many pending batches, wait for some to complete
                    if len(batch_futures) >= MAX_WORKERS * 2:
                        # Wait for at least one batch to complete
                        done, batch_futures = wait_for_first_completion(batch_futures)
                        for future in done:
                            try:
                                kept_count = future.result()
                                logging.info(f"Batch completed, kept {kept_count} molecules")
                            except Exception as e:
                                logging.error(f"Batch processing error: {e}")
                                logging.error(traceback.format_exc())
                    
                    # Print progress update
                    with total_processed.get_lock(), total_kept.get_lock(), total_errors.get_lock():
                        elapsed_time = time.time() - start_time
                        rate = total_processed.value / max(elapsed_time, 1)
                        logging.info(f"Progress: {total_processed.value:,} processed, "
                                   f"{total_kept.value:,} kept ({total_kept.value/max(total_processed.value,1)*100:.2f}%), "
                                   f"{total_errors.value} errors, "
                                   f"Rate: {rate:.2f} molecules/second")
                
                # Wait for all remaining batches to complete
                for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc="Finishing batches"):
                    try:
                        kept_count = future.result()
                        logging.info(f"Batch completed, kept {kept_count} molecules")
                    except Exception as e:
                        logging.error(f"Batch processing error: {e}")
                        logging.error(traceback.format_exc())
        
        # Create merged outputs
        merge_outputs(output_dir)
        
        # Final report
        with total_processed.get_lock(), total_kept.get_lock(), total_errors.get_lock():
            end_time = time.time()
            elapsed_time = end_time - start_time
            rate = total_processed.value / max(elapsed_time, 1)
            
            logging.info(f"Processing complete: {elapsed_time:.2f} seconds")
            logging.info(f"Total molecules processed: {total_processed.value:,}")
            logging.info(f"Total molecules kept: {total_kept.value:,} ({total_kept.value/max(total_processed.value,1)*100:.2f}%)")
            logging.info(f"Total processing errors: {total_errors.value}")
            logging.info(f"Processing rate: {rate:.2f} molecules/second")
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        logging.error(traceback.format_exc())
        return False

# Main Function
def main():
    if len(sys.argv) != 3:
        print("python postnumerical.py <input_file> <output_dir>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Initialize CUDA
    try:
        prepare_gpu_memory()
        logging.info("CUDA initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing CUDA: {e}")
        sys.exit(1)
    
    # Process the file
    success = process_file(input_file, output_dir)
    
    if success:
        logging.info("Processing completed successfully")
        sys.exit(0)
    else:
        logging.error("Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()