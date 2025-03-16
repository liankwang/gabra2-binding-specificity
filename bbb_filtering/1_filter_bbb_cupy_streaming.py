import os
import sys
import cupy as cp
import pandas as pd
import lz4.frame

# Filtering criteria
MW_min, MW_max = 180, 400
sLogP_min, sLogP_max = 1.8, 3.6
TPSA_min, TPSA_max = 20, 60
HBA_max = 6
HBD_max = 2
RotBonds_max = 6
Fsp3_min, Fsp3_max = 0.3, 0.6

# Critical columns (all must be present for complete rows)
critical_cols = ["MW", "sLogP", "TPSA", "FSP3", "HBA", "HBD", "RotBonds", "PPI_modulators"]

# Only use col. 0: smiles, 2: MW, 4: sLogP, 5: HBA, 6: HBD, 7: RotBonds, 8: FSP3, 9: TPSA, 15: PPI_modulators
usecols = [0, 2, 4, 5, 6, 7, 8, 9, 15]
col_names = ["smiles", "MW", "sLogP", "HBA", "HBD", "RotBonds", "FSP3", "TPSA", "PPI_modulators"]

# Filter molecules that meet all the BBB criteria and are NOT PPI modulators
def filter_complete(df):
    MW = cp.array(df["MW"])
    sLogP = cp.array(df["sLogP"])
    TPSA = cp.array(df["TPSA"])
    HBA = cp.array(df["HBA"].astype("int8"))
    HBD = cp.array(df["HBD"].astype("int8"))
    RotBonds = cp.array(df["RotBonds"].astype("int8"))
    FSP3 = cp.array(df["FSP3"])
    
    # Apply numeric filters
    mask = (
        (MW >= MW_min) & (MW <= MW_max) &
        (sLogP >= sLogP_min) & (sLogP <= sLogP_max) &
        (TPSA >= TPSA_min) & (TPSA <= TPSA_max) &
        (HBA <= HBA_max) & (HBD <= HBD_max) &
        (RotBonds <= RotBonds_max) &
        (FSP3 >= Fsp3_min) & (FSP3 <= Fsp3_max)
    )
    
    # Convert mask to NumPy
    numeric_mask_np = mask.get()
    
    # Apply PPI modulator filter (keeping only rows where PPI_modulators is empty or NULL)
    filtered_df = df[numeric_mask_np & df["PPI_modulators"].isnull()]

    return filtered_df["smiles"].to_numpy()

# Filter molecules that have exactly one missing value in the critical columns
def filter_incomplete(df):
    cond = True
    for col in critical_cols:
        if col != "PPI_modulators":  # PPI_modulators should not be missing
            cond &= (df[col].isnull()) | (df[col].between(MW_min, MW_max) if col == "MW" else True)
            cond &= (df[col].between(sLogP_min, sLogP_max) if col == "sLogP" else True)
            cond &= (df[col].between(TPSA_min, TPSA_max) if col == "TPSA" else True)
            cond &= (df[col].between(Fsp3_min, Fsp3_max) if col == "FSP3" else True)
            cond &= (df[col] <= HBA_max if col == "HBA" else True)
            cond &= (df[col] <= HBD_max if col == "HBD" else True)
            cond &= (df[col] <= RotBonds_max if col == "RotBonds" else True)

    missing_count = df[critical_cols].isnull().sum(axis=1)
    filtered_df = df[(missing_count == 1) & df["PPI_modulators"].isnull()]

    return filtered_df["smiles"].to_numpy()

# Read chunk, filter molecules, and return complete/incomplete sets
def filter_molecules_dual(chunk_file):
    print(f"DEBUG: Processing chunk file: {chunk_file}")
    
    try: # Some chunks have corrupted rows due to chunk splitting
        with open(chunk_file, 'r') as f:
            first_line = f.readline().strip()
        skip_header = 1 if first_line.lower().startswith('smiles') else 0
        print("DEBUG: " + ("Detected header row; skipping first row." if skip_header else "No header detected."))
        
        # Read the CSV file
        print("DEBUG: Reading CSV with optimized parameters...")
        df = pd.read_csv(
            chunk_file,
            sep="\t",
            header=None,
            skiprows=skip_header,
            usecols=usecols,
            names=col_names,
            dtype={
                "smiles": "string",
                "MW": "float32",
                "sLogP": "float32",
                "HBA": "float32",
                "HBD": "float32",
                "RotBonds": "float32",
                "FSP3": "float32",
                "TPSA": "float32",
                "PPI_modulators": "string"  # Read as string (to check for NULL/empty)
            },
            engine="c",
            on_bad_lines='skip'
        )
        print(f"DEBUG: Loaded DataFrame with {len(df)} rows.")

        # Count missing values
        missing_count = df[critical_cols].isnull().sum(axis=1)
        df_complete = df[missing_count == 0].copy()
        df_incomplete = df[missing_count == 1].copy()
        
        print(f"DEBUG: {len(df_complete)} complete rows, {len(df_incomplete)} incomplete rows (max 1 missing).")
        
        # Apply filtering
        complete_smiles = filter_complete(df_complete) if not df_complete.empty else None
        incomplete_smiles = filter_incomplete(df_incomplete) if not df_incomplete.empty else None

        return complete_smiles, incomplete_smiles

    except Exception as e:
        print(f"ERROR: Exception processing {chunk_file}: {e}")
        return None, None

# Process a chunk, apply filtering, and write results
def process_chunk(chunk_file, output_file):
    print(f"DEBUG: Processing chunk {chunk_file} -> {output_file}")
    
    try:
        # Remove old files if they exist
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"DEBUG: Removed previous output file: {output_file}")

        incomplete_output_file = os.path.splitext(output_file)[0] + "_incomplete.lz4"
        if os.path.exists(incomplete_output_file):
            os.remove(incomplete_output_file)
            print(f"DEBUG: Removed previous incomplete output file: {incomplete_output_file}")
            
        # Filter molecules
        complete_smiles, incomplete_smiles = filter_molecules_dual(chunk_file)

        if (complete_smiles is None or complete_smiles.size == 0) and (incomplete_smiles is None or incomplete_smiles.size == 0):
            print(f"DEBUG: No valid molecules in {chunk_file}")
            return

        # Write complete filtered molecules
        if complete_smiles is not None and complete_smiles.size > 0:
            print(f"DEBUG: Writing {complete_smiles.size} complete filtered molecules to {output_file}")
            with lz4.frame.open(output_file, "at", encoding="utf-8") as fout:
                for smi in complete_smiles:
                    fout.write(smi + "\n")

        print(f"DEBUG: Finished processing chunk {chunk_file}; removing file.")
        os.remove(chunk_file)

    except Exception as e:
        print(f"ERROR: Exception in processing chunk {chunk_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("python filter_bbb_cupy_streaming.py <input_chunk> <output_file>")
    input_chunk = sys.argv[1]
    output_file = sys.argv[2]
    print(f"DEBUG: Starting processing for {input_chunk}")
    process_chunk(input_chunk, output_file)
    print("DEBUG: Finished processing input chunk.")
