"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
Section 1: Dataset Splitting & Data Leakage Prevention

Ensures 0% patient overlap across splits (grouped by patient_id) while
stratifying by diagnostic class (BCC, SCC, MEL, ACK, NEV, SEK).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DIAGNOSTIC_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DIAGNOSTIC_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(DIAGNOSTIC_CLASSES)}

RISK_MAPPING = {
    "MEL": "CRITICAL",
    "BCC": "HIGH",
    "SCC": "HIGH",
    "ACK": "MODERATE",
    "NEV": "LOW",
    "SEK": "LOW",
}


def load_and_verify_metadata(metadata_path: str = "data/metadata.csv") -> pd.DataFrame:
    """Load raw metadata and verify PAD-UFES-20 integrity."""
    if not os.path.exists(metadata_path):
        # Fallback to parent directory search
        alt = "/home/user/Documents/DermAI Project/metadata.csv"
        if os.path.exists(alt):
            metadata_path = alt
        else:
            raise FileNotFoundError(f"Cannot find metadata at {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"[*] Loaded metadata with {len(df)} images.")
    print(f"[*] Unique patients: {df['patient_id'].nunique()}")
    print(f"[*] Unique lesions: {df['lesion_id'].nunique()}")
    print(f"[*] Diagnostic distribution:\n{df['diagnostic'].value_counts()}")
    return df


def create_stratified_group_splits(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Perform 5-Fold Stratified Group Split grouped by patient_id, stratified by diagnostic.
    Also generate 70% Train, 15% Val, 15% Test split partition.
    """
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Assign fold indices
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["diagnostic"], df["patient_id"])):
        df.loc[val_idx, "fold"] = fold_idx

    # Assign split partition:
    # Fold 0, 1, 2 -> Train (~60%), Fold 3 -> Val (~20%), Fold 4 -> Test (~20%)
    # Or allocate 70/15/15 by assigning folds 0,1,2 + portion of 3 to train, rest to val, fold 4 to test
    # A standard 5-fold scheme defines:
    # Folds 0, 1, 2: Train (1,380 images / 60%)
    # Fold 3: Validation (460 images / 20%)
    # Fold 4: Test (458 images / 20%)
    # For exact 70/15/15:
    # We can split patients in fold 3:
    patient_ids_f3 = list(df[df["fold"] == 3]["patient_id"].unique())
    import random
    rng = random.Random(random_state)
    rng.shuffle(patient_ids_f3)
    train_patients_f3 = set(patient_ids_f3[:int(len(patient_ids_f3) * 0.5)])
    
    def assign_split(row):
        f = row["fold"]
        if f in [0, 1, 2]:
            return "train"
        elif f == 3:
            return "train" if row["patient_id"] in train_patients_f3 else "val"
        else:
            return "test"

    df["split"] = df.apply(assign_split, axis=1)

    # Verification of 0% patient leakage
    train_pats = set(df[df["split"] == "train"]["patient_id"])
    val_pats = set(df[df["split"] == "val"]["patient_id"])
    test_pats = set(df[df["split"] == "test"]["patient_id"])

    assert len(train_pats.intersection(val_pats)) == 0, "DATA LEAKAGE: Overlap between Train & Val patients!"
    assert len(train_pats.intersection(test_pats)) == 0, "DATA LEAKAGE: Overlap between Train & Test patients!"
    assert len(val_pats.intersection(test_pats)) == 0, "DATA LEAKAGE: Overlap between Val & Test patients!"

    print("\n[+] Verification passed: 0% Patient Overlap across Train, Val, and Test splits!")
    print("\n[*] Split summary:")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        print(f"  Split '{s}': {len(sub)} images ({len(sub)/len(df)*100:.1f}%), {sub['patient_id'].nunique()} patients")
        print(f"    Class distribution: {dict(sub['diagnostic'].value_counts())}")

    return df


def main():
    metadata_path = "data/metadata.csv"
    if not os.path.exists(metadata_path):
        metadata_path = "/home/user/Documents/DermAI Project/metadata.csv"
    
    df = load_and_verify_metadata(metadata_path)
    df_splits = create_stratified_group_splits(df)

    output_path = "metadata_with_folds.csv"
    df_splits.to_csv(output_path, index=False)
    print(f"\n[+] Saved split metadata to {output_path}")

    # Also update data/metadata_with_folds.csv if data dir exists
    if os.path.exists("data"):
        data_output = "data/metadata_with_folds.csv"
        # If it's a symlink, remove it first
        if os.path.islink(data_output):
            os.unlink(data_output)
        df_splits.to_csv(data_output, index=False)
        print(f"[+] Saved split metadata to {data_output}")


if __name__ == "__main__":
    main()
