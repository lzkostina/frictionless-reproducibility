"""
Create a safe reduced artifact set for reproducibility.

This script:
1. Loads processed Version B data (requires raw data access).
2. Saves demographic/behavioral features and subject lists to artifacts/.
3. Computes safe network summaries (no raw connectomes).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.pipeline.connectomes import merge_features_and_connectomes, enforce_same_subjects
from src.pipeline.preprocess import preprocess_features
from src.pipeline.clean import (
    handle_missing_values,
    merge_family_ids,
    enforce_common_subjects,
    drop_siblings,
    link_with_g_scores,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed/version_B")
ARTIFACTS_DIR = Path("artifacts/version_B")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_network_summaries(connectomes, subjects):
    """
    Compute safe summary stats from connectomes.
    Right now: only global mean connectivity.
    Extend with PCA or system-level metrics if needed.
    """
    summaries = []
    for subj_id, conn in zip(subjects, connectomes):
        stats = {
            "Subject": subj_id,
            "global_strength": float(np.mean(conn)),
        }
        summaries.append(stats)
    return pd.DataFrame(summaries)


def main():
    print("=== Creating artifacts from processed Version B data ===")

    # -------------------------------------------------------------------
    # Load processed features
    # -------------------------------------------------------------------
    features_baseline = pd.read_csv(PROCESSED_DIR / "features_baseline.csv")
    features_followup = pd.read_csv(PROCESSED_DIR / "features_followup.csv")

    features_baseline.to_csv(ARTIFACTS_DIR / "features_baseline.csv", index=False)
    features_followup.to_csv(ARTIFACTS_DIR / "features_followup.csv", index=False)

    # -------------------------------------------------------------------
    # Load subject IDs
    # -------------------------------------------------------------------
    subs_baseline = pd.read_csv(PROCESSED_DIR / "subjects_baseline.csv")["Subject"].tolist()
    subs_followup = pd.read_csv(PROCESSED_DIR / "subjects_followup.csv")["Subject"].tolist()

    pd.DataFrame({"Subject": subs_baseline}).to_csv(ARTIFACTS_DIR / "subjects_baseline.csv", index=False)
    pd.DataFrame({"Subject": subs_followup}).to_csv(ARTIFACTS_DIR / "subjects_followup.csv", index=False)

    # -------------------------------------------------------------------
    # Load connectomes (local only, not saved to artifacts)
    # -------------------------------------------------------------------
    conn_baseline = np.load(PROCESSED_DIR / "connectomes_baseline.npy")
    conn_followup = np.load(PROCESSED_DIR / "connectomes_followup.npy")

    # -------------------------------------------------------------------
    # Compute safe summaries
    # -------------------------------------------------------------------
    summaries_baseline = compute_network_summaries(conn_baseline, subs_baseline)
    summaries_followup = compute_network_summaries(conn_followup, subs_followup)

    summaries_baseline.to_csv(ARTIFACTS_DIR / "network_summaries_baseline.csv", index=False)
    summaries_followup.to_csv(ARTIFACTS_DIR / "network_summaries_followup.csv", index=False)

    print("Artifacts created in:", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
