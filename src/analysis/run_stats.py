"""
Statistical analysis pipeline with two modes:

(1) Full mode: uses connectomes from data/processed/version_B/
(2) Safe artifacts mode: uses network summaries from artifacts/version_B/
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.features.extraction import (
    extract_article_features,
    extract_ses_features,
    combine_data,
)
from src.evaluation.cross_validation import (
    run_cross_validation,
    print_cross_val_results,
)


RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_data_dir():
    """Decide whether to use full data or artifacts."""
    if os.environ.get("USE_ARTIFACTS") == "1":
        print("Mode: using SAFE ARTIFACTS (no raw connectomes)")
        return Path("artifacts/version_B"), True
    else:
        print("Mode: using FULL processed data (requires raw connectomes)")
        return Path("data/processed/version_B"), False


def run():
    """Main entry point for statistical analysis."""

    DATA_DIR, use_artifacts = _get_data_dir()

    # -----------------------------------------------------------------
    # Load features + subjects
    # -----------------------------------------------------------------
    features_baseline = pd.read_csv(DATA_DIR / "features_baseline.csv")
    features_followup = pd.read_csv(DATA_DIR / "features_followup.csv")

    subs_baseline = pd.read_csv(DATA_DIR / "subjects_baseline.csv")["Subject"].tolist()
    subs_followup = pd.read_csv(DATA_DIR / "subjects_followup.csv")["Subject"].tolist()

    # -----------------------------------------------------------------
    # Only load connectomes if not in safe mode
    # -----------------------------------------------------------------
    if not use_artifacts:
        conn_baseline = np.load(DATA_DIR / "connectomes_baseline.npy")
        conn_followup = np.load(DATA_DIR / "connectomes_followup.npy")

        conn_baseline_df = pd.DataFrame(
            conn_baseline, columns=[f"conn_{i}" for i in range(conn_baseline.shape[1])]
        )
        conn_baseline_df.insert(0, "Subject", subs_baseline)

        conn_followup_df = pd.DataFrame(
            conn_followup, columns=[f"conn_{i}" for i in range(conn_followup.shape[1])]
        )
        conn_followup_df.insert(0, "Subject", subs_followup)
    else:
        conn_baseline_df, conn_followup_df = None, None  # not used

    # -----------------------------------------------------------------
    # Extract article/SES features
    # -----------------------------------------------------------------
    X_baseline, y_baseline = extract_article_features(features_baseline)
    X_followup, y_followup = extract_article_features(features_followup)

    ses_baseline = extract_ses_features(features_baseline)
    ses_followup = extract_ses_features(features_followup)

    main_baseline = combine_data(ses_baseline, X_baseline)
    main_followup = combine_data(ses_followup, X_followup)

    # -----------------------------------------------------------------
    # Cross-validation helper
    # -----------------------------------------------------------------
    def save_barplot(results, title, filename, metric="r2"):
        """
        Save a simple bar plot showing mean CV score.

        Parameters
        ----------
        results : list or dict
            If list of dicts: extract `metric` from each dict
            If list of floats: plot directly
            If dict: keys = labels, values = list of floats
        title : str
            Plot title
        filename : str
            File name for saved figure
        metric : str
            Which metric to extract if results is a list of dicts
        """
        plt.figure(figsize=(6, 4))

        if isinstance(results, dict):
            labels = list(results.keys())
            values = [np.mean(v) for v in results.values()]
        elif isinstance(results, list):
            if isinstance(results[0], dict):  # list of dicts
                values = [r[metric] for r in results if metric in r]
            else:  # list of floats
                values = results
            labels = [metric]
            values = [np.mean(values)]
        else:
            labels, values = ["value"], [results]

        plt.bar(labels, values)
        plt.ylabel(f"Mean CV {metric.upper()}")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename)
        plt.close()

    def run_and_save(name, X, y, use_network, X_network=None, metric="r2"):
        results, _ = run_cross_validation(
            X, y, use_network=use_network, X_network=X_network, num_pc=100
        )
        print_cross_val_results(results)

        # Save table
        pd.DataFrame(results).to_csv(RESULTS_DIR / f"{name}.csv", index=False)

        # Save figure
        save_barplot(results, title=f"{name} performance", filename=f"{name}.png", metric=metric)

        return results

    # -----------------------------------------------------------------
    # Analyses
    # -----------------------------------------------------------------
    print("\n=== Baseline no network ===")
    run_and_save("baseline_no_network", X_baseline, y_baseline, use_network=False)

    print("\n=== Baseline SES + article no network ===")
    run_and_save("baseline_ses_no_network", main_baseline, y_baseline, use_network=False)

    print("\n=== Followup no network ===")
    run_and_save("followup_no_network", X_followup, y_followup, use_network=False)

    print("\n=== Followup SES + article no network ===")
    run_and_save("followup_ses_no_network", main_followup, y_followup, use_network=False)

    # --- Only run with-network analyses in full mode ---
    if not use_artifacts:
        print("\n=== Baseline with network ===")
        run_and_save("baseline_with_network", X_baseline, y_baseline, use_network=True, X_network=conn_baseline_df)

        print("\n=== Baseline SES + article with network ===")
        run_and_save("baseline_ses_with_network", main_baseline, y_baseline, use_network=True, X_network=conn_baseline_df)

        print("\n=== Followup with network ===")
        run_and_save("followup_with_network", X_followup, y_followup, use_network=True, X_network=conn_followup_df)

        print("\n=== Followup SES + article with network ===")
        run_and_save("followup_ses_with_network", main_followup, y_followup, use_network=True, X_network=conn_followup_df)

    print("\n Analysis complete. Results saved to:", RESULTS_DIR)
