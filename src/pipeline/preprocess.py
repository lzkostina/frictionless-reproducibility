import os
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

EVENT_TO_SESSION = {
    "baseline_year_1_arm_1": "baselineYear1Arm1",
    "2_year_follow_up_y_arm_1": "2YearFollowUpYArm1",
}



from src.pipeline.load import filter_subjects_with_two_timepoints
from src.pipeline.connectomes import merge_features_and_connectomes, enforce_same_subjects
from src.pipeline.clean import (
    handle_missing_values,
    merge_family_ids,
    enforce_common_subjects,
    drop_siblings,
    link_with_g_scores,
)
# from src.pipeline.preprocess import (
#     preprocess_features,
#     filter_sites_by_threshold,
#     add_income_group,
#     print_data_summary,
# )

def filter_sites_by_threshold(df: pd.DataFrame, site_col: str, threshold: int) -> pd.DataFrame:
    """
    Keep only rows from sites with number of observations > threshold.

    Args:
        df (pd.DataFrame): Input dataframe.
        site_col (str): Column name with site IDs.
        threshold (int): Minimum number of observations required for a site to be kept.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    site_counts = df[site_col].value_counts()
    keep_sites = site_counts[site_counts > threshold].index
    return df[df[site_col].isin(keep_sites)]


def add_income_group(data: pd.DataFrame, t1: int = 50000, t2: int = 100000) -> pd.DataFrame:

    """
    Add income group column.
    Args:
        pd.DataFrame: Input dataframe.
        t1, t2 - optional values for income group division
    Returns:
        pd.DataFrame: Updated dataframe.
    """

    data['income_group'] = pd.cut(data['IncCombinedMidpoint'],
                                    bins=[-float("inf"), t1, t2, float("inf")],
                                    labels=["low", "medium", "high"]
    )
    return data


def preprocess_features(data: pd.DataFrame, g_score: pd.DataFrame, eventname: str) -> pd.DataFrame:
    """
    Preprocess features and merge with g_score for a given event (baseline or follow-up).
    Ensures output matches canonical schema across preprocessing strategies.
    """

    # Copy inputs
    features = data.copy()
    g_factor = g_score.copy()

    # Drop constant column
    if "Task" in features.columns:
        features = features.drop(columns=["Task"])

    # Rename subjectkey for merge
    g_factor = g_factor.rename(columns={"subjectkey": "src_subject_id"})

    # Filter to one event
    result_df = features[features["eventname"] == eventname]

    # Merge demographics with g_score
    result_df_fg = pd.merge(g_factor, result_df, on="src_subject_id")

    # Drop duplicate column
    if "site_id_l" in result_df_fg.columns:
        result_df_fg = result_df_fg.drop(columns=["site_id_l"])

    # Drop columns that are *always* irrelevant
    drop_cols = ["src_subject_id", "Session", "rel_family_id"]
    result_df_fg = result_df_fg.drop(columns=[c for c in drop_cols if c in result_df_fg.columns])

    # Handle event-specific renames/drops
    if eventname == "baseline_year_1_arm_1":
        result_df_fg = result_df_fg.drop(columns=["site_id_l.2Year", "G_lavaan.2Year"], errors="ignore")
        result_df_fg = result_df_fg.rename(columns={
            "site_id_l.baseline": "site_id_l",
            "G_lavaan.baseline": "g_lavaan"
        })
    elif eventname == "2_year_follow_up_y_arm_1":
        result_df_fg = result_df_fg.drop(columns=["site_id_l.baseline", "G_lavaan.baseline"], errors="ignore")
        result_df_fg = result_df_fg.rename(columns={
            "site_id_l.2Year": "site_id_l",
            "G_lavaan.2Year": "g_lavaan"
        })

    # Harmonize column names
    result_df_fg = result_df_fg.rename(columns={
        "interview_age": "age",
        "demo_sex_v2": "sex"
    })

    # Drop rows with missing values
    result_df_fg = result_df_fg.dropna()

    # Keep only subjects with >= 4 minutes (240 sec) of uncensored data
    if "confounds_nocensor" in result_df_fg.columns:
        result_df_fg = result_df_fg[result_df_fg["confounds_nocensor"] >= 240]

    # Filter sites by threshold
    threshold = 75 if eventname == "baseline_year_1_arm_1" else 50
    result_df_fg = filter_sites_by_threshold(result_df_fg, site_col="site_id_l", threshold=threshold)

    # Add income group
    result_df_fg = add_income_group(result_df_fg)

    # Canonical schema
    canonical_cols = [
        "g_lavaan", "site_id_l", "age", "Subject", "meanFD", "race.4level",
        "hisp", "sex", "EdYearsHighest", "IncCombinedMidpoint",
        "Income2Needs", "Married", "income_group"
    ]
    for col in canonical_cols:
        if col not in result_df_fg.columns:
            result_df_fg[col] = pd.NA

    result_df_fg = result_df_fg[canonical_cols]

    return result_df_fg


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics for key demographic features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Must contain:
        - 'age' (numeric, months)
        - 'sex'
        - 'race.4level'
        - 'hisp'
        - 'income_group'
    """
    n_subjects = len(df)
    print(f"Number of subjects: {n_subjects}")

    # Age (convert months → years)
    if "age" in df.columns:
        print(f"Age mean: {df['age'].mean() / 12:.2f} years")
        print(f"Age std: {df['age'].std() / 12:.2f} years")

    # Categorical summaries
    categorical_cols = ["sex", "race.4level", "hisp", "income_group"]
    for col in categorical_cols:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            print(f"\n{col}:")
            for value, count in counts.items():
                perc = count / n_subjects * 100
                print(f"  {value}: {count} ({perc:.2f}%)")



def run():
    """Main preprocessing entry point."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load raw data
    # -------------------------------------------------------------------------
    features = pd.read_csv(RAW_DATA_DIR / "Demographics.csv")
    g_factor = pd.read_csv(RAW_DATA_DIR / "ABCD_new_G_all.csv")
    features = features.drop(columns=["Task"], inplace=False)

    # -------------------------------------------------------------------------
    # 2. Cleaning
    # -------------------------------------------------------------------------

    # Version A preprocessing

    # Keep only participants with both baseline and follow-up
    df_filtered = filter_subjects_with_two_timepoints(features)
    # Split by event
    df_baseline = df_filtered[df_filtered["eventname"] == "baseline_year_1_arm_1"]
    df_followup = df_filtered[df_filtered["eventname"] == "2_year_follow_up_y_arm_1"]

    # Drop rows with missing values (except family IDs)
    df_baseline, df_followup = handle_missing_values(df_baseline, df_followup)
    # Fill in family IDs for follow-up from baseline
    df_followup = merge_family_ids(df_baseline, df_followup)

    df_baseline, df_followup = enforce_common_subjects(df_baseline, df_followup)
    df_baseline = drop_siblings(df_baseline)
    df_followup = drop_siblings(df_followup)

    merged_df_baseline_A, merged_df_followup_A = link_with_g_scores(df_baseline, df_followup, g_factor)

    rename_map = {
        "G_lavaan.baseline": "g_lavaan",
        "G_lavaan.2Year": "g_lavaan",
        "demo_sex_v2": "sex",
        "interview_age": "age",
    }
    merged_df_baseline_A = merged_df_baseline_A.rename(columns=rename_map)
    merged_df_followup_A = merged_df_followup_A.rename(columns=rename_map)

    drop_cols = [
        "src_subject_id", "eventname", "rel_family_id", "Session",
        "GoodRun_5", "censor_5", "TRs", "confounds_nocensor",
    ]
    merged_df_baseline_A = merged_df_baseline_A.drop(columns=drop_cols)
    merged_df_followup_A = merged_df_followup_A.drop(columns=drop_cols)

    merged_df_baseline_A = add_income_group(merged_df_baseline_A, t1=50000, t2=100000)
    merged_df_followup_A = add_income_group(merged_df_followup_A, t1=50000, t2=100000)

    A_MIN = 50
    merged_df_baseline_A = filter_sites_by_threshold(merged_df_baseline_A, site_col="site_id_l", threshold=A_MIN)
    merged_df_followup_A = filter_sites_by_threshold(merged_df_followup_A, site_col="site_id_l", threshold=A_MIN)

    print_data_summary(merged_df_baseline_A)
    print_data_summary(merged_df_followup_A)

    # Version B preprocessing

    merged_df_baseline_B = preprocess_features(features, g_factor, "baseline_year_1_arm_1")
    merged_df_followup_B = preprocess_features(features, g_factor, "2_year_follow_up_y_arm_1")

    B_MIN = 75
    merged_df_baseline_B = filter_sites_by_threshold(merged_df_baseline_B, site_col="site_id_l", threshold=B_MIN)
    merged_df_followup_B = filter_sites_by_threshold(merged_df_followup_B, site_col="site_id_l", threshold=B_MIN)

    print_data_summary(merged_df_baseline_B)
    print_data_summary(merged_df_followup_B)

    # Final missing value handling
    df_baseline, df_followup = handle_missing_values(df_baseline, df_followup)
    df_followup = merge_family_ids(df_baseline, df_followup)
    df_baseline, df_followup = enforce_common_subjects(df_baseline, df_followup)

    # -------------------------------------------------------------------------
    # 3. Align features with connectomes (Version A)
    # -------------------------------------------------------------------------
    features_A_baseline_aligned, conn_A_baseline, subs_A_baseline = merge_features_and_connectomes(
        merged_df_baseline_A, visit=EVENT_TO_SESSION["baseline_year_1_arm_1"]
    )
    features_A_followup_aligned, conn_A_followup, subs_A_followup = merge_features_and_connectomes(
        merged_df_followup_A, visit=EVENT_TO_SESSION["2_year_follow_up_y_arm_1"]
    )

    (
        features_A_baseline_final, conn_A_baseline_final,
        features_A_followup_final, conn_A_followup_final,
        subs_A_final
    ) = enforce_same_subjects(
        features_A_baseline_aligned, conn_A_baseline, subs_A_baseline,
        features_A_followup_aligned, conn_A_followup, subs_A_followup
    )

    outA = PROCESSED_DIR / "version_A"
    outA.mkdir(parents=True, exist_ok=True)
    features_A_baseline_final.to_csv(outA / "features_baseline.csv", index=False)
    features_A_followup_final.to_csv(outA / "features_followup.csv", index=False)
    np.save(outA / "connectomes_baseline.npy", conn_A_baseline_final)
    np.save(outA / "connectomes_followup.npy", conn_A_followup_final)
    pd.DataFrame({"Subject": subs_A_final}).to_csv(outA / "subjects.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. Align features with connectomes (Version B)
    # -------------------------------------------------------------------------
    features_B_baseline_aligned, conn_B_baseline, subs_B_baseline = merge_features_and_connectomes(
        merged_df_baseline_B, visit=EVENT_TO_SESSION["baseline_year_1_arm_1"]
    )
    features_B_followup_aligned, conn_B_followup, subs_B_followup = merge_features_and_connectomes(
        merged_df_followup_B, visit=EVENT_TO_SESSION["2_year_follow_up_y_arm_1"]
    )

    outB = PROCESSED_DIR / "version_B"
    outB.mkdir(parents=True, exist_ok=True)
    features_B_baseline_aligned.to_csv(outB / "features_baseline.csv", index=False)
    features_B_followup_aligned.to_csv(outB / "features_followup.csv", index=False)
    np.save(outB / "connectomes_baseline.npy", conn_B_baseline)
    np.save(outB / "connectomes_followup.npy", conn_B_followup)
    pd.DataFrame({"Subject": subs_B_baseline}).to_csv(outB / "subjects_baseline.csv", index=False)
    pd.DataFrame({"Subject": subs_B_followup}).to_csv(outB / "subjects_followup.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. Done
    # -------------------------------------------------------------------------
    print("Preprocessing complete. Processed outputs saved under:", PROCESSED_DIR)


if __name__ == "__main__":
    run()
