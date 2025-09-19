import pandas as pd
from pathlib import Path


def handle_missing_values(df_baseline: pd.DataFrame, df_followup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Since we want to keep only subjects with no missing values, we drop rows
    with missing values. The function checks all the columns except "rel_family_id"
    column since df_followup has no values in this column.

    Parameters:
        df_baseline: pd.DataFrame
        Baseline dataframe.
        df_followup: pd.DataFrame
        Followup dataframe.
    Returns:
        tuple of pd.DataFrames
        Filtered dataframes.
    """
    cols_to_check = [col for col in df_baseline.columns if col != 'rel_family_id']
    return df_baseline.dropna(subset=cols_to_check), df_followup.dropna(subset=cols_to_check)


def merge_family_ids(df_baseline: pd.DataFrame, df_followup: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing family IDs in follow-up data using baseline data.
    Parameters:
        df_baseline: pd.DataFrame
        Baseline dataframe.
        df_followup: pd.DataFrame
        Followup dataframe.
    Returns:
        pd.DataFrame
        Followup dataframe with "rel_family_id" column.
    """
    df_followup = df_followup.merge(
        df_baseline[['Subject', 'rel_family_id']],
        on='Subject',
        how='left',
        suffixes=('', '_from_baseline')
    )
    df_followup['rel_family_id'] = df_followup['rel_family_id'].fillna(df_followup['rel_family_id_from_baseline'])
    return df_followup.drop(columns=['rel_family_id_from_baseline'])


def enforce_common_subjects(df_baseline: pd.DataFrame, df_followup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforce common subjects in both data frames.

    Parameters:
        df_baseline: pd.DataFrame
        Baseline dataframe.
        df_followup: pd.DataFrame
        Followup dataframe.
        Both dataframes must have "Subject" column.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame)
        Subset of baseline dataframe and subset of followup dataframe with common subjects.
    """

    common_subjects = set(df_baseline["Subject"]).intersection(df_followup["Subject"])
    return (
        df_baseline[df_baseline["Subject"].isin(common_subjects)],
        df_followup[df_followup["Subject"].isin(common_subjects)],
    )

def drop_siblings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop siblings in both data frames.

    Parameters:
        df: pd.DataFrame
        Input dataframe.
        Must have "rel_family_id" column.

    Returns:
        pd.DataFrame
        Filtered dataframe with only one subject from a family.
    """

    # Drop duplicate rows based on 'rel_family_id' while keeping the first occurrence
    return df.drop_duplicates(subset='rel_family_id', keep='first')


def save_aligned_features(df_baseline_A, df_followup_A, df_baseline_B, df_followup_B,
                          output_dir: str = "../data/processed/"):
    """
    Save processed feature datasets (Version A and B) for baseline and follow-up.
    Ensures canonical schema and consistent filenames.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    canonical_cols = [
        "g_lavaan", "site_id_l", "age", "Subject", "meanFD", "race.4level",
        "hisp", "sex", "EdYearsHighest", "IncCombinedMidpoint",
        "Income2Needs", "Married", "income_group"
    ]

    datasets = {
        "features_A_baseline.csv": df_baseline_A,
        "features_A_followup.csv": df_followup_A,
        "features_B_baseline.csv": df_baseline_B,
        "features_B_followup.csv": df_followup_B,
    }

    for fname, df in datasets.items():
        df = df.copy()
        for col in canonical_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[canonical_cols]
        df.to_csv(f"{output_dir}/{fname}", index=False)
        print(f"Saved {fname} with {df.shape[0]} subjects and {df.shape[1]} features")



def link_with_g_scores(
    df_baseline: pd.DataFrame,
    df_followup: pd.DataFrame,
    g_factor: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Link baseline and follow-up datasets with g-scores, keeping only
    subjects with valid g-scores at both timepoints.

    Parameters
    ----------
    df_baseline : pd.DataFrame
        Baseline dataset with 'src_subject_id' column.
    df_followup : pd.DataFrame
        Follow-up dataset with 'src_subject_id' column.
    g_factor : pd.DataFrame
        Dataset containing g-scores. Must include:
        - 'subjectkey' (renamed to 'src_subject_id')
        - 'G_lavaan.baseline'
        - 'G_lavaan.2Year'
        - 'site_id_l.baseline'
        - 'site_id_l.2Year'

    Returns
    -------
    tuple of pd.DataFrame
        (baseline_merged, followup_merged), each merged with g-scores
        and restricted to subjects who have valid scores at both
        baseline and follow-up.
    """
    # Standardize subject ID
    g_factor = g_factor.rename(columns={"subjectkey": "src_subject_id"})

    # Keep only subjects with both baseline and follow-up g-scores
    filtered_g = g_factor[
        g_factor["G_lavaan.baseline"].notna()
        & g_factor["G_lavaan.2Year"].notna()
    ]

    # Merge baseline
    merged_baseline = pd.merge(
        filtered_g,
        df_baseline,
        on="src_subject_id",
        how="inner"
    ).drop(columns=["G_lavaan.2Year", "site_id_l.2Year", "site_id_l.baseline"])

    # Merge follow-up
    merged_followup = pd.merge(
        filtered_g,
        df_followup,
        on="src_subject_id",
        how="inner"
    ).drop(columns=["G_lavaan.baseline", "site_id_l.baseline", "site_id_l.2Year"])

    return merged_baseline, merged_followup

