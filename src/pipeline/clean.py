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

def save_processed_data(
        df_baseline: pd.DataFrame,
        df_followup: pd.DataFrame,
        output_dir: str = "../data/processed/",
        prefix: str = "features"
) -> None:
    """
    Save baseline and follow-up DataFrames to CSV files.

    Args:
        df_baseline: Baseline timepoint data.
        df_followup: Follow-up timepoint data.
        output_dir: Directory to save files (default: '../data/processed/').
        prefix: Filename prefix (default: 'features' -> outputs 'features_0.csv').
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save files
    df_baseline.to_csv(f"{output_dir}/{prefix}_0.csv", index=False)
    df_followup.to_csv(f"{output_dir}/{prefix}_1.csv", index=False)
    print(f"Data saved to {output_dir}{prefix}_0.csv and {output_dir}{prefix}_1.csv")
