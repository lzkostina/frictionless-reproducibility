import pandas as pd
from pathlib import Path

def handle_missing_values(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """Drop rows with missing values (except rel_family_id)."""
    cols_to_check = [col for col in df1.columns if col != 'rel_family_id']
    return df1.dropna(subset=cols_to_check), df2.dropna(subset=cols_to_check)

def merge_family_ids(df_baseline: pd.DataFrame, df_followup: pd.DataFrame) -> pd.DataFrame:
    """Fill missing family IDs in follow-up data using baseline data."""
    df_followup = df_followup.merge(
        df_baseline[['Subject', 'rel_family_id']],
        on='Subject',
        how='left',
        suffixes=('', '_from_baseline')
    )
    df_followup['rel_family_id'] = df_followup['rel_family_id'].fillna(df_followup['rel_family_id_from_baseline'])
    return df_followup.drop(columns=['rel_family_id_from_baseline'])

def enforce_common_subjects(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """Enforce common subjects in both data frames."""
    # Identify unique values in each dataset
    subjects_0 = set(df1['Subject'].unique())
    subjects_1 = set(df2['Subject'].unique())

    # Find the common values between both datasets
    common_values = subjects_0.intersection(subjects_1)

    #  Filter both datasets to keep only rows with common values
    df1 = df1[df1['Subject'].isin(common_values)]
    df2 = df2[df2['Subject'].isin(common_values)]

    return df1, df2

def drop_siblings(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """Drop siblings in both data frames."""
    # Drop duplicate rows based on 'rel_family_id' while keeping the first occurrence
    df1 = df1.drop_duplicates(subset='rel_family_id', keep='first')
    df2 = df2.drop_duplicates(subset='rel_family_id', keep='first')
    return df1, df2

def convert_sex(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    df1['sex'] = df1['demo_sex_v2'].map({1.0: 'male', 2.0: 'female'})
    df2['sex'] = df2['demo_sex_v2'].map({1.0: 'male', 2.0: 'female'})

    df1 = df1.drop(columns=['demo_sex_v2'], axis=0, inplace=False)
    df2 = df2.drop(columns=['demo_sex_v2'], axis=0, inplace=False)
    return df1, df2

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
