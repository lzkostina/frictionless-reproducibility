import pandas as pd

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

