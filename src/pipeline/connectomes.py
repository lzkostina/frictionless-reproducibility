import pandas as pd
from typing import Tuple


def align_subjects(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    subject_col: str = "Subject",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two dataframes by keeping only overlapping subjects.
    Sorts both dataframes by subject ID and ensures identical subject order.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (e.g., features).
    df2 : pd.DataFrame
        Second dataframe (e.g., connectomes).
    subject_col : str, default="Subject"
        Column name that identifies subjects.
    verbose : bool, default=True
        If True, prints information about mismatches and alignment.

    Returns
    -------
    df1_aligned : pd.DataFrame
        Subset of df1 with only subjects present in both.
    df2_aligned : pd.DataFrame
        Subset of df2 with only subjects present in both.

    Raises
    ------
    ValueError
        If no common subjects exist.
    """
    subjects1 = set(df1[subject_col].unique())
    subjects2 = set(df2[subject_col].unique())
    common_subjects = subjects1 & subjects2

    if not common_subjects:
        raise ValueError("No overlapping subjects found between the two datasets.")

    if verbose:
        missing_in_df1 = subjects2 - subjects1
        missing_in_df2 = subjects1 - subjects2

        if missing_in_df1:
            print(f"Subjects in df2 but not in df1: {sorted(missing_in_df1)}")
        if missing_in_df2:
            print(f"Subjects in df1 but not in df2: {sorted(missing_in_df2)}")

        print(f"Keeping {len(common_subjects)} common subjects.")

    # Keep only common subjects and sort
    df1_aligned = df1[df1[subject_col].isin(common_subjects)].copy()
    df2_aligned = df2[df2[subject_col].isin(common_subjects)].copy()

    df1_aligned = df1_aligned.sort_values(by=subject_col).reset_index(drop=True)
    df2_aligned = df2_aligned.sort_values(by=subject_col).reset_index(drop=True)

    return df1_aligned, df2_aligned
