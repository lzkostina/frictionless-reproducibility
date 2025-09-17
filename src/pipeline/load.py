import pandas as pd

def filter_subjects_with_two_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only subjects that have both baseline and follow-up data.
    Parameters:
        df: pd.DataFrame
        Input dataframe with "Subject" column.
    Returns:
        pd.DataFrame:
        Filtered dataframe which is the subset of the input dataframe
        containing only subjects with more than one record (i.e.) 2 time points.
    """

    subject_counts = df['Subject'].value_counts()
    subjects_to_keep = subject_counts[subject_counts > 1].index

    return df[df['Subject'].isin(subjects_to_keep)]