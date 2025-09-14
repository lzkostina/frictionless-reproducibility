import pandas as pd

def filter_subjects_with_two_timepoints(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only subjects with both baseline and follow-up data."""
    subject_counts = df['Subject'].value_counts()
    subjects_to_keep = subject_counts[subject_counts > 1].index
    return df[df['Subject'].isin(subjects_to_keep)]