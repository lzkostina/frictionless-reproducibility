import pandas as pd
from typing import List, Tuple


def list_columns_with_missing_values(df: pd.DataFrame) -> List[str]:
    """
    Identify columns in a DataFrame that contain missing values.

    Parameters:
    df : pd.DataFrame
        Input dataset.

    Returns:
    list of str
        Names of columns with at least one missing value.
    """
    return df.columns[df.isnull().any()].tolist()


def count_columns_with_missing_values(df: pd.DataFrame) -> int:
    """
    Count the number of columns in a DataFrame with missing values.

    Parameters:
    df : pd.DataFrame
        Input dataset.

    Returns:
    int
        Number of columns containing at least one missing value.
    """
    return (df.isnull().sum() > 0).sum()


def percentage_missing_values(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    """
    Calculate the percentage of missing values for specified columns.

    Parameters:
    df : pd.DataFrame
        Input dataset.
    columns : list of str
        List of column names for which to calculate missing percentages.

    Returns:
    pd.Series
        Index: column names, Values: percentage of missing values.
    """
    return df[columns].isnull().mean() * 100


def divide_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Divide dataset features into numerical and categorical.

    Parameters:
    df : pd.DataFrame
        Input dataset.

    Returns:
    tuple of list
        - numerical_features: list of numerical column names.
        - categorical_features: list of categorical column names.
    """
    numerical_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numerical_features, categorical_features


def categorical_summary_stats(df: pd.DataFrame, categorical_features: List[str]) -> pd.Series:
    """
    Calculate the number of unique values for each categorical feature.

    Parameters:
    df : pd.DataFrame
        Input dataset.
    categorical_features : list of str
        List of categorical feature names.
    Returns:

    pd.Series
        Index: categorical feature names, Values: number of unique values.
    """
    return df[categorical_features].nunique()
