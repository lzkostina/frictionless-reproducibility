import pandas as pd
from typing import Tuple

def extract_article_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target variable for downstream analysis.

    Steps:
      - Remove unused demographic/economic columns.
      - Add squared terms for age and meanFD.
      - Encode binary categorical features (hispanic yes/no).
      - Map race categories into ordinal values.
      - Separate target variable ("g_lavaan").

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing demographic and behavioral features.

    Returns
    -------
    features : pd.DataFrame
        Transformed feature dataframe ready for modeling.
    y : pd.Series
        Target variable (g_lavaan).
    """

    if "g_lavaan" not in df.columns:
        raise ValueError("Expected column 'g_lavaan' not found in dataframe.")

    y = df["g_lavaan"]

    drop_cols = [
        "g_lavaan", "IncCombinedMidpoint", "Income2Needs",
        "income_group", "Married", "EdYearsHighest"
    ]
    features = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Squared terms
    if "age" in features.columns:
        features["age_squared"] = features["age"] ** 2
    if "meanFD" in features.columns:
        features["meanFD_squared"] = features["meanFD"] ** 2

    # Binary mapping for hispanic
    if "hisp" in features.columns:
        features["hisp_encoded"] = features["hisp"].map({"Yes": 1, "No": 0})

    # Race mapping
    race_mapping = {"White": 0, "Other/Mixed": 1, "Black": 2, "Asian": 3}
    if "race.4level" in features.columns:
        features["race_encoded"] = features["race.4level"].map(race_mapping)
        features = features.drop(columns=["race.4level"])

    # Drop redundant columns
    if "hisp" in features.columns:
        features = features.drop(columns=["hisp"])

    return features, y


def extract_ses_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract socioeconomic status (SES) features from the dataset.

    Steps:
      - Drop unrelated columns (e.g., cognitive scores, demographics).
      - Encode binary categorical features (married status).
      - Encode ordinal categorical features (income group).
      - Return only SES-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns including "Married" and "income_group".

    Returns
    -------
    ses_features : pd.DataFrame
        SES features with encoded columns:
          - married_encoded: 1 = Currently Married, 0 = Not Currently Married
          - income_encoded: 0 = low, 1 = medium, 2 = high
    """

    drop_cols = ['g_lavaan', 'meanFD', 'age', 'sex', 'hisp', 'race.4level']
    ses_features = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode marital status
    if "Married" in ses_features.columns:
        ses_features['married_encoded'] = ses_features['Married'].map({
            'Currently Married': 1,
            'Not Currently Married': 0
        })
        ses_features = ses_features.drop(columns=["Married"])

    # Encode income group
    if "income_group" in ses_features.columns:
        income_mapping = {'low': 0, 'medium': 1, 'high': 2}
        ses_features['income_encoded'] = ses_features['income_group'].map(income_mapping)
        ses_features = ses_features.drop(columns=["income_group"])

    return ses_features

