import numpy as np
import pandas as pd

from src.utils.matrix_io import read_file_to_matrix, convert_matrix_to_array
from typing import Callable, List, Tuple, Optional

from src.utils.matrix_io import read_file_to_matrix, convert_matrix_to_array


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


def load_connectomes(
    df: pd.DataFrame,
    visit: str,
    directory: str = "../connectomes/connectomes",
    expected_shape: Tuple[int, int] = (418, 418),
    subject_col: str = "Subject",
    reader: Optional[Callable[[str, str, str], np.ndarray]] = None,
    converter: Optional[Callable[[object], np.ndarray]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load and flatten connectome matrices for unique subjects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subject IDs in `subject_col`.
    visit : str
        Visit identifier passed to the file reader (e.g., "baseline_year_1_arm_1").
    directory : str, optional
        Directory where connectome files are stored.
    expected_shape : tuple of int, optional
        Expected matrix shape (rows, cols), default (418, 418).
    subject_col : str, optional
        Column name containing subject IDs, default "Subject".
    reader : callable, optional
        Function to read a matrix: reader(subject_id, visit, directory) -> np.ndarray-like.
        Defaults to a function named `read_file_to_matrix` in this module's scope.
    converter : callable, optional
        Function to convert raw matrix to numpy array: converter(matrix) -> np.ndarray-like.
        Defaults to a function named `convert_matrix_to_array` in this module's scope.
    verbose : bool, optional
        If True, prints progress and skipped subjects.

    Returns
    -------
    flattened : np.ndarray
        2D array of shape (n_subjects, n_edges) with upper-triangle values (k=1).
    subjects : list of str
        Subject IDs corresponding to the rows of `flattened`.

    Raises
    ------
    ValueError
        If no valid connectomes are loaded.
    """

    # Fallback to in-scope helpers if custom callables not supplied
    if reader is None:
        try:
            reader = read_file_to_matrix  # type: ignore[name-defined]
        except NameError:
            raise NameError("`reader` is None and `read_file_to_matrix` is not defined in scope.")
    if converter is None:
        try:
            converter = convert_matrix_to_array  # type: ignore[name-defined]
        except NameError:
            raise NameError("`converter` is None and `convert_matrix_to_array` is not defined in scope.")

    # Collect unique subjects (deterministic order)
    if subject_col not in df.columns:
        raise KeyError(f"Column '{subject_col}' not found in dataframe.")
    unique_subjects = sorted(pd.Series(df[subject_col]).dropna().astype(str).unique().tolist())

    n_rows, n_cols = expected_shape
    if n_rows != n_cols:
        raise ValueError(f"expected_shape must be square; got {expected_shape}.")

    # Precompute the indices for the upper triangle (excluding diagonal)
    tri_idx = np.triu_indices(n_rows, k=1)

    connectomes: List[np.ndarray] = []
    subjects: List[str] = []

    for subject in unique_subjects:
        try:
            raw = reader(subject, visit, directory=directory)
            if raw is None:
                if verbose:
                    print(f"[skip] No matrix for subject {subject}")
                continue

            arr = np.asarray(converter(raw))
            if arr.shape != expected_shape:
                if verbose:
                    print(f"[skip] {subject}: shape {arr.shape} != expected {expected_shape}")
                continue

            connectomes.append(arr)
            subjects.append(subject)

        except Exception as e:
            if verbose:
                print(f"[skip] {subject}: {e}")
            continue

    if not connectomes:
        raise ValueError("No valid connectomes were loaded (all missing or wrong shape).")

    # Flatten each matrix's upper triangle and stack
    flattened = np.vstack([conn[tri_idx].ravel() for conn in connectomes])

    if verbose:
        print(f"Loaded {len(subjects)} subjects")
        print(f"Flattened shape: {flattened.shape}  (n_subjects x n_edges)")

    return flattened, subjects


def merge_features_and_connectomes(
    df_features: pd.DataFrame,
    visit: str,
    directory: str = "/Users/kostina/my-project~607/data/connectomes",
    expected_shape: Tuple[int, int] = (418, 418),
    subject_col: str = "Subject",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Merge a features dataframe with connectomes, keeping only overlapping subjects.

    Parameters
    ----------
    df_features : pd.DataFrame
        Features dataframe containing subject IDs.
    visit : str
        Visit identifier (e.g., "baseline_year_1_arm_1").
    directory : str, default="data/connectomes"
        Directory where connectome files are stored.
    expected_shape : tuple of int, default (418, 418)
        Expected shape of each connectome matrix.
    subject_col : str, default="Subject"
        Column containing subject IDs.
    verbose : bool, default=True
        If True, prints progress.

    Returns
    -------
    df_features_aligned : pd.DataFrame
        Features dataframe aligned with connectomes.
    flattened_aligned : np.ndarray
        Flattened connectomes aligned with features (subjects x edges).
    subjects : list
        List of aligned subject IDs.
    """
    # Load flattened connectomes + subjects
    flattened, subjects = load_connectomes(
        df=df_features,
        visit=visit,
        directory=directory,
        expected_shape=expected_shape,
        subject_col=subject_col,
        reader=read_file_to_matrix,
        converter=convert_matrix_to_array,
        verbose=verbose,
    )

    # Build connectome dataframe
    df_connectomes = pd.DataFrame({subject_col: subjects})
    df_connectomes["connectome"] = list(flattened)

    # Align both
    df_features_aligned, df_connectomes_aligned = align_subjects(
        df_features, df_connectomes, subject_col=subject_col, verbose=verbose
    )

    # Extract aligned flattened matrices
    flattened_aligned = np.vstack(df_connectomes_aligned["connectome"].values)
    aligned_subjects = df_features_aligned[subject_col].tolist()

    return df_features_aligned, flattened_aligned, aligned_subjects


def enforce_same_subjects(features_baseline, conn_baseline, subs_baseline,
                          features_followup, conn_followup, subs_followup):
    """
    Keep only subjects present in both baseline and followup datasets.
    Aligns features and connectomes accordingly.
    """
    baseline_set = set(subs_baseline)
    followup_set = set(subs_followup)
    common_subjects = baseline_set & followup_set

    # Filter indices for common subjects
    baseline_idx = [i for i, s in enumerate(subs_baseline) if s in common_subjects]
    followup_idx = [i for i, s in enumerate(subs_followup) if s in common_subjects]

    # Align features
    features_baseline_aligned = features_baseline.iloc[baseline_idx].reset_index(drop=True)
    features_followup_aligned = features_followup.iloc[followup_idx].reset_index(drop=True)

    # Align connectomes
    conn_baseline_aligned = conn_baseline[baseline_idx, :]
    conn_followup_aligned = conn_followup[followup_idx, :]

    # Align subject lists
    subs_baseline_aligned = [subs_baseline[i] for i in baseline_idx]
    subs_followup_aligned = [subs_followup[i] for i in followup_idx]

    assert subs_baseline_aligned == subs_followup_aligned, "Subjects not properly aligned!"

    return (features_baseline_aligned, conn_baseline_aligned,
            features_followup_aligned, conn_followup_aligned,
            subs_baseline_aligned)

