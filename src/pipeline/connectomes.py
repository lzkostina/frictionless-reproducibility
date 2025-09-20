import numpy as np
import pandas as pd
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
