import os
import numpy as np
import pandas as pd
from typing import Optional


def read_file_to_matrix(
    subject: str,
    session: str,
    directory: str = "../connectomes/connectomes",
    delimiter: str = ","
) -> Optional[np.ndarray]:
    """
    Read a text file containing a subject's connectome and return it as a NumPy array.

    Parameters
    ----------
    subject : str
        Subject identifier (used in filename).
    session : str
        Session/visit identifier (used in filename).
    directory : str, default="../connectomes/connectomes"
        Path to the directory containing the files.
    delimiter : str, default=","
        Delimiter used in the file (default assumes CSV-style formatting).

    Returns
    -------
    np.ndarray or None
        The loaded connectome matrix if successful, otherwise None.
    """
    # Build file path
    file_name = f"{subject}_{session}_p5.txt"
    file_path = os.path.join(directory, file_name)

    if not os.path.isfile(file_path):
        print(f"[skip] File not found: {file_path}")
        return None

    try:
        matrix = pd.read_csv(file_path, delimiter=delimiter, header=None).values
        return matrix
    except Exception as e:
        print(f"[skip] Error reading {file_name}: {e}")
        return None


def convert_matrix_to_array(input_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a raw text-based matrix (list of strings) into a NumPy array.

    Parameters
    ----------
    input_matrix : np.ndarray
        Input matrix (e.g., loaded from text where each row may contain strings).

    Returns
    -------
    np.ndarray
        Cleaned numeric NumPy array.
    """
    try:
        if isinstance(input_matrix, np.ndarray) and np.issubdtype(input_matrix.dtype, np.number):
            # Already numeric
            return input_matrix

        matrix = []
        for line in input_matrix:
            # Assume values are space-separated inside the first column
            row = np.array([float(value) for value in str(line[0]).split()])
            matrix.append(row)

        return np.array(matrix)

    except Exception as e:
        raise ValueError(f"Failed to convert matrix to array: {e}")


def vector_to_symmetric_matrix(vector, size, diag_value=0.0):
    """
    Convert a vector of upper-triangular values into a full symmetric matrix.

    Args:
        vector (array-like): Values for the upper triangle (excluding diagonal).
        size (int): Dimension of the square matrix.
        diag_value (float, optional): Value to fill the diagonal with.
                                      Defaults to 0.0.

    Returns:
        np.ndarray: Symmetric (size x size) matrix.
    """
    vector = np.asarray(vector)

    # Number of elements in the upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(size, k=1)
    expected_len = len(triu_indices[0])

    if vector.shape[0] != expected_len:
        raise ValueError(
            f"Vector length {vector.shape[0]} doesn't match expected "
            f"length {expected_len} for size {size}"
        )

    # Create empty matrix and fill upper triangle
    matrix = np.zeros((size, size), dtype=vector.dtype)
    matrix[triu_indices] = vector

    # Mirror to lower triangle
    matrix = matrix + matrix.T

    # Fill diagonal
    np.fill_diagonal(matrix, diag_value)

    return matrix
