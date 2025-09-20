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
