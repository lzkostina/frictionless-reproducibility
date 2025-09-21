import numpy as np

def compute_positive_strength(correlation_matrix, exclude_diag=True):
    """
    Compute node positive strength: sum of positive connection weights for each node.

    Args:
        correlation_matrix (np.ndarray): Symmetric connectivity matrix.
        exclude_diag (bool): If True, diagonal entries are ignored (no self-loops).

    Returns:
        np.ndarray: Vector of positive strengths (length = number of nodes).
    """
    mat = correlation_matrix.copy()
    if exclude_diag:
        np.fill_diagonal(mat, 0.0)
    mat[mat < 0] = 0.0
    return np.sum(mat, axis=0)


def compute_negative_strength(correlation_matrix, exclude_diag=True, magnitude=True):
    """
    Compute node negative strength: sum of negative connection weights for each node.

    Args:
        correlation_matrix (np.ndarray): Symmetric connectivity matrix.
        exclude_diag (bool): If True, diagonal entries are ignored (no self-loops).
        magnitude (bool): If True, return absolute magnitudes of negative connections.
                          If False, return signed (negative) sums.

    Returns:
        np.ndarray: Vector of negative strengths (length = number of nodes).
    """
    mat = correlation_matrix.copy()
    if exclude_diag:
        np.fill_diagonal(mat, 0.0)

    if magnitude:
        mat[mat > 0] = 0.0
        return np.sum(np.abs(mat), axis=0)
    else:
        mat[mat > 0] = 0.0
        return np.sum(mat, axis=0)  # negative values


def compute_global_positive_strength(correlation_matrix, exclude_diag=True):
    """
    Compute global positive strength: average of node-level positive strengths.

    Args:
        correlation_matrix (np.ndarray): Symmetric connectivity matrix.
        exclude_diag (bool): If True, diagonal entries are ignored.

    Returns:
        float: Global positive strength.
    """
    return compute_positive_strength(correlation_matrix, exclude_diag=exclude_diag).mean()


def compute_global_negative_strength(correlation_matrix, exclude_diag=True, magnitude=True):
    """
    Compute global negative strength: average of node-level negative strengths.

    Args:
        correlation_matrix (np.ndarray): Symmetric connectivity matrix.
        exclude_diag (bool): If True, diagonal entries are ignored.
        magnitude (bool): If True, return magnitude of negative connections.

    Returns:
        float: Global negative strength.
    """
    return compute_negative_strength(
        correlation_matrix,
        exclude_diag=exclude_diag,
        magnitude=magnitude
    ).mean()
