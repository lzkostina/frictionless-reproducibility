import numpy as np
import pandas as pd
from src.utils.matrix_io import vector_to_symmetric_matrix
from tqdm import tqdm

def build_network_indexer(regions: pd.DataFrame,
                          node_order: list | np.ndarray | None,
                          parcel_col: str = "ParcelID",
                          net_col: str = "NetworkNumber"):
    """
    Returns (network_order, network_to_indices, labels):

    - regions: df with parcel->network mapping
    - node_order: list/array of ParcelIDs in the SAME order as your matrix rows/cols
                  If None, assumes regions[parcel_col] is already in matrix order.
    """
    if node_order is None:
        node_order = regions[parcel_col].tolist()

    reg_ord = regions.set_index(parcel_col).loc[node_order].reset_index()
    labels = reg_ord[net_col].to_numpy()
    network_order = list(pd.unique(labels))  # preserves first-seen order
    network_to_indices = {net: np.flatnonzero(labels == net) for net in network_order}

    # sanity checks
    n = len(node_order)
    covered = sum(len(ix) for ix in network_to_indices.values())
    assert covered == n, f"Indexer mismatch: covered={covered}, n={n}"
    return network_order, network_to_indices, labels


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

def compute_network_statistics_by_index(M: np.ndarray,
                                        network_to_indices: dict,
                                        network_order: list,
                                        mode: str = "all") -> pd.DataFrame:
    """
    Compute network-level statistics for each network in the connectivity matrix.

    Args:
        M (np.ndarray): Connectivity matrix.
        network_to_indices (dict): Dictionary mapping network names to indices.
        network_order (list): Ordered list of network identifiers.
        mode (str): Which statistics to compute ("all", "strengths", "connectivity", "segregation").

    Returns:
        pd.DataFrame: Table of network statistics, one row per network.

    Notes:
        - Strength metrics (avg_positive_strength, avg_negative_strength, avg_total_strength) follow
          definitions of node strength in weighted networks [Barrat et al., 2004; Rubinov & Sporns, 2010].
          Positive/negative separation follows signed-network convention [Rubinov, 2011].

        - Within- and between-system means (within_positive_mean, etc.) are standard summaries of
          functional connectivity [Power et al., 2011].

        - Segregation metrics:
            * segregation_ratio: within-system positive mean / between-system positive mean (used in several FC studies).
            * segregation_index: (within-system positive mean – between-system positive mean) / within-system positive mean,
              as defined by Chan et al. (2014, PNAS).
    """
    n = M.shape[0]
    stats = []

    # strengths (only if requested)
    if mode in ["all", "strengths"]:
        pos_str = compute_positive_strength(M, exclude_diag=True)
        neg_str = compute_negative_strength(M, exclude_diag=True, magnitude=False)
        tot_str = pos_str + neg_str

    all_idx = np.arange(n)

    for net in network_order:
        idx = np.asarray(network_to_indices[net])
        size = idx.size
        row = {"network_id": net, "size": int(size)}

        # strengths
        if mode in ["all", "strengths"]:
            row.update({
                "avg_positive_strength": float(pos_str[idx].mean()) if size else np.nan,
                "avg_negative_strength": float(neg_str[idx].mean()) if size else np.nan,
                "avg_total_strength":    float(tot_str[idx].mean()) if size else np.nan,
            })

        # connectivity / segregation
        if mode in ["all", "connectivity", "segregation"]:
            # within: use upper triangle ONLY to avoid double counting
            W = M[np.ix_(idx, idx)]
            iu = np.triu_indices(size, k=1)
            W_ut = W[iu] if size >= 2 else np.array([])

            within_connections = iu[0].size  # = size*(size-1)/2
            within_positive_sum = W_ut[W_ut > 0].sum() if within_connections else 0.0
            within_negative_sum = W_ut[W_ut < 0].sum() if within_connections else 0.0  # signed

            within_positive_mean = (within_positive_sum / within_connections) if within_connections else np.nan
            within_negative_mean = (within_negative_sum / within_connections) if within_connections else np.nan

            # between: rectangular block counts each edge once
            other = np.setdiff1d(all_idx, idx, assume_unique=True)
            B = M[np.ix_(idx, other)]
            between_connections = idx.size * other.size
            if between_connections:
                flatB = B.ravel()
                between_positive_sum = flatB[flatB > 0].sum()
                between_negative_sum = flatB[flatB < 0].sum()  # signed
                between_positive_mean = between_positive_sum / between_connections
                between_negative_mean = between_negative_sum / between_connections
            else:
                between_positive_mean = between_negative_mean = np.nan

            row.update({
                "within_positive_mean":  float(within_positive_mean),
                "within_negative_mean":  float(within_negative_mean),
                "between_positive_mean": float(between_positive_mean),
                "between_negative_mean": float(between_negative_mean),
            })

            # segregation (two variants, based on positive means)
            if mode in ["all", "segregation"]:
                rp = (within_positive_mean / between_positive_mean
                      if (between_positive_mean is not np.nan and between_positive_mean not in [0, np.nan])
                      else np.nan)
                si = ((within_positive_mean - between_positive_mean) / within_positive_mean
                      if (within_positive_mean is not np.nan and within_positive_mean not in [0, np.nan])
                      else np.nan)
                row.update({
                    "segregation_ratio": rp,
                    "segregation_index": si,
                })

        stats.append(row)

    return pd.DataFrame(stats).set_index("network_id")


def compute_one_statistic(df_features: pd.DataFrame,
                          df_network: pd.DataFrame,
                          regions: pd.DataFrame,
                          stat_name: str,
                          mode: str = "all",
                          node_order: list | np.ndarray | None = None,
                          matrix_size: int = 418,
                          diag_value: float = 0.0) -> pd.DataFrame:
    """
    For each subject row in df_network (flattened vector columns), reconstruct M and compute one statistic
    per network. Returns a wide DF: Subject, site_id_l, and one column per network for `stat_name`.
    """
    # site mapping
    subject_site_map = df_features.set_index('Subject')['site_id_l']
    dfN = df_network.copy()
    dfN['site_id_l'] = dfN['Subject'].map(subject_site_map)

    # extract vector columns in stable order
    vector_cols = [c for c in dfN.columns if c not in ('Subject', 'site_id_l')]
    expected_len = matrix_size * (matrix_size - 1) // 2
    if len(vector_cols) != expected_len:
        raise ValueError(f"Found {len(vector_cols)} vector columns, expected {expected_len} for size {matrix_size}")

    # build indexer once
    network_order, network_to_indices, _ = build_network_indexer(
        regions=regions,
        node_order=node_order,   # pass the parcel order used when building the vectors
        parcel_col="ParcelID",
        net_col="NetworkNumber"
    )

    # prepare outputs
    result = []
    subj_arr = dfN['Subject'].to_numpy()
    site_arr = dfN['site_id_l'].to_numpy()
    vecs = dfN[vector_cols].to_numpy(dtype=float)  # shape: (n_subjects, 87153)

    for subj, site, vec in tqdm(zip(subj_arr, site_arr, vecs), total=len(dfN)):
        M = vector_to_symmetric_matrix(vec, size=matrix_size, diag_value=diag_value)
        stats_df = compute_network_statistics_by_index(M, network_to_indices, network_order, mode=mode)

        if stat_name not in stats_df.columns:
            raise KeyError(f"`stat_name` '{stat_name}' not in computed statistics: {list(stats_df.columns)}")

        row = {"Subject": subj, "site_id_l": site}
        for net in network_order:
            row[f"{stat_name}[{net}]"] = stats_df.loc[net, stat_name]
        result.append(row)

    return pd.DataFrame(result)
