from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def cross_val_pearson(model, X, y, groups, cv_strategy):
    """
    Calculate cross-validated Pearson correlation between predicted and observed values

    Returns:
    - List of (r, p-value) tuples for each fold
    - Array of predictions
    """
    correlations = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in cv_strategy.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        r, p = pearsonr(y_test, y_pred)
        correlations.append((r, p))
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

    return correlations, np.concatenate(all_y_true), np.concatenate(all_y_pred)


def partial_eta_squared(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    if sst == 0:
        return np.nan  # clearer than returning 0
    return 1 - (sse / sst)

def cross_val_partial_eta(model, X, y, groups, cv_strategy):
    """Calculate cross-validated partial eta squared"""
    eta_scores = []

    for train_idx, test_idx in cv_strategy.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        eta = partial_eta_squared(y_test, y_pred)
        eta_scores.append(eta)

    return np.mean(eta_scores), eta_scores


def run_cross_validation(X_features, y, use_network=False, X_network=None, num_pc=50, model=None, verbose=True):
    """
    Run Leave-One-Site-Out cross-validation and track progress with tqdm.

    Parameters
    ----------
    X_features : pd.DataFrame
        Main feature dataframe (must contain 'site_id_l' and 'Subject').
    y : pd.Series
        Target variable.
    use_network : bool, default False
        Whether to include network PCA features.
    X_network : pd.DataFrame, optional
        Network-level features (must include 'Subject').
    num_pc : int, default 50
        Number of principal components if use_network=True.
    model : estimator, default LinearRegression()
        Scikit-learn estimator to use.
    verbose : bool, default True
        If True, print per-fold metrics during CV.

    Returns
    -------
    results : list of dict
        Metrics per fold.
    trained_models : list
        Trained models for each fold.
    """
    if model is None:
        model = LinearRegression()

    logo = LeaveOneGroupOut()
    groups = X_features['site_id_l']
    results = []
    trained_models = []

    for train_idx, test_idx in tqdm(logo.split(X_features, y, groups=groups), total=len(np.unique(groups)),
                                    desc="Cross-validation progress"):
        test_site = X_features.iloc[test_idx]['site_id_l'].iloc[0]

        X_train_main = X_features.iloc[train_idx].drop(columns=['site_id_l', 'Subject'])
        X_test_main = X_features.iloc[test_idx].drop(columns=['site_id_l', 'Subject'])
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if use_network:
            if X_network is None:
                raise ValueError("X_network must be provided when use_network=True")

            subject_site_map = X_features[['Subject', 'site_id_l']].set_index('Subject')['site_id_l'].to_dict()
            X_network['site_id_l'] = X_network['Subject'].map(subject_site_map)
            X_network.fillna(0, inplace=True)

            train_subjects = X_features.iloc[train_idx]['Subject'].unique()
            test_subjects = X_features.iloc[test_idx]['Subject'].unique()

            X_train_network = X_network[(X_network['Subject'].isin(train_subjects)) &
                                        (X_network['site_id_l'] != test_site)].drop(columns=['site_id_l', 'Subject'])
            X_test_network = X_network[X_network['Subject'].isin(test_subjects)].drop(columns=['site_id_l', 'Subject'])

            pca_pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=num_pc, random_state=123, svd_solver="full")
            )
            X_train_network_pca = pca_pipe.fit_transform(X_train_network)
            X_test_network_pca = pca_pipe.transform(X_test_network)

            X_train = np.hstack([X_train_main.values, X_train_network_pca])
            X_test = np.hstack([X_test_main.values, X_test_network_pca])
        else:
            X_train = X_train_main.values
            X_test = X_test_main.values

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        trained_models.append(fold_model)
        y_pred = fold_model.predict(X_test)

        # Metrics
        r, p = pearsonr(y_test, y_pred)
        sse = np.sum((y_test - y_pred) ** 2)
        sst = np.sum((y_test - np.mean(y_test)) ** 2)
        eta = 1 - (sse / sst) if sst != 0 else np.nan
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'test_site': test_site,
            'pearson_r': r,
            'pearson_p': p,
            'partial_eta_squared': eta,
            'mse': mse,
            'r2': r2,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

        if verbose:
            print(f"\nSite: {test_site} | Pearson r = {r:.3f} (p={p:.3f}) | Partial η² = {eta:.3f}")

    return results, trained_models


def print_cross_val_results(df):
    # Convert results to DataFrame
    results_df = pd.DataFrame(df)

    # Calculate mean metrics across folds
    mean_metrics = {
        'Mean Pearson r': results_df['pearson_r'].mean(),
        'Mean Partial η²': results_df['partial_eta_squared'].mean(),
        'Mean R²': results_df['r2'].mean(),
        'Mean MSE': results_df['mse'].mean()
    }

    print("Cross-validated Performance Metrics:")
    print(f"Average Pearson's r: {mean_metrics['Mean Pearson r']:.3f}")
    print(f"Partial eta squared: {mean_metrics['Mean Partial η²']:.3f}")
    print(f"R²: {mean_metrics['Mean R²']:.3f}")