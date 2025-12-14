import os, sys
import numpy as np

# --- normalization ----

def data_matrix_normalization(X_tr, X_te, use_both = False):
    if use_both:        # this is rare!
        X_all = np.vstack([X_tr, X_te])
        X_mu = X_all.mean(axis = 0)
        X_std = X_all.std(axis = 0) + 1e-12
    else:
        X_mu  = X_tr.mean(axis=0)
        X_std = X_tr.std(axis=0) + 1e-12
        
    X_train = (X_tr - X_mu) / X_std
    X_test  = (X_te - X_mu) / X_std
    return X_train, X_test, X_mu, X_std

def inverse_data_matrix_normalization(X_norm, X_mu, X_std):
    """
    Invert the normalization done by data_matrix_normalization().

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized data (e.g., predictions, standardized states)
    X_mu : np.ndarray
        Mean of training data used in normalization (shape = [d_x])
    X_std : np.ndarray
        Standard deviation of training data used in normalization (shape = [d_x])
    """
    return X_norm * X_std + X_mu

# Chronological validation split on TRAIN (last 20% for val)
# ------------------------------------------------------------
def data_split(X, alpha): # assume data matrix X is organized as Sample size by dimension of each sample
    n = X.shape[0] 
    n_val = max(1, int(alpha * n)) 
    idx_tr = np.arange(0, n - n_val)   # first alpha%
    idx_va = np.arange(n - n_val, n)   # last  alpha%
    return X[idx_tr], X[idx_va], idx_tr, idx_va # return X_training, X_Validation in order



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def flatten_time_series(X):
    if X.ndim == 3:
        return X.reshape(-1, X.shape[-1])
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Unexpected X.ndim={X.ndim}")

def unflatten_time_series(X_flat, n_traj, T):
    d = X_flat.shape[1]
    return X_flat.reshape(n_traj, T, d)

# --------------------------------------------------------------
# Data matrix normalization
# --------------------------------------------------------------

def Normalization_3D(X_tr_raw, X_te_raw):
    n_tr, n_te, T = X_tr_raw.shape[0], X_te_raw.shape[0], X_tr_raw.shape[1]
    X_tr_flat = flatten_time_series(X_tr_raw)                  # (n_tr*T, d), that is, np.vstack(X). Stack the trajactories, from 3D to 2D
    X_te_flat = flatten_time_series(X_te_raw)                  # (n_te*T, d)
    X_train, X_test, X_mu, X_std = data_matrix_normalization(X_tr_flat, X_te_flat)
    # d_x = X_train.shape[1]

    # reshape back to 3D for clean per-trajectory pairing
    X_train_3d = unflatten_time_series(X_train, n_tr, T)       # (n_tr, T, d)
    X_test_3d  = unflatten_time_series(X_test,  n_te, T)       # (n_te, T, d)
    d_x = X_train_3d.shape[-1]
    return X_train_3d, X_test_3d, d_x

def Normalization_2D(X_tr_raw, X_te_raw):
    X_tr_flat = flatten_time_series(X_tr_raw)                  # (n_tr*T, d), that is, np.vstack(X). Stack the trajactories, from 3D to 2D
    X_te_flat = flatten_time_series(X_te_raw)                  # (n_te*T, d)
    X_train, X_test, X_mu, X_std = data_matrix_normalization(X_tr_flat, X_te_flat)
    return X_train, X_test, X_mu, X_std