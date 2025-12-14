import os, sys
import numpy as np
from scipy.spatial.distance import cdist
 


def wendland_kernel(x, y, k=2, d_x=2, ell=None):
    if ell is None:
        raise ValueError("Provide a lengthscale ell.")
    r = np.linalg.norm(x - y) / ell
    if r >= 1.0:
        return 0.0
    ell_param = (d_x // 2) + k + 1
    u = 1.0 - r
    # polynomial for k=2
    P2 = ((ell_param**2 + 4*ell_param + 3)*r**2 + (3*ell_param + 6)*r + 3)/3.0
    return (u**(ell_param + 2)) * P2

def median_pairwise_dist(X):
    D = cdist(X, X)
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]; vals = vals[vals > 0]
    return np.median(vals) if vals.size else 1.0

def pick_ell_for_sparsity(Xs, d_x, sparsity_target=(0.3, 0.5), max_tries=8):
    """
    Heuristic: choose ell so Wendland Gram has ~30-50% zeros.
    """
    ell = median_pairwise_dist(Xs) # ρ(‖x−x′‖ / ell) with support on [0,1], Entries are exactly zero when ‖x−x′‖ > ell.  
    def build_W(ell_):             # start with midian_pairwise_dist, then Roughly half the pairs fall inside the support (nonzeros) and half outside (zeros) 
        return kernel_gram_matrix(Xs, Xs, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell_)
    W = build_W(ell); s = np.mean(W == 0.0)
    tries = 0
    while (s > sparsity_target[1] or s < sparsity_target[0]) and tries < max_tries:
        ell = ell * (2.0 if s > sparsity_target[1] else 0.5)
        W = build_W(ell); s = np.mean(W == 0.0); tries += 1
        print(f"[pick_ell] ell={ell:.4g}, sparsity={s:.3f}")
    return ell

def kernel_gram_matrix(X1, X2, linear_kernel, wendland_kernel, d_x, ell):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        xi = X1[i]
        for j in range(n2):
            xj = X2[j]
            K[i, j] = linear_kernel(xi, xj) * wendland_kernel(xi, xj, d_x=d_x, ell=ell)
    return K

def ensure_min_neighbors(X_train_in, X_test_in, ell, min_deg=5, step=1.2, max_steps=8):
    for _ in range(max_steps):
        D = cdist(X_test_in, X_train_in)
        deg = np.sum(D <= ell, axis=1)
        if np.min(deg) >= min_deg:
            return ell, deg
        ell *= step
    return ell, deg  # return whatever we ended with