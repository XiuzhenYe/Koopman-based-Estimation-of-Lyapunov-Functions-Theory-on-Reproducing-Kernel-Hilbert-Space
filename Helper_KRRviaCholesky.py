import os, sys
import numpy as np




def krr_predict_chol(Ktt, y_train, KtT, eps, jitter=1e-10):
    """
    Stable KRR via Cholesky.
    """
    A = (Ktt + Ktt.T) * 0.5 + (eps + jitter) * np.eye(Ktt.shape[0])
    L = np.linalg.cholesky(A)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    return KtT @ alpha, alpha