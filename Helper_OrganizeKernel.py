import os, sys
import numpy as np

 

 

def normalize_product_kernels(K_tr, K_va, s_tr, s_va, gamma, tau=1e-8):
    """
    Wendland mass normalization with exponent gamma:
    K_tr_norm = D_tr^{-gamma} K_tr D_tr^{-gamma}
    K_va_norm = D_va^{-gamma} K_va D_tr^{-gamma}   (rows=test/val, cols=train)
    """
    Dtr = s_tr ** (-gamma)
    Dva = s_va ** (-gamma)
    K_tr_norm = (K_tr * Dtr[:, None]) * Dtr[None, :]
    K_va_norm = (K_va * Dtr[None, :]) * Dva[:, None]
    return K_tr_norm, K_va_norm