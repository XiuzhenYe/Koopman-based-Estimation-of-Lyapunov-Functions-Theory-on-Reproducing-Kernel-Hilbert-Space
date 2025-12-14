import os, sys
import numpy as np




# --- kernel gram matrix for linear * wendland kernel
def kernel_gram_matrix(X1, X2, linear_kernel, wendland_kernel, d_x, ell):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        xi = X1[i]
        for j in range(n2):
            xj = X2[j]
            K[i, j] = linear_kernel(xi, xj) * wendland_kernel(xi, xj, d_x=d_x, ell=ell)
    return K