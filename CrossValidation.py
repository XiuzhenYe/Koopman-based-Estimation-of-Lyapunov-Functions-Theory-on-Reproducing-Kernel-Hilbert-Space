import Helper_OrganizeData as HelpDataOrg 
from Helper_Wendland_Kernel import wendland_kernel, pick_ell_for_sparsity, ensure_min_neighbors
from Helper_KernelGramMatrix import kernel_gram_matrix
import numpy as np
from Helper_OrganizeKernel import normalize_product_kernels

def CV(X0_tr, X1_tr):
    X0_tr_cv, X0_va_cv, idx_tr, idx_va = HelpDataOrg.data_split(X0_tr, alpha=0.2)
    X1_tr_cv, X1_va_cv = X1_tr[idx_tr], X1_tr[idx_va]
    d_x = X0_tr_cv.shape[-1]
# ------------------------------------------------------------
# Choose ell on training inputs
# ------------------------------------------------------------
    best_ell = pick_ell_for_sparsity(X0_tr, d_x=d_x, sparsity_target=(0.3, 0.5))
# Pure Wendland blocks for CV
    W_trtr = kernel_gram_matrix(X0_tr_cv, X0_tr_cv, lambda a, b: 1.0, wendland_kernel, d_x=d_x, ell=best_ell)
    W_vatr = kernel_gram_matrix(X0_va_cv, X0_tr_cv, lambda a, b: 1.0, wendland_kernel, d_x=d_x, ell=best_ell)

    eps_mass = 1e-12
    s_tr = W_trtr.sum(axis=1) + eps_mass
    s_va = W_vatr.sum(axis=1) + eps_mass
    #=================== Hyperparameter search ==========================================
    gamma_grid = np.arange(0.5, 1.00 + 1e-9, 0.025)
    c_grid     = np.arange(0.0,  0.6  + 1e-9, 0.1)
    eps_range  = np.logspace(-12, 2, 30)
    best = {"err": np.inf}
    for c_bias in c_grid:
        K_tr = (X0_tr_cv @ X0_tr_cv.T + c_bias) * W_trtr
        K_va = (X0_va_cv @ X0_tr_cv.T + c_bias) * W_vatr
        for gamma in gamma_grid:
            K_tr_n, K_va_n = normalize_product_kernels(K_tr, K_va, s_tr, s_va, gamma)
            for eps in eps_range:
                A = (K_tr_n + K_tr_n.T) * 0.5 + (eps+1e-10)*np.eye(K_tr_n.shape[0])
                L = np.linalg.cholesky(A)
                Z = np.linalg.solve(L, X1_tr_cv)
                B = np.linalg.solve(L.T, Z)
                X1_va_pred = K_va_n @ B
                rel = np.linalg.norm(X1_va_pred - X1_va_cv) / (np.linalg.norm(X1_va_cv) + 1e-12)
                if rel < best["err"]:
                    best = {"err": rel, "eps": eps, "gamma": gamma, "c_bias": c_bias}
        print(f"[EDMD CV] best rel={best['err']:.6f}, eps={best['eps']:.3e}, gamma={best['gamma']}, c={best['c_bias']}")
    return best, best_ell
 