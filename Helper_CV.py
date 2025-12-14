import numpy as np
from Helper_KRRviaCholesky import krr_predict_chol

def KRR_CV(
    X0_tr_cv, X1_tr_cv, X0_va_cv, X1_va_cv,
    wendland_kernel, normalize_product_kernels, kernel_gram_matrix,
    fixed_ell,
    d_x,
    gamma_grid=None, c_grid=None, eps_range=None,
    sparsity_target=(0.3, 0.5),
    eps_mass=1e-12,
    verbose=True
):
    """
    Cross-validate Kernel Ridge Regression (KRR) hyperparameters for the product kernel
    (linear × Wendland) with graph-style normalization.
    """
    # Step 1: choose kernel length scale ell
    best_ell = fixed_ell
    # Step 2: compute Wendland kernel blocks
    W_trtr = kernel_gram_matrix(X0_tr_cv, X0_tr_cv, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=best_ell)
    W_trva = kernel_gram_matrix(X0_va_cv, X0_tr_cv, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=best_ell)

    # Step 3: compute degree (mass) vectors
    s_tr = W_trtr.sum(axis=1) + eps_mass
    s_va = W_trva.sum(axis=1) + eps_mass

    # Step 4: hyperparameter search
    best = {"err": np.inf, "eps": None, "gamma": None, "c_bias": None, "ell": best_ell}

    for c_bias in c_grid:
        # Build product kernel (linear × Wendland)
        K_tr = (X0_tr_cv @ X0_tr_cv.T + c_bias) * W_trtr
        K_va = (X0_va_cv @ X0_tr_cv.T + c_bias) * W_trva

        for gamma in gamma_grid:
            # Normalize kernels
            K_tr_n, K_va_n = normalize_product_kernels(K_tr, K_va, s_tr, s_va, gamma)

            for eps in eps_range:
                # Predict validation next-states using KRR via Cholesky
                X1_va_pred = krr_predict_chol(K_tr_n, K_va_n, X1_tr_cv, eps)

                # Compute relative error
                rel = np.linalg.norm(X1_va_pred - X1_va_cv) / (np.linalg.norm(X1_va_cv) + 1e-12)

                if rel < best["err"]:
                    best.update({"err": rel, "eps": eps, "gamma": gamma, "c_bias": c_bias})

    return best
