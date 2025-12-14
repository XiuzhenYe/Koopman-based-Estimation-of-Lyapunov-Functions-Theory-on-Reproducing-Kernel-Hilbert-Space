# predict_kedmd_multi.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist
from scipy.linalg import solve_discrete_lyapunov
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle
from Helper_OrganizeData import data_matrix_normalization, data_split, inverse_data_matrix_normalization
from Helper_OrganizeKernel import normalize_product_kernels
from Helper_Wendland_Kernel import wendland_kernel, pick_ell_for_sparsity, ensure_min_neighbors
from Helper_KernelGramMatrix import kernel_gram_matrix
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# ------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)

# ------------------------------------------------------------
# Helperss
# ------------------------------------------------------------ 
def kernel_row_normed(x_raw, X0_all_std, K_tr_all, s_tr_all, X_mu, X_std, c_bias, ell, d_x, gamma, eps_mass):
    x_std = (x_raw - X_mu) / (X_std + 1e-12)
    W_row = kernel_gram_matrix(x_std[None, :], X0_all_std, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell)
    s_row = W_row.sum(axis=1) + eps_mass
    K_row = (x_std @ X0_all_std.T + c_bias) * W_row
    _, K_row_n = normalize_product_kernels(K_tr_all, K_row, s_tr_all, s_row, gamma)
    return K_row_n

def flatten_time_series(X):
    if X.ndim == 3:
        n, T, d = X.shape
        return X.reshape(n*T, d)
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Unexpected X.ndim={X.ndim}")

def unflatten_time_series(X_flat, n_traj, T):
    d = X_flat.shape[1]
    return X_flat.reshape(n_traj, T, d)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
X_tr_raw = np.load("X_train.npy")  # (n_traj_train, T, 2)
X_te_raw = np.load("X_test.npy")   # (n_traj_test,  T, 2)
time_step = float(np.load("time_step.npy"))
mu = float(np.load("mu.npy"))

n_tr, T, d = X_tr_raw.shape
n_te, _, _ = X_te_raw.shape

# ------------------------------------------------------------
# Normalize using training stats (no leakage)
# ------------------------------------------------------------
X_tr_flat = flatten_time_series(X_tr_raw)                  # (n_tr*T, d)
X_te_flat = flatten_time_series(X_te_raw)                  # (n_te*T, d)
X_train, X_test, X_mu, X_std = data_matrix_normalization(X_tr_flat, X_te_flat)
d_x = X_train.shape[1]

# reshape back to per-trajectory for clean per-trajectory pairing
X_train_3d = unflatten_time_series(X_train, n_tr, T)       # (n_tr, T, d)
X_test_3d  = unflatten_time_series(X_test,  n_te, T)       # (n_te, T, d)

# ------------------------------------------------------------
# Build snapshot pairs on TRAIN, per trajectory (avoid cross-boundary pairs)
# ------------------------------------------------------------
X0_tr_list, X1_tr_list = [], []
for i in range(n_tr):
    Xi = X_train_3d[i]         # (T, d)
    X0_tr_list.append(Xi[:-1]) # (T-1, d)
    X1_tr_list.append(Xi[1:])  # (T-1, d)
X0_tr = np.vstack(X0_tr_list)  # (N_tr_pairs, d)
X1_tr = np.vstack(X1_tr_list)  # (N_tr_pairs, d)

# ------------------------------------------------------------
# CV split on the TRAIN pairs
# ------------------------------------------------------------
X0_tr_cv, X0_va_cv, idx_tr, idx_va = data_split(X0_tr, alpha=0.2)
X1_tr_cv, X1_va_cv = X1_tr[idx_tr], X1_tr[idx_va]

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

# # ------------------------------------------------------------
# # Hyperparameter search
# # ------------------------------------------------------------
# gamma_grid = np.arange(0.5, 1.00 + 1e-9, 0.025)
# c_grid     = np.arange(0.0,  0.6  + 1e-9, 0.1)
# eps_range  = np.logspace(-12, 2, 10)

# best = {"err": np.inf}
# for c_bias in c_grid:
#     K_tr = (X0_tr_cv @ X0_tr_cv.T + c_bias) * W_trtr
#     K_va = (X0_va_cv @ X0_tr_cv.T + c_bias) * W_vatr
#     for gamma in gamma_grid:
#         K_tr_n, K_va_n = normalize_product_kernels(K_tr, K_va, s_tr, s_va, gamma)
#         for eps in eps_range:
#             A = (K_tr_n + K_tr_n.T) * 0.5 + (eps+1e-10)*np.eye(K_tr_n.shape[0])
#             L = np.linalg.cholesky(A)
#             Z = np.linalg.solve(L, X1_tr_cv)
#             B = np.linalg.solve(L.T, Z)
#             X1_va_pred = K_va_n @ B
#             rel = np.linalg.norm(X1_va_pred - X1_va_cv) / (np.linalg.norm(X1_va_cv) + 1e-12)
#             if rel < best["err"]:
#                 best = {"err": rel, "eps": eps, "gamma": gamma, "c_bias": c_bias}
#     print(c_bias)

# print(f"[EDMD CV] best rel={best['err']:.6f}, eps={best['eps']:.3e}, gamma={best['gamma']}, c={best['c_bias']}")
# np.save("best.npy", best)

# ------------------------------------------------------------
# Final train on ALL training pairs, test rollouts on ALL test trajectories
# ------------------------------------------------------------
best = np.load("best.npy", allow_pickle=True).item()
c_best, gamma_best, eps_best = best["c_bias"], best["gamma"], best["eps"]
 
# neighbor coverage sanity check (test vs train inputs)
D_te_tr = cdist(X_test[:-1, :], X_train[:-1, :])
deg_te = np.sum(D_te_tr <= best_ell, axis=1)
print("test min/median/max neighbors within ell:", np.min(deg_te), np.median(deg_te), np.max(deg_te))

# ensure no poor tail
ell_fixed, deg_te = ensure_min_neighbors(X_train[:-1, :], X_test[:-1, :], best_ell, min_deg=5, step=1.15)
print("Adjusted ell:", ell_fixed, "  min/median/max deg:", np.min(deg_te), np.median(deg_te), np.max(deg_te))

# Build blocks on ALL training pairs + ALL test inputs
# (training pairs:)
W_tr_all = kernel_gram_matrix(X0_tr, X0_tr, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell_fixed)
# (test inputs vs training pair inputs:)
W_te_all = kernel_gram_matrix(X_test[:-1, :], X0_tr, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell_fixed)

K_tr_all = (X0_tr @ X0_tr.T + c_best) * W_tr_all
K_te_all = (X_test[:-1, :] @ X0_tr.T + c_best) * W_te_all

s_tr_all = W_tr_all.sum(axis=1) + 1e-12
s_te_all = W_te_all.sum(axis=1) + 1e-12
K_tr_all_n, K_te_all_n = normalize_product_kernels(K_tr_all, K_te_all, s_tr_all, s_te_all, gamma_best)


# ==============================
# Step 1: Get Theta = G_xx^-1, learning Koopman operator
# ==============================

# --- (i) Build W_11 and masses for X1 (same Wendland & ell)
W_11 = kernel_gram_matrix(X1_tr, X1_tr, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell_fixed)
s_1  = W_11.sum(axis=1) + 1e-12

# --- (ii) Cross Gram matrices between X1 and X0 (rows=X1, cols=X0),
#          then normalize to get rows normalized by s_1 and columns by s_tr_all.
W_10 = kernel_gram_matrix(X1_tr, X0_tr, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell_fixed)
K_10 = (X1_tr @ X0_tr.T + c_best) * W_10

# Normalize: returns (K_tr_all_n_again, K_10_n). We only need K_10_n.
K_tr_all_n_again, K_10_n = normalize_product_kernels(K_tr_all, K_10, s_tr_all, s_1, gamma_best)

# For EDMD we want A = K(X0, X1), which is the transpose of K_10 (since K_10 is rows=X1, cols=X0)
K_01_n = K_10_n.T  # shape: (N, N), matches K_tr_all_n
# --- (iii) Discrete-time Koopman finite matrix and eigenvalues
Gxx, Gamma = K_tr_all_n, K_01_n 

reg_theta = max(eps_best, 1e-5)   # start with 1e-6; you can try 1e-5 too
Gxx_reg = (Gxx + Gxx.T)*0.5 + reg_theta*np.eye(Gxx.shape[0])
Theta = np.linalg.solve(Gxx_reg, np.eye(Gxx.shape[0]))
print("Theta shape:", Theta.shape)

res_reg = np.linalg.norm(Gxx_reg @ Theta - np.eye(Gxx.shape[0]), ord='fro')
print("||Gxx_reg*Theta - I||_F =", res_reg) # this needs to be small, so that it means the inverse is numerically accurate
print("cond(Gxx_reg) =", np.linalg.cond(Gxx_reg))
print("min eig(Gxx_reg) =", np.min(np.linalg.eigvalsh(Gxx_reg)))


# ==============================
# Step 2: Build H (baseline choice)
# ==============================
X0_raw = inverse_data_matrix_normalization(X0_tr, X_mu, X_std) 
alpha = np.linalg.solve(Gxx_reg, X0_raw) 
H = alpha @ alpha.T
H = (H + H.T) * 0.5  # symmetrize

# ==============================
# Step 3: Compute M = Gamma*Theta 
# ==============================
M = np.linalg.solve(Gxx_reg, Gamma.T).T   
rho = np.max(np.abs(np.linalg.eigvals(M))) 
print("spectral radius rho(M) =", rho)

# ==============================
# Step 4: Solve for Pi
# ==============================
Pi = solve_discrete_lyapunov(M.T, H) 
# Check residual of the equation: M^T Pi M - Pi + H  ~= 0
res = np.linalg.norm(M.T @ Pi @ M - Pi + H, ord='fro')
print("Lyapunov residual ||M^T Pi M - Pi + H||_F =", res) # residual should be small (relative to ||H||_F)
# PSD check
lam_min = np.min(np.linalg.eigvalsh((Pi + Pi.T)*0.5))
print("min eigenvalue of sym(Pi) =", lam_min) 
Pi += np.abs(lam_min) * np.eye(Pi.shape[0]) # shift to ensure PSD

# ==============================
# Step 5: Evaluate the learned Lyapunov function hat{v}(x)
# ==============================
def vhat_from_xraw(x_raw):
    k_row = kernel_row_normed(x_raw, X0_tr, K_tr_all, s_tr_all, X_mu, X_std, c_best, ell_fixed, d_x, gamma_best, 1e-12)
    return (k_row @ Pi @ k_row.T).item() 

# ==============================
# Step 6: Contour plot of vhat(x)
# ==============================

# pick plotting bounds from training data (raw)
X0_raw = inverse_data_matrix_normalization(X0_tr, X_mu, X_std)
x1_min, x1_max = np.percentile(X0_raw[:,0], [1, 99])
x2_min, x2_max = np.percentile(X0_raw[:,1], [1, 99])
nx, ny = 101, 101
x1, x2 = np.linspace(-1.5, 1.5, nx), np.linspace(-1.5, 1.5, ny)
X1g, X2g = np.meshgrid(x1, x2)
V = np.zeros_like(X1g)
for i in range(ny):
    for j in range(nx):
        V[i, j] = vhat_from_xraw(np.array([X1g[i,j], X2g[i,j]])) 

# after V is filled
Vmin, Vmax = np.percentile(V, 1), np.percentile(V, 99)
print('Minimum and maximum of the estimated V:', Vmin, Vmax) 
levels = np.logspace(-1, 2, 10)
plt.figure(figsize=(6, 5))
cs = plt.contour(X1g, X2g, V, levels=levels)
plt.clabel(cs, inline=1, fontsize=7)
# plt.xlim(-1.5, 1.5), plt.ylim(-1.5, 1.5)
plt.xlabel(r"$x_1$"), plt.ylabel(r"$x_2$")
plt.tight_layout(), plt.savefig("vhat_contours.png", dpi=300)
plt.show()

# ==============================
# Step 7: True Lyapunov contours
# w(x) = |x|^2, so V(x) is the sum of squared states over time
# ==============================
Vtrue = np.zeros_like(X1g)
for i in range(ny):
    for j in range(nx): 
        # Simulate
        sol = solve_ivp(lambda t,x: [x[1], -x[1]-x[0]/(1.0+x[0]**2)], [0.0, 5.0], np.array([X1g[i,j], X2g[i,j]]), 
                        method='RK45', t_eval=np.arange(0.0, 5.0, time_step))
        Vtrue[i, j] = np.sum(np.sum(sol.y**2))
plt.figure(figsize=(6, 5))
levels = np.logspace(-1, 2, 10)
cs = plt.contour(X1g, X2g, Vtrue, levels=levels)
plt.clabel(cs, inline=1, fontsize=7)
plt.xlim(-1.5, 1.5), plt.ylim(-1.5, 1.5)
plt.xlabel(r"$x_1$"), plt.ylabel(r"$x_2$")
plt.tight_layout(), plt.savefig("vtrue_contours.png", dpi=300)
plt.show()