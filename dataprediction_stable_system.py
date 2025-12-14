# predict_kedmd_multi.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import cdist
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
# Helpers
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Hyperparameter search
# ------------------------------------------------------------
gamma_grid = np.arange(0.5, 1.00 + 1e-9, 0.025)
c_grid     = np.arange(0.0,  0.6  + 1e-9, 0.1)
eps_range  = np.logspace(-12, 2, 10)

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
np.save("best.npy", best)

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

# Solve for multi-output mapping X0_tr -> X1_tr
A = (K_tr_all_n + K_tr_all_n.T) * 0.5 + (eps_best + 1e-10) * np.eye(K_tr_all_n.shape[0])
L = np.linalg.cholesky(A)
Z = np.linalg.solve(L, X1_tr)
B = np.linalg.solve(L.T, Z)  # shape (N_tr_pairs, d)


# ==============================
# Learned Koopman spectrum (Kernel EDMD)
# ==============================
# We form G = K(X0,X0) and A = K(X0,X1) with the SAME kernel/normalization used for training,
# then approximate K ≈ (G + eps*I)^{-1} A and plot its eigenvalues.

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
# Use the SAME regularization used during training (eps_best).
Gn = K_tr_all_n            # from earlier normalize_product_kernels(...) on (X0,X0)
reg = eps_best + 1e-10
# Solve (Gn + reg*I)^{-1} A via numerically stable Cholesky solution 
A_mat = K_01_n
G_reg = (Gn + Gn.T) * 0.5 + reg * np.eye(Gn.shape[0])
Lk = np.linalg.cholesky(G_reg)
Y  = np.linalg.solve(Lk, A_mat)
K_tilde = np.linalg.solve(Lk.T, Y)

# --- (iv) Eigenvalues 
evals = np.linalg.eigvals(K_tilde)
K_tilde_adj = np.linalg.solve(G_reg, A_mat.T)
evals_adj = np.linalg.eigvals(K_tilde_adj)    # this is the dual Koopman operator, eigs are the same as K_tilde
 
# Spectrum 
fig, ax = plt.subplots(1, 2)
ax[0].scatter(evals_adj.real, evals_adj.imag, s=9, alpha=0.7)
# Theoretical spectrum
spect_points = [np.exp(time_step*(pp*complex(-1/2, np.sqrt(3)/2) + qq*complex(-1/2, -np.sqrt(3)/2))) 
                for pp in range(100) for qq in range(100)]
spect_points.pop(0) 
ax[1].scatter([pt.real for pt in spect_points], [pt.imag for pt in spect_points], color='k', s=9, alpha=0.5)
for i in range(2):
    ax[i].plot(np.cos(np.linspace(0, 2*np.pi, 300)), np.sin(np.linspace(0, 2*np.pi, 300)), 'k--', linewidth=0.5, alpha=0.5)
    ax[i].axhline(0, linewidth=0.5, alpha=0.8), ax[i].axvline(0, linewidth=0.5, alpha=0.4)
    ax[i].set_aspect('equal', 'box') 
    ax[i].set_xlim(-1.2, 1.2), ax[i].set_ylim(-1.2, 1.2)
    ax[i].set_xlabel(r'Re$(\lambda)$'), ax[i].set_ylabel(r'Im$(\lambda)$')
    ax[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("koopman_spectrum.png", dpi=300)


# ------------------------------------------------------------
def kernel_row_normed(x_raw, X0_all_std, K_tr_all, s_tr_all, X_mu, X_std, c_bias, ell, d_x, gamma, eps_mass):
    x_std = (x_raw - X_mu) / (X_std + 1e-12)
    W_row = kernel_gram_matrix(x_std[None, :], X0_all_std, lambda a,b: 1.0, wendland_kernel, d_x=d_x, ell=ell)
    s_row = W_row.sum(axis=1) + eps_mass
    K_row = (x_std @ X0_all_std.T + c_bias) * W_row
    _, K_row_n = normalize_product_kernels(K_tr_all, K_row, s_tr_all, s_row, gamma)
    return K_row_n

def rollout_from_x0(x0_raw, H,
                    X0_all_std, K_tr_all, s_tr_all, B,
                    X_mu, X_std, c_bias, ell, d_x, gamma, eps_mass):
    dloc = X0_all_std.shape[1]
    X_roll_raw = np.zeros((H+1, dloc))
    X_roll_raw[0] = x0_raw.copy()
    x_curr = x0_raw.copy()
    for k in range(1, H+1):
        K_row_n = kernel_row_normed(x_curr, X0_all_std, K_tr_all, s_tr_all,
                                    X_mu, X_std, c_bias, ell, d_x, gamma, eps_mass)
        x_next_std = K_row_n @ B
        x_next_raw = inverse_data_matrix_normalization(x_next_std, X_mu, X_std).reshape(-1)
        X_roll_raw[k] = x_next_raw
        x_curr = x_next_raw
    return X_roll_raw

# For rollouts, the “dictionary” inputs are the standardized training X0 nodes:
X0_all_std = X0_tr                                # already standardized
K_tr_all, _ = normalize_product_kernels(K_tr_all, K_tr_all, s_tr_all, s_tr_all, gamma_best)  # K_tr_all_n, but we reuse structure
K_tr_all = K_tr_all  # just to keep variable naming consistent in the function calls

# --- Closed-loop rollout for each TEST trajectory ---
H = T - 1
rollouts = np.zeros((n_te, H+1, d))
rmse_roll = np.zeros(n_te)

for i in range(n_te):
    x0_i = X_test_3d[i, 0]  # standardized initial state? No—this is RAW after inverse… Wait:
    # IMPORTANT: we want x0 in RAW units for plotting RMSE vs raw test states.
    # But kernel_row_normed expects raw and will standardize internally using X_mu/X_std.
    # X_test_3d is standardized. So convert it back to raw for the initial state:
    x0_raw = inverse_data_matrix_normalization(X_test_3d[i, 0], X_mu, X_std)
    X_roll_i = rollout_from_x0(x0_raw, H, X0_all_std, K_tr_all, s_tr_all, B, X_mu, X_std,
                               c_best, ell_fixed, d_x, gamma_best, 1e-12)
    rollouts[i] = X_roll_i

    # compute RMSE in raw units against raw ground truth:
    X_true_raw = inverse_data_matrix_normalization(X_test_3d[i], X_mu, X_std)
    err_i = X_true_raw[1:, :] - X_roll_i[1:, :]
    rmse_roll[i] = np.sqrt((err_i**2).sum(axis=1)).mean()

print("Closed-loop RMSE per trajectory:", rmse_roll)
print("Mean closed-loop RMSE:", rmse_roll.mean())

# --- Phase plot (true vs predicted rollout) ---
plt.figure(figsize=(6,5))
t_vec = np.arange(T) * time_step
norm = mpl.colors.Normalize(vmin=float(t_vec.min()), vmax=float(t_vec.max()))
cmap = plt.get_cmap('plasma')
for i in range(n_te):
    X_true_raw = inverse_data_matrix_normalization(X_test_3d[i], X_mu, X_std)
    # True trajectory (dashed gray)
    plt.plot(X_true_raw[:,0], X_true_raw[:,1], color="gray", lw=1, alpha=0.7, label="Actual" if i==0 else None)
    # Predicted scatter (time colored)
    sc = plt.scatter(rollouts[i,:,0], rollouts[i,:,1], c=t_vec, cmap=cmap, norm=norm, s=10, marker='o')
    # Start point (single dot, but colored consistently)
    plt.scatter(rollouts[i,0,0], rollouts[i,0,1], c=[t_vec[0]], cmap=cmap, norm=norm, s=40, marker='*')

cbar = plt.colorbar(sc)
cbar.set_label('Time')

# -----------------------------------------------------------------------
true_line = mlines.Line2D([], [], color='gray', lw=2, linestyle='--', label='Actual')
pred_points = mlines.Line2D([], [], marker='o', markersize = 5.0, linestyle='None', color='navy', label='Predicted')
start_point = mlines.Line2D([], [], marker='*', markersize = 5.0, linestyle='None', markerfacecolor='k', color='k', label='Initial')
plt.legend(handles=[true_line, pred_points, start_point], loc='best', fontsize=6)

plt.xlabel('$x_1$'), plt.ylabel('$x_2$'), plt.tight_layout()
plt.savefig("kedmd_trajectory.png", dpi=300), plt.show()