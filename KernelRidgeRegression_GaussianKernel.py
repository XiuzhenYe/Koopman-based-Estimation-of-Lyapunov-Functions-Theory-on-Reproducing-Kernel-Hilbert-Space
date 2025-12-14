import os, sys 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist 
script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) 
os.chdir(script_dir)

# --- data --- 
T_cut = np.load("T_cut.npy") 
x1_train = np.load("x1_train.npy"); x1_test = np.load("x1_test.npy") 
x2_train = np.load("x2_train.npy"); x2_test = np.load("x2_test.npy"); t_cutoff = np.load("t_cutoff.npy")

X_train = np.vstack([x1_train, x2_train]).T; X_test = np.vstack([x1_test, x2_test]).T
X_mu  = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-12
 
TT = X_train.shape[0] # full train length 
def rbf_kernel_gram_matrix(X1, X2, sigma=1.0): return np.exp(-cdist(X1, X2, 'sqeuclidean') / (2 * sigma**2))

# target
y_tr_x = np.linalg.norm(X_train, axis=1); y_te_x = np.linalg.norm(X_test, axis=1)
n = X_train.shape[0]; n_val = max(1, int(0.2 * n)); idx_tr = np.arange(0, n - n_val); idx_va = np.arange(n - n_val, n)

# data split
X_tr_cv, X_va_cv = X_train[idx_tr], X_train[idx_va] 
y_tr_cv, y_va_cv = y_tr_x[idx_tr], y_tr_x[idx_va] 
T_cv = X_tr_cv.shape[0]

# grids
eps_range = np.logspace(-5, -3, 10) 
sigma_range = np.logspace(-1, 1, 10)

best_err = float('inf') 
best_sigma_x, best_eps_x = None, None

for sigma in sigma_range: 
    K_trtr = rbf_kernel_gram_matrix(X_tr_cv, X_tr_cv, sigma) / T_cv 
    K_vatr = rbf_kernel_gram_matrix(X_va_cv, X_tr_cv, sigma) / T_cv 
    for eps in eps_range: 
        v = np.linalg.solve(K_trtr + eps * np.eye(T_cv), y_tr_cv) 
        y_pred = K_vatr @ v 
        rel_err = np.linalg.norm(y_pred - y_va_cv) / (np.linalg.norm(y_va_cv) + 1e-12) 
        if rel_err < best_err: 
            best_err = rel_err 
            best_eps_x = eps 
            best_sigma_x = sigma 

print('CV best — sigma:', best_sigma_x, 'eps:', best_eps_x, 'val rel err:', best_err)

K_trtr_all = rbf_kernel_gram_matrix(X_train, X_train, best_sigma_x) / TT 
K_tetr_all = rbf_kernel_gram_matrix(X_test, X_train, best_sigma_x) / TT 
v = np.linalg.solve(K_trtr_all + best_eps_x * np.eye(TT), y_tr_x) 
y_pred = K_tetr_all @ v

test_rel_err_x = np.linalg.norm(y_pred - y_te_x) / (np.linalg.norm(y_te_x) + 1e-12) 
print('X TEST rel err:', test_rel_err_x)

plt.figure(figsize=(10, 4)) 
plt.plot(t_cutoff, y_pred, marker='o', label='pred') 
plt.plot(t_cutoff, y_te_x, marker='x', label='true') 
plt.xlabel(r"$t\ \mathrm{(min)}$") 
plt.ylabel(r'$\|x_t\|$') 
plt.legend(); plt.tight_layout() 
plt.savefig(f"Gaussian Kernel with cutoff time {T_cut}.png", dpi=300, bbox_inches='tight') 
plt.show()