import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# System: Liénard system in the Koopman–Lyapunov example
#   x1_dot = x2
#   x2_dot = -x2 - gamma(x1)
#   gamma(x1) = x1 / (1 + x1^2)
# ------------------------------------------------------------

def gamma_fun(x1):
    return x1 / (1.0 + x1**2)
def lienard_ode(t, x):
    x1, x2 = x
    dx1 = x2
    dx2 = -x2 - gamma_fun(x1)
    return [dx1, dx2]

# ------------------------------------------------------------
# Single-trajectory simulator
# ------------------------------------------------------------

def simulate_trajectory(x0, t_final=20.0, dt=0.1):
    t_span = (0.0, t_final)
    t_eval = np.arange(0.0, t_final + 1e-12, dt)

    sol = solve_ivp(
        fun=lienard_ode,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("solve_ivp failed: " + sol.message)
    # sol.y shape = (2, T)
    return t_eval, sol.y

# ------------------------------------------------------------
# Multi-trajectory simulation (return X with shape (n_traj, T, 2))
# ------------------------------------------------------------

def simulate_multiple_trajectories(n_traj=20, x1_range=(-1.0, 1.0), x2_range=(-1.0, 1.0),
    t_final=20.0, dt=0.1, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    x0_list = []
    traj_list = []
    for _ in range(n_traj):
        x0 = np.array([rng.uniform(*x1_range), rng.uniform(*x2_range)], dtype=float)
        t_eval, X_traj = simulate_trajectory(x0, t_final=t_final, dt=dt)
        # X_traj is (2, T) → convert to (T, 2) for stacking
        traj_list.append(X_traj.T)
        x0_list.append(x0)
    # Stack into one array: (n_traj, T, 2)
    X = np.stack(traj_list, axis=0)
    return t_eval, X, x0_list

# ------------------------------------------------------------
# Run and plot
# ------------------------------------------------------------
 
if __name__ == "__main__":
    n_traj, t_final, time_step = 50, 5.0, 0.2
    x1_range, x2_range = (-2.0, 2.0), (-2.0, 2.0)
    t_eval, X, x0_list = simulate_multiple_trajectories(n_traj=n_traj, x1_range=x1_range, x2_range=x2_range,
        t_final=t_final, dt=time_step, rng_seed=5)

    np.save("X_train.npy", X) 
    np.save("time_step.npy", time_step)
    np.save("t_multi.npy", t_eval)
    plt.figure(figsize=(6, 6))
    for i in range(n_traj):
        plt.plot(X[i, :, 0], X[i, :, 1], color='k', markersize=4, marker='o', alpha=0.3)
    plt.scatter(0, 0, marker="x", s=80)
    plt.xlabel("$x_1$"), plt.ylabel("$x_2$")
    plt.tight_layout(), plt.show()

    t_eval, X, x0_list = simulate_multiple_trajectories(n_traj=20, x1_range=x1_range, x2_range=x2_range,
        t_final=t_final, dt=time_step, rng_seed=15) 
    np.save("X_test.npy", X) 