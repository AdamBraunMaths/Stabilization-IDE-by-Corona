import numpy as np

# ============================================================
# USER PARAMETERS
# Example data from the article
# ============================================================

nx = 2

A_list = [
    np.array([[0.20, 0.05],
              [0.10, 0.15]], dtype=float),
    np.array([[0.10, 0.02],
              [0.03, 0.08]], dtype=float),
]

tau_list = [1.0, 2.0]

B_list = [
    np.array([1.0, 0.5], dtype=float),
    np.array([0.2, 1.2], dtype=float),
]

theta_list = [0.8, np.pi / 2]

tau_max = max(tau_list) if len(tau_list) else 0.0
theta_max = max(theta_list) if len(theta_list) else 0.0


def N_fun(eta):
    if eta < 0.0 or eta > tau_max:
        return np.zeros((2, 2))
    return np.array([
        [np.sin(eta), 0.04 * np.cos(eta)],
        [0.06 * np.sin(2 * eta), np.sin(eta)]
    ], dtype=float)


def M_fun(eta):
    if eta < 0.0 or eta > theta_max:
        return np.zeros(2)
    return np.array([
        np.sqrt(max(2.0 * eta, 0.0)),
        0.5 * np.sqrt(max(eta, 0.0))
    ], dtype=float)


T_max = 35.0
time_grid = np.linspace(0.0, T_max, 400)
dt = time_grid[1] - time_grid[0]

nu = 0.2

# Tikhonov regularization parameter in the least-squares solve.
lam = 1e-5

# Initial support index used for every gain at the first solve.
m_init = 180

# Number of alternating support minimization / least-squares iterations.
n_outer_iterations = 8

# Number of sweeps inside the support-minimization step.
n_independent_sweeps = 10

# Admissible degradation factors during support minimization.
tol_factor_weighted = 1.20
tol_factor_plain = 1.20