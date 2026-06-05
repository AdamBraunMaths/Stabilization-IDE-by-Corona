import numpy as np
import matplotlib.pyplot as plt
import itertools
from parameters import (
    nx,
    A_list,
    tau_list,
    B_list,
    theta_list,
    N_fun,
    M_fun,
    T_max,
    time_grid,
    dt,
    nu,
    lam,
    m_init,
    n_outer_iterations,
    n_independent_sweeps,
    tol_factor_weighted,
    tol_factor_plain,
)
# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.size": 11,
    "axes.labelsize": 20,
    "axes.titlesize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16
})



# ============================================================
# BASIC HELPERS
# ============================================================
def delay_to_steps(delay, dt):
    return int(np.rint(delay / dt))

def conv_trunc(a, b, L):
    return np.convolve(a, b)[:L]

def build_conv_matrix(mu):
    L = len(mu)
    C = np.zeros((L, L))
    for i in range(L):
        C[i, :i+1] = mu[:i+1][::-1]
    return C

def laplace_of_measure_samples(mu, time_grid, z):
    t = np.asarray(time_grid, dtype=float)
    return np.sum(mu * np.exp(-z * t))

def laplace_of_function_samples(f, time_grid, z):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    return dt * np.sum(f * np.exp(-z * t))

def permutation_sign(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inv += 1
    return -1 if inv % 2 else 1


# ============================================================
# DETERMINANT / MINORS IN THE CONVOLUTION ALGEBRA
# Each matrix entry is a measure array; multiplication is conv.
# ============================================================
def determinant_of_measure_matrix(M_entries, L):
    n = len(M_entries)
    out = np.zeros(L)
    for perm in itertools.permutations(range(n)):
        sign = permutation_sign(perm)
        term = np.zeros(L)
        term[0] = 1.0  # delta_0
        for i in range(n):
            term = conv_trunc(term, M_entries[i][perm[i]], L)
        out += sign * term
    return out

def minor_measure_of_augmented(q_entries, p_entries, remove_col, L):
    """
    [q,-p] is nx x (nx+1), remove column remove_col (0-based),
    then take determinant in the convolution algebra.
    """
    n = len(q_entries)
    aug = []
    for i in range(n):
        row = [q_entries[i][j] for j in range(n)]
        row.append(-p_entries[i])
        aug.append(row)

    sub = []
    for i in range(n):
        row = []
        for j in range(n + 1):
            if j != remove_col:
                row.append(aug[i][j])
        sub.append(row)

    return determinant_of_measure_matrix(sub, L)


# ============================================================
# BUILD Delta0, q, p AS MEASURE ARRAYS
# ============================================================
def build_Delta0_entries(A_list, tau_list, time_grid, nx):
    L = len(time_grid)
    Delta0 = [[np.zeros(L) for _ in range(nx)] for _ in range(nx)]

    for i in range(nx):
        Delta0[i][i][0] = 1.0

    for A, tau in zip(A_list, tau_list):
        k = delay_to_steps(tau, time_grid[1] - time_grid[0])
        if 0 <= k < L:
            for i in range(nx):
                for j in range(nx):
                    Delta0[i][j][k] += -A[i, j]
    return Delta0

def build_q_entries(A_list, tau_list, N_fun, time_grid, nx):
    Delta0 = build_Delta0_entries(A_list, tau_list, time_grid, nx)
    dt = time_grid[1] - time_grid[0]

    q = [[Delta0[i][j].copy() for j in range(nx)] for i in range(nx)]

    for m, tm in enumerate(time_grid):
        Nij = N_fun(tm)
        for i in range(nx):
            for j in range(nx):
                q[i][j][m] += -dt * Nij[i, j]

    return q, Delta0

def build_p_entries(B_list, theta_list, M_fun, time_grid, nx):
    L = len(time_grid)
    dt = time_grid[1] - time_grid[0]

    p = [np.zeros(L) for _ in range(nx)]

    for B, th in zip(B_list, theta_list):
        k = delay_to_steps(th, dt)
        if 0 <= k < L:
            for i in range(nx):
                p[i][k] += B[i]

    for m, tm in enumerate(time_grid):
        Mv = M_fun(tm)
        for i in range(nx):
            p[i][m] += dt * Mv[i]

    return p


# ============================================================
# BUILD ALL R_j AND NTILDE FOR ARBITRARY nx
# ============================================================
def build_corona_objects(A_list, tau_list, B_list, theta_list, N_fun, M_fun, time_grid, nx):
    L = len(time_grid)
    dt = time_grid[1] - time_grid[0]

    q_entries, Delta0_entries = build_q_entries(A_list, tau_list, N_fun, time_grid, nx)
    p_entries = build_p_entries(B_list, theta_list, M_fun, time_grid, nx)

    R_list = []
    for j in range(nx + 1):
        R_list.append(minor_measure_of_augmented(q_entries, p_entries, remove_col=j, L=L))

    det_q = determinant_of_measure_matrix(q_entries, L)
    det_D0 = determinant_of_measure_matrix(Delta0_entries, L)

    Ntilde_mass = det_q - det_D0
    Ntilde_fun = Ntilde_mass / dt

    return {
        "R_list": R_list,
        "det_q": det_q,
        "det_D0": det_D0,
        "Ntilde_mass": Ntilde_mass,
        "Ntilde_fun": Ntilde_fun,
        "q_entries": q_entries,
        "p_entries": p_entries,
        "Delta0_entries": Delta0_entries,
    }


# ============================================================
# SIGNS IN THE CORONA EQUATION
# T(v)=sum_{j=1}^{nx+1} (-1)^{nx+1+j} R_j * v_j
# with v=(g1,...,gnx,f)
# ============================================================
def corona_signs(nx):
    return np.array([(-1)**(nx + 1 + (j + 1)) for j in range(nx + 1)], dtype=float)


# ============================================================
# WEIGHTED LS FOR ARBITRARY nx
# v_list = [g1,...,gnx,f]
# ============================================================
def solve_weighted_multi_ls(R_list, Ntilde_fun, nu, m_list, lam, time_grid, nx):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    w = np.exp(nu * t) * np.sqrt(dt)
    signs = corona_signs(nx)

    blocks = []
    for j in range(nx + 1):
        Cj = build_conv_matrix(signs[j] * R_list[j])[:, :m_list[j] + 1]
        blocks.append(Cj)

    A = np.hstack(blocks)
    b = Ntilde_fun.copy()

    W = w[:, None]
    Aw = W * A
    bw = w * b

    AtA = Aw.T @ Aw
    rhs = Aw.T @ bw
    if lam > 0:
        AtA += lam * np.eye(AtA.shape[0])

    x = np.linalg.solve(AtA, rhs)

    L = len(Ntilde_fun)
    v_list = []
    pos = 0
    for j in range(nx + 1):
        mj = m_list[j]
        v = np.zeros(L)
        v[:mj+1] = x[pos:pos+mj+1]
        pos += mj + 1
        v_list.append(v)

    approx_fun = np.zeros(L)
    for j in range(nx + 1):
        approx_fun += build_conv_matrix(R_list[j]) @ (signs[j] * v_list[j])

    err = approx_fun - Ntilde_fun
    weighted_res = np.sqrt(np.sum((w * err)**2))
    plain_res = np.linalg.norm(err)

    return v_list, approx_fun, weighted_res, plain_res


# ============================================================
# SUPPORT MINIMIZATION FOR ARBITRARY nx
# ============================================================
def trim_kernel_to_support(arr, m):
    out = arr.copy()
    if m + 1 < len(out):
        out[m+1:] = 0.0
    return out

def evaluate_multi_residuals(R_list, Ntilde_fun, v_list, nu, time_grid, nx):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    w = np.exp(nu * t) * np.sqrt(dt)
    signs = corona_signs(nx)

    approx_fun = np.zeros(len(Ntilde_fun))
    for j in range(nx + 1):
        approx_fun += build_conv_matrix(R_list[j]) @ (signs[j] * v_list[j])

    err = approx_fun - Ntilde_fun
    weighted_res = np.sqrt(np.sum((w * err)**2))
    plain_res = np.linalg.norm(err)
    return approx_fun, weighted_res, plain_res

def smallest_support_binary(which, R_list, Ntilde_fun, v_list, m_list,
                            nu, time_grid, weighted_tol, plain_tol, nx):
    lo, hi = 0, m_list[which]
    best_m = m_list[which]
    best_v = trim_kernel_to_support(v_list[which], m_list[which])

    while lo <= hi:
        mid = (lo + hi) // 2
        v_try = [v.copy() for v in v_list]
        v_try[which] = trim_kernel_to_support(v_try[which], mid)

        _, wres, pres = evaluate_multi_residuals(R_list, Ntilde_fun, v_try, nu, time_grid, nx)

        if wres <= weighted_tol and pres <= plain_tol:
            best_m = mid
            best_v = v_try[which]
            hi = mid - 1
        else:
            lo = mid + 1

    return best_v, best_m

def minimize_supports_independently(R_list, Ntilde_fun, v_init, m_init,
                                    nu, time_grid, weighted_tol, plain_tol,
                                    nx, n_sweeps=10, verbose=True):
    v_list = [trim_kernel_to_support(v_init[j], m_init[j]) for j in range(nx + 1)]
    m_list = list(m_init)

    approx_fun, wres, pres = evaluate_multi_residuals(R_list, Ntilde_fun, v_list, nu, time_grid, nx)
    if wres > weighted_tol or pres > plain_tol:
        raise ValueError(f"Initial solution does not satisfy tolerances: weighted={wres}, plain={pres}")

    for sweep in range(n_sweeps):
        old = tuple(m_list)

        for j in range(nx + 1):
            v_list[j], m_list[j] = smallest_support_binary(
                j, R_list, Ntilde_fun, v_list, m_list,
                nu, time_grid, weighted_tol, plain_tol, nx
            )

        approx_fun, wres, pres = evaluate_multi_residuals(R_list, Ntilde_fun, v_list, nu, time_grid, nx)

        if verbose:
            print(f"    sweep {sweep+1}: supports={m_list}, weighted={wres:.3e}, plain={pres:.3e}")

        if tuple(m_list) == old:
            break

    return v_list, m_list, approx_fun, wres, pres


# ============================================================
# HISTORY HELPERS
# ============================================================
def make_history_function_scalar(t_hist, y_hist):
    t_hist = np.asarray(t_hist, dtype=float)
    y_hist = np.asarray(y_hist, dtype=float)
    tmin = t_hist[0]
    tmax = t_hist[-1]

    def h(t):
        if t < tmin - 1e-12 or t > tmax + 1e-12:
            raise ValueError(f"scalar history queried outside range: t={t}, range=[{tmin},{tmax}]")
        return float(np.interp(t, t_hist, y_hist))
    return h

def make_history_function_vector(t_hist, Y_hist):
    t_hist = np.asarray(t_hist, dtype=float)
    Y_hist = np.asarray(Y_hist, dtype=float)
    tmin = t_hist[0]
    tmax = t_hist[-1]
    n = Y_hist.shape[1]

    def h(t):
        if t < tmin - 1e-12 or t > tmax + 1e-12:
            raise ValueError(f"vector history queried outside range: t={t}, range=[{tmin},{tmax}]")
        return np.array([np.interp(t, t_hist, Y_hist[:, k]) for k in range(n)], dtype=float)
    return h


# ============================================================
# SAMPLE KERNELS
# ============================================================
def sample_matrix_kernel_on_grid(kernel_fun, support, dt, shape):
    mmax = delay_to_steps(support, dt)
    vals = np.zeros((mmax + 1,) + shape)
    for m in range(mmax + 1):
        vals[m] = kernel_fun(m * dt)
    return vals, mmax


# ============================================================
# OPEN-LOOP PRINCIPAL PART
# ============================================================
def simulate_openloop_principal(time_grid, A_list, tau_list, x_hist_fun, nx):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    L = len(t)

    tau_steps = [delay_to_steps(tau, dt) for tau in tau_list]
    X = np.zeros((L, nx), dtype=float)

    for i in range(L):
        ti = t[i]
        xi = np.zeros(nx, dtype=float)
        for A, d in zip(A_list, tau_steps):
            j = i - d
            if j >= 0:
                xi += A @ X[j]
            else:
                xi += A @ x_hist_fun(ti - d * dt)
        X[i] = xi
    return X


# ============================================================
# OPEN-LOOP FULL HOMOGENEOUS
# ============================================================
def simulate_openloop_full_homogeneous(time_grid, A_list, tau_list, N_fun, x_hist_fun, nx):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    L = len(t)

    N_samp, mN = sample_matrix_kernel_on_grid(N_fun, max(tau_list), dt, (nx, nx))
    tau_steps = [delay_to_steps(tau, dt) for tau in tau_list]
    X = np.zeros((L, nx), dtype=float)

    for i in range(L):
        ti = t[i]
        rhs = np.zeros(nx, dtype=float)

        for A, d in zip(A_list, tau_steps):
            j = i - d
            if j >= 0:
                rhs += A @ X[j]
            else:
                rhs += A @ x_hist_fun(ti - d * dt)

        for m in range(1, mN + 1):
            j = i - m
            if j >= 0:
                rhs += dt * (N_samp[m] @ X[j])
            else:
                rhs += dt * (N_samp[m] @ x_hist_fun(ti - m * dt))

        MAT = np.eye(nx) - dt * N_samp[0]
        X[i] = np.linalg.solve(MAT, rhs)

    return X


# ============================================================
# CLOSED-LOOP SIMULATION FOR ARBITRARY nx
# v_list = [g1,...,gnx,f]
# ============================================================
def simulate_closedloop_discrete(time_grid, A_list, tau_list, B_list, theta_list,
                                 N_fun, M_fun, v_list, m_list,
                                 x_hist_fun, u_hist_fun, nx):
    t = np.asarray(time_grid, dtype=float)
    dt = t[1] - t[0]
    L = len(t)

    N_samp, mN = sample_matrix_kernel_on_grid(N_fun, max(tau_list), dt, (nx, nx))
    M_samp, mM = sample_matrix_kernel_on_grid(M_fun, max(theta_list), dt, (nx,))
    tau_steps = [delay_to_steps(tau, dt) for tau in tau_list]
    theta_steps = [delay_to_steps(th, dt) for th in theta_list]

    X = np.zeros((L, nx), dtype=float)
    U = np.zeros(L, dtype=float)

    g_list = v_list[:nx]
    f_arr = v_list[nx]
    mg_list = m_list[:nx]
    mf = m_list[nx]

    for i in range(L):
        ti = t[i]
        rhs_x = np.zeros(nx, dtype=float)

        for A, d in zip(A_list, tau_steps):
            xpast = X[i-d] if (i-d) >= 0 else x_hist_fun(ti - d * dt)
            rhs_x += A @ xpast

        for m in range(1, mN + 1):
            xpast = X[i-m] if (i-m) >= 0 else x_hist_fun(ti - m * dt)
            rhs_x += dt * (N_samp[m] @ xpast)

        for B, d in zip(B_list, theta_steps):
            upast = U[i-d] if (i-d) >= 0 else u_hist_fun(ti - d * dt)
            rhs_x += B * upast

        for m in range(1, mM + 1):
            upast = U[i-m] if (i-m) >= 0 else u_hist_fun(ti - m * dt)
            rhs_x += dt * (M_samp[m] * upast)

        rhs_u = 0.0
        for k in range(nx):
            for m in range(1, mg_list[k] + 1):
                rhs_u += g_list[k][m] * (X[i-m, k] if i-m >= 0 else x_hist_fun(ti - m * dt)[k])

        for m in range(1, mf + 1):
            rhs_u += f_arr[m] * (U[i-m] if i-m >= 0 else u_hist_fun(ti - m * dt))

        rhs_u *= dt

        A11 = np.eye(nx) - dt * N_samp[0]
        A12 = -dt * M_samp[0].reshape(nx, 1)
        A21 = -dt * np.array([[g_list[k][0] for k in range(nx)]], dtype=float)
        A22 = np.array([[1.0 - dt * f_arr[0]]], dtype=float)

        MAT = np.block([[A11, A12],
                        [A21, A22]])
        RHS = np.concatenate([rhs_x, np.array([rhs_u])])

        sol = np.linalg.solve(MAT, RHS)
        X[i, :] = sol[:nx]
        U[i] = sol[nx]

    return X, U


# ============================================================
# VISUALIZATION
# ============================================================
def visualize_corona(time_grid, Ntilde_fun, approx_fun, v_list, S_list, nx):
    res = approx_fun - Ntilde_fun

    print("||Ntilde_fun||_2 =", np.linalg.norm(Ntilde_fun))
    print("||res||_2        =", np.linalg.norm(res))
    for k in range(nx):
        print(f"||g{k+1}||_2        =", np.linalg.norm(v_list[k]))
    print("||f||_2          =", np.linalg.norm(v_list[nx]))
    print("Supports =", S_list)

    # Compact controller-gain figure
    fig, ax = plt.subplots(figsize=(5.2, 3.0))

    for k in range(nx):
        ax.plot(time_grid, v_list[k], linewidth=1.4, label=rf"$g_{k+1}(t)$")

    ax.plot(time_grid, v_list[nx], linewidth=1.4, label=r"$f(t)$")

    # Vertical dotted lines for compact supports
    for k in range(nx):
        ax.axvline(
            S_list[k],
            linestyle=":",
            linewidth=1.4,
            label=rf"$S_{k+1}={S_list[k]:.3g}$"
        )

    ax.axvline(
        S_list[nx],
        linestyle=":",
        linewidth=1.4,
        label=rf"$S_{{{nx+1}}}={S_list[nx]:.3g}$"
    )

    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, ncol=2, frameon=True, loc="best")

    ax.set_xlabel(r"$t$", fontsize=10)

    # Force compact tick labels despite global rcParams
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=8)

    # Also shrink scientific-notation offset text if it appears
    ax.xaxis.get_offset_text().set_fontsize(9)
    ax.yaxis.get_offset_text().set_fontsize(9)

    fig.tight_layout(pad=0.6)
    plt.show()
# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    data = build_corona_objects(A_list, tau_list, B_list, theta_list, N_fun, M_fun, time_grid, nx)

    R_list = data["R_list"]
    Ntilde_fun = data["Ntilde_fun"]

    m_list = [m_init] * (nx + 1)

    v_list, approx_fun, weighted_res, plain_res = solve_weighted_multi_ls(
        R_list, Ntilde_fun, nu=nu, m_list=m_list, lam=lam, time_grid=time_grid, nx=nx
    )

    print("\nInitial large-support solve")
    print("weighted residual =", weighted_res)
    print("plain residual    =", plain_res)
    print("supports =", m_list)

    for outer in range(n_outer_iterations):
        print(f"\n=== Outer iteration {outer+1} ===")
        weighted_tol = tol_factor_weighted * weighted_res
        plain_tol = tol_factor_plain * plain_res

        print("Current tolerances:")
        print("  weighted_tol =", weighted_tol)
        print("  plain_tol    =", plain_tol)

        v_trim, m_new, approx_trim, wres_trim, pres_trim = minimize_supports_independently(
            R_list, Ntilde_fun, v_list, m_list,
            nu, time_grid, weighted_tol, plain_tol,
            nx, n_sweeps=n_independent_sweeps, verbose=True
        )

        print("After independent support minimization:")
        print("  supports =", m_new)
        print("  weighted residual =", wres_trim)
        print("  plain residual    =", pres_trim)

        v_res, approx_res, wres_res, pres_res = solve_weighted_multi_ls(
            R_list, Ntilde_fun, nu=nu, m_list=m_new, lam=lam, time_grid=time_grid, nx=nx
        )

        print("After re-solving on reduced supports:")
        print("  weighted residual =", wres_res)
        print("  plain residual    =", pres_res)

        if tuple(m_new) == tuple(m_list):
            v_list, approx_fun = v_res, approx_res
            weighted_res, plain_res = wres_res, pres_res
            print("Supports unchanged; stopping outer loop.")
            break

        m_list = m_new
        v_list, approx_fun = v_res, approx_res
        weighted_res, plain_res = wres_res, pres_res

    S_list = [time_grid[m] for m in m_list]

    print("\nFinal result")
    print("weighted residual =", weighted_res)
    print("plain residual    =", plain_res)
    print("supports =", m_list)
    print("support times =", S_list)

    visualize_corona(time_grid, Ntilde_fun, approx_fun, v_list, S_list, nx)

    D_max = max(max(tau_list), max(theta_list), max(S_list))
    t_hist = np.arange(-D_max, 0.0 + 1e-12, dt)

    X0_hist = np.zeros((len(t_hist), nx), dtype=float)
    X0_hist[:, 0] = 0.2 * np.cos(2.0 * t_hist) * np.exp(t_hist / 3.0)
    if nx >= 2:
        X0_hist[:, 1] = -0.15 * np.sin(1.5 * t_hist) * np.exp(t_hist / 4.0)
    if nx >= 3:
        X0_hist[:, 2] = 0.1 * np.cos(1.2 * t_hist) * np.exp(t_hist / 5.0)
    for k in range(3, nx):
        X0_hist[:, k] = 0.05 * np.cos((1.0 + 0.2*k) * t_hist) * np.exp(t_hist / (4.0 + k))

    U0_hist = np.zeros(len(t_hist), dtype=float)

    x_hist_fun = make_history_function_vector(t_hist, X0_hist)
    u_hist_fun = make_history_function_scalar(t_hist, U0_hist)

    X_principal = simulate_openloop_principal(time_grid, A_list, tau_list, x_hist_fun, nx)
    X_open = simulate_openloop_full_homogeneous(time_grid, A_list, tau_list, N_fun, x_hist_fun, nx)
    Xcl, Ucl = simulate_closedloop_discrete(
        time_grid, A_list, tau_list, B_list, theta_list,
        N_fun, M_fun, v_list, m_list,
        x_hist_fun, u_hist_fun, nx
    )

    print("\nClosed-loop diagnostics")
    print("max ||Xcl|| =", np.max(np.linalg.norm(Xcl, axis=1)))
    print("max |Ucl|   =", np.max(np.abs(Ucl)))

    # one-line determinant diagnostic
    omega = np.linspace(-20.0, 20.0, 1500)
    vals = []
    for w in omega:
        z = 0.05 + 1j*w
        detA = (
            laplace_of_measure_samples(data["det_D0"], time_grid, z)
            + laplace_of_measure_samples(data["Ntilde_mass"], time_grid, z)
            - laplace_of_function_samples(approx_fun, time_grid, z)
        )
        vals.append(abs(detA))
    print(f"min |detA(0.05+i w)| = {np.min(vals):.6e}")

                     # ============================================================
        # ============================================================
    # DYNAMICS: OPEN-LOOP / PRINCIPAL AND CLOSED-LOOP LOG-SCALE
    # Same vertical log scale on both subplots
    # ============================================================
    eps_log = 1e-14  # avoids log(0)

    # Common log-scale limits for fair comparison
    open_vals = []
    closed_vals = []

    for k in range(nx):
        open_vals.append(np.abs(X_principal[:, k]) + eps_log)
        open_vals.append(np.abs(X_open[:, k]) + eps_log)
        closed_vals.append(np.abs(Xcl[:, k]) + eps_log)

    closed_vals.append(np.abs(Ucl) + eps_log)

    all_vals = np.concatenate(open_vals + closed_vals)

    ymin = np.min(all_vals)
    ymax = np.max(all_vals)

    # Add a small margin on the log scale
    ymin = 10 ** (np.floor(np.log10(ymin)))
    ymax = 10 ** (np.ceil(np.log10(ymax)))

    fig, axs = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True)

    # Top subplot: open-loop and principal part in log scale
    for k in range(nx):
        axs[0].semilogy(
            time_grid,
            np.abs(X_principal[:, k]) + eps_log,
            linestyle="--",
            linewidth=1.4,
            label=rf"$|x_{k+1}|$ principal"
        )

    for k in range(nx):
        axs[0].semilogy(
            time_grid,
            np.abs(X_open[:, k]) + eps_log,
            linewidth=1.4,
            label=rf"$|x_{k+1}|$ open-loop"
        )

    axs[0].set_ylim(ymin, ymax)
    axs[0].grid(True, which="major", axis="y", alpha=0.35)
    axs[0].grid(True, which="major", axis="x", alpha=0.25)
    axs[0].grid(True, which="minor", axis="x", alpha=0.12)
    axs[0].legend(fontsize=9, ncol=2, frameon=True, loc="best")
    axs[0].set_ylabel("")

    # Bottom subplot: closed-loop components in log scale
    for k in range(nx):
        axs[1].semilogy(
            time_grid,
            np.abs(Xcl[:, k]) + eps_log,
            linewidth=1.4,
            label=rf"$|x_{k+1}|$ closed-loop"
        )

    axs[1].semilogy(
        time_grid,
        np.abs(Ucl) + eps_log,
        linewidth=1.4,
        label=r"$|U|$ closed-loop",
        linestyle=":"
    )

    axs[1].set_ylim(ymin, ymax)
    axs[1].grid(True, which="major", axis="y", alpha=0.35)
    axs[1].grid(True, which="major", axis="x", alpha=0.25)
    axs[1].grid(True, which="minor", axis="x", alpha=0.12)

    axs[1].legend(fontsize=9, ncol=2, frameon=True, loc="best")
    axs[1].set_xlabel(r"$t$", fontsize=10)
    axs[1].set_ylabel("")

    # Force compact tick labels on both subplots
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)

        # Shrink scientific/log exponent text
        ax.xaxis.get_offset_text().set_fontsize(9)
        ax.yaxis.get_offset_text().set_fontsize(9)

    fig.tight_layout(pad=0.6)
    plt.show()