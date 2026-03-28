import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.optimize import differential_evolution, minimize


def fold_and_average(p, M, P_inf, p_c=0.5, P_inf_sem=None):
    """
    Fold peaked symmetric P_inf onto the right half-axis p >= p_c.
    """
    p = np.asarray(p, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    P_inf = np.asarray(P_inf, dtype=np.float64)

    p_right = np.round(np.where(p >= p_c, p, 2.0 * p_c - p), 8)
    unique_M = np.unique(M)

    p_fold, M_fold, P_fold, sem_fold = [], [], [], []
    has_sem = P_inf_sem is not None

    if has_sem:
        P_inf_sem = np.asarray(P_inf_sem, dtype=np.float64)

    for m_val in unique_M:
        mask_m = M == m_val
        pr_m = p_right[mask_m]
        Pi_m = P_inf[mask_m]

        if has_sem:
            si_m = P_inf_sem[mask_m]
            with np.errstate(invalid='ignore', divide='ignore'):
                rel_sem_m = np.where(Pi_m > 1e-12, si_m / Pi_m, np.nan)

            valid_rel = rel_sem_m[np.isfinite(rel_sem_m) & (rel_sem_m > 0)]
            median_rel = float(np.median(valid_rel)) if len(valid_rel) > 0 else 0.05
            rel_sem_m = np.where(np.isfinite(rel_sem_m) & (rel_sem_m > 0), rel_sem_m, median_rel)

            valid_abs = si_m[(si_m > 0) & np.isfinite(si_m)]
            median_abs = float(np.median(valid_abs)) if len(valid_abs) > 0 else median_rel * np.mean(Pi_m)
            si_m = np.where((si_m > 0) & np.isfinite(si_m), si_m, median_abs)

        for p_key in np.unique(pr_m):
            idx = pr_m == p_key
            Pi_group = Pi_m[idx]

            if has_sem:
                rel_si = rel_sem_m[idx]
                si_abs = si_m[idx]
                w = 1.0 / rel_si ** 2
                W = np.sum(w)
                P_avg = np.sum(w * Pi_group) / W
                abs_sem_avg = np.sqrt(np.sum((w / W) ** 2 * si_abs ** 2))
            else:
                P_avg = np.mean(Pi_group)
                abs_sem_avg = 1.0

            p_fold.append(p_key)
            M_fold.append(m_val)
            P_fold.append(P_avg)
            sem_fold.append(abs_sem_avg)

    order = np.lexsort((np.array(p_fold), np.array(M_fold)))
    return (np.array(p_fold)[order], np.array(M_fold)[order],
            np.array(P_fold)[order], np.array(sem_fold)[order])


# ---------------------------------------------------------------------------
# COST FUNCTIONS
# ---------------------------------------------------------------------------

def apply_scaling(params, p, M, P_inf, p_c):
    """Calculates corrected scaled variables X and Y."""
    k1, k2, c, omega = params

    # Guard against unphysical parameters or singularities
    if k1 <= 0 or k2 < 0:
        return None, None


    if np.any(c * (M ** -omega) <= -0.99):
        return None, None  # Prevent negative or near-zero denominators

    correction_factor = 1.0 + c * (M ** -omega)
    X = (p - p_c) * (M ** k1)
    Y = (P_inf * (M ** k2)) / correction_factor
    return X, Y


def pairwise_alignment_cost(params, p, M, P_inf, p_c=0.5, weights=None):
    """Pairwise curve alignment cost over 4D parameter space."""
    X, Y = apply_scaling(params, p, M, P_inf, p_c)
    if X is None:
        return 1e6

    w = np.ones_like(p) if weights is None else np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    unique_M = np.unique(M)
    if len(unique_M) < 2:
        return 1e6

    M_max = float(np.max(unique_M))
    total_cost, total_pair_w = 0.0, 0.0
    peak_Y = np.max(Y)

    for i in range(len(unique_M)):
        m1 = unique_M[i]
        mask1 = M == m1
        X1, Y1, w1 = X[mask1], Y[mask1], w[mask1]
        s1 = np.argsort(X1)
        X1s, Y1s, w1s = X1[s1], Y1[s1], w1[s1]

        for j in range(i + 1, len(unique_M)):
            m2 = unique_M[j]
            mask2 = M == m2
            X2, Y2 = X[mask2], Y[mask2]
            s2 = np.argsort(X2)
            X2s, Y2s = X2[s2], Y2[s2]

            pair_w = (m1 * m2) / (M_max ** 2)

            x_lo = max(X1s[0], X2s[0])
            x_hi = min(X1s[-1], X2s[-1])
            if x_hi <= x_lo:
                continue

            in_overlap = (X1s >= x_lo) & (X1s <= x_hi)
            if np.sum(in_overlap) < 3:
                continue

            Y2_interp = np.interp(X1s[in_overlap], X2s, Y2s)
            diff = Y1s[in_overlap] - Y2_interp
            w_ovlp = w1s[in_overlap]

            Y_local = 0.5 * (Y1s[in_overlap] + Y2_interp)
            denom = np.maximum(Y_local, 1e-3 * peak_Y)
            rel_diff = diff / denom

            pair_cost = np.sum(w_ovlp * rel_diff ** 2) / np.sum(w_ovlp)
            total_cost += pair_w * pair_cost
            total_pair_w += pair_w

    return total_cost / total_pair_w if total_pair_w > 0 else 1e6


# ---------------------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------------------

def extract_exponents_scaled(alpha, q, df, p_c=0.5,
                             nu_guess=4 / 3, beta_guess=5 / 36,
                             show_plot=True, n_boot=0, verbose=True):

    """
    Robust FSS extraction including leading-order corrections.
    """
    #df = df[df["P_sum"] > 0].copy()
    df = df[df["M"] > 200].copy()
    p_raw = df['p'].values.astype(np.float64)
    M_raw = df['M'].values.astype(np.float64)
    P_raw = df['P_inf'].values.astype(np.float64)
    sem_raw = df['P_inf_sem'].values.astype(np.float64) if 'P_inf_sem' in df.columns else None

    p_vals, M_vals, P_inf_vals, sem_vals = fold_and_average(
        p_raw, M_raw, P_raw, p_c=p_c, P_inf_sem=sem_raw
    )


    if sem_raw is not None:
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_sem = np.where(P_inf_vals > 1e-12, sem_vals / P_inf_vals, np.nan)
        rel_sem = np.where(np.isfinite(rel_sem) & (rel_sem > 0), rel_sem, float(np.nanmedian(rel_sem)))
        weights = 1.0 / rel_sem ** 2
        weights /= np.mean(weights)
    else:
        weights = None
    if not True:
        nu_opt, beta_opt, c_opt, omega_fixed = extract_exponents_decoupled(p_vals, M_vals, P_inf_vals, weights, 0.5,
                                                                           nu_guess)
        return nu_opt, 0.0, beta_opt, 0.0, c_opt, omega_fixed


    if verbose:
        print(f"\n[Robust FSS with Corrections] α={alpha}, q={q}")

    k1_guess, k2_guess = 1.0 / nu_guess, beta_guess / nu_guess


    # Parameter vector: [k1, k2, c, omega]
    bounds = [(1.1, 1.5), (0.15, 0.75), (0, 0), (0.0, 5.0)]

    seeds = [
        [k1_guess, k2_guess, 0.0, 1.0],  # Uncorrected start
        [k1_guess, k2_guess, 0.0, 0.5],  # Positive amplitude

    ]

    def _cost(params):
        return pairwise_alignment_cost(params, p_vals, M_vals, P_inf_vals, p_c, weights)

    best_cost, best_params = np.inf, None

    for seed in seeds:
        try:
            res = differential_evolution(
                _cost, bounds=bounds, x0=seed,
                strategy='best1bin', popsize=15, tol=1e-5,
                maxiter=1000, seed=42, polish=True
            )
            if res.fun < best_cost:
                best_cost, best_params = res.fun, res.x
        except Exception as e:
            warnings.warn(f"DE failed for seed {seed}: {e}")

    if best_params is None:
        return [np.nan] * 6

    res_local = minimize(_cost, best_params, bounds=bounds, method='Nelder-Mead', options={'maxiter': 5000})
    k1_opt, k2_opt, c_opt, omega_opt = res_local.x

    nu_opt = 1.0 / k1_opt
    beta_opt = k2_opt * nu_opt

    if verbose:
        print(f"  Result: ν={nu_opt:.3f}, β={beta_opt:.4f}")
        print(f"  Corrections: c={c_opt:.3f}, ω={omega_opt:.3f}")
        print(f"  Cost: {res_local.fun:.5f}")

    if show_plot:
        plot_data_collapse_corrected(p_vals, M_vals, P_inf_vals, p_c, res_local.x, alpha, q, sem=sem_vals)

    return nu_opt, 0.0, beta_opt, 0.0, c_opt, omega_opt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize, differential_evolution
import warnings


# ---------------------------------------------------------------------------
# 1. ISOLATE y-AXIS SCALING AT p = p_c
# ---------------------------------------------------------------------------

def P_inf_pc_model(M, A, beta_nu, c0, omega):
    """Analytic form for P_inf at the critical point."""
    return A * (M ** -beta_nu) * (1.0 + c0 * (M ** -omega))


def fit_critical_scaling(p, M, P_inf, p_c=0.5):
    """
    Extracts beta/nu and omega using only data at p = p_c.
    Returns beta_nu, omega, and plots the effective exponent.
    """
    # Isolate data strictly at p_c
    mask_pc = np.isclose(p, p_c, atol=1e-6)
    M_pc = M[mask_pc]
    P_pc = P_inf[mask_pc]

    # Sort by M just in case
    sort_idx = np.argsort(M_pc)
    M_pc = M_pc[sort_idx]
    P_pc = P_pc[sort_idx]

    if len(M_pc) < 4:
        raise ValueError("Not enough distinct M values at p=p_c to fit 4 parameters.")

    # Bounds: A > 0, beta/nu in [0.01, 0.5], c0 unrestricted, omega in [0.1, 4.0]
    bounds = (
        [0.0, 0.01, -np.inf, 0.1],
        [np.inf, 0.5, np.inf, 4.0]
    )

    # Guess parameters
    A_guess = P_pc[0] * (M_pc[0] ** 0.1)
    p0 = [A_guess, 0.1, 0.0, 1.0]

    popt, _ = curve_fit(P_inf_pc_model, M_pc, P_pc, p0=p0, bounds=bounds, maxfev=10000)
    A_opt, beta_nu_opt, c0_opt, omega_opt = popt

    # --- Calculate discrete effective exponents for visualization ---
    M_mid = []
    beta_nu_eff = []
    for i in range(len(M_pc) - 1):
        m1, m2 = M_pc[i], M_pc[i + 1]
        p1, p2 = P_pc[i], P_pc[i + 1]
        b_eff = -np.log(p2 / p1) / np.log(m2 / m1)
        M_mid.append(np.sqrt(m1 * m2))
        beta_nu_eff.append(b_eff)

    # Calculate analytical effective exponent from the fit for the plot
    M_smooth = np.linspace(min(M_pc), max(M_pc), 100)
    # The logarithmic derivative of the fit model
    beta_nu_eff_fit = beta_nu_opt + (c0_opt * omega_opt * M_smooth ** -omega_opt) / (
                1 + c0_opt * M_smooth ** -omega_opt)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(M_mid, beta_nu_eff, 'ko', label=r'Discrete $\beta_{eff}/\nu$')
    ax.plot(M_smooth, beta_nu_eff_fit, 'r-', label=f'Analytic Fit\nAsymptote: {beta_nu_opt:.4f}')
    ax.axhline(beta_nu_opt, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('$M$')
    ax.set_ylabel(r'$\beta_{eff} / \nu$')
    ax.set_xscale('log')
    ax.set_title(r'Effective Exponent Drift at $p=p_c$')
    ax.legend()
    plt.show()

    return beta_nu_opt, omega_opt


# ---------------------------------------------------------------------------
# 2. GLOBAL COLLAPSE (LOCKED k2 AND omega)
# ---------------------------------------------------------------------------

def apply_scaling_locked(params, p, M, P_inf, p_c, beta_nu_fixed, omega_fixed):
    """Calculates X and Y with beta/nu and omega locked."""
    k1, c = params
    k2 = beta_nu_fixed

    if k1 <= 0:
        return None, None
    if np.any(c * (M ** -omega_fixed) <= -0.99):
        return None, None

    correction_factor = 1.0 + c * (M ** -omega_fixed)
    X = (p - p_c) * (M ** k1)
    Y = (P_inf * (M ** k2)) / correction_factor
    return X, Y


def pairwise_alignment_cost_locked(params, p, M, P_inf, p_c, beta_nu_fixed, omega_fixed, weights=None):
    """Modified cost function using locked parameters."""
    X, Y = apply_scaling_locked(params, p, M, P_inf, p_c, beta_nu_fixed, omega_fixed)
    if X is None:
        return 1e6

    w = np.ones_like(p) if weights is None else weights
    unique_M = np.unique(M)
    M_max = float(np.max(unique_M))
    total_cost, total_pair_w = 0.0, 0.0
    peak_Y = np.max(Y)

    for i in range(len(unique_M)):
        m1 = unique_M[i]
        mask1 = M == m1
        X1, Y1, w1 = X[mask1], Y[mask1], w[mask1]
        s1 = np.argsort(X1)
        X1s, Y1s, w1s = X1[s1], Y1[s1], w1[s1]

        for j in range(i + 1, len(unique_M)):
            m2 = unique_M[j]
            mask2 = M == m2
            X2, Y2 = X[mask2], Y[mask2]
            s2 = np.argsort(X2)
            X2s, Y2s = X2[s2], Y2[s2]

            pair_w = (m1 * m2) / (M_max ** 2)

            x_lo = max(X1s[0], X2s[0])
            x_hi = min(X1s[-1], X2s[-1])
            if x_hi <= x_lo:
                continue

            in_overlap = (X1s >= x_lo) & (X1s <= x_hi)
            if np.sum(in_overlap) < 3:
                continue

            Y2_interp = np.interp(X1s[in_overlap], X2s, Y2s)
            diff = Y1s[in_overlap] - Y2_interp
            w_ovlp = w1s[in_overlap]

            Y_local = 0.5 * (Y1s[in_overlap] + Y2_interp)
            denom = np.maximum(Y_local, 1e-3 * peak_Y)
            rel_diff = diff / denom

            pair_cost = np.sum(w_ovlp * rel_diff ** 2) / np.sum(w_ovlp)
            total_cost += pair_w * pair_cost
            total_pair_w += pair_w

    return total_cost / total_pair_w if total_pair_w > 0 else 1e6




# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_data_collapse_corrected(p, M, P_inf, p_c, params, alpha, q, sem=None):
    """Plot data collapse for corrected variables."""
    k1, k2, c, omega = params
    nu = 1.0 / k1
    beta = k2 * nu

    X, Y = apply_scaling(params, p, M, P_inf, p_c)
    correction_factor = 1.0 + c * (M ** -omega)
    dY = (sem * (M ** k2)) / correction_factor if sem is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for m_val in np.sort(np.unique(M)):
        mask = M == m_val
        xs = np.argsort(X[mask])
        Xm, Ym = X[mask][xs], Y[mask][xs]
        if dY is not None:
            ax.errorbar(Xm, Ym, yerr=dY[mask][xs], fmt='o-', markersize=3, alpha=0.8, label=f'M={m_val:.0f}')
        else:
            ax.plot(Xm, Ym, 'o-', markersize=3, alpha=0.8, label=f'M={m_val:.0f}')

    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_title(f'Corrected Collapse: ν={nu:.3f}, β={beta:.4f}\nc={c:.3f}, ω={omega:.3f}')
    ax.set_xlabel(r'$(p - p_c)\,M^{1/\nu}$')
    ax.set_ylabel(r'$P_\infty\,M^{\beta/\nu} \,/\, (1 + cM^{-\omega})$')
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.4)

    ax2 = axes[1]
    for m_val in np.sort(np.unique(M)):
        mask = M == m_val
        s = np.argsort(p[mask])
        pm, Pm = p[mask][s], P_inf[mask][s]
        if sem is not None:
            ax2.errorbar(pm, Pm, yerr=sem[mask][s], fmt='o-', markersize=3, label=f'M={m_val:.0f}')
        else:
            ax2.plot(pm, Pm, 'o-', markersize=3, label=f'M={m_val:.0f}')

    ax2.axvline(p_c, color='k', linestyle='--', alpha=0.4)
    ax2.set_title('Folded Raw Data ($p \geq p_c$)')
    ax2.set_xlabel('$p$')
    ax2.set_ylabel('$P_\infty$')
    ax2.legend(fontsize='small', ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()

def discover_datasets(folder, pattern):
    """
    Scans the target folder for files matching the regex pattern
    and extracts alpha and q values.
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Data directory '{folder}' not found.")

    regex = re.compile(pattern)
    files = glob.glob(os.path.join(folder, "*.csv"))
    datasets = []

    for f_path in files:
        basename = os.path.basename(f_path)
        match = regex.search(basename)
        if match:
            alpha_val = float(match.group(1))
            q_val = float(match.group(2))
            datasets.append((alpha_val, q_val, f_path))

    # Sort by alpha, then q for clean terminal output
    return sorted(datasets, key=lambda x: (x[0], x[1]))


def main():
    DATA_FOLDER = "results/anti_correlated"
    FILE_PATTERN = r"a=([\d.]+),q=([\d.]+)\.csv"
    P_C = 0.5

    datasets = discover_datasets(DATA_FOLDER, FILE_PATTERN)
    if not datasets:
        print("No datasets found. Check your folder path and regex.")
        return

    print(f"Found {len(datasets)} dataset(s). Processing...")

    data_list = []

    for alpha_val, q_val, data_path in datasets:
        # Heuristic for initial \nu guess (corrected operator precedence)
        if q_val == 0 or alpha_val > 0.75:
            nu_rough = 1.333
        elif q_val < 0.4:
            nu_rough = ((q_val + 0.05) / 0.4) * (2.0 / alpha_val)
        else:
            nu_rough = 2.0 / alpha_val

        # Load data
        try:
            df_current = pd.read_csv(data_path)
        except Exception as e:
            print(f"Error reading {data_path}: {e}")
            continue

        # Run the FSS extraction with corrections
        # show_plot=False to avoid halting the loop for every file
        print(f"Analyzing a={alpha_val}, q={q_val}...")

        nu_opt, nu_err, beta_opt, beta_err, c_opt, omega_opt = extract_exponents_scaled(
            alpha=alpha_val,
            q=q_val,
            df=df_current,
            p_c=P_C,
            nu_guess=0.8,
            beta_guess=1/3,
            show_plot=True,
            n_boot=200,  # Set >0 if you want actual bootstrap errors
            verbose=False
        )

        data_list.append({
            'a': alpha_val,
            'q': q_val,
            'nu_B': nu_opt,
            'nu_B_err': nu_err,  # Will be 0.0 unless n_boot > 0
            'beta': beta_opt,
            'beta_err': beta_err,
            'c_opt': c_opt,
            'omega_opt': omega_opt
        })

    # Compile results
    df_results = pd.DataFrame(data_list)
    out_path = os.path.join(DATA_FOLDER, "nu_scaling.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # =======================================================================
    # PLOTTING
    # =======================================================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # ---- 1. \nu_B vs q ----
    sns.lineplot(ax=axes[0], data=df_results, x='q', y='nu_B', hue='a', marker='o', palette='viridis')
    if df_results['nu_B_err'].sum() > 0:
        axes[0].errorbar(df_results['q'], df_results['nu_B'], yerr=df_results['nu_B_err'], fmt='none', ecolor='gray',
                         alpha=0.5)

    axes[0].set_title(r'$\nu_B$ vs $q$ for varying $a$')
    axes[0].set_ylabel(r'$\nu_B$')
    axes[0].set_xlabel(r'$q$')
    axes[0].legend(title='$a$')

    # ---- 2. \nu_B vs a ----
    sns.lineplot(ax=axes[1], data=df_results, x='a', y='nu_B', hue='q', marker='D', palette='viridis')

    if df_results['nu_B_err'].sum() > 0:
        axes[1].errorbar(df_results['a'], df_results['nu_B'], yerr=df_results['nu_B_err'], fmt='none', ecolor='gray',
                         alpha=0.5)

    # Analytical Comparison: \nu = 2/a
    a_theory = np.linspace(df_results['a'].min(), df_results['a'].max(), 100)
    # Avoid division by zero if a=0 is in the domain
    a_theory_safe = a_theory[a_theory > 0]
    v_theory = 2.0 / a_theory_safe
    axes[1].plot(a_theory_safe, v_theory, color="red", linestyle='--', label=r'Theory: $2/a$')

    axes[1].set_title(r'$\nu_B$ vs $a$ for varying $q$')
    axes[1].set_ylabel(r'$\nu_B$')
    axes[1].set_xlabel(r'$a$')
    axes[1].legend(title='$q$')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
