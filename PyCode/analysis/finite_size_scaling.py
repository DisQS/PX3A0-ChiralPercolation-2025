"""
FSS collapse for localisation length xi — Chalker-Coddington classical limit.
p_c = 0.5 fixed by symmetry.

Usage
-----
Set DATA_FOLDER at the top.  The script discovers every file matching
    a=<float>,q=<float>.csv
in that folder and processes each one independently.

Workflow per dataset
--------------------
1.  Load CSV, fold onto p >= p_c (inverse-variance weighted average of
    mirror pairs), build per-size arrays with relative inverse-variance
    weights.

2.  Finite-size corrections at p_c:
        xi(p_c, M) / M  =  f_inf * (1 + a * M^{-omega})
    Fit via nonlinear least squares over a grid of starting omegas.
    Divide each curve by C_M = 1 + a*M^{-omega} to get corrected data.
    If drift < 0.5% or non-monotonic, corrections are skipped.

3.  Find nu by minimising the generalised-Gaussian collapse cost on the
    corrected scaled data.

    Cost function — no interpolation, no bin edges:
    -----------------------------------------------
    At each candidate nu form the rescaled cloud
        X_i = (p_i - p_c) * M_i^{1/nu}
        Y_i = xi_corr_i / M_i
    Fit a generalised Gaussian
        f(X) = A * exp( -|X/sigma|^beta )
    to the pooled (X, Y, W) data by weighted nonlinear least squares.
    Cost = weighted residual sum of squares / total weight  (WRSS/W).

    Physical motivation: at criticality the master curve for a symmetric
    peaked observable is well described by a generalised Gaussian.  Using
    the correct functional form gives a far sharper minimum in nu than a
    free polynomial, and requires no bin edges or curve interpolation.

4.  Profile scan for uncertainty (adaptive width from cost curvature).

5.  Six-panel figure:
    Top row   : folded raw xi/M  |  folded corrected  |  best collapse
    Bottom row: FSS correction   |  profile scan       |  full nu scan

"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar, curve_fit
from scipy.special import gamma as _gamma
import seaborn as sns
# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FOLDER  = "results/anti_correlated"   # folder containing a=*,q=*.csv files
FILE_PATTERN = r"a=([\d.]+),q=([\d.]+)\.csv"   # regex to extract alpha, q

P_C = 0.5   # fixed by symmetry

# nu scan range as multiples of nu_rough
NU_LO_FRAC = 0.25
NU_HI_FRAC = 4.0
N_SCAN_COARSE = 0.01   # coarse scan points
N_SCAN_FINE   = 0.01   # fine scan points in ±20% window

COLORS = ["#534AB7", "#0F6E56", "#D85A30", "#BA7517", "#993556",
          "#185FA5", "#3B6D11", "#A32D2D", "#0C447C", "#633806"]
LGRAY  = "#B4B2A9"

# =============================================================================
# 1.  FILE DISCOVERY
# =============================================================================

def discover_datasets(folder, pattern):
    datasets = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"DATA_FOLDER not found: {folder!r}")
    for fname in sorted(os.listdir(folder)):
        m = re.fullmatch(pattern, fname)
        if m:
            alpha = float(m.group(1))
            q     = float(m.group(2))
            datasets.append((alpha, q, os.path.join(folder, fname)))
    if not datasets:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {folder!r}")
    return datasets


# =============================================================================
# 2.  DATA LOADING AND FOLDING
# =============================================================================

def load_csv(path):
    df = pd.read_csv(path, header=0)
    df.columns = df.columns.str.strip()
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if   cl == "m":                                   rename[c] = "M"
        elif cl == "p" and "p" not in rename.values():    rename[c] = "p"
        elif cl in ("xi_sem","xisem","xi_err","xi_std"):  rename[c] = "xi_sem"
        elif cl == "xi":                                  rename[c] = "xi"
    df = df.rename(columns=rename)
    for col in ("M", "p", "xi"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    df = df.dropna(subset=["M", "p", "xi"])
    df["M"] = df["M"].astype(int)
    return df.sort_values(["M", "p"]).reset_index(drop=True)


def fold(df, p_c=0.5):
    """
    Mirror p < p_c onto p' = 2*p_c - p.
    Average mirrored pairs using inverse-variance weights; propagate error.
    Returns DataFrame: M, p, xi, xi_sem (NaN if no SEM in input).
    """
    has_sem = "xi_sem" in df.columns
    rows = []
    for M in df["M"].unique():
        sub    = df[df["M"] == M]
        p_arr  = sub["p"].to_numpy()
        xi_arr = sub["xi"].to_numpy()
        sem_arr = sub["xi_sem"].to_numpy() if has_sem else np.ones_like(xi_arr)

        p_right = np.round(np.where(p_arr >= p_c, p_arr, 2*p_c - p_arr), 10)

        for pk in np.unique(p_right):
            g    = p_right == pk
            xi_g = xi_arr[g]
            s_g  = sem_arr[g].copy()

            # Guard bad SEMs
            bad = ~(np.isfinite(s_g) & (s_g > 0))
            if bad.any():
                s_g[bad] = float(np.nanmedian(s_g[~bad])) if (~bad).any() else 1.0

            w       = 1.0 / s_g**2
            W       = w.sum()
            xi_avg  = float(np.dot(w, xi_g) / W)
            sem_avg = float(1.0 / np.sqrt(W))
            rows.append({"M": M, "p": pk, "xi": xi_avg,
                         "xi_sem": sem_avg if has_sem else np.nan})

    return (pd.DataFrame(rows)
              .sort_values(["M", "p"])
              .reset_index(drop=True))


# =============================================================================
# 3.  FINITE-SIZE CORRECTIONS
# =============================================================================

def fit_fss_corrections(df_fold, sizes, p_c=0.5):
    """
    Fits the leading finite-size correction at the critical point:
    xi(p_c, M) / M = f_inf * (1 + c * M^{-omega}) = f_inf + A * M^{-omega}

    Returns the fit parameters and the correction factors evaluated for each size M.
    """
    # Isolate data strictly at the critical point

    df_pc = df_fold[df_fold["p"] == 0.5].sort_values("M")
    df_1pc = ((df_fold[np.isclose(df_fold["p"],0.5053030303030304)]).sort_values("M"))
    # Need at least 3 points to fit 3 parameters (f_inf, A, omega)
    if len(df_pc) < 3:
        return {"fit_ok": False, "f0_vals": {}}

    M_arr = df_pc["M"].to_numpy(dtype=float)
    xi_arr = df_pc["xi"].to_numpy(dtype=float)
    f0_arr = xi_arr / M_arr  # observable y = xi/M evaluated at x=0

    f0_vals = {int(m): f for m, f in zip(M_arr, f0_arr)}

    # Check the finite-size drift range
    drift_range = np.ptp(f0_arr)  # Equivalent to f0_arr.max() - f0_arr.min()
    if drift_range < 0.01:
        print(f"    FSS fit skipped: drift range ({drift_range:.4f}) < 0.1 cutoff.")
        return {
            "fit_ok": False,
            "f0_vals": f0_vals,
            "M_arr": M_arr,
            "f0_arr": f0_arr,
            "monotonic": np.all(np.diff(f0_arr) <= 0) or np.all(np.diff(f0_arr) >= 0),
            "drift_frac": drift_range
        }

    # Model: y(M) = f_inf + A * M^(-omega)
    def model(M, f_inf, A, omega):
        return f_inf + A * M ** (-omega)

    # Initial estimates
    f_inf_0 = f0_arr[-1]
    A_0 = f0_arr[0] - f_inf_0
    omega_0 = 1.0

    try:
        # Non-linear least squares fit
        popt, pcov = curve_fit(
            model, M_arr, f0_arr,
            p0=[f_inf_0, A_0, omega_0],
            bounds=([0, -np.inf, 0.01], [np.inf, np.inf, 5.0]),
            maxfev=5000
        )
        f_inf, A, omega = popt
        f_inf_err, A_err, omega_err = np.sqrt(np.diag(pcov))

        # Calculate the relative amplitude c = A / f_inf
        c = A / f_inf

        # Precompute the correction divisor C_M = 1 + c * M^(-omega)
        corr_factors = {int(M): 1.0 + c * M ** (-omega) for M in M_arr}

        # Arrays for the log-log FSS plot (Panel 4)
        log_M = np.log(M_arr)
        log_drift = np.log(np.abs(f0_arr - f_inf))
        log_fit = np.log(np.abs(A * M_arr ** (-omega)))

        return {
            "fit_ok": True,
            "f0_vals": f0_vals,
            "M_arr": M_arr,
            "f0_arr": f0_arr,
            "f_inf": f_inf, "f_inf_err": f_inf_err,
            "A": A, "A_err": A_err,
            "c": c,
            "omega": omega, "omega_err": omega_err,
            "corr_factors": corr_factors,
            "log_M": log_M,
            "log_drift": log_drift,
            "log_fit": log_fit,
            "monotonic": np.all(np.diff(f0_arr) <= 0) or np.all(np.diff(f0_arr) >= 0),
            "drift_frac": abs(c * M_arr[0] ** (-omega))
        }
    except Exception as e:
        print(f"    FSS fit failed: {e}")
        return {
            "fit_ok": False,
            "f0_vals": f0_vals,
            "M_arr": M_arr,
            "f0_arr": f0_arr,
            "monotonic": np.all(np.diff(f0_arr) <= 0) or np.all(np.diff(f0_arr) >= 0),
            "drift_frac": drift_range
        }


# =============================================================================
# 4.  GENERALISED GAUSSIAN COLLAPSE COST
# =============================================================================

def gen_gauss(X, A, sigma, beta):
    """Generalised Gaussian: A * exp(-|X/sigma|^beta)."""
    return A * np.exp(-np.abs(X / (sigma + 1e-14))**beta)


def _fit_gen_gauss(X, Y, W):
    """
    Fit a generalised Gaussian to weighted (X,Y) data.
    Returns (Y_fit, success).
    """
    # Starting guesses from moments of the weighted distribution
    W_sum  = W.sum() + 1e-14
    A0     = float(np.dot(W, Y) / W_sum * 1.2)
    # Weighted variance of X gives sigma estimate
    X_mean = float(np.dot(W, X) / W_sum)
    X_var  = float(np.dot(W, (X - X_mean)**2) / W_sum)
    sig0   = float(np.sqrt(X_var + 1e-8))
    A0     = max(A0, float(Y.max()) * 0.5)

    best_Y_fit, best_sse = None, np.inf

    for beta0 in (0.8, 1.5, 2.0, 3.0):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gen_gauss, X, Y,
                    p0=[A0, sig0, beta0],
                    bounds=([0, 1e-6, 0.3], [np.inf, np.inf, 8.0]),
                    sigma=1.0 / np.sqrt(W + 1e-14),
                    absolute_sigma=True,
                    maxfev=3000,
                )
            Y_fit = gen_gauss(X, *popt)
            sse   = float(np.dot(W, (Y - Y_fit)**2))
            if sse < best_sse:
                best_sse, best_Y_fit = sse, Y_fit
        except Exception:
            continue

    return best_Y_fit, best_Y_fit is not None


def make_collapse_cost_gauss(curve_arrays_corr, sizes):
    """
    Returns cost(nu): WRSS/W of a generalised-Gaussian fit to the
    pooled corrected scaled data at the given nu.

    No interpolation. No bin edges. Pure weighted nonlinear least squares.
    """
    def cost(nu):
        if nu <= 0.05 or nu > 60.0:
            return 1e8
        inv_nu = 1.0 / nu
        X_parts, Y_parts, W_parts = [], [], []
        for M in sizes:
            c = curve_arrays_corr[M]
            X_parts.append(c["dp"] * (float(M) ** inv_nu))
            Y_parts.append(c["y_corr"])
            W_parts.append(c["w"])

        X = np.concatenate(X_parts)
        Y = np.concatenate(Y_parts)
        W = np.concatenate(W_parts)

        # Only keep positive Y (gen-Gaussian is non-negative)
        mask = Y > 0
        if mask.sum() < 6:
            return 1e8
        X, Y, W = X[mask], Y[mask], W[mask]

        Y_fit, ok = _fit_gen_gauss(X, Y, W)
        if not ok:
            return 1e8
        wrss = float(np.dot(W, (Y - Y_fit)**2))
        return wrss / (W.sum() + 1e-14)

    return cost


def make_collapse_cost(curve_arrays_corr, sizes):
    """
    Returns cost(nu): The reduced chi-squared of a non-parametric
    leave-one-out data collapse.
    """

    def cost(nu):
        inv_nu = 1.0 / nu

        # Pre-calculate scaled X and extract variances for all sizes
        scaled_data = {}
        for M in sizes:
            c = curve_arrays_corr[M]
            X = c["dp"] * (float(M) ** inv_nu)
            Y = c["y_corr"]
            # Convert weights back to variance for standard error propagation
            Var = 1.0 / (c["w"] + 1e-16)
            scaled_data[M] = (X, Y, Var)

        total_chi_sq = 0.0
        total_points = 0

        # Leave-one-out cross-validation
        for target_M in sizes:
            X_target, Y_target, Var_target = scaled_data[target_M]

            # Pool data from all OTHER sizes to form the reference master curve
            X_ref_parts, Y_ref_parts, Var_ref_parts = [], [], []
            for M in sizes:
                if M != target_M:
                    X_ref_parts.append(scaled_data[M][0])
                    Y_ref_parts.append(scaled_data[M][1])
                    Var_ref_parts.append(scaled_data[M][2])

            X_ref = np.concatenate(X_ref_parts)
            Y_ref = np.concatenate(Y_ref_parts)
            Var_ref = np.concatenate(Var_ref_parts)

            # Sort the reference data by X to allow interpolation
            sort_idx = np.argsort(X_ref)
            X_ref = X_ref[sort_idx]
            Y_ref = Y_ref[sort_idx]
            Var_ref = Var_ref[sort_idx]

            # Only evaluate target points that fall within the reference domain
            x_min, x_max = X_ref[0], X_ref[-1]
            valid_mask = (X_target >= x_min) & (X_target <= x_max)

            if valid_mask.sum() == 0:
                continue

            X_eval = X_target[valid_mask]
            Y_eval = Y_target[valid_mask]
            Var_eval = Var_target[valid_mask]

            # Interpolate the reference Y and Reference Variance onto the target X
            Y_interp = np.interp(X_eval, X_ref, Y_ref)
            Var_interp = np.interp(X_eval, X_ref, Var_ref)

            # Compute the chi-squared for this size
            residuals = Y_eval - Y_interp
            combined_variance = Var_eval + Var_interp

            total_chi_sq += np.sum((residuals ** 2) / combined_variance)
            total_points += valid_mask.sum()

        # Return reduced chi-squared. If no points overlapped, return a heavy penalty.
        return total_chi_sq / total_points if total_points > 0 else 1e8

    return cost

# =============================================================================
# 5.  NU OPTIMISATION AND UNCERTAINTY
# =============================================================================

def find_nu(curve_arrays_corr, sizes, nu_rough):
    nu_lo = max(0.05, nu_rough * NU_LO_FRAC)
    nu_hi = min(60.0, nu_rough * NU_HI_FRAC)

    cost = make_collapse_cost(curve_arrays_corr, sizes)

    # Coarse scan
    nu_coarse   = np.linspace(nu_lo, nu_hi, int(np.ceil((nu_hi-nu_lo)/N_SCAN_COARSE)))
    cost_coarse = np.array([cost(nu) for nu in nu_coarse])
    nu_c_best   = nu_coarse[np.argmin(cost_coarse)]

    # Fine scan in ±20% window around coarse best
    r_lo = max(nu_lo, nu_c_best * 0.50)
    r_hi = min(nu_hi, nu_c_best * 1.50)
    nu_fine   = np.linspace(r_lo, r_hi, int(np.ceil(r_hi-r_lo)/N_SCAN_FINE))
    cost_fine = np.array([cost(nu) for nu in nu_fine])
    nu_f_best = nu_fine[np.argmin(cost_fine)]

    # Brent polish in tight bracket
    b_lo = max(nu_lo, nu_f_best * 0.90)
    b_hi = min(nu_hi, nu_f_best * 1.10)
    res  = minimize_scalar(cost, bounds=(b_lo, b_hi), method="bounded",
                           options={"xatol": 1e-8, "maxiter": 500})
    nu_opt = float(res.x)

    # Full scan array for plotting
    nu_scan   = np.unique(np.concatenate([nu_coarse, nu_fine]))
    cost_scan = np.array([cost(nu) for nu in nu_scan])

    return nu_opt, nu_scan, cost_scan, cost


def profile_uncertainty(cost_fn, nu_opt, threshold=0.002):
    """
    Derivative-free profile scan. Iteratively expands the grid until the
    threshold is bracketed on both sides, then extracts conservative asymmetric errors.
    """
    cost_min = cost_fn(nu_opt)
    target_cost = cost_min * (1.0 + threshold)

    # Start with a conservative 15% window
    width = max(0.1, nu_opt * 0.15)

    for _ in range(4):  # Max 4 expansions
        nu_profile = np.linspace(max(0.01, nu_opt - width), nu_opt + width, 400)
        cost_profile = np.array([cost_fn(nu) for nu in nu_profile])

        inside = nu_profile[cost_profile <= target_cost]

        if len(inside) < 2:
            width *= 2.0
            continue

        # Verify the threshold is contained strictly within the profile bounds
        bracketed_left = inside.min() > nu_profile.min()
        bracketed_right = inside.max() < nu_profile.max()

        if bracketed_left and bracketed_right:
            err_minus = nu_opt - inside.min()
            err_plus = inside.max() - nu_opt

            # Take the maximum deviation to establish a conservative symmetric bound,
            # or return both if you prefer asymmetric reporting.
            nu_err = max(err_minus, err_plus)
            return nu_err, nu_profile, cost_profile

        # Expand and recalculate if not fully bracketed
        width *= 2.0

    # Fallback if the landscape is entirely flat (indicates failed collapse)
    return 0.0, nu_profile, cost_profile


# =============================================================================
# 6.  MAIN LOOP
# =============================================================================



def analyse(alpha_val,q_val,data_path):
    print(f"\n{'='*62}")
    print(f"  alpha={alpha_val}   q={q_val}   ->   {os.path.basename(data_path)}")
    print(f"{'='*62}")

    # ── Load ─────────────────────────────────────────────────────────────
    df      = load_csv(data_path)
    #df = df[df["M"] > 400].copy()
    has_sem = "xi_sem" in df.columns
    sizes   = np.array(sorted(df["M"].unique()))
    n_sizes = len(sizes)

    print(f"  Sizes  : {sizes}")
    print(f"  p range: [{df['p'].min():.4f}, {df['p'].max():.4f}]")
    print(f"  has SEM: {has_sem}")

    if n_sizes < 2:
        print("  Skipping — need ≥ 2 sizes.")
        return

    # ── nu_rough — shared formula across all alpha/q ──────────────────
    if q_val == 0 or alpha_val >0.75:
        nu_rough = 1.333
    elif q_val < 0.4:
        nu_rough = (q_val+0.05 / 0.4) * (2.0 / alpha_val)
    else:
        nu_rough = 2.0 / alpha_val
    nu_rough =0.6
    print(f"  nu_rough = {nu_rough:.4f}")

    # ── Fold ─────────────────────────────────────────────────────────────
    df_fold = fold(df, p_c=P_C)

    # ── FSS corrections ───────────────────────────────────────────────
    print("\n[1] Finite-size corrections …")
    corr = fit_fss_corrections(df_fold, sizes, p_c=P_C)

    # Extract the precomputed correction factors. Default to 1.0 if fit failed.
    corr_factors = corr.get("corr_factors", {})

    curve_arrays_raw = {}
    curve_arrays_corr = {}

    for M in sizes:
        sub = df_fold[df_fold["M"] == M].sort_values("p")
        p_M = sub["p"].to_numpy(dtype=float)
        xi_M = sub["xi"].to_numpy(dtype=float)
        y_M = xi_M / float(M)
        dp_M = p_M - P_C

        if has_sem:
            sem_M = sub["xi_sem"].to_numpy(dtype=float)
            bad   = ~(np.isfinite(sem_M) & (sem_M > 0))
            if bad.any():
                sem_M[bad] = float(np.nanmedian(sem_M[~bad])) if (~bad).any() else 1.0
            # Relative inverse-variance weights
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(xi_M > 1e-12, sem_M / xi_M, np.nan)
            med_rel = float(np.nanmedian(rel[np.isfinite(rel) & (rel > 0)]))
            rel     = np.where(np.isfinite(rel) & (rel > 0), rel, med_rel)
            w_M     = 1.0 / rel**2
            w_M     = w_M / w_M.mean()
        else:
            w_M   = np.ones(len(p_M))
            sem_M = np.full(len(p_M), np.nan)

        cf = corr_factors.get(int(M), 1.0)

        curve_arrays_raw[M] = {
            "p": p_M, "dp": dp_M, "y": y_M, "w": w_M,
            "sem": sem_M / float(M),
        }
        curve_arrays_corr[M] = {
            "p": p_M, "dp": dp_M,
            "y_corr": y_M / cf,
            "w": w_M,
            "sem": sem_M / float(M) / cf,
        }

    # ── Find nu ───────────────────────────────────────────────────────
    print("\n[2] Finding nu …")
    nu_opt, nu_scan, cost_scan, cost_fn = find_nu(
        curve_arrays_corr, sizes, nu_rough)

    nu_err, nu_profile, cost_profile = profile_uncertainty(cost_fn, nu_opt)




    print(f"    nu = {nu_opt:.5f} +/- {nu_err:.5f}")
    print(f"    (reference 4/3 = {4/3:.5f})")

    # ── Best-collapse gen-Gaussian fit for plotting ────────────────────
    inv_nu = 1.0 / nu_opt
    X_all  = np.concatenate([
        curve_arrays_corr[M]["dp"] * (float(M)**inv_nu) for M in sizes])
    Y_all  = np.concatenate([curve_arrays_corr[M]["y_corr"] for M in sizes])
    W_all  = np.concatenate([curve_arrays_corr[M]["w"]      for M in sizes])
    mask   = Y_all > 0
    X_fit, Y_fit_d, W_fit = X_all[mask], Y_all[mask], W_all[mask]

    Y_master_fit, fit_ok = _fit_gen_gauss(X_fit, Y_fit_d, W_fit)
    X_master = np.linspace(X_fit.min(), X_fit.max(), 400)
    if fit_ok:
        # Refit on dense grid using the parameters from the best fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt_m, _ = curve_fit(
                    gen_gauss, X_fit, Y_fit_d,
                    p0=[Y_fit_d.max(), float(np.std(X_fit)), 2.0],
                    bounds=([0, 1e-6, 0.3], [np.inf, np.inf, 8.0]),
                    sigma=1.0/np.sqrt(W_fit+1e-14), absolute_sigma=True,
                    maxfev=3000)
            Y_master = gen_gauss(X_master, *popt_m)
            A_m, sig_m, beta_m = popt_m
        except Exception:
            Y_master = np.interp(X_master, np.sort(X_fit),
                                  Y_fit_d[np.argsort(X_fit)])
            beta_m   = float("nan")
    else:
        Y_master = np.interp(X_master, np.sort(X_fit),
                              Y_fit_d[np.argsort(X_fit)])
        beta_m   = float("nan")

    # ── nu vs M_min stability ─────────────────────────────────────────
    nu_vs_Mmin = []
    if n_sizes >= 4:
        print("\n[3] nu vs M_min stability …")
        for n_drop in range(n_sizes - 2):
            sub_sizes = sizes[n_drop:]
            sub_corr  = {M: curve_arrays_corr[M] for M in sub_sizes}
            sub_cost  = make_collapse_cost(sub_corr, sub_sizes)
            nu_lo_s   = max(0.05, nu_opt * 0.50)
            nu_hi_s   = min(60.0, nu_opt * 2.50)
            res_s     = minimize_scalar(sub_cost,
                                        bounds=(nu_lo_s, nu_hi_s),
                                        method="bounded",
                                        options={"xatol": 1e-6, "maxiter": 300})
            nu_sub = float(res_s.x)
            nu_vs_Mmin.append((int(sub_sizes[0]), nu_sub))
            print(f"    M_min={sub_sizes[0]:6d}: nu={nu_sub:.5f}")

    # =================================================================
    # PLOTTING
    # =================================================================

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    ax_raw  = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_coll = fig.add_subplot(gs[0, 2])
    ax_fss  = fig.add_subplot(gs[1, 0])
    ax_prof = fig.add_subplot(gs[1, 1])
    ax_scan = fig.add_subplot(gs[1, 2])

    def _plot_curve(ax, c_arr, y_key, M, col, label, has_sem):
        kw = dict(color=col, ms=3, lw=1.1, alpha=0.8, label=label)
        if has_sem and np.any(np.isfinite(c_arr["sem"])):
            ax.errorbar(c_arr["p"], c_arr[y_key], yerr=c_arr["sem"],
                        fmt="o-", capsize=2, elinewidth=0.7, **kw)
        else:
            ax.plot(c_arr["p"], c_arr[y_key], "o-", **kw)

    # ── Panel 1: folded raw ───────────────────────────────────────────
    for idx, M in enumerate(sizes):
        _plot_curve(ax_raw, curve_arrays_raw[M], "y", M,
                    COLORS[idx % len(COLORS)], f"M={M}", has_sem)

    # Mark crossing values
    if corr.get("f0_vals"):
        f0s = list(corr["f0_vals"].values())
        f0m = float(np.mean(f0s))
        f0s_std = float(np.std(f0s))
        ax_raw.axhline(f0m, color="black", lw=1.0, ls=":",
                       label=rf"$f(0)={f0m:.4f}$")
        if f0s_std > 0:
            ax_raw.axhspan(f0m - f0s_std, f0m + f0s_std,
                           color="black", alpha=0.07)

    ax_raw.axvline(P_C, color=LGRAY, lw=1.2, ls="--")
    ax_raw.set_xlabel(r"$p$", fontsize=12)
    ax_raw.set_ylabel(r"$\xi\,/\,M$", fontsize=12)
    ax_raw.set_title("Folded raw", fontsize=11)
    ax_raw.legend(fontsize=7, framealpha=0.85)

    # ── Panel 2: folded corrected ─────────────────────────────────────
    for idx, M in enumerate(sizes):
        _plot_curve(ax_corr, curve_arrays_corr[M], "y_corr", M,
                    COLORS[idx % len(COLORS)], f"M={M}", has_sem)

    ax_corr.axvline(P_C, color=LGRAY, lw=1.2, ls="--")
    ax_corr.set_xlabel(r"$p$", fontsize=12)
    ax_corr.set_ylabel(r"$\xi\,/\,(M\,C_M)$", fontsize=12)
    ax_corr.set_title(
        "Corrected  $C_M=1+aM^{-\\omega}$" if corr["fit_ok"]
        else "Corrected (no correction needed)", fontsize=11)
    ax_corr.legend(fontsize=7, framealpha=0.85)

    # ── Panel 3: best collapse ────────────────────────────────────────
    for idx, M in enumerate(sizes):
        c   = curve_arrays_corr[M]
        col = COLORS[idx % len(COLORS)]
        X_M = c["dp"] * (float(M) ** inv_nu)
        s   = np.argsort(X_M)
        if has_sem and np.any(np.isfinite(c["sem"])):
            ax_coll.errorbar(X_M[s], c["y_corr"][s], yerr=c["sem"][s],
                             fmt="o", color=col, ms=3, alpha=0.75,
                             capsize=2, elinewidth=0.7,
                             label=f"M={M}", zorder=3)
        else:
            ax_coll.plot(X_M[s], c["y_corr"][s], "o", color=col,
                         ms=3, alpha=0.75, label=f"M={M}", zorder=3)

    lbl_master = (rf"Gen-Gauss $\beta={beta_m:.2f}$"
                  if np.isfinite(beta_m) else "master")
    ax_coll.axvline(0, color=LGRAY, lw=1.0, ls="--", zorder=2)
    ax_coll.set_xlabel(r"$(p-p_c)\,M^{1/\nu}$", fontsize=12)
    ax_coll.set_ylabel(r"$\xi\,/\,(M\,C_M)$", fontsize=12)
    ax_coll.set_title(
        rf"Collapse  $\nu={nu_opt:.4f}\pm{nu_err:.4f}$", fontsize=11)
    ax_coll.legend(fontsize=7, framealpha=0.85)

    # ── Panel 4: FSS correction fit ───────────────────────────────────
    if corr.get("f0_vals"):
        M_a  = corr.get("M_arr",
               np.array(sorted(corr["f0_vals"].keys()), dtype=float))
        f0_a = corr.get("f0_arr",
               np.array([corr["f0_vals"][int(m)] for m in M_a]))

        if corr["fit_ok"]:
            # Show log|f0 - f_inf| vs log(M) — the straight line fit
            ax_fss.plot(corr["log_M"], corr["log_drift"], "o",
                        color=COLORS[0], ms=7, zorder=3,
                        label=r"$\log|f_0(M)-f_\infty|$")
            ax_fss.plot(corr["log_M"], corr["log_fit"], "-",
                        color=COLORS[2], lw=2.0,
                        label=rf"slope $=-\omega={-corr['omega']:.3f}$")
            ax_fss.set_xlabel(r"$\log\,M$", fontsize=12)
            ax_fss.set_ylabel(r"$\log|\xi(p_c,M)/M - f_\infty|$", fontsize=12)
            ax_fss.set_title(
                rf"FSS correction  $\omega={corr['omega']:.3f}"
                rf"\pm{corr['omega_err']:.3f}$", fontsize=11)
        else:
            # No fit: raw crossing values on semilog
            ax_fss.semilogx(M_a, f0_a, "o-", color=COLORS[0], ms=7,
                            label=r"$\xi(p_c,M)/M$")
            ax_fss.set_xlabel(r"$M$", fontsize=12)
            ax_fss.set_ylabel(r"$\xi(p_c,M)/M$", fontsize=12)
            ax_fss.set_title(
                f"FSS crossing (drift={corr['drift_frac']:.4f}, "
                f"{'monotonic' if corr['monotonic'] else 'non-monotonic'})",
                fontsize=11)
        ax_fss.legend(fontsize=8)
    else:
        ax_fss.text(0.5, 0.5, "p_c not in data range",
                    ha="center", va="center", transform=ax_fss.transAxes,
                    fontsize=11, color=LGRAY)

    # ── Panel 5: profile scan ─────────────────────────────────────────
    cost_min_p = cost_profile.min()
    threshold  = 0.002
    ax_prof.plot(nu_profile, cost_profile, "-", color=COLORS[0], lw=1.5)
    ax_prof.axvline(nu_opt, color="#D85A30", lw=1.8,
                    label=rf"$\nu={nu_opt:.4f}\pm{nu_err:.4f}$")
    ax_prof.axvline(4/3, color=LGRAY, lw=1.2, ls="--", label=r"$\nu=4/3$")
    ax_prof.axhline(cost_min_p * (1 + threshold), color=LGRAY,
                    lw=0.8, ls=":", label="0.2% threshold")
    ax_prof.fill_betweenx(
        [cost_min_p, cost_min_p * (1+threshold) * 1.05],
        nu_opt - nu_err, nu_opt + nu_err,
        alpha=0.15, color="#D85A30")
    ax_prof.set_xlabel(r"$\nu$", fontsize=12)
    ax_prof.set_ylabel("WRSS / W", fontsize=12)
    ax_prof.set_title("Profile scan", fontsize=11)
    ax_prof.legend(fontsize=8)

    # ── Panel 6: full nu scan ─────────────────────────────────────────
    ax_scan.semilogy(nu_scan, cost_scan, "-", color=COLORS[1], lw=1.4,
                     label="Collapse cost (WRSS/W)")
    ax_scan.axvline(nu_opt, color="#D85A30", lw=1.5,
                    label=rf"$\nu_\mathrm{{opt}}={nu_opt:.4f}$")
    ax_scan.axvline(4/3, color=LGRAY, lw=1.0, ls="--",
                    label=r"$\nu=4/3$  (2D perc.)")

    if nu_vs_Mmin:
        ax_scan.axvspan(
            min(v for _, v in nu_vs_Mmin),
            max(v for _, v in nu_vs_Mmin),
            alpha=0.10, color=COLORS[3],
            label=r"$\nu$ range vs $M_{\min}$")

    ax_scan.set_xlabel(r"$\nu$", fontsize=12)
    ax_scan.set_ylabel("WRSS / W", fontsize=12)
    ax_scan.set_title("Full $\\nu$ scan", fontsize=11)
    ax_scan.legend(fontsize=8)

    # ── Supertitle & save ─────────────────────────────────────────────
    corr_str = (rf", $\omega={corr['omega']:.3f}$" if corr["fit_ok"] else "")
    fig.suptitle(
        rf"Chalker-Coddington  $\alpha={alpha_val},\;q={q_val}$"
        rf"  —  $p_c={P_C}$ (fixed)"
        rf"  —  $\nu={nu_opt:.4f}\pm{nu_err:.4f}${corr_str}",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = data_path.replace(
        ".csv", f"_collapse_a{alpha_val}_q{q_val}.png")
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  RESULT  alpha={alpha_val}  q={q_val}")
    print(f"    nu    = {nu_opt:.6f}  +/-  {nu_err:.5f}")
    if corr["fit_ok"]:
        print(f"    omega = {corr['omega']:.4f}  +/-  {corr['omega_err']:.4f}")
        print(f"    f_inf = {corr['f_inf']:.6f}  +/-  {corr['f_inf_err']:.6f}")
    if nu_vs_Mmin:
        nu_range = max(v for _, v in nu_vs_Mmin) - min(v for _, v in nu_vs_Mmin)
        print(f"    nu stability range = {nu_range:.5f}  "
              f"({'stable' if nu_range < nu_err else 'drifting — consider larger M'})")
    print(f"    plot  -> {os.path.basename(out_path)}")

    return(nu_opt,nu_err)


datasets = discover_datasets(DATA_FOLDER, FILE_PATTERN)
print(f"Found {len(datasets)} dataset(s): "
      + ", ".join(f"(a={a},q={q})" for a, q, _ in datasets))
data_list = []
for alpha_val, q_val, data_path in datasets:
    if q_val != 1:
        continue
    nu_opt, nu_err = analyse(alpha_val,q_val,data_path)

    data_list.append({
        'a': alpha_val,
        'q': q_val,
        'nu_B': nu_opt,
        'nu_B_err': nu_err
    })

# Create DataFrame
df = pd.DataFrame(data_list)

# Save to CSV
df.to_csv(os.path.join('results/anticorrelated', "nu_scaling.csv"), index=False)

# --- Plotting ---
sns.set_theme(style="whitegrid")
# Fixed to 2 subplots as requested by the logic
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# 1. nu_B vs q (Lines for each 'a')
sns.lineplot(ax=axes[0], data=df, x='q', y='nu_B', hue='a', marker='o', palette='viridis')
axes[0].errorbar(df['q'], df['nu_B'], yerr=df['nu_B_err'], fmt='none', ecolor='gray', alpha=0.5)
axes[0].set_title(r'$\nu_B$ vs $q$ for varying $a$')
axes[0].set_ylabel(r'$\nu_B$')
axes[1].legend()

# 2. nu_B vs a (Lines for each 'q')
# Use axes[1] instead of axes[2]
sns.lineplot(ax=axes[1], data=df, x='a', y='nu_B', hue='q', marker='D', palette='viridis')

# Analytical Comparison: ν = 2/a
a_theory = np.linspace(df['a'].min(), df['a'].max(), 100)
v_theory = 2 / a_theory
axes[1].plot(a_theory, v_theory, color="red", linestyle='--', label=r'Theory: $2/a$')

axes[1].errorbar(df['a'], df['nu_B'], yerr=df['nu_B_err'], fmt='none', ecolor='gray', alpha=0.5)
axes[1].set_title(r'$\nu_B$ vs $a$ for varying $q$')
axes[1].set_ylabel(r'$\nu_B$')
axes[1].legend()

plt.tight_layout()
plt.show()
