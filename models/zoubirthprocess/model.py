"""Dixon-Robinson birth process model + Zou Bayesian updating.

Implements:
1. Inhomogeneous Poisson process with score-state dependent intensities (Dixon & Robinson 1998)
2. Parameter estimation via vectorized MLE (L-BFGS-B)
3. Gamma-Poisson conjugate Bayesian updating (Zou et al. 2020)
4. Recursive algorithm for exact outcome probabilities (Zou et al. 2018)
"""

import numpy as np
from rich.console import Console
from scipy.optimize import minimize, LinearConstraint

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_GOALS = 10  # Max goals per team in recursive algorithm
EPS = 1e-12  # Numerical floor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INTENSITY FUNCTIONS (Eqs 1-4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def rho(minute, params):
    """Injury time multiplier (Eq 3). Minutes 45 and 90 have inflated rates."""
    if minute == 45:
        return params["rho1"]
    if minute == 90:
        return params["rho2"]
    return 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def shots_before(cov_lookup, dev0, mid, upto=90):
    """Shots deviation (home, away) known before each of Zou's minutes 1..upto: the value at the end of
    the previous minute, so a minute's own shots never enter its intensity. dev0 is the kick-off value."""
    h, a = [dev0], [dev0]
    last1 = max(t for t in range(44, 70) if (mid, t, 1) in cov_lookup)   # end of the first half
    for p in range(2, upto + 1):
        c = cov_lookup[(mid, p - 2, 1) if p <= 45 else (mid, last1, 1) if p == 46 else (mid, p - 2, 2)]
        h.append(float(c["dev_shots_home"])); a.append(float(c["dev_shots_away"]))
    return h, a


def extract_match_data(ws_events, align, cov_lookup=None, dev0=0.0):
    """Extract goal-time data from WhoScored events for birth process fitting.

    cov_lookup: period-keyed covariates with shots deviations (None fits the model without shots).
    Returns list of dicts with keys:
        home_id, away_id, goal_times: [(minute, is_home_goal), ...], psi_h, psi_a (90 per-minute values)
    """
    ws_to_home = dict(zip(align["ws_game_id"], align["ws_home_team"]))
    matches = []

    for mid in ws_events["ws_match_id"].unique():
        if mid not in ws_to_home:
            continue
        home_name = ws_to_home[mid]
        evts = ws_events[ws_events["ws_match_id"] == mid]

        # Get team IDs
        home_evts = evts[evts["team_name"] == home_name]
        away_evts = evts[evts["team_name"] != home_name]
        if len(home_evts) == 0 or len(away_evts) == 0:
            continue
        home_id = int(home_evts["team_id"].iloc[0])
        away_id = int(away_evts["team_id"].iloc[0])

        # Goals on Zou's clock: minute t covers (t-1, t], first-half stoppage is minute 45, second-half stoppage 90
        goals_df = evts[evts["is_goal"] == True]
        goal_times = []
        for _, g in goals_df.iterrows():
            if g["period"] not in ("FirstHalf", "SecondHalf"):
                raise ValueError(f"match {mid}: goal in unexpected period {g['period']!r}")
            m = min(int(g["minute"]) + 1, 45 if g["period"] == "FirstHalf" else 90)  # WhoScored minutes count from 0
            is_own = bool(g.get("is_own_goal", False))
            scorer_is_home = g["team_name"] == home_name
            # Own goal: credit to opposite team
            is_home_goal = scorer_is_home if not is_own else not scorer_is_home
            goal_times.append((m, is_home_goal))

        goal_times.sort(key=lambda x: x[0])
        psi_h, psi_a = shots_before(cov_lookup, dev0, int(mid)) if cov_lookup else ([0.0] * 90, [0.0] * 90)
        matches.append(
            {
                "match_id": int(mid),
                "home_id": home_id,
                "away_id": away_id,
                "goal_times": goal_times,
                "psi_h": psi_h, "psi_a": psi_a,
            }
        )

    return matches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VECTORIZED LOG-LIKELIHOOD + GRADIENT (used by fit_birth_process)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Parameter layout: xi = [log_alpha(K), log_beta(K), log_a,
#   log_tau10, log_tau01, log_tau21, log_tau12,
#   log_kappa10, log_kappa01, log_kappa21, log_kappa12,
#   log_rho1, log_rho2, xi1, xi2]
# Total: 2K + 13

_TAU_KEYS = ["tau10", "tau01", "tau21", "tau12"]
_KAPPA_KEYS = ["kappa10", "kappa01", "kappa21", "kappa12"]
_P_A, _P_T10, _P_T01, _P_T21, _P_T12 = 0, 1, 2, 3, 4
_P_K10, _P_K01, _P_K21, _P_K12 = 5, 6, 7, 8
_P_RHO1, _P_RHO2, _P_XI1, _P_XI2, _P_ZETA = 9, 10, 11, 12, 13   # zeta: shots-deviation coefficient
N_GLOBAL_ZOU = 14


def _precompute_zou(matches, tid_to_idx):
    """Build minute-level arrays for vectorized Zou LL.

    For each match, we walk through 90 minutes tracking score state.
    Each minute produces one row with: home_idx, away_idx, minute,
    tau_code, kappa_code, is_rho1, is_rho2, n_home_goals, n_away_goals.
    """
    # We'll build flat arrays
    all_hi, all_ai = [], []       # Team indices
    all_minute = []                # Minute number
    all_tau_code = []              # Which tau applies (0=none, 1=tau10, 2=tau01, 3=tau21, 4=tau12)
    all_kappa_code = []            # Which kappa applies
    all_is_rho1 = []               # Minute 45
    all_is_rho2 = []               # Minute 90
    all_n_hg = []                  # Number of home goals AT this minute
    all_n_ag = []                  # Number of away goals AT this minute
    all_psi_h, all_psi_a = [], []  # Shots deviation known before this minute

    for m in matches:
        hi = tid_to_idx[m["home_id"]]
        ai = tid_to_idx[m["away_id"]]
        goals = m["goal_times"]
        x, y, gi = 0, 0, 0

        for minute in range(1, 91):
            # Count goals at this minute
            nhg, nag = 0, 0
            while gi < len(goals) and goals[gi][0] == minute:
                if goals[gi][1]:
                    nhg += 1
                else:
                    nag += 1
                gi += 1

            # Tau/kappa code based on current score BEFORE goals counted
            tc = _tau_code(x, y)
            kc = _kappa_code(x, y)

            all_hi.append(hi); all_ai.append(ai)
            all_minute.append(minute)
            all_tau_code.append(tc); all_kappa_code.append(kc)
            all_is_rho1.append(minute == 45)
            all_is_rho2.append(minute == 90)
            all_n_hg.append(nhg); all_n_ag.append(nag)
            all_psi_h.append(m["psi_h"][minute - 1]); all_psi_a.append(m["psi_a"][minute - 1])

            # Update score AFTER recording
            x += nhg; y += nag
        if gi != len(goals):
            raise ValueError(f"match {m['match_id']}: {len(goals) - gi} goals fall outside minutes 1-90")

    return {
        "hi": np.array(all_hi, dtype=np.int32),
        "ai": np.array(all_ai, dtype=np.int32),
        "minute": np.array(all_minute, dtype=np.float64),
        "tau_code": np.array(all_tau_code, dtype=np.int32),
        "kappa_code": np.array(all_kappa_code, dtype=np.int32),
        "is_rho1": np.array(all_is_rho1, dtype=bool),
        "is_rho2": np.array(all_is_rho2, dtype=bool),
        "n_hg": np.array(all_n_hg, dtype=np.int32),
        "n_ag": np.array(all_n_ag, dtype=np.int32),
        "psi_h": np.array(all_psi_h), "psi_a": np.array(all_psi_a),
    }


def _tau_code(x, y):
    """Encode score state -> tau index. 0=baseline(1.0)."""
    if x == 1 and y == 0: return 1
    if x == 0 and y == 1: return 2
    if x + y > 1 and x - y >= 1: return 3
    if x + y > 1 and x - y <= -1: return 4
    return 0


def _kappa_code(x, y):
    """Encode score state -> kappa index. 0=baseline(1.0)."""
    if x == 1 and y == 0: return 1
    if x == 0 and y == 1: return 2
    if x + y > 1 and x - y >= 1: return 3
    if x + y > 1 and x - y <= -1: return 4
    return 0


def _neg_loglik_grad_zou(xi, S, K):
    """Vectorized negative log-likelihood and gradient for Zou birth process.

    Intensity at minute t for match with home=H, away=A, score (x,y):
      Λ_h = a * α_H * β_A * τ_xy * ρ(t) + ξ₁ * t/90
      Λ_a = α_A * β_H * κ_xy * ρ(t) + ξ₂ * t/90

    LL = Σ_goals log(Λ) - Σ_minutes (Λ_h + Λ_a)/90
    """
    P = 2 * K
    log_alpha, log_beta = xi[:K], xi[K:P]
    log_a = xi[P + _P_A]
    log_tau = xi[P + _P_T10: P + _P_T12 + 1]   # 4 values
    log_kappa = xi[P + _P_K10: P + _P_K12 + 1]  # 4 values
    log_rho1 = xi[P + _P_RHO1]
    log_rho2 = xi[P + _P_RHO2]
    xi1 = xi[P + _P_XI1]
    xi2 = xi[P + _P_XI2]
    zeta = xi[P + _P_ZETA]

    a = np.exp(log_a)
    tau_vals = np.concatenate([[0.0], np.exp(log_tau)])   # Index 0 = log(1)=0 -> needs special handling
    kappa_vals = np.concatenate([[0.0], np.exp(log_kappa)])
    tau_vals[0] = 1.0   # tau_code=0 means multiplier=1
    kappa_vals[0] = 1.0

    hi, ai = S["hi"], S["ai"]
    t = S["minute"] / 90.0  # Rescaled time

    # Theta per minute-row
    log_theta1 = log_alpha[hi] + log_beta[ai]  # Home scoring
    log_theta2 = log_alpha[ai] + log_beta[hi]  # Away scoring

    # Tau/kappa multipliers
    tau_m = tau_vals[S["tau_code"]]
    kappa_m = kappa_vals[S["kappa_code"]]

    # Rho multiplier
    rho = np.ones(len(hi))
    rho[S["is_rho1"]] = np.exp(log_rho1)
    rho[S["is_rho2"]] = np.exp(log_rho2)

    # Base intensity (without ξ additive term)
    base_h = a * np.exp(log_theta1 + zeta * S["psi_h"]) * tau_m * rho   # a * θ₁ * τ * ρ * e^{ζψ}
    base_a = np.exp(log_theta2 + zeta * S["psi_a"]) * kappa_m * rho       # θ₂ * κ * ρ * e^{ζψ}

    # Full intensity
    lam_h = base_h + xi1 * t
    lam_a = base_a + xi2 * t

    n_hg, n_ag = S["n_hg"], S["n_ag"]

    # Log-likelihood: goal events + survival
    has_hg = n_hg > 0
    has_ag = n_ag > 0
    ll = 0.0
    ll += (n_hg[has_hg] * np.log(np.maximum(lam_h[has_hg], EPS))).sum()
    ll += (n_ag[has_ag] * np.log(np.maximum(lam_a[has_ag], EPS))).sum()
    ll -= (lam_h + lam_a).sum() / 90.0  # Integrated intensity

    # ── Gradient ─────────────────────────────────────────────────────────────
    grad = np.zeros_like(xi)

    # For gradient, we need d(LL)/d(param). The tricky part is that lam_h has
    # an additive ξ₁*t term, so d(log lam_h)/d(log_theta1) = base_h / lam_h
    frac_h = base_h / np.maximum(lam_h, EPS)  # Fraction of intensity from base (not ξ)
    frac_a = base_a / np.maximum(lam_a, EPS)

    # Residuals: (observed - expected) weighted by base fraction
    # For goals: n_hg * (base_h/lam_h) contributes to theta1 gradient
    # For survival: -lam_h/90 contributes -(base_h)/90 to theta1 gradient
    res_h = n_hg * frac_h - base_h / 90.0  # d(LL)/d(log_theta1) per row, partial
    res_a = n_ag * frac_a - base_a / 90.0  # d(LL)/d(log_theta2) per row

    # log_alpha gradient: alpha_i appears in theta1 when i=home, theta2 when i=away
    np.add.at(grad[:K], hi, res_h)   # Home matches: d/d(log_alpha_H) via theta1
    np.add.at(grad[:K], ai, res_a)   # Away matches: d/d(log_alpha_A) via theta2

    # log_beta gradient: beta_i appears in theta1 when i=away, theta2 when i=home
    np.add.at(grad[K:P], ai, res_h)  # Away defence in theta1
    np.add.at(grad[K:P], hi, res_a)  # Home defence in theta2

    # log_a: only affects home intensity base_h
    # d(LL)/d(log_a) = Σ (n_hg * base_h/lam_h - base_h/90) * 1  (since d(base_h)/d(log_a) = base_h)
    grad[P + _P_A] = res_h.sum()

    # Tau gradients: d(base_h)/d(log_tau_k) = base_h when tau_code==k, else 0
    for k in range(1, 5):
        mask = S["tau_code"] == k
        grad[P + _P_T10 + k - 1] = res_h[mask].sum()

    # Kappa gradients
    for k in range(1, 5):
        mask = S["kappa_code"] == k
        grad[P + _P_K10 + k - 1] = res_a[mask].sum()

    # Rho gradients: affects both home and away base
    mask1 = S["is_rho1"]
    mask2 = S["is_rho2"]
    grad[P + _P_RHO1] = res_h[mask1].sum() + res_a[mask1].sum()
    grad[P + _P_RHO2] = res_h[mask2].sum() + res_a[mask2].sum()

    # Xi gradients (additive, not multiplicative)
    # d(LL)/d(xi1) = Σ (n_hg * t/lam_h - t/90)
    xi1_contrib = np.where(has_hg, n_hg * t / np.maximum(lam_h, EPS), 0.0) - t / 90.0
    xi2_contrib = np.where(has_ag, n_ag * t / np.maximum(lam_a, EPS), 0.0) - t / 90.0
    grad[P + _P_XI1] = xi1_contrib.sum()
    grad[P + _P_XI2] = xi2_contrib.sum()
    grad[P + _P_ZETA] = (res_h * S["psi_h"]).sum() + (res_a * S["psi_a"]).sum()   # d(base)/dζ = base·ψ

    return -ll, -grad


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARAMETER ESTIMATION (Vectorized MLE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def fit_birth_process(matches, verbose=True):
    """Fit Dixon-Robinson birth process model via L-BFGS-B with analytical gradient.

    Returns: team_alpha, team_beta, params
    """
    all_ids = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    K = len(all_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(all_ids)}

    if verbose:
        console.print(f"[cyan]Fitting birth process: {K} teams, {len(matches)} matches, {2*K+N_GLOBAL_ZOU} params[/]")

    S = _precompute_zou(matches, tid_to_idx)

    # Initial parameter vector
    xi0 = np.zeros(2 * K + N_GLOBAL_ZOU)
    # log_alpha, log_beta = 0 (alpha=beta=1)
    P = 2 * K
    xi0[P + _P_A] = np.log(1.35)      # log(a)
    # tau: all start at log(1)=0 except tau21, tau12
    xi0[P + _P_T21] = np.log(0.92)
    xi0[P + _P_T12] = np.log(1.14)
    xi0[P + _P_K10] = np.log(1.12)
    xi0[P + _P_K01] = np.log(1.08)
    xi0[P + _P_K21] = np.log(1.10)
    xi0[P + _P_K12] = np.log(1.04)
    xi0[P + _P_RHO1] = np.log(4.6)
    xi0[P + _P_RHO2] = np.log(7.7)
    xi0[P + _P_XI1] = 0.5
    xi0[P + _P_XI2] = 0.4

    # Identifiability constraints: sum(log_alpha) = 0 AND sum(log_beta) = 0
    A = np.zeros((2, 2 * K + N_GLOBAL_ZOU))
    A[0, :K] = 1.0       # sum(log_alpha) = 0
    A[1, K:2*K] = 1.0    # sum(log_beta) = 0
    constraint = LinearConstraint(A, [0.0] * 2, [0.0] * 2)

    # Bounds: xi1, xi2 >= 0
    bounds = [(None, None)] * (2 * K + N_GLOBAL_ZOU)
    bounds[P + _P_XI1] = (0.0, None)
    bounds[P + _P_XI2] = (0.0, None)

    res = minimize(_neg_loglik_grad_zou, xi0, args=(S, K), method="trust-constr",
                   jac=True, constraints=[constraint], bounds=bounds,
                   options={"maxiter": 500, "gtol": 1e-6})

    if not res.success:
        raise RuntimeError(f"birth process fit did not converge: {res.message}")
    if verbose:
        console.print(f"  LL={-res.fun:.2f}, nit={res.nit}")

    # Unpack results
    log_alpha, log_beta = res.x[:K], res.x[K:2*K]
    team_alpha = {tid: np.exp(log_alpha[i]) for i, tid in enumerate(all_ids)}
    team_beta = {tid: np.exp(log_beta[i]) for i, tid in enumerate(all_ids)}

    params = {
        "a": np.exp(res.x[P + _P_A]),
        "tau10": np.exp(res.x[P + _P_T10]), "tau01": np.exp(res.x[P + _P_T01]),
        "tau21": np.exp(res.x[P + _P_T21]), "tau12": np.exp(res.x[P + _P_T12]),
        "kappa10": np.exp(res.x[P + _P_K10]), "kappa01": np.exp(res.x[P + _P_K01]),
        "kappa21": np.exp(res.x[P + _P_K21]), "kappa12": np.exp(res.x[P + _P_K12]),
        "rho1": np.exp(res.x[P + _P_RHO1]), "rho2": np.exp(res.x[P + _P_RHO2]),
        "xi1": res.x[P + _P_XI1], "xi2": res.x[P + _P_XI2],
        "xi_shots": res.x[P + _P_ZETA],
    }

    if verbose:
        console.print(f"  a={params['a']:.3f}, xi1={params['xi1']:.3f}, xi2={params['xi2']:.3f}")
        console.print(f"  rho1={params['rho1']:.2f}, rho2={params['rho2']:.2f}")

    return team_alpha, team_beta, params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RECURSIVE ALGORITHM (Zou et al. 2018) — Exact outcome probabilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_tau_kappa_grids(params):
    """Precompute (M+1, M+1) tau and kappa multiplier grids for all score states."""
    M = MAX_GOALS
    X, Y = np.meshgrid(np.arange(M + 1), np.arange(M + 1), indexing='ij')
    lead = (X + Y > 1) & (X - Y >= 1)   # Home leading by 2+
    trail = (X + Y > 1) & (X - Y <= -1)  # Home trailing by 2+

    tau_g = np.ones((M + 1, M + 1))
    tau_g[1, 0], tau_g[0, 1] = params["tau10"], params["tau01"]
    tau_g[lead], tau_g[trail] = params["tau21"], params["tau12"]

    kap_g = np.ones((M + 1, M + 1))
    kap_g[1, 0], kap_g[0, 1] = params["kappa10"], params["kappa01"]
    kap_g[lead], kap_g[trail] = params["kappa21"], params["kappa12"]

    return tau_g, kap_g


# Precomputed comparison grids for H/D/A aggregation
_XG, _YG = np.meshgrid(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1), indexing='ij')
_HOME_MASK = _XG > _YG
_DRAW_MASK = _XG == _YG
_AWAY_MASK = _XG < _YG
_TOTAL_GOALS = _XG + _YG


def compute_outcome_probs(theta1, theta2, params, from_minute=0, start_x=0, start_y=0,
                          to_minute=90, shots_dev_h=0.0, shots_dev_a=0.0):
    """Compute exact P(H), P(D), P(A) via vectorized Markov chain over (M+1)^2 state grid.

    For minutes > 90: redistributes rho2 across stoppage minutes (rho2/S per minute)
    so that total expected scoring matches the paper's single-spike formulation.
    """
    a, M = params["a"], MAX_GOALS
    xi1, xi2 = params["xi1"], params["xi2"]
    mult_h, mult_a = np.exp(params["xi_shots"] * shots_dev_h), np.exp(params["xi_shots"] * shots_dev_a)   # shots held at their current value
    tau_g, kap_g = _build_tau_kappa_grids(params)
    stoppage = max(1, to_minute - 90) if to_minute > 90 else 0  # Estimated stoppage minutes

    prob = np.zeros((M + 1, M + 1))
    prob[start_x, start_y] = 1.0

    for minute in range(from_minute + 1, to_minute + 1):
        if minute == 45:
            r = params["rho1"]
        elif minute == 90 and to_minute <= 90:
            r = params["rho2"]  # Paper-standard: all stoppage compressed into min 90
        elif minute > 90:
            r = params["rho2"] / stoppage  # Redistribute rho2 across stoppage minutes
        else:
            r = 1.0
        t = min(minute, 90) / 90.0  # Cap at 1.0 for stoppage minutes
        lam_h = (a * theta1 * mult_h * tau_g * r + xi1 * t) / 90.0
        lam_a = (theta2 * mult_a * kap_g * r + xi2 * t) / 90.0

        scale = np.minimum(1.0, 0.95 / np.maximum(lam_h + lam_a, EPS))
        lam_h *= scale; lam_a *= scale

        new = prob * (1.0 - lam_h - lam_a)          # No-goal stay
        new[1:, :] += prob[:-1, :] * lam_h[:-1, :]  # Home goal: (x,y) -> (x+1,y)
        new[:, 1:] += prob[:, :-1] * lam_a[:, :-1]  # Away goal: (x,y) -> (x,y+1)
        prob = new

    p_h, p_d, p_a = prob[_HOME_MASK].sum(), prob[_DRAW_MASK].sum(), prob[_AWAY_MASK].sum()
    tot = p_h + p_d + p_a
    return (p_h / tot, p_d / tot, p_a / tot) if tot > 0 else (0.0, 0.0, 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BAYESIAN UPDATING (Zou Eqs 18-21)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_expected_goals_pair(theta1, theta2, params, from_min=0, to_min=45, psi_h=None, psi_a=None):
    """Compute E_H(T), E_A(T) jointly via vectorized Markov chain (ξ=0, Eq 17).
    psi_h/psi_a: shots deviations before minutes 1..to_min (None: typical shots, deviation 0)."""
    a, M = params["a"], MAX_GOALS
    tau_g, kap_g = _build_tau_kappa_grids(params)

    prob = np.zeros((M + 1, M + 1))
    prob[0, 0] = 1.0
    E_h, E_a = 0.0, 0.0

    for minute in range(from_min + 1, to_min + 1):
        r = params["rho1"] if minute == 45 else (params["rho2"] if minute == 90 else 1.0)
        mh = 1.0 if psi_h is None else np.exp(params["xi_shots"] * psi_h[minute - 1])
        ma = 1.0 if psi_a is None else np.exp(params["xi_shots"] * psi_a[minute - 1])
        lam_h = (a * theta1 * mh * tau_g * r) / 90.0  # No xi term (Bayesian conjugacy)
        lam_a = (theta2 * ma * kap_g * r) / 90.0

        scale = np.minimum(1.0, 0.95 / np.maximum(lam_h + lam_a, EPS))
        lam_h *= scale; lam_a *= scale

        E_h += (prob * lam_h).sum()
        E_a += (prob * lam_a).sum()

        new = prob * (1.0 - lam_h - lam_a)
        new[1:, :] += prob[:-1, :] * lam_h[:-1, :]
        new[:, 1:] += prob[:, :-1] * lam_a[:, :-1]
        prob = new

    return E_h, E_a


def bayesian_update(
    theta1_prior, theta2_prior, goals_h, goals_a, params, current_minute, psi_h=None, psi_a=None
):
    """Apply Zou's Bayesian update (Eqs 18-19).

    θ̂₁ = (r₁ + X(T)) / (r₁ + E_H(T)) × θ̂₀₁
    θ̂₂ = (r₂ + Y(T)) / (r₂ + E_A(T)) × θ̂₀₂

    Where r₁ = E_H(45), r₂ = E_A(45) (expected goals at halftime, under typical shots).
    E_H(T), E_A(T) use the shots deviations observed up to T: a known multiplier of the exposure,
    like τ and ρ, so the Gamma update stays conjugate.
    """
    # r = expected goals at halftime (shape parameter of Gamma prior)
    r1, r2 = compute_expected_goals_pair(
        theta1_prior, theta2_prior, params, from_min=0, to_min=45
    )

    # E[X(T)] and E[Y(T)] = expected goals from 0 to current_minute
    E_h, E_a = compute_expected_goals_pair(
        theta1_prior, theta2_prior, params, from_min=0, to_min=current_minute, psi_h=psi_h, psi_a=psi_a
    )

    # Bayesian posterior mean
    theta1_post = ((r1 + goals_h) / max(r1 + E_h, EPS)) * theta1_prior
    theta2_post = ((r2 + goals_a) / max(r2 + E_a, EPS)) * theta2_prior

    return theta1_post, theta2_post


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FULL PREDICTION PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def predict_match(
    home_id, away_id, team_alpha, team_beta, params,
    minute=0, goals_h=0, goals_a=0, to_minute=90, psi_h=None, psi_a=None, shots_dev_h=0.0, shots_dev_a=0.0,
):
    """Predict match outcome at given minute using Zou's method.

    At minute 0: pure birth process (no Bayesian update).
    At minute > 0: Bayesian update of theta using observed goals, then recursive algorithm.

    Returns: {"home": float, "draw": float, "away": float}
    """
    theta1_prior = team_alpha[home_id] * team_beta[away_id]
    theta2_prior = team_alpha[away_id] * team_beta[home_id]

    if minute == 0:
        theta1, theta2 = theta1_prior, theta2_prior
    else:
        theta1, theta2 = bayesian_update(
            theta1_prior, theta2_prior, goals_h, goals_a, params, minute, psi_h, psi_a
        )

    p_h, p_d, p_a = compute_outcome_probs(
        theta1, theta2, params, from_minute=minute,
        start_x=goals_h, start_y=goals_a, to_minute=to_minute,
        shots_dev_h=shots_dev_h, shots_dev_a=shots_dev_a,
    )

    return {"home": p_h, "draw": p_d, "away": p_a}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BETFAIR CALIBRATION — find θ₁, θ₂ matching target 1X2 probabilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_outcome_probs_full(theta1, theta2, params, from_minute=0, start_x=0, start_y=0,
                               to_minute=90, shots_dev_h=0.0, shots_dev_a=0.0):
    """Like compute_outcome_probs but also returns O/U probabilities."""
    a, M = params["a"], MAX_GOALS
    xi1, xi2 = params["xi1"], params["xi2"]
    mult_h, mult_a = np.exp(params["xi_shots"] * shots_dev_h), np.exp(params["xi_shots"] * shots_dev_a)   # shots held at their current value
    tau_g, kap_g = _build_tau_kappa_grids(params)
    stoppage = max(1, to_minute - 90) if to_minute > 90 else 0

    prob = np.zeros((M + 1, M + 1))
    prob[start_x, start_y] = 1.0

    for minute in range(from_minute + 1, to_minute + 1):
        if minute == 45:
            r = params["rho1"]
        elif minute == 90 and to_minute <= 90:
            r = params["rho2"]
        elif minute > 90:
            r = params["rho2"] / stoppage
        else:
            r = 1.0
        t = min(minute, 90) / 90.0  # Cap at 1.0 for stoppage minutes
        lam_h = (a * theta1 * mult_h * tau_g * r + xi1 * t) / 90.0
        lam_a = (theta2 * mult_a * kap_g * r + xi2 * t) / 90.0

        scale = np.minimum(1.0, 0.95 / np.maximum(lam_h + lam_a, EPS))
        lam_h *= scale; lam_a *= scale

        new = prob * (1.0 - lam_h - lam_a)
        new[1:, :] += prob[:-1, :] * lam_h[:-1, :]
        new[:, 1:] += prob[:, :-1] * lam_a[:, :-1]
        prob = new

    p_h, p_d, p_a = prob[_HOME_MASK].sum(), prob[_DRAW_MASK].sum(), prob[_AWAY_MASK].sum()
    tot = p_h + p_d + p_a
    if tot > 0:
        p_h, p_d, p_a = p_h / tot, p_d / tot, p_a / tot
    ou = {t: prob[_TOTAL_GOALS > t].sum() / tot if tot > 0 else 0.5
          for t in [0.5, 1.5, 2.5, 3.5, 4.5]}
    return p_h, p_d, p_a, ou


def calibrate_thetas_full(target_1x2, ou_targets, params, t1_init=1.0, t2_init=1.0, shots_dev=0.0):
    """Find (θ₁, θ₂) matching 1X2 + all available O/U markets (same as Weibull calibration).
    shots_dev: kick-off shots deviation, so the minute-0 forecast reproduces the fitted prices."""
    log_t1_init = np.log(max(t1_init, 0.01))
    log_t2_init = np.log(max(t2_init, 0.01))

    def obj(log_t):
        t1, t2 = np.exp(log_t)
        ph, pd, pa, ou_probs = compute_outcome_probs_full(t1, t2, params, shots_dev_h=shots_dev, shots_dev_a=shots_dev)
        err_1x2 = ((ph - target_1x2['p_home'])**2 +
                   (pd - target_1x2['p_draw'])**2 +
                   (pa - target_1x2['p_away'])**2)
        err_ou = sum((ou_probs[t] - p)**2 for t, p in ou_targets.items())
        return err_1x2 + err_ou

    res = minimize(obj, [log_t1_init, log_t2_init],
                   method="Powell", options={"maxiter": 200, "ftol": 1e-6})
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"Zou calibration failed: {res.message}")
    return np.exp(res.x[0]), np.exp(res.x[1]), res.fun


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPANDING-WINDOW ESTIMATION (mirrors Weibull team_params pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os, pickle
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from models.data import load_seasons
from models.weibull.clock import training_events


WORKERS = 12  # parallel snapshot fits


def _fit_quiet(matches):
    return fit_birth_process(matches, verbose=False)


def estimate_expanding(seasons, root, eval_season=None, out=None):
    """Fit birth process with an expanding window, one snapshot per eval-season match date (see
    eval.utils.snapshot_cuts). Historical seasons = fixed pool.

    Args:
        seasons: list of season strings, e.g. ["21-22", "22-23", "23-24", "24-25"]
        eval_season: season for expanding window (default: last in list)

    Run: python -m models.zoubirthprocess.model --seasons 21-22 22-23 23-24 24-25
    """
    if isinstance(seasons, str):
        seasons = [seasons]
    eval_season = eval_season or seasons[-1]

    ws, align, sched = load_seasons(seasons, root)
    ws = training_events(ws)   # same reviewed exclusion as the Weibull fit

    # Map ws_match_id -> gameweek, date, home team
    fbref_to_gw = dict(zip(sched["match_id"], sched["week"]))
    ws_to_gw = {r["ws_game_id"]: fbref_to_gw.get(r["fbref_match_id"])
                for _, r in align.iterrows()}
    ws_to_date = dict(zip(align["ws_game_id"], pd.to_datetime(align["date"])))
    ws_to_home = dict(zip(align["ws_game_id"], align["ws_home_team"]))
    ws["gameweek"] = ws["ws_match_id"].map(ws_to_gw)
    ws["match_date"] = ws["ws_match_id"].map(ws_to_date)
    ws["is_home"] = ws.apply(lambda r: r["team_name"] == ws_to_home.get(r["ws_match_id"]), axis=1)

    # Match-level metadata
    matches = (ws.groupby("ws_match_id")
               .agg({"match_date": "first", "gameweek": "first", "season": "first"})
               .sort_values("match_date").dropna(subset=["gameweek"]))

    # Historical match IDs (always in training pool)
    hist_mids = set(matches[matches["season"] != eval_season].index)
    eval_matches = matches[matches["season"] == eval_season].copy()
    n_hist = len(hist_mids)
    console.print(f"Historical: {n_hist} matches from {[s for s in seasons if s != eval_season]}")
    from eval.utils import snapshot_cuts
    cuts = snapshot_cuts(eval_matches["match_date"], eval_season)
    console.print(f"Eval season {eval_season}: {len(eval_matches)} matches, {len(cuts)} snapshots from {cuts[0].date()}")

    out = out or os.path.join(root, "models", "zoubirthprocess", "params",
                              f"pl_{eval_season.replace('-', '_')}")
    os.makedirs(out, exist_ok=True)
    # Shots deviation from the pre-evaluation population curve, as for the Cox process
    from models.strategies.extended import build_cov_lookup, compute_dev_fit
    from eval.utils import SEASON_CFG
    fb = pd.concat([pd.read_parquet(os.path.join(root, "data", s, "fbref_shots.parquet")).assign(season=s)
                    for s in seasons], ignore_index=True)
    pre = align.loc[pd.to_datetime(align["date"]) < pd.to_datetime(SEASON_CFG[eval_season][1]), "ws_game_id"]
    dev_shots_fit = compute_dev_fit(ws[ws["ws_match_id"].isin(pre)], fb, align, "shots")
    console.print(f"shots population fit: slope={dev_shots_fit[0]:.6f}, intercept={dev_shots_fit[1]:.6f}")
    cov_lookup = build_cov_lookup(ws, fb, align, period_keys=True, fit_meta={"dev_shots_fit": dev_shots_fit})
    dev0 = -dev_shots_fit[1]   # no shots yet at kick-off
    by_id = {m["match_id"]: m for m in extract_match_data(ws[ws["ws_match_id"].isin(matches.index)], align,
                                                          cov_lookup=cov_lookup, dev0=dev0)}
    jobs = []
    for cut in cuts:   # one fit per snapshot: history plus eval-season matches up to the cut
        mids = hist_mids | set(eval_matches[eval_matches["match_date"] <= cut].index)
        jobs.append((cut, [by_id[i] for i in sorted(mids) if i in by_id]))
    date_to_path = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for (cut, train), fit in zip(jobs, pool.map(_fit_quiet, [j[1] for j in jobs])):
            console.print(f"{cut.date()}: {len(train)} matches fitted")
            fit[2]["dev_shots_fit"] = tuple(float(v) for v in dev_shots_fit)
            path = os.path.join(out, f"params_{cut:%Y%m%d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(dict(zip(("team_alpha", "team_beta", "bp_params"), fit)), f)
            date_to_path[str(cut.date())] = os.path.basename(path)  # name only: readers resolve it in the index's dir

    idx = {"season": eval_season, "param_files": date_to_path, "dates": sorted(date_to_path)}
    with open(os.path.join(out, "index.pkl"), "wb") as f:
        pickle.dump(idx, f)
    console.print(f"[green]Saved {len(date_to_path)} param files ({cuts[0].date()} to {cuts[-1].date()}) to {out}[/]")
    return date_to_path


class ZouParamBaseline:
    """Load pre-computed birth process params, dispatch by match date."""

    def __init__(self, params_dir):
        idx_path = os.path.join(params_dir, "index.pkl")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"No index.pkl in {params_dir}. "
                "Run: python -m models.zoubirthprocess.model --season 24-25")
        with open(idx_path, "rb") as f:
            self.index = pickle.load(f)
        self.dir = params_dir
        self.param_dates = sorted(self.index["dates"])
        self.cache = {}
        console.print(f"Zou params: {len(self.param_dates)} dates "
                       f"({self.param_dates[0]} to {self.param_dates[-1]})")

    def get_params_for_date(self, match_date):
        """Return (team_alpha, team_beta, bp_params) for most recent fit before match_date."""
        dt = str(pd.to_datetime(match_date).date())
        valid = None
        for d in self.param_dates:
            if d < dt:
                valid = d
            else:
                break
        if valid is None:
            raise ValueError(f"No params before {dt}. Earliest: {self.param_dates[0]}")
        if valid not in self.cache:
            # basename: pre-2026-09-23 indexes stored absolute paths
            with open(os.path.join(self.dir, os.path.basename(self.index["param_files"][valid])), "rb") as f:
                self.cache[valid] = pickle.load(f)
        p = self.cache[valid]
        return p["team_alpha"], p["team_beta"], p["bp_params"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", default=None, help="Default: all seasons through --eval-season")
    p.add_argument("--eval-season", default="24-25", help="Season for expanding window")
    args = p.parse_args()
    from eval.utils import seasons_through
    args.seasons = args.seasons or seasons_through(args.eval_season)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    estimate_expanding(args.seasons, root, eval_season=args.eval_season)
