"""Maia et al. (2025) Cox process model with dynamic regressors.

Full joint model: goals + red cards + stoppage time (minus market value regressor).

Goal intensity (Model G0+G4, no market value, +shots per-team deviation):
  lambda_home(t) = alpha_h * beta_a * exp(gamma + xi_half*I(2H) + xi_goals*gd_h + xi_rc*rc_diff_h + xi_shots*shots_dev_h)
  lambda_away(t) = alpha_a * beta_h * exp(xi_half*I(2H) - xi_goals*gd_h - xi_rc*rc_diff_h + xi_shots*shots_dev_a)
  where shots_dev_k(t) = cum_shots_k(t) - population_mean(t), mirroring Weibull.

Red card intensity (NHPP, power-law, team-independent):
  rho_home(t) = exp(c_h) * t^(b_h)
  rho_away(t) = exp(c_a) * t^(b_a)

Stoppage time (Poisson, model S5):
  U1 ~ Poisson(exp(s1_0 + s1_rc * total_rc_1h + s1_g * goals_1h))
  U2 ~ Poisson(exp(s2_0 + s2_rc * rc_2h + s2_g * goals_2h + s2_close * I(|gd|<=1)))

Parameter vector: xi = [log_alpha(K), log_beta(K), gamma, xi_half, xi_goals, xi_rc, xi_shots,
                         c_h, b_h, c_a, b_a, s1_0, s1_rc, s1_g, s2_0, s2_rc, s2_g, s2_close]
Total: 2K + 16 parameters.

Identifiability: sum(alpha) = sum(beta)  (nonlinear constraint in log-space).
Simulation: continuous-time event-driven (exact for piecewise-constant intensity).

Reference: Maia, Pennanen, Da Silva & Targino (2025), EJOR.
R code: https://github.com/luizfgnmaia/football-dynamic-regressors
"""

import os, pickle
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.optimize import minimize, LinearConstraint
from models.data import load_seasons
from scipy.special import gammaln  # log(n!)
from scipy.stats import poisson

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_SIMS = 100000
# Indices into the non-team part of xi (offset from 2*K)
_G, _H, _GD, _RC, _SHOTS = 0, 1, 2, 3, 4  # Goal model: gamma, xi_half, xi_goals, xi_rc, xi_shots
_CH, _BH, _CA, _BA = 5, 6, 7, 8          # Red card: c_h, b_h, c_a, b_a
_S10, _S1R, _S1G, _S20, _S2R, _S2G, _S2C = 9, 10, 11, 12, 13, 14, 15  # Stoppage (S5)
N_GLOBAL = 16  # Total non-team parameters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def extract_match_data(ws_events, align, cov_lookup=None):
    """Extract goal/red card times + half boundary + stoppage from WhoScored events.

    Args:
        cov_lookup: optional {(match_id, minute): {shots_home, shots_away, ...}}
                    from build_cov_lookup. Provides shots covariate for goal intensity.

    Returns list of dicts with actual minute values and derived stoppage fields.
    """
    ws_to_home = dict(zip(align["ws_game_id"], align["ws_home_team"]))
    matches = []

    for mid in ws_events["ws_match_id"].unique():
        if mid not in ws_to_home:
            continue
        home_name = ws_to_home[mid]
        evts = ws_events[ws_events["ws_match_id"] == mid]

        home_evts = evts[evts["team_name"] == home_name]
        away_evts = evts[evts["team_name"] != home_name]
        if len(home_evts) == 0 or len(away_evts) == 0:
            continue
        home_id = int(home_evts["team_id"].iloc[0])
        away_id = int(away_evts["team_id"].iloc[0])

        fh = evts[evts["period"] == "FirstHalf"]
        sh = evts[evts["period"] == "SecondHalf"]
        half_end = int(fh["minute"].max()) if len(fh) > 0 else 45
        match_end = int(sh["minute"].max()) if len(sh) > 0 else 90

        played = evts[evts["period"].isin(["FirstHalf", "SecondHalf"])]
        goals_df = played[played["is_goal"] == True]
        if len(goals_df) != int((evts["is_goal"] == True).sum()):
            raise ValueError(f"match {mid}: goal outside the two halves")
        goal_times = []
        for _, g in goals_df.iterrows():
            is_own = bool(g.get("is_own_goal", False))
            scorer_home = g["team_name"] == home_name
            goal_times.append((int(g["minute"]), scorer_home if not is_own else not scorer_home, g["period"] == "FirstHalf"))
        goal_times.sort(key=lambda x: (not x[2], x[0]))   # first half, then second: their minutes overlap in stoppage

        cards = played[(played["event_type"] == "Card") & (played["card_type"].isin(["Red", "SecondYellow"]))]
        rc_times = [(int(c["minute"]), c["team_name"] == home_name, c["period"] == "FirstHalf") for _, c in cards.iterrows()]
        rc_times.sort(key=lambda x: (not x[2], x[0]))

        # Derived fields for stoppage model
        u1 = max(half_end - 45, 0)  # 1H stoppage minutes
        u2 = max(match_end - 90, 0)  # 2H stoppage minutes
        rc_1h = sum(1 for _, _, is_1h in rc_times if is_1h)       # Total red cards in 1H
        rc_2h = sum(1 for _, _, is_1h in rc_times if not is_1h)   # Total red cards in 2H only
        goals_1h = sum(1 for _, _, is_1h in goal_times if is_1h)  # Goals in 1H
        goals_2h = sum(1 for _, _, is_1h in goal_times if not is_1h)  # Goals in 2H
        # Final score for close-match indicator
        final_h = sum(1 for _, is_h, _ in goal_times if is_h)
        final_a = sum(1 for _, is_h, _ in goal_times if not is_h)

        # Shots deviation per (period, minute); a minute past the covariate grid carries the half's last value
        shots_by_min = {}
        if cov_lookup:
            for period, t0, t1 in [(1, 0, half_end), (2, 45, match_end)]:
                last = None
                for t in range(t0, t1 + 1):
                    c = cov_lookup.get((int(mid), t, period))
                    if c is not None:
                        last = (float(c["dev_shots_home"]), float(c["dev_shots_away"]))
                    if last is None:
                        raise KeyError(f"match {mid}: no shots covariate at period {period} minute {t}")
                    shots_by_min[(period, t)] = last

        matches.append({
            "match_id": int(mid), "home_id": home_id, "away_id": away_id,
            "goal_times": goal_times, "redcard_times": rc_times,   # (minute, is_home, is_first_half)
            "half_end": half_end, "match_end": match_end,
            # Stoppage model data
            "u1": u1, "u2": u2, "rc_1h": rc_1h, "rc_2h": rc_2h,
            "goals_1h": goals_1h, "goals_2h": goals_2h,
            "close_match": int(abs(final_h - final_a) <= 1),
            # Continuous-time red card event times for NHPP LL
            "rc_home_times": [t for t, is_h, _ in rc_times if is_h],
            "rc_away_times": [t for t, is_h, _ in rc_times if not is_h],
            "shots_by_min": shots_by_min,
        })
    return matches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONTINUOUS-TIME JOINT LOG-LIKELIHOOD WITH ANALYTICAL GRADIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_intervals(match):
    """Piecewise-constant intervals for the goal process, built half by half: the first half runs
    [0, half_end] and the second [45, match_end], so stoppage-time minutes shared by both halves stay apart.

    Returns list of dicts: {t_start, t_end, is_2h, gd_h, rc_diff_h, shots_dev_h, shots_dev_a, n_hg, n_ag}
    """
    shots = match.get("shots_by_min", {})
    gd_h, rc_h, rc_a, intervals = 0, 0, 0, []
    for period, t0, t1 in [(1, 0, match["half_end"]), (2, 45, match["match_end"])]:
        first = period == 1
        events = sorted([(t, 0, h) for t, h, p in match["redcard_times"] if p == first] +
                        [(t, 1, h) for t, h, p in match["goal_times"] if p == first])   # red cards before goals in a minute
        half, t_cur = [], t0

        def add(a, b):
            dev_h, dev_a = shots[(period, int(a))] if shots else (0.0, 0.0)
            half.append({"t_start": a, "t_end": b, "is_2h": float(period == 2), "gd_h": gd_h,
                         "rc_diff_h": rc_a - rc_h, "shots_dev_h": dev_h, "shots_dev_a": dev_a, "n_hg": 0, "n_ag": 0})

        for t, kind, is_home in events:
            if not t0 <= t <= t1:
                raise ValueError(f"match {match['match_id']}: event at minute {t} outside period {period}")
            if t > t_cur:
                add(t_cur, t); t_cur = t
            if kind == 1:
                if not half or half[-1]["t_end"] < t:
                    add(t, t)
                half[-1]["n_hg" if is_home else "n_ag"] += 1
                gd_h += 1 if is_home else -1
            elif is_home:
                rc_h += 1
            else:
                rc_a += 1
        if t_cur < t1:
            add(t_cur, t1)
        intervals += half
    return intervals


def _precompute_stacked(matches, tid_to_idx):
    """Precompute all match data into padded arrays for vectorised LL."""
    N = len(matches)
    all_intervals = [_build_intervals(m) for m in matches]
    max_iv = max(len(iv) for iv in all_intervals)

    hids = np.empty(N, dtype=np.int32)
    aids = np.empty(N, dtype=np.int32)
    dt = np.zeros((N, max_iv))
    is_2h = np.zeros((N, max_iv))
    gd_h = np.zeros((N, max_iv))
    rc_diff_h = np.zeros((N, max_iv))
    shots_dev_h = np.zeros((N, max_iv))
    shots_dev_a = np.zeros((N, max_iv))
    n_hg = np.zeros((N, max_iv), dtype=np.int32)
    n_ag = np.zeros((N, max_iv), dtype=np.int32)

    # Per-match scalars for stoppage + red card LL
    match_end = np.empty(N)
    u1 = np.empty(N, dtype=np.int32)
    u2 = np.empty(N, dtype=np.int32)
    rc_1h = np.empty(N, dtype=np.int32)
    rc_2h = np.empty(N, dtype=np.int32)
    goals_1h = np.empty(N, dtype=np.int32)
    goals_2h = np.empty(N, dtype=np.int32)
    close_match = np.empty(N, dtype=np.int32)
    # Red card event times: ragged -> store as lists
    rc_home_times = []
    rc_away_times = []

    for mi, m in enumerate(matches):
        hids[mi] = tid_to_idx[m["home_id"]]
        aids[mi] = tid_to_idx[m["away_id"]]
        ivs = all_intervals[mi]
        for j, iv in enumerate(ivs):
            dt[mi, j] = iv["t_end"] - iv["t_start"]
            is_2h[mi, j] = iv["is_2h"]
            gd_h[mi, j] = iv["gd_h"]
            rc_diff_h[mi, j] = iv["rc_diff_h"]
            shots_dev_h[mi, j] = iv["shots_dev_h"]
            shots_dev_a[mi, j] = iv["shots_dev_a"]
            n_hg[mi, j] = iv["n_hg"]
            n_ag[mi, j] = iv["n_ag"]
        match_end[mi] = m["match_end"]
        u1[mi] = m["u1"]
        u2[mi] = m["u2"]
        rc_1h[mi] = m["rc_1h"]
        rc_2h[mi] = m["rc_2h"]
        goals_1h[mi] = m["goals_1h"]
        goals_2h[mi] = m["goals_2h"]
        close_match[mi] = m["close_match"]
        rc_home_times.append(np.array(m["rc_home_times"], dtype=np.float64))
        rc_away_times.append(np.array(m["rc_away_times"], dtype=np.float64))

    mask = np.zeros((N, max_iv), dtype=bool)
    for mi in range(N):
        mask[mi, :len(all_intervals[mi])] = True

    return {"hids": hids, "aids": aids, "dt": dt, "is_2h": is_2h,
            "gd_h": gd_h, "rc_diff_h": rc_diff_h,
            "shots_dev_h": shots_dev_h, "shots_dev_a": shots_dev_a,
            "n_hg": n_hg, "n_ag": n_ag, "mask": mask,
            "match_end": match_end, "u1": u1, "u2": u2,
            "rc_1h": rc_1h, "rc_2h": rc_2h,
            "goals_1h": goals_1h, "goals_2h": goals_2h, "close_match": close_match,
            "rc_home_times": rc_home_times, "rc_away_times": rc_away_times}


def neg_loglik_grad(xi, S, K):
    """Joint negative log-likelihood and gradient: goals + red cards + stoppage."""
    P = 2*K  # Offset for global params
    log_alpha, log_beta = xi[:K], xi[K:P]
    gamma    = xi[P + _G]
    xi_half  = xi[P + _H]
    xi_goals = xi[P + _GD]
    xi_rc    = xi[P + _RC]
    xi_shots  = xi[P + _SHOTS]
    c_h, b_h = xi[P + _CH], xi[P + _BH]
    c_a, b_a = xi[P + _CA], xi[P + _BA]
    s1_0, s1_rc, s1_g = xi[P + _S10], xi[P + _S1R], xi[P + _S1G]
    s2_0, s2_rc, s2_g, s2_close = xi[P + _S20], xi[P + _S2R], xi[P + _S2G], xi[P + _S2C]

    hids, aids = S["hids"], S["aids"]
    mask = S["mask"]
    N = len(hids)
    grad = np.zeros_like(xi)

    # ── GOAL PROCESS LL ──────────────────────────────────────────────────────
    log_t1 = log_alpha[hids] + log_beta[aids]
    log_t2 = log_alpha[aids] + log_beta[hids]

    log_lam_h = (log_t1[:, None] + gamma + xi_half * S["is_2h"] + xi_goals * S["gd_h"]
                 + xi_rc * S["rc_diff_h"] + xi_shots * S["shots_dev_h"])
    log_lam_a = (log_t2[:, None] + xi_half * S["is_2h"] - xi_goals * S["gd_h"]
                 - xi_rc * S["rc_diff_h"] + xi_shots * S["shots_dev_a"])
    lam_h = np.exp(log_lam_h)
    lam_a = np.exp(log_lam_a)

    ll_goals = (S["n_hg"] * log_lam_h + S["n_ag"] * log_lam_a
                - lam_h * S["dt"] - lam_a * S["dt"])
    ll_goals = (ll_goals * mask).sum()

    res_h = ((S["n_hg"] - lam_h * S["dt"]) * mask)
    res_a = ((S["n_ag"] - lam_a * S["dt"]) * mask)
    res_h_sum = res_h.sum(axis=1)
    res_a_sum = res_a.sum(axis=1)

    np.add.at(grad[:K], hids, res_h_sum)
    np.add.at(grad[:K], aids, res_a_sum)
    np.add.at(grad[K:P], aids, res_h_sum)
    np.add.at(grad[K:P], hids, res_a_sum)
    grad[P + _G]    = res_h.sum()
    grad[P + _H]    = (res_h * S["is_2h"]).sum() + (res_a * S["is_2h"]).sum()
    grad[P + _GD]   = (res_h * S["gd_h"]).sum() - (res_a * S["gd_h"]).sum()
    grad[P + _RC]   = (res_h * S["rc_diff_h"]).sum() - (res_a * S["rc_diff_h"]).sum()
    grad[P + _SHOTS] = (res_h * S["shots_dev_h"]).sum() + (res_a * S["shots_dev_a"]).sum()

    # ── RED CARD PROCESS LL (NHPP, power-law) ────────────────────────────────
    # rho(t) = exp(c) * t^b  =>  log rho(t) = c + b*log(t)
    # Lambda(T) = exp(c) * T^(b+1) / (b+1)
    # LL = sum_events [c + b*log(t)] - sum_matches [Lambda(T)]
    T = S["match_end"]  # (N,)
    log_T = np.log(np.maximum(T, 1e-10))

    # Home red cards — clamp b+1 to avoid log(0)
    bp1_h = max(b_h + 1, 1e-6)
    surv_h = np.exp(c_h + bp1_h * log_T - np.log(bp1_h))  # Lambda_h(T) per match
    ll_rc_h = -surv_h.sum()
    grad[P + _CH] += -surv_h.sum()
    grad[P + _BH] += -(surv_h * (log_T - 1.0 / bp1_h)).sum()

    for mi in range(N):
        ts = S["rc_home_times"][mi]
        if len(ts) > 0:
            log_ts = np.log(np.maximum(ts, 0.5))  # Floor at 0.5 min to avoid log(0)
            ll_rc_h += (c_h + b_h * log_ts).sum()
            grad[P + _CH] += len(ts)       # d/dc_h of event term = 1 per event
            grad[P + _BH] += log_ts.sum()  # d/db_h of event term = log(t) per event

    # Away red cards — clamp b+1 to avoid log(0)
    bp1_a = max(b_a + 1, 1e-6)
    surv_a = np.exp(c_a + bp1_a * log_T - np.log(bp1_a))
    ll_rc_a = -surv_a.sum()
    grad[P + _CA] += -surv_a.sum()
    grad[P + _BA] += -(surv_a * (log_T - 1.0 / bp1_a)).sum()

    for mi in range(N):
        ts = S["rc_away_times"][mi]
        if len(ts) > 0:
            log_ts = np.log(np.maximum(ts, 0.5))
            ll_rc_a += (c_a + b_a * log_ts).sum()
            grad[P + _CA] += len(ts)
            grad[P + _BA] += log_ts.sum()

    # ── STOPPAGE TIME LL (Poisson, model S5) ─────────────────────────────────
    log_pi1 = s1_0 + s1_rc * S["rc_1h"] + s1_g * S["goals_1h"]
    log_pi2 = s2_0 + s2_rc * S["rc_2h"] + s2_g * S["goals_2h"] + s2_close * S["close_match"]
    pi1, pi2 = np.exp(log_pi1), np.exp(log_pi2)
    u1, u2 = S["u1"].astype(np.float64), S["u2"].astype(np.float64)

    ll_st = (u1 * log_pi1 - pi1 - gammaln(u1 + 1)
             + u2 * log_pi2 - pi2 - gammaln(u2 + 1)).sum()

    # Gradient: d/d(param) = (U - pi) * d(log_pi)/d(param)
    res_st1 = u1 - pi1  # (N,)
    res_st2 = u2 - pi2
    grad[P + _S10] = res_st1.sum()
    grad[P + _S1R] = (res_st1 * S["rc_1h"]).sum()
    grad[P + _S1G] = (res_st1 * S["goals_1h"]).sum()
    grad[P + _S20] = res_st2.sum()
    grad[P + _S2R] = (res_st2 * S["rc_2h"]).sum()
    grad[P + _S2G] = (res_st2 * S["goals_2h"]).sum()
    grad[P + _S2C] = (res_st2 * S["close_match"]).sum()

    # ── TOTAL ────────────────────────────────────────────────────────────────
    ll = ll_goals + ll_rc_h + ll_rc_a + ll_st
    return -ll, -grad


def fit_maia(matches, verbose=True):
    """Fit full Maia joint model (goals + red cards + stoppage) via MLE.

    Uses trust-constr with LinearConstraint: sum(log_alpha) = sum(log_beta).
    """
    all_ids = sorted({m["home_id"] for m in matches} | {m["away_id"] for m in matches})
    K = len(all_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(all_ids)}

    if verbose:
        console.print(f"[cyan]Fitting Maia model: {K} teams, {len(matches)} matches, "
                       f"{2*K + N_GLOBAL} params[/]")

    S = _precompute_stacked(matches, tid_to_idx)

    xi0 = np.zeros(2*K + N_GLOBAL)
    xi0[:K] = -2.15       # log(alpha) ~ attack
    xi0[K:2*K] = -2.15    # log(beta) ~ defense  (satisfies sum(log_a)=sum(log_b))
    P = 2*K
    xi0[P + _G]    = 0.4    # gamma (home advantage)
    xi0[P + _H]    = 0.19   # xi_half
    xi0[P + _GD]   = -0.09  # xi_goals
    xi0[P + _RC]   = 0.37   # xi_rc
    xi0[P + _SHOTS] = 0.0    # xi_shots (per-team shots deviation effect)
    xi0[P + _CH]   = -6.0   # c_h (red cards are rare: exp(-6) * t^1 ~ 0.002/min)
    xi0[P + _BH] = 1.0    # b_h (increasing power-law)
    xi0[P + _CA] = -6.0   # c_a
    xi0[P + _BA] = 1.0    # b_a
    xi0[P + _S10] = 1.0   # s1_0: log(pi1) ~ 1.0 => pi1 ~ 2.7 min
    xi0[P + _S1R] = 0.1   # s1_rc
    xi0[P + _S1G] = 0.0   # s1_g: goals in 1H (near-zero effect expected)
    xi0[P + _S20] = 1.7   # s2_0: log(pi2) ~ 1.7 => pi2 ~ 5.5 min
    xi0[P + _S2R] = 0.1   # s2_rc
    xi0[P + _S2G] = 0.0   # s2_g: goals in 2H
    xi0[P + _S2C] = 0.1   # s2_close

    # Identifiability: sum(log_alpha) = sum(log_beta)  (Maia et al. Sec 5.1)
    A = np.zeros((1, len(xi0)))
    A[0, :K] = 1.0       # sum(log_alpha)
    A[0, K:2*K] = -1.0   # - sum(log_beta)
    constraint = LinearConstraint(A, 0.0, 0.0)

    # Bounds: b_h, b_a > -0.5 (so b+1 > 0.5, ensuring Lambda is finite)
    bounds = [(None, None)] * (2*K + N_GLOBAL)
    bounds[P + _BH] = (-0.5, None)
    bounds[P + _BA] = (-0.5, None)

    res = minimize(neg_loglik_grad, xi0, args=(S, K), method="trust-constr",
                   jac=True, constraints=[constraint], bounds=bounds,
                   options={"maxiter": 1000, "gtol": 1e-8})

    if verbose:
        console.print(f"  Converged: {res.success}, LL={-res.fun:.2f}, nit={res.nit}")

    log_alpha, log_beta = res.x[:K], res.x[K:2*K]
    team_alpha = {tid: np.exp(log_alpha[i]) for i, tid in enumerate(all_ids)}
    team_beta = {tid: np.exp(log_beta[i]) for i, tid in enumerate(all_ids)}

    params = {
        "gamma": res.x[P + _G], "xi_half": res.x[P + _H],
        "xi_goals": res.x[P + _GD], "xi_rc": res.x[P + _RC],
        "xi_shots": res.x[P + _SHOTS],
        "rc_home": (res.x[P + _CH], res.x[P + _BH]),   # (c_h, b_h)
        "rc_away": (res.x[P + _CA], res.x[P + _BA]),    # (c_a, b_a)
        "st1": (res.x[P + _S10], res.x[P + _S1R], res.x[P + _S1G]),         # (s1_0, s1_rc, s1_g)
        "st2": (res.x[P + _S20], res.x[P + _S2R], res.x[P + _S2G], res.x[P + _S2C]),  # (s2_0, s2_rc, s2_g, s2_close)
    }

    if verbose:
        console.print(f"  gamma={params['gamma']:.3f} (home={np.exp(params['gamma']):.2f}x)")
        console.print(f"  xi_half={params['xi_half']:.3f} (2H={np.exp(params['xi_half']):.2f}x)")
        console.print(f"  xi_goals={params['xi_goals']:.3f} (gd={np.exp(params['xi_goals']):.2f}x)")
        console.print(f"  xi_rc={params['xi_rc']:.3f} (rc={np.exp(params['xi_rc']):.2f}x)")
        console.print(f"  xi_shots={params['xi_shots']:.3f} (shots={np.exp(params['xi_shots']):.2f}x)")
        c_h, b_h = params["rc_home"]
        c_a, b_a = params["rc_away"]
        console.print(f"  RC home: c={c_h:.3f}, b={b_h:.3f}  |  away: c={c_a:.3f}, b={b_a:.3f}")
        s1_0, s1_rc, s1_g = params["st1"]
        s2_0, s2_rc, s2_g, s2_close = params["st2"]
        console.print(f"  ST 1H: mean={np.exp(s1_0):.1f}min (rc={s1_rc:.3f}, goals={s1_g:.3f})")
        console.print(f"  ST 2H: mean={np.exp(s2_0):.1f}min (rc={s2_rc:.3f}, goals={s2_g:.3f}, close={s2_close:.3f})")
        avg = np.mean(list(team_alpha.values())) * np.mean(list(team_beta.values()))
        console.print(f"  avg goal rate={avg:.4f}/min ({avg*90:.2f}/90)")

    return team_alpha, team_beta, params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MONTE CARLO SIMULATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _sim_rc_times(c, b, t_start, t_max, n_sims):
    """Simulate red card times via NHPP inversion. Returns (n_sims, 5) padded with inf."""
    bp1 = b + 1.0
    s = np.full(n_sims, np.exp(c) * t_start**bp1 / bp1) if t_start > 0 else np.zeros(n_sims)
    times = np.full((n_sims, 5), np.inf)
    for j in range(5):
        s += np.random.exponential(1.0, n_sims)
        times[:, j] = ((s * bp1) / np.exp(c)) ** (1.0 / bp1)
    return times


def _sim_phase_ct(x, y, rc_h_t, rc_a_t, log_base_h, log_base_a,
                  half_val, xi_goals, xi_rc, t_start, t_end, n):
    """Simulate one phase via continuous-time competing events. Mutates x, y in-place.

    t_end: scalar or (n,) array for per-sim endpoints (variable stoppage).
    At each step: draw next goal time from Exp(lam_h + lam_a), check against
    next pre-generated RC time and phase end. Whichever comes first determines
    the event. RC state is recomputed from rc_*_t arrays each iteration.
    """
    MAX_EVENTS = 50
    t_end_arr = np.broadcast_to(np.asarray(t_end, dtype=np.float64), n).copy()
    t = np.full(n, float(t_start) if np.isscalar(t_start) else t_start, dtype=np.float64)

    for _ in range(MAX_EVENTS):
        alive = t < t_end_arr
        if not alive.any():
            break
        # RC state from pre-generated times
        gd = (x - y).astype(np.float64)
        rc_diff = ((rc_a_t <= t[:, None]).sum(axis=1)
                    - (rc_h_t <= t[:, None]).sum(axis=1)).astype(np.float64)
        # Per-minute goal rates
        lam_h = np.exp(log_base_h + half_val + xi_goals * gd + xi_rc * rc_diff)
        lam_a = np.exp(log_base_a + half_val - xi_goals * gd - xi_rc * rc_diff)
        lam_tot = lam_h + lam_a
        # Next goal time ~ Exp(lam_tot)
        t_goal = t + (-np.log(np.random.uniform(1e-15, 1.0, n)) / np.maximum(lam_tot, 1e-15))
        # Next RC time (min of all future pre-generated RC events)
        future = np.minimum(
            np.where(rc_h_t > t[:, None], rc_h_t, np.inf).min(axis=1),
            np.where(rc_a_t > t[:, None], rc_a_t, np.inf).min(axis=1))
        # Advance to whichever comes first
        t_next = np.minimum(np.minimum(t_goal, future), t_end_arr)
        is_goal = alive & (t_goal <= future) & (t_goal <= t_end_arr)
        is_home = np.random.uniform(0, 1, n) < (lam_h / np.maximum(lam_tot, 1e-15))
        x += (is_goal & is_home).astype(np.int32)
        y += (is_goal & ~is_home).astype(np.int32)
        t = np.where(alive, t_next, t)


def _stoppage(pi, played, cap):
    """Poisson(pi) stoppage minutes given that more than `played` have elapsed (played < 0: unconditional)."""
    if played < 0:
        return np.minimum(np.random.poisson(pi), cap).astype(np.float64)
    lo = poisson.cdf(played, pi)
    u = poisson.ppf(np.minimum(lo + np.random.uniform(0, 1, np.shape(pi)) * (1 - lo), 1 - 1e-12), pi)
    return np.maximum(np.minimum(np.where(np.isfinite(u), u, played + 1), cap), played + 1).astype(np.float64)


def _sim_match_core(theta1, theta2, params, from_min=0, start_x=0, start_y=0,
                    rc_h=0, rc_a=0, shots_dev_h=0.0, shots_dev_a=0.0, n_sims=N_SIMS, period=None):
    """Full MC sim returning raw (x, y) score arrays of shape (n_sims,).

    Continuous-time event-driven: goals via competing exponentials,
    red cards via pre-generated NHPP times, stoppage via Poisson draws.
    shots_dev_h/a: held constant at from_min value (future shots unknown).
    period: 1 or 2 (required in play, since stoppage minutes 45+ occur in both halves).
    """
    if period is None:
        if from_min != 0:
            raise ValueError("period is required for in-play simulation")
        period = 1
    if period not in (1, 2) or (period == 2 and from_min < 45):
        raise ValueError(f"invalid state: period {period!r}, minute {from_min}")
    gamma, xi_half = params["gamma"], params["xi_half"]
    xi_goals, xi_rc = params["xi_goals"], params["xi_rc"]
    xi_shots = params.get("xi_shots", 0.0)
    c_h_rc, b_h_rc = params["rc_home"]
    c_a_rc, b_a_rc = params["rc_away"]
    s1_0, s1_rc, s1_g = params["st1"]
    s2_0, s2_rc, s2_g, s2_close = params["st2"]

    log_base_h = np.log(theta1) + gamma + xi_shots * shots_dev_h
    log_base_a = np.log(theta2) + xi_shots * shots_dev_a
    x = np.full(n_sims, start_x, dtype=np.int32)
    y = np.full(n_sims, start_y, dtype=np.int32)

    # Pre-generate RC times; prepend synthetic t=0 events for pre-existing RCs
    rc_h_t = _sim_rc_times(c_h_rc, b_h_rc, from_min, 150, n_sims)
    rc_a_t = _sim_rc_times(c_a_rc, b_a_rc, from_min, 150, n_sims)
    if rc_h > 0:
        rc_h_t = np.hstack([np.zeros((n_sims, rc_h)), rc_h_t])
    if rc_a > 0:
        rc_a_t = np.hstack([np.zeros((n_sims, rc_a)), rc_a_t])

    _ct = lambda *a: _sim_phase_ct(x, y, rc_h_t, rc_a_t, log_base_h, log_base_a, *a)
    _rc_count = lambda t: (rc_h_t <= t).sum(axis=1) + (rc_a_t <= t).sum(axis=1)

    if period == 1:
        if from_min < 45:
            _ct(0.0, xi_goals, xi_rc, from_min, 45, n_sims)  # 1H regular
        # 1H stoppage, conditional on the stoppage already played
        g1h = (x + y).astype(np.float64)
        pi1 = np.exp(s1_0 + s1_rc * _rc_count(45).astype(np.float64) + s1_g * g1h)
        u1 = _stoppage(pi1, int(from_min) - 45, 15)
        _ct(0.0, xi_goals, xi_rc, max(from_min, 45), 45 + u1, n_sims)
        score_after_1h = (x + y).astype(np.float64)  # Score after 1H incl stoppage
        _ct(xi_half, xi_goals, xi_rc, 45, 90, n_sims)  # 2H regular
    else:
        if from_min < 90:
            _ct(xi_half, xi_goals, xi_rc, from_min, 90, n_sims)
        score_after_1h = np.float64(start_x + start_y)

    # 2H stoppage, conditional on the stoppage already played
    g2h = (x + y).astype(np.float64) - score_after_1h  # Goals in 2H at min 90
    rc_2h = (_rc_count(90) - _rc_count(45)).astype(np.float64)
    close = (np.abs(x - y) <= 1).astype(np.float64)
    pi2 = np.exp(s2_0 + s2_rc * rc_2h + s2_g * g2h + s2_close * close)
    late = period == 2 and from_min >= 90
    u2 = _stoppage(pi2, int(from_min) - 90 if late else -1, 20)
    _ct(xi_half, xi_goals, xi_rc, from_min if late else 90, 90 + u2, n_sims)
    return x, y


def simulate_match(theta1, theta2, params, from_min=0, start_x=0, start_y=0,
                   rc_h=0, rc_a=0, shots_dev_h=0.0, shots_dev_a=0.0, n_sims=N_SIMS, period=None):
    """Returns (p_home, p_draw, p_away) via continuous-time MC simulation."""
    x, y = _sim_match_core(theta1, theta2, params, from_min, start_x, start_y,
                            rc_h, rc_a, shots_dev_h, shots_dev_a, n_sims, period)
    diff = x - y
    ph = float((diff > 0).mean())
    pd_ = float((diff == 0).mean())
    return ph, pd_, 1.0 - ph - pd_


def compute_outcome_probs(theta1, theta2, params, from_minute=0, start_x=0, start_y=0,
                          rc_h=0, rc_a=0, shots_dev_h=0.0, shots_dev_a=0.0, n_sims=N_SIMS, period=None):
    """Wrapper returning (p_home, p_draw, p_away) via Monte Carlo."""
    return simulate_match(theta1, theta2, params, from_minute, start_x, start_y,
                          rc_h, rc_a, shots_dev_h, shots_dev_a, n_sims, period)


def compute_outcome_probs_full(theta1, theta2, params, from_minute=0, start_x=0, start_y=0,
                               rc_h=0, rc_a=0, shots_dev_h=0.0, shots_dev_a=0.0, n_sims=N_SIMS):
    """Returns (p_home, p_draw, p_away, ou_dict) via continuous-time MC simulation."""
    x, y = _sim_match_core(theta1, theta2, params, from_minute, start_x, start_y,
                            rc_h, rc_a, shots_dev_h, shots_dev_a, n_sims)
    diff, total = x - y, x + y
    ph = float((diff > 0).mean())
    pd_ = float((diff == 0).mean())
    ou = {t: float((total > t).mean()) for t in [0.5, 1.5, 2.5, 3.5, 4.5]}
    return ph, pd_, 1.0 - ph - pd_, ou


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BETFAIR CALIBRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def calibrate_thetas_full(target_1x2, ou_targets, params, t1_init, t2_init, *, n_sims, seed, shots_dev=0.0):
    """Find (theta1, theta2) matching 1X2 + O/U from Betfair. Common random numbers (reseeded per
    evaluation) make the simulated objective deterministic, as in models/weibull/calibrate.py."""
    log_t1_init = np.log(max(t1_init, 1e-6))
    log_t2_init = np.log(max(t2_init, 1e-6))

    def obj(log_t):
        t1, t2 = np.exp(log_t)
        np.random.seed(seed)
        ph, pd_, pa, ou = compute_outcome_probs_full(t1, t2, params, n_sims=n_sims,
                                                     shots_dev_h=shots_dev, shots_dev_a=shots_dev)   # kick-off shots deviation
        err = (ph - target_1x2['p_home'])**2 + (pd_ - target_1x2['p_draw'])**2 + (pa - target_1x2['p_away'])**2
        err += sum((ou[t] - p)**2 for t, p in ou_targets.items())
        return err
    res = minimize(obj, [log_t1_init, log_t2_init],
                   method="Powell", options={"maxiter": 200, "ftol": 1e-5})
    if not res.success or not np.isfinite(res.fun):
        raise RuntimeError(f"Maia calibration failed: {res.message}")
    return np.exp(res.x[0]), np.exp(res.x[1]), res.fun


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPANDING-WINDOW ESTIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


WORKERS = 6  # parallel snapshot fits per run (two runs go side by side)


def _fit_quiet(train):
    return fit_maia(train, verbose=False)


def estimate_expanding(seasons, root, eval_season=None, out=None):
    """Fit Maia model with an expanding window, one snapshot per eval-season match date (see
    eval.utils.snapshot_cuts). Historical seasons = fixed pool.

    Args:
        seasons: list of season strings, e.g. ["21-22", "22-23", "23-24", "24-25"]
        eval_season: season for expanding window (default: last in list)

    Run: python -m models.maia.model --seasons 21-22 22-23 23-24 24-25
    """
    if isinstance(seasons, str):
        seasons = [seasons]
    eval_season = eval_season or seasons[-1]

    ws, align, sched = load_seasons(seasons, root)

    # Load FBref shots for shots covariate (per-team deviation from population baseline)
    from models.strategies.extended import build_cov_lookup, compute_dev_fit
    from models.weibull.clock import training_events, EXCLUDED_MATCHES
    from eval.utils import SEASON_CFG
    ws = ws[~ws["ws_match_id"].isin(EXCLUDED_MATCHES)]   # same reviewed exclusion as the Weibull fit
    fb_shots = pd.concat([pd.read_parquet(os.path.join(root, "data", s, "fbref_shots.parquet")).assign(season=s)
                          for s in seasons], ignore_index=True)
    pre = align.loc[pd.to_datetime(align["date"]) < pd.to_datetime(SEASON_CFG[eval_season][1]), "ws_game_id"]
    train = training_events(ws[ws["ws_match_id"].isin(pre)])   # baseline from pre-evaluation matches only, as the Weibull
    dev_shots_fit = compute_dev_fit(train, fb_shots, align, "shots")
    console.print(f"shots population fit: slope={dev_shots_fit[0]:.6f}, intercept={dev_shots_fit[1]:.6f}")
    cov_lookup = build_cov_lookup(ws, fb_shots, align, period_keys=True, fit_meta={"dev_shots_fit": dev_shots_fit})
    console.print(f"shots covariate: {len(cov_lookup)} entries")

    fbref_to_gw = dict(zip(sched["match_id"], sched["week"]))
    ws_to_gw = {r["ws_game_id"]: fbref_to_gw.get(r["fbref_match_id"])
                for _, r in align.iterrows()}
    ws_to_date = dict(zip(align["ws_game_id"], pd.to_datetime(align["date"])))
    ws["gameweek"] = ws["ws_match_id"].map(ws_to_gw)
    ws["match_date"] = ws["ws_match_id"].map(ws_to_date)

    meta = (ws.groupby("ws_match_id")
            .agg({"match_date": "first", "gameweek": "first", "season": "first"})
            .sort_values("match_date").dropna(subset=["gameweek"]))

    # Historical match IDs (always in training pool)
    hist_mids = set(meta[meta["season"] != eval_season].index)
    eval_meta = meta[meta["season"] == eval_season].copy()
    n_hist = len(hist_mids)
    console.print(f"Historical: {n_hist} matches from {[s for s in seasons if s != eval_season]}")
    from eval.utils import snapshot_cuts
    cuts = snapshot_cuts(eval_meta["match_date"], eval_season)
    console.print(f"Eval season {eval_season}: {len(eval_meta)} matches, {len(cuts)} snapshots from {cuts[0].date()}")

    out = out or os.path.join(root, "models", "maia", "params",
                              f"pl_{eval_season.replace('-', '_')}")
    os.makedirs(out, exist_ok=True)
    by_id = {m["match_id"]: m for m in extract_match_data(ws[ws["ws_match_id"].isin(meta.index)], align, cov_lookup=cov_lookup)}
    jobs = []
    for cut in cuts:   # one fit per snapshot: history plus eval-season matches up to the cut
        mids = hist_mids | set(eval_meta[eval_meta["match_date"] <= cut].index)
        jobs.append((cut, [by_id[i] for i in sorted(mids) if i in by_id]))
    date_to_path = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for (cut, train), (team_alpha, team_beta, maia_params) in zip(jobs, pool.map(_fit_quiet, [j[1] for j in jobs])):
            console.print(f"{cut.date()}: {len(train)} matches fitted")
            if dev_shots_fit is not None:
                maia_params["dev_shots_fit"] = dev_shots_fit
            path = os.path.join(out, f"params_{cut:%Y%m%d}.pkl")
            with open(path, "wb") as f:
                pickle.dump({"team_alpha": team_alpha, "team_beta": team_beta, "maia_params": maia_params}, f)
            date_to_path[str(cut.date())] = os.path.basename(path)  # name only: readers resolve it in the index's dir

    idx = {"season": eval_season, "param_files": date_to_path, "dates": sorted(date_to_path)}
    with open(os.path.join(out, "index.pkl"), "wb") as f:
        pickle.dump(idx, f)
    console.print(f"[green]Saved {len(date_to_path)} param files ({cuts[0].date()} to {cuts[-1].date()}) to {out}[/]")
    return date_to_path


class MaiaParamBaseline:
    """Load pre-computed Maia params, dispatch by match date."""

    def __init__(self, params_dir):
        idx_path = os.path.join(params_dir, "index.pkl")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"No index.pkl in {params_dir}. "
                "Run: python -m models.maia.model --season 24-25")
        with open(idx_path, "rb") as f:
            self.index = pickle.load(f)
        self.dir = params_dir
        self.param_dates = sorted(self.index["dates"])
        self.cache = {}
        console.print(f"Maia params: {len(self.param_dates)} dates "
                       f"({self.param_dates[0]} to {self.param_dates[-1]})")

    def get_params_for_date(self, match_date):
        """Return (team_alpha, team_beta, maia_params) for most recent fit before match_date."""
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
        return p["team_alpha"], p["team_beta"], p["maia_params"]


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
