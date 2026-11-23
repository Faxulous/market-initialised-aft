"""Team attack/defense parameter estimation via Weibull AFT MLE (Leriou & Ntzoufras 2025)."""
import os
import numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn, gammaln, digamma
from models.weibull.clock import match_clock, training_events

XI_DEFAULT = 0.0065  # Time decay factor for weighted MLE


def extract_intervals(df: pd.DataFrame, mcol: str = "ws_match_id") -> list:
    """Extract goal arrival intervals with score state for variable gamma."""
    req = ["is_home", "is_goal", "is_own_goal", "minute", "second", "team_id"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    intervals = []
    df = training_events(df, mcol)
    for mid, events in df.groupby(mcol, sort=False):
        m, ht, dur = match_clock(events)
        teams = m["team_id"].dropna().unique()
        if len(teams) != 2:
            raise ValueError(f"Match {mid} has {len(teams)} teams")
        if not (m["is_home"] == True).any():
            raise ValueError(f"Match {mid} has no home team")
        home = m.loc[m["is_home"] == True, "team_id"].iloc[0]
        away = [t for t in teams if t != home][0]
        goals = sorted(m[m["is_goal"] == True][["time", "team_id", "is_own_goal"]].values.tolist(), key=lambda x: x[0])
        prev, h, a = 0.0, 0, 0
        for gmin, gtid, og in goals:
            if og:
                gtid = away if gtid == home else home  # Own goal = opponent scores
            if gmin - prev > 0:
                intervals.append((str(mid), gmin - prev, int(home), int(away), int(gtid), False, h - a))
            prev, (h, a) = gmin, (h + 1, a) if gtid == home else (h, a + 1)
        if dur - prev > 0:
            intervals.append((str(mid), dur - prev, int(home), int(away), None, True, h - a))  # Censored interval
    return intervals


def extract_intervals_with_period(df: pd.DataFrame, mcol: str = "ws_match_id",
                                   split_at_ht: bool = False) -> list:
    """Extract goal arrival intervals with period info for half-specific gamma.

    split_at_ht: if True, break any interval crossing minute 45 into a censored 1H
    part and a fresh 2H part (treats halves as independent survival processes).
    """
    req = ["is_home", "is_goal", "is_own_goal", "minute", "second", "team_id"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    intervals = []
    for mid in df[mcol].unique():
        m = df[df[mcol] == mid].sort_values(["minute", "second"])
        teams = m["team_id"].dropna().unique()
        if len(teams) != 2:
            continue
        if not (m["is_home"] == True).any():
            continue
        home = m.loc[m["is_home"] == True, "team_id"].iloc[0]
        away = [t for t in teams if t != home][0]
        goals = sorted(m[m["is_goal"] == True][["minute", "team_id", "is_own_goal"]].values.tolist(), key=lambda x: x[0])
        dur, prev, h, a = max(90.0, m["minute"].max()), 0.0, 0, 0
        mid_s, hi, ai = str(mid), int(home), int(away)
        for gmin, gtid, og in goals:
            if og:
                gtid = away if gtid == home else home
            if gmin - prev > 0:
                if split_at_ht and prev < 45 and gmin > 45:
                    # Split: censored 1H part + uncensored 2H part
                    intervals.append((mid_s, 45 - prev, hi, ai, None, True, h - a, 1))
                    intervals.append((mid_s, gmin - 45, hi, ai, int(gtid), False, h - a, 2))
                else:
                    period = 1 if gmin <= 45 else 2
                    intervals.append((mid_s, gmin - prev, hi, ai, int(gtid), False, h - a, period))
            prev, (h, a) = gmin, (h + 1, a) if gtid == home else (h, a + 1)
        if dur - prev > 0:
            if split_at_ht and prev < 45 and dur > 45:
                intervals.append((mid_s, 45 - prev, hi, ai, None, True, h - a, 1))
                intervals.append((mid_s, dur - 45, hi, ai, None, True, h - a, 2))
            else:
                period = 1 if prev < 45 else 2
                intervals.append((mid_s, dur - prev, hi, ai, None, True, h - a, period))
    return intervals


def exposure_intervals(df, mcol="ws_match_id", reset_at_ht=False):
    """Half-labelled exposure with goal age; resetting is an explicit sensitivity."""
    rows = []
    for mid, events in training_events(df, mcol).groupby(mcol, sort=False):
        m, ht, end = match_clock(events)
        hi = int(m.loc[m.is_home == True, 'team_id'].iloc[0])
        ai, = [int(t) for t in m.team_id.unique() if t != hi]
        prev, h, a = 0., 0, 0
        goals = [(float(r.time), ai if r.team_id == hi else hi)
                 if r.is_own_goal else (float(r.time), int(r.team_id))
                 for r in m[m.is_goal == True].itertuples()]
        for t, scorer in goals + [(end, None)]:
            cuts = [prev] + ([ht] if prev < ht < t else []) + [t]
            for u, v in zip(cuts[:-1], cuts[1:]):
                if v <= u:  # Recorded-minute ties carry no measurable exposure.
                    continue
                origin = max(prev, ht) if reset_at_ht and u >= ht else prev
                final = v == t and scorer is not None
                rows.append((str(mid), v-origin, hi, ai, scorer if final else None,
                             not final, h-a, 2 if u >= ht else 1, u-origin))
            prev = t
            if scorer is not None:
                h, a = (h+1, a) if scorer == hi else (h, a+1)
    return rows


def _prep_data(intervals: list, tids: list, match_dates: dict = None, ref_date=None, xi: float = 0.0) -> dict:
    """Precompute static arrays for vectorized NLL."""
    idx = {t: i for i, t in enumerate(tids)}
    valid = [(iv, idx.get(iv[2]), idx.get(iv[3])) for iv in intervals if idx.get(iv[2]) is not None and idx.get(iv[3]) is not None]
    t = np.array([max(iv[1], 1e-6) for iv, _, _ in valid])
    hi, ai = np.array([h for _, h, _ in valid]), np.array([a for _, _, a in valid])
    cens = np.array([iv[5] for iv, _, _ in valid], dtype=bool)
    is_home_scorer = np.array([iv[4] == iv[2] if not iv[5] else False for iv, _, _ in valid])
    sd = np.array([iv[6] for iv, _, _ in valid])
    if xi > 0 and match_dates and ref_date:
        ref = pd.to_datetime(ref_date)
        w = np.array([np.exp(-xi * (ref - pd.to_datetime(match_dates[iv[0]])).days / 3.5) for iv, _, _ in valid])
    else:
        w = np.ones(len(valid))
    return {"t": t, "log_t": np.log(t), "hi": hi, "ai": ai, "cens": cens, "is_home_scorer": is_home_scorer,
            "home_leading": sd > 0, "home_trailing": sd < 0, "tied": sd == 0, "weights": w}


def _prep_data_with_period(intervals: list, tids: list, exposure=False) -> dict:
    """Precompute arrays for NLL with period info."""
    idx = {t: i for i, t in enumerate(tids)}
    valid = [(iv, idx.get(iv[2]), idx.get(iv[3])) for iv in intervals
             if idx.get(iv[2]) is not None and idx.get(iv[3]) is not None]
    t = np.array([max(iv[1], 1e-6) for iv, _, _ in valid])
    hi, ai = np.array([h for _, h, _ in valid]), np.array([a for _, _, a in valid])
    cens = np.array([iv[5] for iv, _, _ in valid], dtype=bool)
    is_home_scorer = np.array([iv[4] == iv[2] if not iv[5] else False for iv, _, _ in valid])
    sd = np.array([iv[6] for iv, _, _ in valid])
    period = np.array([iv[7] for iv, _, _ in valid])
    return {"t": t, "log_t": np.log(t), "hi": hi, "ai": ai, "cens": cens, "is_home_scorer": is_home_scorer,
            "home_leading": sd > 0, "home_trailing": sd < 0, "tied": sd == 0,
            "is_1h": period == 1, "is_2h": period == 2,
            "start": np.array([iv[8] for iv, _, _ in valid]) if exposure else np.zeros(len(t))}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLL functions (used by the MLE estimators below)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def weibull_nll_single_gamma(params: np.ndarray, data: dict, baseline: dict, tids: list) -> float:
    """NLL for Weibull AFT with single gamma (no score-state dependency)."""
    g = params[0]
    mu, home = baseline["mu"], baseline["home"]
    att = np.array([baseline["attack"][t] for t in tids])
    def_ = np.array([baseline["defense"][t] for t in tids])

    lgf = np.log(gamma_fn(1 + 1/g))
    t, log_t, hi, ai = data["t"], data["log_t"], data["hi"], data["ai"]

    eta_h = mu + home + att[hi] + def_[ai]
    eta_a = mu + att[ai] + def_[hi]
    ll_h, ll_a = -g * (eta_h - lgf), -g * (eta_a - lgf)
    lam_h, lam_a = np.exp(ll_h), np.exp(ll_a)

    cens = data["cens"]
    nll = (lam_h[cens] * t[cens]**g + lam_a[cens] * t[cens]**g).sum()
    hs, aws = ~cens & data["is_home_scorer"], ~cens & ~data["is_home_scorer"]
    if hs.any():
        nll -= (np.log(g) + ll_h[hs] + (g-1)*log_t[hs] - lam_h[hs]*t[hs]**g).sum()
        nll += (lam_a[hs] * t[hs]**g).sum()
    if aws.any():
        nll -= (np.log(g) + ll_a[aws] + (g-1)*log_t[aws] - lam_a[aws]*t[aws]**g).sum()
        nll += (lam_h[aws] * t[aws]**g).sum()
    return nll


def weibull_nll_single_gamma_phi(params: np.ndarray, data: dict, baseline: dict, tids: list) -> float:
    """NLL for single gamma + phi multiplier in 2H (no score-state dependency)."""
    g_base, phi = params[0], params[1]
    mu, home = baseline["mu"], baseline["home"]
    att = np.array([baseline["attack"][t] for t in tids])
    def_ = np.array([baseline["defense"][t] for t in tids])

    is_2h = data["is_2h"]
    g = np.where(is_2h, phi * g_base, g_base)
    lgf = np.log(gamma_fn(1 + 1/g))

    t, log_t, hi, ai = data["t"], data["log_t"], data["hi"], data["ai"]
    eta_h = mu + home + att[hi] + def_[ai]
    eta_a = mu + att[ai] + def_[hi]
    ll_h, ll_a = -g * (eta_h - lgf), -g * (eta_a - lgf)
    lam_h, lam_a = np.exp(ll_h), np.exp(ll_a)

    nll = ((lam_h + lam_a) * (t**g - data['start']**g)).sum()
    event = ~data['cens']
    rate = np.where(data['is_home_scorer'], ll_h, ll_a)
    nll -= (np.log(g[event]) + rate[event] + (g[event]-1)*log_t[event]).sum()
    return nll


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Estimation functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def common_nll(params, data, n):
    """Common-shape joint NLL and analytic gradient, including match weights."""
    mu, home, g = params[0], params[1], params[-1]
    att, dff = np.zeros(n), np.zeros(n)
    att[:-1], dff[:-1] = params[2:n+1], params[n+1:2*n]
    att[-1], dff[-1] = -att[:-1].sum(), -dff[:-1].sum()
    hi, ai, w = data['hi'], data['ai'], data['weights']
    lh = gammaln(1+1/g) - mu - home - att[hi] - dff[ai] + data['log_t']
    la = gammaln(1+1/g) - mu - att[ai] - dff[hi] + data['log_t']
    hh, ha = np.exp(g*lh), np.exp(g*la)
    eh = ~data['cens'] & data['is_home_scorer']
    ea = ~data['cens'] & ~data['is_home_scorer']
    value = np.sum(w*(hh+ha-eh*(np.log(g)+g*lh-data['log_t'])-ea*(np.log(g)+g*la-data['log_t'])))
    dh, da = w*g*(eh-hh), w*g*(ea-ha)
    gat = np.bincount(hi, dh, n) + np.bincount(ai, da, n)
    gdf = np.bincount(ai, dh, n) + np.bincount(hi, da, n)
    z = digamma(1+1/g)/g
    gg = np.sum(w*(hh*(lh-z)+ha*(la-z)-eh*(1/g+lh-z)-ea*(1/g+la-z)))
    grad = np.r_[np.sum(dh+da), dh.sum(), gat[:-1]-gat[-1], gdf[:-1]-gdf[-1], gg]
    return float(value), grad


def fit_team_intervals(intervals, *, xi=0., match_dates=None, ref_date=None):
    """Jointly fit team strengths and one shape from pre-extracted intervals."""
    tids = sorted({t for iv in intervals for t in (iv[2], iv[3])})
    n = len(tids)
    data = _prep_data(intervals, tids, match_dates=match_dates, ref_date=ref_date, xi=xi)
    x0 = np.zeros(2*n+1)
    x0[0], x0[1], x0[-1] = np.log(60), -.2, 1.2
    res = minimize(common_nll, x0, args=(data, n), jac=True, method="L-BFGS-B",
                   bounds=[(None, None)]*(2*n)+[(.1, 5.)], options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Weibull team fit failed: {res.message}")
    mu, home = res.x[0], res.x[1]
    att, def_ = np.zeros(n), np.zeros(n)
    att[:-1], def_[:-1] = res.x[2:n+1], res.x[n+1:2*n]
    att[-1], def_[-1] = -att[:-1].sum(), -def_[:-1].sum()
    return {"mu": float(mu), "home": float(home), "gamma": float(res.x[-1]),
            "attack": {t: float(att[i]) for i, t in enumerate(tids)},
            "defense": {t: float(def_[i]) for i, t in enumerate(tids)},
            "nll": float(res.fun), "n_intervals": len(intervals)}


def estimate_team_params(df: pd.DataFrame, mcol: str = "ws_match_id", verbose: bool = True,
                         xi: float = 0.0, match_dates: dict = None, ref_date=None) -> dict:
    """Joint MLE of team strengths and a common shape, without published shapes."""
    result = fit_team_intervals(extract_intervals(df, mcol), xi=xi, match_dates=match_dates, ref_date=ref_date)
    result['team_names'] = df.drop_duplicates('team_id').set_index('team_id')['team_name'].to_dict()
    if verbose:
        from rich import print
        print(f"mu={result['mu']:.3f}, home={result['home']:.3f}, gamma={result['gamma']:.3f}")
    return result


def estimate_single_gamma(df: pd.DataFrame, baseline: dict, mcol: str = "ws_match_id", verbose: bool = True) -> dict:
    """Estimate single gamma (no score-state dependency)."""
    intervals = extract_intervals(df, mcol)
    tids = sorted(set(baseline["attack"].keys()))
    data = _prep_data(intervals, tids)

    x0 = np.array([1.0])
    bounds = [(0.1, 5.0)]
    res = minimize(weibull_nll_single_gamma, x0, args=(data, baseline, tids), method="L-BFGS-B", bounds=bounds)

    from scipy.optimize import approx_fprime
    eps = 1e-5
    hess = approx_fprime(res.x, lambda x: approx_fprime(x, lambda p: weibull_nll_single_gamma(p, data, baseline, tids), eps)[0], eps)
    se = np.sqrt(1 / max(hess[0], 1e-10)) if hess[0] > 0 else np.nan

    if verbose:
        print(f"gamma={res.x[0]:.4f} (SE={se:.4f})")
        print(f"NLL: {res.fun:.2f}, converged: {res.success}")

    return {"gamma": float(res.x[0]), "_meta": {"nll": float(res.fun), "converged": res.success, "se": float(se)}}


def estimate_single_gamma_phi(df: pd.DataFrame, baseline: dict, mcol: str = "ws_match_id",
                               verbose: bool = True, split_at_ht: bool = True,
                               reset_at_ht: bool = False) -> dict:
    """Fit half shapes with continuous goal age; False selects legacy attribution."""
    intervals = (exposure_intervals(df, mcol, reset_at_ht) if split_at_ht
                 else extract_intervals_with_period(df, mcol, split_at_ht=False))
    tids = sorted(baseline['attack'])
    data = _prep_data_with_period(intervals, tids, exposure=split_at_ht)

    if verbose:
        n_1h, n_2h = data["is_1h"].sum(), data["is_2h"].sum()
        print(f"Fitting single gamma + phi on {len(intervals)} intervals ({n_1h} 1H, {n_2h} 2H)")

    x0 = np.array([1.0, 1.0])  # gamma_base, phi
    bounds = [(0.1, 5.0), (0.5, 2.0)]
    res = minimize(weibull_nll_single_gamma_phi, x0, args=(data, baseline, tids), method="L-BFGS-B", bounds=bounds)

    if not res.success:
        raise RuntimeError(f"Weibull shape fit failed: {res.message}")
    from scipy.optimize import approx_fprime
    eps = 1e-5
    hess = np.zeros((2, 2))
    for i in range(2):
        def grad_i(x):
            return approx_fprime(x, lambda p: weibull_nll_single_gamma_phi(p, data, baseline, tids), eps)[i]
        hess[i] = approx_fprime(res.x, grad_i, eps)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
    except np.linalg.LinAlgError:
        se = np.full(2, np.nan)

    if verbose:
        print(f"gamma={res.x[0]:.4f} (SE={se[0]:.4f}), phi={res.x[1]:.4f} (SE={se[1]:.4f})")
        print(f"NLL: {res.fun:.2f}, converged: {res.success}")

    return {"gamma": float(res.x[0]), "phi": float(res.x[1]),
            "_meta": {"nll": float(res.fun), "converged": res.success, "se_gamma": float(se[0]), "se_phi": float(se[1])}}


def estimate_expanding(seasons, root: str, eval_season: str = None, out: str = None, xi: float = XI_DEFAULT) -> dict:
    """Estimate team parameters with an expanding window, one snapshot per eval-season match date (see
    eval.utils.snapshot_cuts). Historical seasons = fixed pool.

    Args:
        seasons: list of season strings, e.g. ["21-22", "22-23", "23-24", "24-25"]
        eval_season: season for expanding window (default: last in list)
    """
    if isinstance(seasons, str):
        seasons = [seasons]
    eval_season = eval_season or seasons[-1]
    mcol = "ws_match_id"

    all_df, all_align, all_sched = [], [], []
    for s in seasons:
        df = pd.read_parquet(os.path.join(root, "data", s, "whoscored_events.parquet"))
        align = pd.read_parquet(os.path.join(root, "data", s, "match_alignment.parquet"))
        sched = pd.read_parquet(os.path.join(root, "data", s, "fbref_schedule.parquet"))
        df["season"], align["season"], sched["season"] = s, s, s
        all_df.append(df); all_align.append(align); all_sched.append(sched)
    df = pd.concat(all_df, ignore_index=True)
    align = pd.concat(all_align, ignore_index=True)
    sched = pd.concat(all_sched, ignore_index=True)

    fbref_to_gw = dict(zip(sched["match_id"], sched["week"]))
    ws_to_gw = {row["ws_game_id"]: fbref_to_gw.get(row["fbref_match_id"]) for _, row in align.iterrows()}
    df["gameweek"] = df[mcol].map(ws_to_gw)
    ws_to_date = dict(zip(align["ws_game_id"], pd.to_datetime(align["date"])))
    df["match_date"] = df[mcol].map(ws_to_date)
    ws_to_home = dict(zip(align["ws_game_id"], align["ws_home_team"]))
    df["is_home"] = df.apply(lambda r: r["team_name"] == ws_to_home.get(r[mcol]), axis=1)
    match_dates = {str(k): v for k, v in ws_to_date.items()}

    matches = df.groupby(mcol).agg({"match_date": "first", "gameweek": "first", "season": "first"}).sort_values("match_date").dropna(subset=["gameweek"])

    hist_mids = set(matches[matches["season"] != eval_season].index)
    eval_matches = matches[matches["season"] == eval_season]
    n_hist = len(hist_mids)
    print(f"Historical: {n_hist} matches from {[s for s in seasons if s != eval_season]}")
    from eval.utils import snapshot_cuts
    cuts = snapshot_cuts(eval_matches["match_date"], eval_season)
    print(f"Eval season {eval_season}: {len(eval_matches)} matches, {len(cuts)} snapshots from {cuts[0].date()}")

    out = out or os.path.join(root, "models", "weibull", "team_params", f"pl_{eval_season.replace('-', '_')}")
    os.makedirs(out, exist_ok=True)
    intervals = extract_intervals(df, mcol)
    names = df.drop_duplicates('team_id').set_index('team_id')['team_name'].to_dict()
    date_to_path = {}
    for cut in cuts:
        eval_mids = set(eval_matches[eval_matches["match_date"] <= cut].index)
        train_mids = hist_mids | eval_mids
        print(f"{cut.date()}: {len(train_mids)} matches ({n_hist} hist + {len(eval_mids)} eval)")
        mids = {str(mid) for mid in train_mids}
        p = fit_team_intervals([iv for iv in intervals if iv[0] in mids], xi=xi, match_dates=match_dates, ref_date=cut)
        p['team_names'] = names
        path = os.path.join(out, f"params_{cut:%Y%m%d}.pt")
        torch.save(p, path)
        date_to_path[str(cut.date())] = os.path.basename(path)  # name only: readers resolve it in the index's dir
    from models.weibull.fit import VERSION
    torch.save({"season": eval_season, "clock_version": VERSION, "param_files": date_to_path, "dates": sorted(date_to_path)},
               os.path.join(out, "index.pt"))
    print(f"Saved {len(date_to_path)} param files ({cuts[0].date()} to {cuts[-1].date()}) to {out}")
    return date_to_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", nargs="+", default=None, help="Default: all seasons through --eval-season")
    p.add_argument("--eval-season", default="24-25", help="Season for expanding window")
    p.add_argument("--xi", type=float, default=XI_DEFAULT)
    a = p.parse_args()
    from eval.utils import seasons_through
    a.seasons = a.seasons or seasons_through(a.eval_season)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    estimate_expanding(a.seasons, root, eval_season=a.eval_season, xi=a.xi)
