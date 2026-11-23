"""Reproduce training tables for the estimated-shape Weibull revision."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import gamma as gamma_fn
from rich import print

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.test import load_data
from eval.utils import EVAL_START_DATE, seasons_through
from models.covariates.stoppage import H1_OBSERVED, H2_OBSERVED
from models.weibull.clock import match_clock, training_events
from models.weibull.team_params.estimate import estimate_team_params, extract_intervals, _prep_data, common_nll
from models.weibull.team_params.covariates import extract_subintervals_with_covariates
from models.covariates.grid_search import prep_data, nll_joint
from models.weibull.fit import select_shapes, load_fit, DELTA

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEASON = '24-25'
SEED = 8
OUT = ROOT / 'eval' / 'refit'
COMBOS = [[], ['red_diff'], ['red_diff', 'dev_goals'], ['red_diff', 'dev_psxg'],
          ['red_diff', 'dev_sot'], ['red_diff', 'dev_xg'], ['red_diff', 'dev_shots'],
          ['red_diff', 'dev_shots', 'dev_xg']]
BOUNDS_G = (0.30, 4.00)
SIM_DT = 0.05                   # simulation step (minutes)
ON_TARGET = ('Goal', 'Saved')   # FBref outcomes carrying a PSxG value
COUNT_SRC = {'dev_sot': 'sot'}  # deviation covariates built from the shot-count arrays below
HAZARD_REF = 30.0               # reference minute for the post-goal hazard comparison


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. INTERVAL EXTRACTION (game clock + half boundary + pre/post-goal score state)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract(ws: pd.DataFrame, clock: str = "minute", mcol: str = "ws_match_id") -> pd.DataFrame:
    """Goal inter-arrival intervals with absolute game times, half boundary and score states.

    Mirrors models/weibull/team_params/estimate.extract_intervals (own goals credited to the
    opponent, censored tail to max(90, last minute)) but additionally records the absolute
    start/end of each interval, the match's half-time boundary, and the score state BOTH at
    the interval start (sd0, the causal convention) and after the terminating goal (sd1).
    """
    prank = {"FirstHalf": 0, "SecondHalf": 1}
    w = ws[ws["period"].isin(prank)].copy()
    w["_p"] = w["period"].map(prank)
    rows, n_neg, n_zero = [], 0, 0
    for mid, m in w.groupby(mcol, sort=False):
        teams = m["team_id"].dropna().unique()
        if len(teams) != 2:
            raise ValueError(f"Match {mid} has {len(teams)} teams")
        if not (m["is_home"] == True).any():
            raise ValueError(f"Match {mid} has no home team")
        home = int(m.loc[m["is_home"] == True, "team_id"].iloc[0])
        away = int([t for t in teams if t != home][0])
        h1 = m[m["_p"] == 0][clock]
        ht = max(45.0, float(h1.max()) if len(h1) else 45.0)       # 1H incl. stoppage
        dur = max(90.0, float(m[clock].max()))
        g = m[m["is_goal"] == True].sort_values(["_p", clock, "second"])
        gt = np.maximum.accumulate(g[clock].to_numpy(float)) if len(g) else np.array([])
        prev, h, a = 0.0, 0, 0
        for k in range(len(g)):
            tid, og = int(g["team_id"].iloc[k]), bool(g["is_own_goal"].iloc[k])
            if og:
                tid = away if tid == home else home
            nh, na = (h + 1, a) if tid == home else (h, a + 1)
            if gt[k] - prev > 0:
                rows.append((str(mid), prev, gt[k], ht, dur, home, away, tid, False, h - a, nh - na))
            elif gt[k] - prev < 0:
                n_neg += 1
            else:
                n_zero += 1              # two goals in the same recorded minute: no interval
            prev, h, a = gt[k], nh, na
        if dur - prev > 0:
            rows.append((str(mid), prev, dur, ht, dur, home, away, -1, True, h - a, h - a))
    if n_neg:
        raise ValueError(f"{n_neg} non-monotone goal times survived the running-max clock")
    out = pd.DataFrame(rows, columns=["mid", "t0", "t1", "ht", "dur", "home", "away",
                                      "scorer", "cens", "sd0", "sd1"])
    out.attrs["n_zero"] = n_zero        # same-minute goals dropped, as in extract_intervals
    return out


def _state_idx(sd: np.ndarray, home: bool) -> np.ndarray:
    """0=leading, 1=tied, 2=trailing for the home (or away) team given signed score diff."""
    s = np.sign(sd) if home else -np.sign(sd)
    return np.where(s > 0, 0, np.where(s == 0, 1, 2))


def build(iv: pd.DataFrame, tids: list, state_col: str = "sd0", half: str = "piece") -> dict:
    """Split intervals into pieces and return flat arrays for the NLL.

    Pieces carry elapsed-since-last-goal (renewal clock) at their start/end, so the cumulative
    hazard is accumulated piecewise; with a constant shape inside a piece this telescopes exactly.

    half controls how a piece is assigned to a half:
      "piece" - cut the interval at the half-time boundary; each part sits in the half it
                actually occupies (the only assignment that does not use the outcome);
      "start" - whole interval, labelled by the half it starts in;
      "paper" - whole interval, labelled by the half the terminating goal falls in (censored
                intervals by their start), matching extract_intervals_with_period.
    """
    idx = {t: i for i, t in enumerate(tids)}
    miss = sorted({t for t in pd.concat([iv["home"], iv["away"]]).unique() if t not in idx})
    if miss:
        raise ValueError(f"{len(miss)} team ids absent from baseline: {miss}")
    if half not in ("piece", "start", "paper"):
        raise ValueError(f"unknown half convention {half}")
    P = []   # (interval_idx, rel0, rel1, is2h)
    for i, r in enumerate(iv.itertuples()):
        cuts = [r.t0, r.t1]
        if half == "piece" and r.t0 < r.ht < r.t1:
            cuts.insert(1, r.ht)
        ref = (r.t0 if r.cens else r.t1) if half == "paper" else None
        for u, v in zip(cuts[:-1], cuts[1:]):
            P.append((i, u - r.t0, v - r.t0, 1 if (ref if ref is not None else u) >= r.ht else 0))
    P = np.array(P, float)
    ii = P[:, 0].astype(int)
    sd = iv[state_col].to_numpy()
    return {
        "iv": ii, "t0": P[:, 1], "t1": P[:, 2], "is2h": P[:, 3].astype(int),
        "hi": iv["home"].map(idx).to_numpy()[ii], "ai": iv["away"].map(idx).to_numpy()[ii],
        "sh": _state_idx(sd, True)[ii], "sa": _state_idx(sd, False)[ii],
        "n_iv": len(iv), "n_teams": len(tids),
        "final": np.concatenate([ii[1:] != ii[:-1], [True]]),
        "cens": iv["cens"].to_numpy()[ii],
        "hscore": (iv["scorer"].to_numpy() == iv["home"].to_numpy())[ii],
        "dur": (P[:, 2] - P[:, 1]),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. LIKELIHOOD (one interval-censored Weibull AFT likelihood for every shape model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def nll(mu, home, att, dff, sh, d) -> float:
    """NLL with per-piece shapes and log-mean offsets. E[T]=exp(eta): shape moves spread only."""
    gh, ga, oh, oa = sh
    eta_h = mu + home + att[d["hi"]] + dff[d["ai"]] + oh
    eta_a = mu + att[d["ai"]] + dff[d["hi"]] + oa
    ah = np.exp(eta_h - np.log(gamma_fn(1 + 1 / gh)))
    aa = np.exp(eta_a - np.log(gamma_fn(1 + 1 / ga)))
    t0, t1 = d["t0"], d["t1"]
    z = t0 > 0
    dHh = (t1 / ah) ** gh - np.where(z, (np.maximum(t0, 1e-12) / ah) ** gh, 0.0)
    dHa = (t1 / aa) ** ga - np.where(z, (np.maximum(t0, 1e-12) / aa) ** ga, 0.0)
    out = dHh.sum() + dHa.sum()
    ev = d["final"] & ~d["cens"]
    hs, aw = ev & d["hscore"], ev & ~d["hscore"]
    if hs.any():
        out -= (np.log(gh[hs]) - gh[hs] * np.log(ah[hs]) + (gh[hs] - 1) * np.log(t1[hs])).sum()
    if aw.any():
        out -= (np.log(ga[aw]) - ga[aw] * np.log(aa[aw]) + (ga[aw] - 1) * np.log(t1[aw])).sum()
    return out


def shapes(spec: str, g: np.ndarray, d: dict):
    """Map a free-parameter vector to per-piece (gamma_home, gamma_away, offset_h, offset_a)."""
    z = 0.0
    if spec == "single":
        return np.full(len(d["t0"]), g[0]), np.full(len(d["t0"]), g[0]), z, z
    if spec == "state":
        return g[d["sh"]], g[d["sa"]], z, z
    if spec == "half":
        return g[d["is2h"]], g[d["is2h"]], z, z
    raise ValueError(f"unknown spec {spec}")


SPEC = {"single": [BOUNDS_G], "state": [BOUNDS_G] * 3, "half": [BOUNDS_G] * 2}
NP = {k: len(v) for k, v in SPEC.items()}
X0 = {k: np.full(len(v), 1.15) for k, v in SPEC.items()}


def fit(spec: str, d: dict, base: dict, tids: list, joint: bool = False, x0=None) -> dict:
    """Profile fit (team params frozen) or joint fit (mu, home, att, def and shapes free)."""
    att0 = np.array([base["attack"][t] for t in tids])
    dff0 = np.array([base["defense"][t] for t in tids])
    n, k = len(tids), NP[spec]
    g0 = np.array(X0[spec] if x0 is None else x0, dtype=float)   # copy: never alias the caller's
    bg = [tuple(v) for v in SPEC[spec]]

    def unpack(x):
        if not joint:
            return base["mu"], base["home"], att0, dff0, x
        mu, hm = x[0], x[1]
        att, dff = np.zeros(n), np.zeros(n)
        att[:-1], dff[:-1] = x[2:1 + n], x[1 + n:2 * n]
        att[-1], dff[-1] = -att[:-1].sum(), -dff[:-1].sum()
        return mu, hm, att, dff, x[2 * n:]

    def f(x):
        mu, hm, att, dff, g = unpack(x)
        return nll(mu, hm, att, dff, shapes(spec, g, d), d)

    if joint:
        z0 = np.concatenate([[base["mu"], base["home"]], att0[:-1], dff0[:-1], g0])
        bnds = [(None, None)] * (2 * n) + bg
    else:
        z0, bnds = g0, bg
    res = minimize(f, z0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 5000})
    g = np.array(res.x[-k:], dtype=float)
    se = hess_se(lambda gg: f(np.concatenate([res.x[:-k], gg])) if joint else f(gg), g)
    return {"spec": spec, "g": g, "se": se, "nll": res.fun, "k": k + (2 * n if joint else 0),
            "ok": bool(res.success), "x": res.x}


def hess_se(f, x, eps=1e-4) -> np.ndarray:
    """SEs from the numerical observed information (central second differences)."""
    m = len(x); H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            ei, ej = np.zeros(m), np.zeros(m); ei[i] = ej[j] = eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps * eps)
    try:
        v = np.diag(np.linalg.inv(H))
        return np.where(v > 0, np.sqrt(np.abs(v)), np.nan)
    except np.linalg.LinAlgError:
        return np.full(m, np.nan)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SIMULATION (competing-risks renewal process with a shape table gtab[half, state])
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def simulate(mt: pd.DataFrame, base: dict, gtab: np.ndarray, rng) -> pd.DataFrame:
    """Simulate goal times for the given match list; returns an interval table like extract()."""
    att = np.array([base["attack"][t] for t in mt["home"]])
    dfa = np.array([base["defense"][t] for t in mt["away"]])
    ata = np.array([base["attack"][t] for t in mt["away"]])
    dfh = np.array([base["defense"][t] for t in mt["home"]])
    eta_h, eta_a = base["mu"] + base["home"] + att + dfa, base["mu"] + ata + dfh
    ht, dur = mt["ht"].to_numpy(float), mt["dur"].to_numpy(float)
    M = len(mt)
    lg = np.log(gamma_fn(1 + 1 / gtab))                       # log Gamma(1+1/gamma), 2x3
    e, h, a, t = np.zeros(M), np.zeros(M, int), np.zeros(M, int), 0.0
    ev = []
    while t < dur.max():
        act = t < dur
        sd = h - a
        sh = np.where(sd > 0, 0, np.where(sd == 0, 1, 2))
        sa = np.where(sd > 0, 2, np.where(sd == 0, 1, 0))
        p2 = (t >= ht).astype(int)
        gh, ga = gtab[p2, sh], gtab[p2, sa]
        ah, aa = np.exp(eta_h - lg[p2, sh]), np.exp(eta_a - lg[p2, sa])
        dHh = ((e + SIM_DT) / ah) ** gh - (e / ah) ** gh
        dHa = ((e + SIM_DT) / aa) ** ga - (e / aa) ** ga
        sc_h = act & (rng.random(M) < 1 - np.exp(-dHh))
        sc_a = act & (rng.random(M) < 1 - np.exp(-dHa))
        both, pick = sc_h & sc_a, rng.random(M) < 0.5
        sc_h, sc_a = sc_h & ~(both & ~pick), sc_a & ~(both & pick)
        for i in np.flatnonzero(sc_h): ev.append((i, t + SIM_DT, True))
        for i in np.flatnonzero(sc_a): ev.append((i, t + SIM_DT, False))
        h, a = h + sc_h, a + sc_a
        e = np.where(sc_h | sc_a, 0.0, e + SIM_DT); t += SIM_DT
    bym = {}
    for i, tt, hh in ev: bym.setdefault(i, []).append((tt, hh))
    rows = []
    for i, r in enumerate(mt.itertuples()):
        prev, hg, ag = 0.0, 0, 0
        for tt, hh in bym.get(i, []):
            nh, na = (hg + 1, ag) if hh else (hg, ag + 1)
            rows.append((f"S{i}", prev, tt, r.ht, r.dur, r.home, r.away,
                         r.home if hh else r.away, False, hg - ag, nh - na))
            prev, hg, ag = tt, nh, na
        if r.dur - prev > 0:
            rows.append((f"S{i}", prev, r.dur, r.ht, r.dur, r.home, r.away, -1, True, hg - ag, hg - ag))
    return pd.DataFrame(rows, columns=["mid", "t0", "t1", "ht", "dur", "home", "away",
                                       "scorer", "cens", "sd0", "sd1"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. SHOT / SHOT-ON-TARGET COUNTS (mirrors covariates.compute_match_covariates, which
#    accumulates PSxG but not raw shot counts; cross-checked against it on PSxG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_shot_counts(ws: pd.DataFrame, fb: pd.DataFrame, align: pd.DataFrame) -> dict:
    """Per-match, per-half cumulative-lagged shot/SoT/PSxG arrays keyed (mid, period)."""
    ai = align.set_index('ws_game_id')
    fb_g = {k: v for k, v in fb.groupby('match_id')}
    out = {}
    for mid, m_ws in ws.groupby('ws_match_id'):
        if mid not in ai.index:
            raise KeyError(f'match {mid} missing from alignment')
        arow = ai.loc[mid]
        m_fb = fb_g[arow['fbref_match_id']]
        h1w, h2w = m_ws[m_ws['period'] == 'FirstHalf'], m_ws[m_ws['period'] == 'SecondHalf']
        max_h1 = int(h1w['minute'].max()) if len(h1w) else 45
        max_h2 = int(max(h2w['minute'].max() if len(h2w) else 90, 90))
        warm = None
        for per, mx in ((1, max_h1), (2, max_h2)):
            a = np.zeros((6, mx + 1))  # sh_h, sh_a, sot_h, sot_a, psxg_h, psxg_a
            for _, s in m_fb[m_fb['period'] == per].iterrows():
                t = int(s['minute'])
                if 0 <= t <= mx:
                    o = 0 if s['team'] == arow['fbref_home_team'] else 1
                    a[o, t] += 1
                    if s['outcome'] in ON_TARGET:
                        a[2 + o, t] += 1
                    a[4 + o, t] += float(s['PSxG']) if pd.notna(s['PSxG']) else 0.0
            cum = np.cumsum(a, axis=1) - a          # strictly backward-looking
            if per == 2:
                cum += warm[:, None]                # warm-start 2H with 1H totals
            else:
                warm = a.sum(axis=1)
            out[(str(mid), per)] = cum
    return out


def lookup_counts(subs: list, counts: dict) -> dict:
    """Map sub-intervals onto the cumulative shot arrays (same clamping as get_covariates_at)."""
    res = {k: np.empty(len(subs)) for k in
           ('shots_home', 'shots_away', 'sot_home', 'sot_away', 'psxg_home', 'psxg_away')}
    for i, s in enumerate(subs):
        gt = float(s[17])
        arr = counts[(s[0], int(s[15]))]                    # extractor's half: 1H stoppage has minute >= 45
        t = min(int(gt), arr.shape[1] - 1)
        for j, k in enumerate(res):
            res[k][i] = arr[j, t]
    return res


def add_dev(data: dict, valid: list, counts: dict, names: list) -> None:
    """Add deviation covariates (cumulative minus linear population baseline) in place."""
    gt = np.array([float(s[17]) for s in valid])
    lk = lookup_counts(valid, counts)
    for name in names:
        src = COUNT_SRC[name]
        ch, ca = lk[f'{src}_home'], lk[f'{src}_away']
        coefs = np.polyfit(np.concatenate([gt, gt]), np.concatenate([ch, ca]), 1)
        pop = np.polyval(coefs, gt)
        data[f'{name}_home'], data[f'{name}_away'] = ch - pop, ca - pop
        data[f'_{name}_fit'] = tuple(coefs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. TRAINING TABLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def added_time(events: pd.DataFrame, half: str, start: int) -> np.ndarray:
    """Minutes played past `start` in `half`, one value per match."""
    ends = events[events.period.isin(['FirstHalf', 'SecondHalf'])].groupby(['ws_match_id', 'period']).minute.max().unstack()
    return (ends[half] - start).clip(lower=0).to_numpy()


def training():
    ws, _, align = load_data(seasons_through(SEASON))
    train = training_events(ws[pd.to_datetime(ws.match_date) < pd.to_datetime(EVAL_START_DATE)])
    recent = train[pd.to_datetime(train.match_date) >= pd.Timestamp('2024-08-01')]
    stop = []
    for half, start, observed in [('FirstHalf', 45, H1_OBSERVED), ('SecondHalf', 90, H2_OBSERVED)]:
        values = added_time(recent, half, start)
        np.testing.assert_array_equal(np.sort(values), np.sort(observed))   # shipped arrays == pre-eval 24/25 data
        stop.append(dict(half=half, matches=len(values), mean=float(values.mean())))
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(stop).to_csv(OUT/'stoppage.csv', index=False)
    quoted = [dict(quantity=f'h2_added_mean_{s}', value=float(added_time(train[train.season == s], 'SecondHalf', 90).mean()))
              for s in seasons_through(SEASON)]
    base = estimate_team_params(train)
    tids = sorted(base['attack'])
    intervals = extract_intervals(train)
    p = np.r_[base['mu'], base['home'], [base['attack'][t] for t in tids[:-1]],
              [base['defense'][t] for t in tids[:-1]], base['gamma']]
    data = _prep_data(intervals, tids)
    hessian = approx_derivative(lambda x: common_nll(x, data, len(tids))[1], p)
    se = np.sqrt(np.diag(np.linalg.inv(hessian)))
    summary = dict(matches=int(train.ws_match_id.nunique()), intervals=len(intervals),
                   red_cards=int(train.card_type.isin(['Red', 'SecondYellow']).sum()),
                   mu=base['mu'], home=base['home'], gamma=base['gamma'], nll=base['nll'],
                   se_mu=se[0], se_home=se[1], se_gamma=se[-1])
    OUT.mkdir(exist_ok=True)
    (OUT/'baseline.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Post-goal hazard relative to its level at HAZARD_REF minutes: h(t)/h(ref) = (t/ref)^(gamma-1)
    g = base['gamma']
    quoted += [dict(quantity='se_gamma', value=float(se[-1])),
               dict(quantity='hazard_ratio_5min', value=float((5/HAZARD_REF)**(g-1))),
               dict(quantity='hazard_ratio_mean_first_10min', value=float((10/HAZARD_REF)**(g-1)/g))]
    clocked = pd.concat([match_clock(m)[0] for _, m in train.groupby('ws_match_id', sort=False)])
    iv = extract(clocked, clock='time')
    d = build(iv, tids)
    rows = []
    for name in ['single', 'state', 'half']:
        result = fit(name, d, base, tids, joint=True)
        if not result['ok']:
            raise RuntimeError(f'{name} joint fit failed')
        rows.append(dict(shape=name, shapes=' / '.join(f'{v:.6f}' for v in result['g']),
                         k=result['k'], nll=result['nll'], bic=2*result['nll']+result['k']*np.log(len(iv))))
    shape_tab = pd.DataFrame(rows)
    shape_tab['delta_bic'] = shape_tab.bic-shape_tab.bic.iloc[0]
    shape_tab.to_csv(OUT/'shapes.csv', index=False)
    print(shape_tab.to_string(index=False))

    simulated = simulate(iv.drop_duplicates('mid')[['home', 'away', 'ht', 'dur']],
                         base, np.full((2, 3), base['gamma']), np.random.default_rng(SEED))
    labels = []
    for source, sample in [('observed', iv), ('no-state-effect simulation', simulated)]:
        for convention in ['sd0', 'sd1']:
            result = fit('state', build(sample, tids, state_col=convention), base, tids, joint=True)
            if not result['ok']:
                raise RuntimeError(f'{source}/{convention} fit failed')
            labels.append(dict(source=source, labels=convention, leading=result['g'][0],
                               tied=result['g'][1], trailing=result['g'][2]))
    pd.DataFrame(labels).to_csv(OUT/'labelling.csv', index=False)

    fb = pd.concat([pd.read_parquet(ROOT/'data'/s/'fbref_shots.parquet') for s in seasons_through(SEASON)])
    psxg = fb.PSxG.fillna(0).astype(float).groupby(fb.outcome).mean()   # NaN -> 0, as compute_match_covariates does
    quoted += [dict(quantity=f'psxg_mean_{k.lower()}', value=float(psxg[k])) for k in ON_TARGET]
    sub = extract_subintervals_with_covariates(train, fb, align, delta=DELTA)
    counts = build_shot_counts(train, fb, align)
    selected_shape = load_fit(SEASON)['meta']['selected_shape']
    rows, candidates = [], []
    for i, names in enumerate(COMBOS):
        d = prep_data(sub, tids, [n for n in names if n != 'dev_sot'])
        if 'dev_sot' in names:
            add_dev(d, sub, counts, ['dev_sot'])
        _, fits = select_shapes(d, base, tids, names)
        candidates.extend([dict(model=f'M{i}', **r) for r in fits])
        result, = [r for r in fits if r['shape'] == selected_shape]
        pars = np.array(result['params'])
        free = np.r_[0, np.arange(2, len(pars))] if selected_shape == 'common' else np.arange(len(pars))
        def objective(x):
            full = pars.copy()
            full[free] = x
            return nll_joint(full, d, base, tids, names)
        errors = np.zeros(len(pars))
        errors[free] = hess_se(objective, pars[free])
        row = dict(model=f'M{i}', covariates=', '.join(names), shape=selected_shape,
                   gamma=pars[0], phi=pars[1], se_gamma=errors[0], k=result['k'],
                   nll=result['nll'], bic=result['bic'])
        row.update({f'beta_{name}': pars[j+2] for j, name in enumerate(names)})
        row.update({f'se_{name}': errors[j+2] for j, name in enumerate(names)})
        rows.append(row)
        print(row)
    table = pd.DataFrame(rows)
    table['delta_bic'] = table.bic-table.bic.iloc[0]
    table['lrt'] = [np.nan, 2*(rows[0]['nll']-rows[1]['nll'])]+[2*(rows[1]['nll']-r['nll']) for r in rows[2:7]]+[2*(rows[6]['nll']-rows[7]['nll'])]
    table.to_csv(OUT/'covariates.csv', index=False)
    pd.DataFrame(candidates).to_csv(OUT/'covariate_shapes.csv', index=False)
    quoted.append(dict(quantity='lrt_xg_given_shots', value=float(table.lrt.iloc[-1])))
    pd.DataFrame(quoted).to_csv(OUT/'quoted.csv', index=False)
    print(pd.DataFrame(quoted).to_string(index=False))
    actual = load_fit(SEASON)
    np.testing.assert_allclose([rows[6]['gamma'], rows[6]['phi'], rows[6]['beta_red_diff'], rows[6]['beta_dev_shots']],
                               [actual['gamma'], actual['phi'], actual['betas']['red_diff'], actual['betas']['dev_shots']], rtol=1e-6)


if __name__ == '__main__':
    training()
