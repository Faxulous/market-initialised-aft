"""Market-implied benchmark forecasts: does the AFT model add value beyond importing
pre-match market information?

mkt_poisson    - independent Poisson, (lam_H, lam_A) fitted to the SAME pre-kick-off
                 Betfair 1X2 + O/U targets used by models/weibull/calibrate.py.
                 In-play: remaining goals ~ Poisson(lam * frac_remaining), uniform rate.
mkt_poisson_t  - as mkt_poisson but frac_remaining = empirical league goal-time survival
                 estimated on TRAINING matches only (score-and-time updating).
mkt_naive      - multinomial logit on pre-kick-off market log-odds, score diff,
                 minutes remaining and their interaction. No goal process at all.

Writes eval/analysis/out/benchmark_preds.parquet, consumed by eval/utils.load_predictions.

Run: python eval/analysis/benchmarks.py
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rich.console import Console

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from models.covariates.stoppage import MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE, compute_remaining
from models.market import load_prematch_1x2, load_ou_targets
from eval.utils import SEASON_CFG, ALL_SEASONS, seasons_through

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
SEASON = "24-25"                                   # the paper's single eval season
TOTAL = 90 + MEAN_H1_STOPPAGE + MEAN_H2_STOPPAGE   # 99.31 min expected match length
KMAX = 15                                          # Poisson score-matrix truncation per team
OUT = os.path.join(ROOT, 'eval', 'analysis', 'out')

console = Console()
KS = np.arange(KMAX + 1)
DMAT = KS[:, None] - KS[None, :]          # score-difference lattice (K+1, K+1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POISSON CORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def pmf_pair(lam_h, lam_a):
    """Truncated, renormalised Poisson pmfs for both teams."""
    px, py = poisson.pmf(KS, np.atleast_1d(lam_h)[:, None]), poisson.pmf(KS, np.atleast_1d(lam_a)[:, None])
    return px / px.sum(1, keepdims=True), py / py.sum(1, keepdims=True)


def outcome_probs(lam_h, lam_a, sd):
    """P(H/D/A) for final diff = sd + X - Y, X~Poi(lam_h), Y~Poi(lam_a). Analytic."""
    px, py = pmf_pair(lam_h, lam_a)
    joint = px[:, :, None] * py[:, None, :]                  # (n, K+1, K+1)
    tot = np.atleast_1d(sd)[:, None, None] + DMAT[None]      # final score difference
    return (np.where(tot > 0, joint, 0).sum((1, 2)),
            np.where(tot == 0, joint, 0).sum((1, 2)),
            np.where(tot < 0, joint, 0).sum((1, 2)))


def fit_lambdas(t1x2, ou):
    """Least-squares fit of (lam_H, lam_A) to pre-kick-off 1X2 + available O/U targets."""
    def obj(z):
        lh, la = np.exp(z)
        ph, pdw, pa = outcome_probs(lh, la, 0.0)
        e = (ph[0] - t1x2[0]) ** 2 + (pdw[0] - t1x2[1]) ** 2 + (pa[0] - t1x2[2]) ** 2
        return e + sum((poisson.sf(int(np.floor(t)), lh + la) - p) ** 2 for t, p in ou.items())
    r = minimize(obj, np.log([1.5, 1.1]), method='Nelder-Mead',
                 options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 2000})
    if not r.success:
        raise RuntimeError(f"lambda fit failed: {r.message}")
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def clock(minute, period):
    """Common match clock u in [0, TOTAL]: period 2 offset by mean 1H stoppage."""
    return np.clip(np.where(period == 1, minute, minute + MEAN_H1_STOPPAGE), 0, TOTAL)


def load_grid():
    """Eval grid + truth from the betfair rows of predictions.parquet (all models share it)."""
    p = pd.read_parquet(os.path.join(ROOT, 'eval', 'predictions.parquet'))
    g = p[p['model'] == 'betfair'].drop(columns=['model', 'pred_home', 'pred_draw', 'pred_away']).copy()
    assert set(g['season']) == {SEASON}, f"unexpected eval season in predictions: {set(g['season'])}"
    return p.columns, g.reset_index(drop=True)


def load_targets():
    """Pre-kick-off 1X2 (every season: mkt_naive trains on earlier ones) and eval O/U, by ws_game_id."""
    x12 = {}
    for s in ALL_SEASONS:
        x12.update(load_prematch_1x2(s))
    return x12, load_ou_targets(SEASON)


def load_train_matches():
    """Training matches (seasons through the eval season, dated before its eval start, as the
    models are trained): goals on the u-clock, half ends, outcome."""
    goals, meta, no_events = {}, {}, []
    for s in seasons_through(SEASON):
        al = pd.read_parquet(os.path.join(ROOT, 'data', s, 'match_alignment.parquet'))
        al = al[pd.to_datetime(al['date']) < SEASON_CFG[SEASON][1]]
        if not len(al):
            continue
        ws = pd.read_parquet(os.path.join(ROOT, 'data', s, 'whoscored_events.parquet'),
                             columns=['ws_match_id', 'period', 'minute', 'event_type', 'team_name', 'is_own_goal'])
        ws = ws[ws['ws_match_id'].isin(set(al['ws_game_id']))]
        home = dict(zip(al['ws_game_id'], al['ws_home_team']))
        pnum = ws['period'].map({'FirstHalf': 1, 'SecondHalf': 2})
        ends = ws[pnum.isin([1, 2])].assign(p=pnum).groupby(['ws_match_id', 'p'])['minute'].max().unstack()
        g = ws[ws['event_type'] == 'Goal'].assign(p=pnum)
        g = g[g['p'].isin([1, 2])]
        g['u'] = clock(g['minute'].values, g['p'].values)
        # own goal credits the opposing team
        g['is_home'] = (g['team_name'] == g['ws_match_id'].map(home)) != g['is_own_goal']
        for gid, grp in g.groupby('ws_match_id'):
            goals[int(gid)] = grp[['u', 'is_home']].sort_values('u').values
        for _, r in al.iterrows():
            gid = int(r['ws_game_id'])
            if gid not in ends.index or ends.loc[gid].isna().any():  # incomplete feed -> excluded, counted
                no_events.append(gid)
                continue
            meta[gid] = {'h1_end': int(ends.loc[gid, 1]), 'h2_end': int(ends.loc[gid, 2]), 'season': s,
                         'y': 0 if r['home_score'] > r['away_score'] else (1 if r['home_score'] == r['away_score'] else 2)}
            goals.setdefault(gid, np.empty((0, 2)))
    return goals, meta, no_events


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    np.random.seed(SEED)
    cols, grid = load_grid()
    eval_ids = sorted(grid['match_id'].unique())
    console.print(f"[bold]Eval grid[/]: {len(grid):,} points, {len(eval_ids)} matches")

    # ── targets ────────────────────────────────────────────────────────────────
    pre1x2, ou_all = load_targets()
    no_pre = [m for m in eval_ids if m not in pre1x2]
    if no_pre:
        raise ValueError(f"{len(no_pre)} eval matches have no pre-kick-off Betfair 1X2: {no_pre}")
    t1x2 = {m: pre1x2[m] for m in eval_ids}   # kappa target: last price before kick-off, as the calibrators use
    missing_ou = [m for m in eval_ids if m not in ou_all or not ou_all[m]]
    if missing_ou:
        raise ValueError(f"{len(missing_ou)} eval matches have no O/U targets: {missing_ou}")

    # ── Poisson fit (shared by mkt_poisson/mkt_poisson_t: frac_remaining = 1 at minute 0) ──
    console.print("[yellow]Fitting market-implied Poisson lambdas...[/]")
    lam = {m: fit_lambdas(t1x2[m], ou_all[m]) for m in eval_ids}

    # ── training data (rate profile + logit) ───────────────────────────────────
    console.print(f"[yellow]Training pool (< {SEASON_CFG[SEASON][1]}, {seasons_through(SEASON)})...[/]")
    tg, tmeta, no_ev = load_train_matches()
    train_ids = sorted(tmeta)
    if set(train_ids) & set(eval_ids):
        raise RuntimeError("LEAKAGE: eval match present in the training set")
    console.print(f"  Training matches: {len(train_ids)} "
                  f"({len(no_ev)} scheduled matches dropped: no WhoScored event feed)")

    # mkt_poisson_t: empirical goal-time survival on the u-clock (training goals only)
    gclock = np.sort(np.concatenate([tg[m][:, 0].astype(float) for m in train_ids if len(tg[m])])) + 0.5
    console.print(f"  Training goals for rate profile: {len(gclock):,}")
    surv = lambda u: 1 - np.searchsorted(gclock, u, side='left') / len(gclock)  # S(u), S(0)=1

    # mkt_naive: pre-kick-off market features
    have = [m for m in train_ids if m in pre1x2]
    console.print(f"  Training matches with pre-kick-off Betfair 1X2: [green]{len(have)}[/] "
                  f"of {len(train_ids)} ({len(train_ids) - len(have)} excluded)")
    rows = []
    for m in have:
        p = pre1x2[m]
        lo_h, lo_a = np.log(p[0] / p[1]), np.log(p[2] / p[1])
        gu, gh = tg[m][:, 0].astype(float), tg[m][:, 1].astype(bool)
        mins = np.concatenate([np.arange(0, tmeta[m]['h1_end'] + 1), np.arange(45, tmeta[m]['h2_end'] + 1)])
        per = np.concatenate([np.ones(tmeta[m]['h1_end'] + 1), np.full(tmeta[m]['h2_end'] - 44, 2)])
        u = clock(mins, per)
        sd = np.array([(gh[gu <= x]).sum() - (~gh[gu <= x]).sum() for x in u])  # goals at clock <= u seen
        # rem uses the SAME function as evaluation (0.1-floored) so train/test definitions are identical
        rem = np.array([compute_remaining(mm, pp) for mm, pp in zip(mins, per)])
        rows.append(np.column_stack([np.full(len(u), lo_h), np.full(len(u), lo_a), sd, rem, sd * rem,
                                     np.full(len(u), tmeta[m]['y'])]))
    tr = np.vstack(rows)
    console.print(f"  mkt_naive training rows: {len(tr):,}")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1e6, random_state=SEED))
    clf.fit(tr[:, :5], tr[:, 5].astype(int))

    # ── predict all three baselines on the eval grid ───────────────────────────
    mid = grid['match_id'].values
    sd_ev = (grid['home_goals'] - grid['away_goals']).values
    rem_ev = np.array([compute_remaining(m, p) for m, p in zip(grid['eval_minute'], grid['period'])])
    u_ev = clock(grid['eval_minute'].values, grid['period'].values)
    lh = np.array([lam[m][0] for m in mid]); la = np.array([lam[m][1] for m in mid])

    out = {}
    for tag, f in [('mkt_poisson', rem_ev / TOTAL), ('mkt_poisson_t', surv(u_ev))]:
        out[tag] = np.stack(outcome_probs(lh * f, la * f, sd_ev), 1)
    # eval features use the pre-kick-off Betfair 1X2 already targeted by the Poisson baselines
    Xe = np.column_stack([[np.log(t1x2[m][0] / t1x2[m][1]) for m in mid],
                          [np.log(t1x2[m][2] / t1x2[m][1]) for m in mid],
                          sd_ev, rem_ev, sd_ev * rem_ev])
    out['mkt_naive'] = clf.predict_proba(Xe)

    frames = []
    for tag, P in out.items():
        d = grid.copy()
        d['model'] = tag
        d['pred_home'], d['pred_draw'], d['pred_away'] = P[:, 0], P[:, 1], P[:, 2]
        frames.append(d[cols])
    bench = pd.concat(frames, ignore_index=True)
    bench.to_parquet(os.path.join(OUT, 'benchmark_preds.parquet'), index=False)
    console.print(f"[green]Saved {len(bench):,} benchmark predictions[/]")


if __name__ == '__main__':
    main()
