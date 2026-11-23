"""Generate predictions from strategies. Run: python eval/test.py --strategy weibull"""

import argparse
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from rich import print

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import (
    SEASON_CFG,
    SEASON_CFG,
    seasons_through,
    build_goals_lookup,
    build_team_id_map,
    get_elapsed,
    get_score_at,
    load_betfair_data,
)
from models import TeamParamBaseline, get_strategy, list_strategies
from models.covariates.stoppage import compute_remaining
from models.weibull.fit import COVARIATES, DELTA
from models.covariates.stoppage import MEAN_H1_STOPPAGE


def load_data(seasons):
    """Load WhoScored events, schedule, and alignment across seasons."""
    if isinstance(seasons, str):
        seasons = [seasons]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_ws, all_sched, all_align = [], [], []
    for season in seasons:
        ws = pd.read_parquet(os.path.join(root, "data", season, "whoscored_events.parquet"))
        sched = pd.read_parquet(os.path.join(root, "data", season, "fbref_schedule.parquet"))
        align = pd.read_parquet(os.path.join(root, "data", season, "match_alignment.parquet"))
        ws["season"], align["season"], sched["season"] = season, season, season
        all_ws.append(ws); all_sched.append(sched); all_align.append(align)
    ws = pd.concat(all_ws, ignore_index=True)
    schedule = pd.concat(all_sched, ignore_index=True)
    align = pd.concat(all_align, ignore_index=True)
    # Add is_home and match_date columns
    ws_to_home = dict(zip(align["ws_game_id"], align["ws_home_team"]))
    ws_to_date = dict(zip(align["ws_game_id"], pd.to_datetime(align["date"])))
    ws["is_home"] = ws.apply(
        lambda r: r["team_name"] == ws_to_home.get(r["ws_match_id"]), axis=1
    )
    ws["match_date"] = ws["ws_match_id"].map(ws_to_date)
    return ws, schedule, align


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_SIMS = 10000
SEED = 8
WORKERS = 12
# Strategy attributes keyed by match_id or (match_id, ...): sliced per match before shipping to workers
MATCH_KEYED = ("goals_lookup", "rc_lookup", "cov_lookup", "cov_data", "match_dates", "calibrated_thetas", "odds_lookup")


@dataclass
class TeamState:
    """Minimal team state for strategy.predict_probs()."""

    baseline_log_et: float


def fit_joint(ws, fb_shots, align, train_start_date: str, train_end_date: str,
              delta: int = 5, cov_names: list = None, use_full_history: bool = True):
    """Estimate covariate effects and select the shape specification by training BIC."""
    from rich import print
    from models.covariates.grid_search import prep_data
    from models.weibull.team_params.covariates import extract_subintervals_with_covariates
    from models.weibull.team_params.estimate import estimate_team_params
    from models.weibull.clock import training_events
    from models.weibull.fit import select_shapes
    names = ["red_diff"] if cov_names is None else cov_names
    all_train = training_events(ws[pd.to_datetime(ws["match_date"]) < pd.to_datetime(train_end_date)])
    sub_ws = all_train if use_full_history else all_train[pd.to_datetime(all_train["match_date"]) >= pd.to_datetime(train_start_date)]
    baseline = estimate_team_params(all_train)
    tids = sorted(baseline['attack'])
    sub = extract_subintervals_with_covariates(sub_ws, fb_shots, align, delta=delta)
    data = prep_data(sub, tids, names)
    selected, rows = select_shapes(data, baseline, tids, names)
    print({'shape_comparison': rows, 'selected': selected['shape']})
    gamma, phi, *betas = selected['params']
    meta = dict(shape_selection=rows, selected_shape=selected['shape'],
                baseline={key: baseline[key] for key in ['mu', 'home', 'gamma', 'nll', 'n_intervals']})
    if "game_time_c" in names:
        meta["game_time_center"] = float(data["game_time_c_mean"])
    for name in names:
        if name.startswith('dev_'):
            meta[f'{name}_fit'] = list(data[f'_{name}_fit'])
    return gamma, phi, dict(zip(names, betas)), meta


def fit_gamma_phi_only(ws, train_start_date: str, train_end_date: str,
                       use_full_history: bool = True):
    """Select common/half-specific shapes for the no-covariate comparison."""
    from rich import print
    from models.weibull.team_params.estimate import estimate_team_params, extract_intervals, estimate_single_gamma, estimate_single_gamma_phi
    from models.weibull.clock import training_events
    all_train = training_events(ws[pd.to_datetime(ws["match_date"]) < pd.to_datetime(train_end_date)])
    fit_ws = all_train if use_full_history else all_train[pd.to_datetime(all_train["match_date"]) >= pd.to_datetime(train_start_date)]
    baseline = estimate_team_params(all_train, verbose=False)
    one = estimate_single_gamma(fit_ws, baseline, verbose=False)
    half = estimate_single_gamma_phi(fit_ws, baseline, verbose=False)
    difference = 2*(half['_meta']['nll']-one['_meta']['nll'])+np.log(len(extract_intervals(fit_ws)))
    print({'no_cov_half_minus_common_bic': difference})
    return (one['gamma'], 1.) if difference >= 0 else (half['gamma'], half['phi'])


def seed_call(mid, period, minute):
    """Reseed numpy and torch from (SEED, match, period, minute): each forecast is independent of run order."""
    s = int(np.random.SeedSequence([SEED, int(mid), period, minute]).generate_state(1)[0])
    np.random.seed(s); torch.manual_seed(s)


def predict_match(job):
    """Forecasts for one match at every (period, minute) in `grid` before its gating end time."""
    strategy, strat_name, n_sims, grid, c = job
    mid, goals, rows = c["mid"], c["goals"], []
    for period, minute, gate in grid:
        if minute >= c[gate]:   # match (or first half, for 1H stoppage) already over
            continue
        seen = goals.iloc[:0] if period == 1 and minute == 0 else goals   # minute 0 is forecast before kick-off: no goal of the opening minute is seen
        home_goals, away_goals = get_score_at(seen, minute, period)
        remaining = compute_remaining(minute, period)   # mean stoppage estimate
        if remaining <= 0:
            continue
        # Elapsed time since last goal; first-half time left for 1H forecasts
        elapsed = get_elapsed(seen, minute, period, h1_end=max(45., c["h1"]) if period == 2 else 45.)
        h1_left = min(remaining, max(0.1, 45 + MEAN_H1_STOPPAGE - minute)) if period == 1 else 0.
        seed_call(mid, period, minute)
        probs = strategy.predict_probs(
            c["home"], c["away"], home_goals - away_goals, remaining, None, n_sims, match_id=mid,
            minute=minute, period=period, h1_left=h1_left, elapsed=elapsed, home_id=c["home_id"],
            away_id=c["away_id"], home_goals=home_goals, away_goals=away_goals)
        if probs is None:
            continue
        # Floor probabilities to prevent degenerate predictions (e.g. exact 0), then renormalise
        p = np.maximum([probs["home"], probs["draw"], probs["away"]], 1e-3)
        ph, pd_, pa = (p / p.sum()).tolist()
        rows.append({"match_id": mid, "eval_minute": minute, "period": period, "model": strat_name,
                     "pred_home": ph, "pred_draw": pd_, "pred_away": pa, "actual": c["actual"],
                     "score_diff": home_goals - away_goals, "home_goals": home_goals,
                     "away_goals": away_goals, "home_team": c["info"].get("home_team"),
                     "away_team": c["info"].get("away_team"), "date": c["info"].get("date")})
    return rows


def build_match_end_times(ws: pd.DataFrame, align: pd.DataFrame) -> dict:
    """Map ws_match_id -> last event minute (actual match end time)."""
    return ws.groupby('ws_match_id')['minute'].max().to_dict()


def build_h1_end_times(ws: pd.DataFrame) -> dict:
    """Map ws_match_id -> last FirstHalf event minute."""
    h1 = ws[ws['period'] == 'FirstHalf']
    return h1.groupby('ws_match_id')['minute'].max().to_dict()


def run_predictions(strategy, strat_name, matches, goals_lookup, team_params, team_id_map,
                    outcome_lookup, match_info, n_sims, match_end_times, h1_end_times, workers):
    """Run predictions at every minute, including stoppage time, one match per worker task."""
    # 1H minutes gated by match end, 1H stoppage by first-half end, 2H (incl. stoppage) by match end
    grid = ([(1, t, "end") for t in range(45)] + [(1, t, "h1") for t in range(45, int(max(h1_end_times.values())) + 1)]
            + [(2, t, "end") for t in range(45, int(max(match_end_times.values())) + 1)])
    ctxs = []
    for _, m in matches.iterrows():
        mid = int(m["ws_game_id"])
        home_id, away_id = team_id_map.get(m["ws_home_team"]), team_id_map.get(m["ws_away_team"])
        if mid not in outcome_lookup or home_id is None or away_id is None:
            continue
        ctxs.append(dict(
            mid=mid, home_id=home_id, away_id=away_id, actual=outcome_lookup[mid], info=match_info.get(mid, {}),
            goals=goals_lookup.get(mid, pd.DataFrame()), end=match_end_times[mid], h1=h1_end_times[mid],
            home=TeamState(team_params.get_params(home_id, away_id, is_home=True, date=m["date"])["baseline_log_et"]),
            away=TeamState(team_params.get_params(away_id, home_id, is_home=False, date=m["date"])["baseline_log_et"])))
    # Ship each task a strategy copy holding only its match's entries of the match-keyed lookups
    keyed = [a for a in MATCH_KEYED if a in vars(strategy)]
    parts = {c["mid"]: {a: {} for a in keyed} for c in ctxs}
    for a in keyed:
        for k, v in getattr(strategy, a).items():
            m = int(k[0] if isinstance(k, tuple) else k)
            if m in parts:
                parts[m][a][k] = v
    jobs = []
    for c in ctxs:
        s = copy.copy(strategy)
        vars(s).update(parts[c["mid"]])
        jobs.append((s, strat_name, n_sims, grid, c))
    with ProcessPoolExecutor(workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        preds = [r for rows in tqdm(pool.map(predict_match, jobs), total=len(jobs), desc=strat_name) for r in rows]
    # Original row order: period, minute, then match order
    return sorted(preds, key=lambda r: (r["period"], r["eval_minute"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="maia_calibd", choices=list_strategies())
    p.add_argument("--seasons", nargs="+", default=None, help="Default: all seasons through --eval-season")
    p.add_argument("--eval-season", default="24-25", choices=list(SEASON_CFG), help="Season to evaluate on")
    p.add_argument(
        "--delta",
        type=int,
        default=DELTA,
        help="Sub-interval width for weibull strategy (minutes)",
    )
    p.add_argument(
        "--covariates",
        type=str,
        default=",".join(COVARIATES),  # shots headline covariate (A2/D4); dev_xg / dev_psxg = tested alternatives
        help="Comma-separated covariate names",
    )
    p.add_argument("--fit-only", action="store_true", help="Fit and save the shared Weibull specification")
    p.add_argument("--clear", action="store_true", help="Delete all per-run prediction files and exit")
    p.add_argument("--merge", action="store_true", help="Concatenate per-run files into predictions.parquet and exit")
    p.add_argument("--workers", type=int, default=WORKERS, help="Prediction processes (one match per task)")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)   # match the single-threaded workers

    eval_dir = os.path.dirname(__file__)
    out_path = os.path.join(eval_dir, "predictions.parquet")
    run_dir = os.path.join(eval_dir, "predictions")   # one file per (season, strategy) so runs can go in parallel
    os.makedirs(run_dir, exist_ok=True)

    if args.clear:
        for f in os.listdir(run_dir):
            os.remove(os.path.join(run_dir, f))
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"Cleared {run_dir} and {out_path}")
        return
    if args.merge:
        files = sorted(f for f in os.listdir(run_dir) if f.split("_", 1)[0] in SEASON_CFG)   # reported seasons only
        if not files:
            raise FileNotFoundError(f"No run files for {list(SEASON_CFG)} in {run_dir}")
        df = pd.concat([pd.read_parquet(os.path.join(run_dir, f)) for f in files], ignore_index=True)
        df.to_parquet(out_path)
        print(f"Merged {len(files)} runs -> {out_path} ({len(df)} rows): "
              f"{df.groupby(['season', 'model'])['match_id'].nunique().to_dict()}")
        return

    root = os.path.dirname(os.path.dirname(__file__))
    args.seasons = args.seasons or seasons_through(args.eval_season)
    TRAIN_START_DATE, EVAL_START_DATE = SEASON_CFG[args.eval_season]
    print(f"Strategy: {args.strategy} | Eval {args.eval_season} from {EVAL_START_DATE} | Sims: {N_SIMS}")
    print("-" * 50)

    # Load data (all seasons for training, eval_season for prediction)
    ws, schedule, align = load_data(args.seasons)
    fb_shots = pd.concat([pd.read_parquet(os.path.join(root, "data", s, "fbref_shots.parquet"))
                          for s in args.seasons], ignore_index=True)

    from models.weibull.fit import load_fit, save_fit, fit_id
    if args.fit_only:
        names = [c.strip() for c in args.covariates.split(',') if c.strip()]
        gamma, phi, betas, meta = fit_joint(ws, fb_shots, align, TRAIN_START_DATE,
                                           EVAL_START_DATE, args.delta, names)
        no_cov = fit_gamma_phi_only(ws, TRAIN_START_DATE, EVAL_START_DATE)
        fit = save_fit(args.eval_season, gamma, phi, betas, meta, no_cov, args.delta)
        print(f"Saved shared Weibull fit {fit_id(fit)}")
        return

    # Build lookups (use WhoScored events for goals - includes own goals)
    goals_lookup = build_goals_lookup(ws, align)
    team_id_map = build_team_id_map(ws)

    params_dir = os.path.join(
        root, "models", "weibull", "team_params", f"pl_{args.eval_season.replace('-', '_')}"
    )
    team_params = TeamParamBaseline(params_dir)

    odds_lookup, outcome_lookup, match_info = load_betfair_data(args.eval_season, root)
    betfair = get_strategy("betfair", odds_lookup=odds_lookup or {})

    # Filter to eval period matches (eval_season only, after EVAL_START_DATE)
    eval_start = pd.to_datetime(EVAL_START_DATE)
    align["date"] = pd.to_datetime(align["date"])
    eval_matches = align[(align["date"] >= eval_start) & (align["season"] == args.eval_season)].copy()
    ev = ws[ws.ws_match_id.isin(eval_matches.ws_game_id)]   # per-match lookups need only eval-match events
    print(f"Eval matches: {len(eval_matches)} (season {args.eval_season})")

    # Parse covariates
    cov_names = [c.strip() for c in args.covariates.split(",") if c.strip()]
    strat_label = args.strategy
    print(f"Covariates: {cov_names}")

    # Load strategy
    if args.strategy == "betfair":
        strategy = betfair
        print(f"Using Betfair odds ({len(odds_lookup or {})} entries)")
    elif args.strategy in ("weibull", "weibull_calibd", "weibull_no_cov"):
        from models.strategies.extended import build_cov_lookup
        fit = load_fit(args.eval_season)
        if set(cov_names) != set(fit['betas']) or args.delta != fit['delta']:
            raise ValueError("Prediction arguments differ from the saved Weibull fit")
        if args.strategy == 'weibull_no_cov':
            strategy = get_strategy(args.strategy, gamma=fit['no_cov'][0], phi=fit['no_cov'][1])
        else:
            cov_lookup = build_cov_lookup(ev,
                                          fb_shots, align, fit_meta=fit['meta'], period_keys=True, prematch=True)
            kw = dict(betas=fit['betas'], cov_data=cov_lookup, gamma=fit['gamma'],
                      phi=fit['phi'], fit_meta=fit['meta'])
            if args.strategy == 'weibull_calibd':
                from models.weibull.calibrate import load_calibrated_etas
                kw['calibrated_etas'] = load_calibrated_etas(args.eval_season)
            strategy = get_strategy(args.strategy, **kw)
    elif args.strategy in ("zou", "zou_calibd"):
        from models.strategies.extended import build_cov_lookup
        from models.strategies.zou_strategy import load_calibrated_thetas
        from models.zoubirthprocess.model import ZouParamBaseline

        zou_params = ZouParamBaseline(os.path.join(root, "models", "zoubirthprocess", "params", f"pl_{args.eval_season.replace('-', '_')}"))
        _, _, tmp_p = zou_params.get_params_for_date(zou_params.param_dates[-1])   # period-keyed shots deviations from the saved population curve
        zou_cov = build_cov_lookup(ev, fb_shots, align, fit_meta={"dev_shots_fit": tmp_p["dev_shots_fit"]}, period_keys=True, prematch=True)
        match_dates = {int(r["ws_game_id"]): str(r["date"])[:10] for _, r in eval_matches.iterrows()}
        if args.strategy == "zou_calibd":
            cal_thetas = load_calibrated_thetas(args.eval_season, label="calibd")
            print(f"Loaded {len(cal_thetas)} calibrated theta pairs (1X2 + O/U)")
            strategy = get_strategy("zou_calibd", zou_params=zou_params, match_dates=match_dates,
                                    calibrated_thetas=cal_thetas, cov_lookup=zou_cov)
        else:
            strategy = get_strategy("zou", zou_params=zou_params, match_dates=match_dates, cov_lookup=zou_cov)
    elif args.strategy in ("maia", "maia_calibd"):
        from models.maia.model import MaiaParamBaseline
        from models.strategies.extended import build_cov_lookup
        from models.strategies.maia_strategy import load_calibrated_thetas as load_maia_cal

        maia_params = MaiaParamBaseline(os.path.join(root, "models", "maia", "params", f"pl_{args.eval_season.replace('-', '_')}"))
        # Period-keyed state (red cards; shots deviations from the saved population curve)
        _, _, tmp_p = maia_params.get_params_for_date(maia_params.param_dates[-1])
        maia_cov = build_cov_lookup(ev, fb_shots, align, fit_meta={"dev_shots_fit": tmp_p["dev_shots_fit"]}, period_keys=True, prematch=True)
        print(f"Maia state lookup: {len(maia_cov)} entries")
        match_dates = {int(r["ws_game_id"]): str(r["date"])[:10] for _, r in eval_matches.iterrows()}
        if args.strategy == "maia_calibd":
            cal_thetas = load_maia_cal(args.eval_season, label="calibd")
            print(f"Loaded {len(cal_thetas)} Maia calibrated theta pairs")
            strategy = get_strategy("maia_calibd", maia_params=maia_params, match_dates=match_dates,
                                    calibrated_thetas=cal_thetas, cov_lookup=maia_cov)
        else:
            strategy = get_strategy("maia", maia_params=maia_params, match_dates=match_dates, cov_lookup=maia_cov)
    else:
        strategy = get_strategy(args.strategy)

    # Build per-match end times for stoppage-time gating
    match_end_times = build_match_end_times(ws, align)
    h1_end_times = build_h1_end_times(ws)

    # Run predictions
    all_preds = run_predictions(strategy, strat_label, eval_matches, goals_lookup, team_params, team_id_map,
                                outcome_lookup, match_info, N_SIMS, match_end_times, h1_end_times, args.workers)

    df = pd.DataFrame(all_preds)
    df["season"] = args.eval_season
    for col in ["pred_home", "pred_draw", "pred_away"]:
        df[col] = df[col].round(4)
    run_path = os.path.join(run_dir, f"{args.eval_season}_{strat_label}.parquet")
    if args.strategy.startswith("weibull"):
        df.attrs["weibull_fit"] = fit_id(fit)
    df.to_parquet(run_path)
    print(f"Saved {run_path} ({len(df)} rows); run `eval/test.py --merge` to rebuild predictions.parquet")


if __name__ == "__main__":
    main()
