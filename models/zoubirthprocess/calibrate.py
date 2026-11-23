"""Market calibration for Zou birth process using 1X2 + O/U markets.

Mirrors models/weibull/calibrate.py for fair comparison.
Run: python -m models.zoubirthprocess.calibrate
"""
import os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.zoubirthprocess.model import (
    ZouParamBaseline, calibrate_thetas_full, compute_outcome_probs_full
)

from models.market import load_market_data, eval_match_ids

console = Console()
SEASON = '24-25'


def _calibrate_one(args):
    """Worker: calibrate a single match (must be top-level for pickling)."""
    home_id, away_id, mkt, team_alpha, team_beta, bp = args
    t1_model = team_alpha[home_id] * team_beta[away_id]
    t2_model = team_alpha[away_id] * team_beta[home_id]
    target_1x2 = {'p_home': mkt['p_home'], 'p_draw': mkt['p_draw'], 'p_away': mkt['p_away']}
    t1_cal, t2_cal, err = calibrate_thetas_full(
        target_1x2, mkt['ou_targets'], bp, t1_model, t2_model, shots_dev=-bp['dev_shots_fit'][1])   # no shots yet at kick-off
    return {
        'date': mkt['date'], 'home_id': home_id, 'away_id': away_id,
        'ws_game_id': mkt['ws_game_id'],
        'home_name': mkt['home_name'], 'away_name': mkt['away_name'],
        't1_model': t1_model, 't2_model': t2_model,
        't1_cal': t1_cal, 't2_cal': t2_cal,
        'p_home': mkt['p_home'], 'p_draw': mkt['p_draw'], 'p_away': mkt['p_away'],
        'n_ou_markets': len(mkt['ou_targets']),
        'delta_t1': t1_cal - t1_model, 'delta_t2': t2_cal - t2_model,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='calibd', help='Label for output file')
    p.add_argument('--workers', type=int, default=os.cpu_count() - 1, help='Parallel workers')
    p.add_argument('--season', default='24-25', help='Eval season to calibrate')
    args = p.parse_args()
    global SEASON
    SEASON = args.season

    console.print(f"[bold]Zou Birth Process Calibration: {args.label}[/]")
    console.print("=" * 60)

    params_dir = os.path.join(ROOT, 'models', 'zoubirthprocess', 'params', f'pl_{SEASON.replace("-", "_")}')
    zou_params = ZouParamBaseline(params_dir)
    console.print(f"Zou params: {len(zou_params.param_dates)} dates "
                  f"({zou_params.param_dates[0]} to {zou_params.param_dates[-1]})")

    market_data = load_market_data(SEASON, eval_match_ids(SEASON))  # raises if any eval match lacks targets
    console.print(f"[green]Matches with pre-kick-off 1X2 + O/U: {len(market_data)}[/]\n")

    # Pre-resolve params in main process, build work items
    work = []
    for (home_id, away_id), mkt in market_data.items():
        team_alpha, team_beta, bp = zou_params.get_params_for_date(mkt['date'])
        if home_id not in team_alpha or away_id not in team_alpha:
            raise KeyError(f"no team parameters for {mkt['home_name']} v {mkt['away_name']} on {mkt['date']}")
        work.append((home_id, away_id, mkt, team_alpha, team_beta, bp))

    console.print(f"[yellow]Calibrating {len(work)} matches with {args.workers} workers...[/]\n")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_calibrate_one, w): w for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(work):
                console.print(f"  {i}/{len(work)} matches done")

    results.sort(key=lambda x: (x['date'], x['ws_game_id']))   # workers finish out of order
    out_dir = os.path.join(ROOT, 'models', 'zoubirthprocess', 'calibrated')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f'zou_calibration_{SEASON.replace("-", "_")}_{args.label}.csv')
    df.to_csv(out_path, index=False)
    console.print(f"\n[green]Saved {len(results)} matches to {out_path}[/]")
    console.print(f"t1 delta: mean={df['delta_t1'].mean():.4f}, std={df['delta_t1'].std():.4f}")
    console.print(f"t2 delta: mean={df['delta_t2'].mean():.4f}, std={df['delta_t2'].std():.4f}")


if __name__ == '__main__':
    main()
