"""Market calibration for Maia Cox process using 1X2 + O/U markets.

Mirrors models/zoubirthprocess/calibrate.py for fair comparison.
Run: python -m models.maia.calibrate
"""
import os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.maia.model import MaiaParamBaseline, calibrate_thetas_full

from models.market import load_market_data, eval_match_ids, N_SIMS_CAL, SEED_CAL

console = Console()
SEASON = '24-25'


def _calibrate_one(args):
    """Worker: calibrate a single match (must be top-level for pickling)."""
    home_id, away_id, mkt, team_alpha, team_beta, mp = args
    t1_model = team_alpha[home_id] * team_beta[away_id]
    t2_model = team_alpha[away_id] * team_beta[home_id]
    target_1x2 = {'p_home': mkt['p_home'], 'p_draw': mkt['p_draw'], 'p_away': mkt['p_away']}
    t1_cal, t2_cal, err = calibrate_thetas_full(
        target_1x2, mkt['ou_targets'], mp, t1_model, t2_model, n_sims=N_SIMS_CAL, seed=SEED_CAL,
        shots_dev=-mp['dev_shots_fit'][1])   # no shots yet at kick-off, as in the minute-0 forecast
    return {
        'date': mkt['date'], 'home_id': home_id, 'away_id': away_id,
        'ws_game_id': mkt['ws_game_id'],
        'home_name': mkt['home_name'], 'away_name': mkt['away_name'],
        't1_model': t1_model, 't2_model': t2_model,
        't1_cal': t1_cal, 't2_cal': t2_cal,
        'p_home': mkt['p_home'], 'p_draw': mkt['p_draw'], 'p_away': mkt['p_away'],
        'n_ou_markets': len(mkt['ou_targets']),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--label', default='calibd')
    p.add_argument('--workers', type=int, default=os.cpu_count() - 1, help='Parallel workers')
    p.add_argument('--season', default='24-25', help='Eval season to calibrate')
    args = p.parse_args()
    global SEASON
    SEASON = args.season

    console.print(f"[bold]Maia Cox Process Calibration: {args.label}[/]")
    console.print("=" * 60)

    params_dir = os.path.join(ROOT, 'models', 'maia', 'params', f'pl_{SEASON.replace("-", "_")}')
    maia_baseline = MaiaParamBaseline(params_dir)
    console.print(f"Maia params: {len(maia_baseline.param_dates)} dates "
                  f"({maia_baseline.param_dates[0]} to {maia_baseline.param_dates[-1]})")

    market_data = load_market_data(SEASON, eval_match_ids(SEASON))  # raises if any eval match lacks targets
    console.print(f"[green]Matches with pre-kick-off 1X2 + O/U: {len(market_data)}[/]\n")

    # Pre-resolve params in main process, build work items
    work = []
    for (home_id, away_id), mkt in market_data.items():
        team_alpha, team_beta, mp = maia_baseline.get_params_for_date(mkt['date'])
        if home_id not in team_alpha or away_id not in team_alpha:
            raise KeyError(f"no team parameters for {mkt['home_name']} v {mkt['away_name']} on {mkt['date']}")
        work.append((home_id, away_id, mkt, team_alpha, team_beta, mp))

    console.print(f"[yellow]Calibrating {len(work)} matches with {args.workers} workers...[/]\n")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_calibrate_one, w): w for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(work):
                console.print(f"  {i}/{len(work)} matches done")

    results.sort(key=lambda x: (x['date'], x['ws_game_id']))   # workers finish out of order
    out_dir = os.path.join(ROOT, 'models', 'maia', 'calibrated')
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(out_dir, f'maia_calibration_{SEASON.replace("-", "_")}_{args.label}.csv')
    df.to_csv(out_path, index=False)
    console.print(f"\n[green]Saved {len(results)} matches to {out_path}[/]")
    console.print(f"t1 delta: mean={df['t1_cal'].mean() - df['t1_model'].mean():.4f}")
    console.print(f"t2 delta: mean={df['t2_cal'].mean() - df['t2_model'].mean():.4f}")


if __name__ == '__main__':
    main()
