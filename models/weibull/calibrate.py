"""Market calibration using 1X2 + all available O/U markets (0.5, 1.5, 2.5, 3.5, 4.5).

Fit team etas to joint outcome and total-goal targets using the saved prediction specification.

Run: python -m models.weibull.calibrate
"""
import os, sys, numpy as np, pandas as pd, torch
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from rich.console import Console

from .core import simulate_goals, DEV
from .fit import load_fit, fit_id
from models.covariates.stoppage import MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE
from models.market import load_market_data, eval_match_ids, N_SIMS_CAL as N_SIMS, SEED_CAL as SEED

# Root is two levels up from models/weibull/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

console = Console()
# CONFIG
SEASON = '24-25'  # overridden by --season in main()
WORKERS = 8


def load_calibrated_etas(season='24-25', label='weibull_calibd'):
    """Load calibrated etas from CSV. Returns dict: (home_id, away_id) -> {eta_h_cal, eta_a_cal, ...}"""
    path = os.path.join(ROOT, 'models', 'weibull', 'calibrated', f'match_calibration_{season.replace("-", "_")}_{label}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run calibration first: python -m models.weibull.calibrate\n  Missing: {path}")
    df = pd.read_csv(path)
    if not df["fit_id"].eq(fit_id(load_fit(season))).all():
        raise ValueError("Calibration does not match the current Weibull fit")
    if df.duplicated(["home_id", "away_id"]).any():
        raise ValueError("Duplicate calibrated fixture")
    return {(int(r['home_id']), int(r['away_id'])): {
        'eta_h_cal': r['eta_h_cal'], 'eta_a_cal': r['eta_a_cal'],
        'eta_h_model': r['eta_h_model'], 'eta_a_model': r['eta_a_model'],
        'home_name': r['home_name'], 'away_name': r['away_name'], 'date': r['date']
    } for _, r in df.iterrows()}


def kickoff_covs(fit):
    """The fitted covariates at kick-off, including population-curve intercepts."""
    covs = {}
    for name in fit['betas']:
        if name in ('red_diff', 'score_diff', 'cum_psxg_diff', 'cum_ng_psxg_diff'):
            covs[name] = 0.
        else:
            if name.startswith('dev_'):
                value = -fit['meta'][f'{name}_fit'][1]
            elif name == 'game_time_c':
                value = -fit['meta']['game_time_center']
                covs['_game_time_center'] = -value
            else:
                value = 0.  # Non-deviation cumulative fields are zero at kick-off.
            covs[f'{name}_home'] = covs[f'{name}_away'] = value
    return covs


def simulate_match(eta_h, eta_a, gamma, phi, n_sims, *, betas, covs):
    """Common random numbers for optimisation, without changing the caller's RNG."""
    devices = [torch.cuda.current_device()] if DEV.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        gh, ga = simulate_goals(eta_h, eta_a, 0, 90 + MEAN_H1_STOPPAGE + MEAN_H2_STOPPAGE,
                               betas, covs, gamma, phi, n=n_sims,
                               period=1, h1_left=45 + MEAN_H1_STOPPAGE)
    diff, total = gh-ga, gh+ga
    return {'p_home': float((diff > 0).float().mean()),
            'p_draw': float((diff == 0).float().mean()),
            'p_away': float((diff < 0).float().mean()),
            **{f'p_over_{t}': float((total > t).float().mean()) for t in [.5, 1.5, 2.5, 3.5, 4.5]},
            'expected_goals': float(total.mean())}


def find_calibrated_etas(target_1x2, ou_targets, eta_h_init, eta_a_init,
                         gamma, phi, n_sims, *, betas, covs):
    """Fit market etas using exactly the deployed shapes and covariate effects."""
    def objective(etas):
        result = simulate_match(*etas, gamma, phi, n_sims, betas=betas, covs=covs)
        return (sum((result[k]-p)**2 for k, p in target_1x2.items()) +
                sum((result[f'p_over_{t}']-p)**2 for t, p in ou_targets.items()))
    # Unbounded, as in the Zou/Maia calibrators: scipy's bounded Powell line search can abandon a good start for a worse local minimum
    result = minimize(objective, [eta_h_init, eta_a_init], method='Powell', options={'maxiter': 200, 'ftol': 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f'Weibull calibration failed: {result.message}')
    if result.fun > objective([eta_h_init, eta_a_init]):
        raise RuntimeError(f'Weibull calibration ended worse than its start: {result.fun:.5f}')
    return *result.x, result.fun


def calibrate_one(job):
    """Independent match worker; common random numbers make scheduling irrelevant."""
    hi, ai, m, eh, ea, fit, n_sims = job
    torch.set_num_threads(1)
    targets = {k: m[k] for k in ['p_home', 'p_draw', 'p_away']}
    ch, ca, err = find_calibrated_etas(targets, m['ou_targets'], eh, ea,
                                      fit['gamma'], fit['phi'], n_sims,
                                      betas=fit['betas'], covs=kickoff_covs(fit))
    return dict(date=m['date'], home_id=hi, away_id=ai,
                home_name=m['home_name'], away_name=m['away_name'],
                eta_h_model=eh, eta_a_model=ea, eta_h_cal=ch, eta_a_cal=ca,
                delta_h=ch-eh, delta_a=ca-ea, **targets,
                n_ou_markets=len(m['ou_targets']), sse=err, fit_id=fit_id(fit))


def main():
    import argparse
    from .team_params import TeamParamBaseline
    p = argparse.ArgumentParser()
    p.add_argument('--season', default='24-25')
    p.add_argument('--label', default='weibull_calibd')
    args = p.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    fit = load_fit(args.season)
    baseline = TeamParamBaseline(os.path.join(ROOT, 'models', 'weibull', 'team_params',
                                            f'pl_{args.season.replace("-", "_")}'))
    markets = load_market_data(args.season, eval_match_ids(args.season))  # raises if any eval match lacks targets
    jobs = [(hi, ai, m, baseline.get_log_et(hi, ai, True, m['date']),
             baseline.get_log_et(ai, hi, False, m['date']), fit, N_SIMS)
            for (hi, ai), m in sorted(markets.items())]
    rows = []
    console.print(f'Calibrating {len(markets)} matches with shared fit {fit_id(fit)[:12]}, {WORKERS} workers')
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn')) as pool:
        for i, row in enumerate(pool.map(calibrate_one, jobs)):
            rows.append(row)
            if (i+1) % 10 == 0 or i+1 == len(markets):
                console.print(f'Calibrated {i+1}/{len(markets)}; SSE={row["sse"]:.6f}')
    out = os.path.join(ROOT, 'models', 'weibull', 'calibrated')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f'match_calibration_{args.season.replace("-", "_")}_{args.label}.csv')
    pd.DataFrame(rows).to_csv(path, index=False)
    console.print(f'Saved {path}')


if __name__ == '__main__':
    main()
