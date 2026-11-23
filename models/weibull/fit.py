"""One fitted specification shared by Weibull calibration and prediction."""
import hashlib
import json
from pathlib import Path
from models.covariates.stoppage import MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE

# CONFIG
ROOT = Path(__file__).resolve().parents[2]
VERSION = 'estimated-common-baseline-bic-shapes-v2'
COVARIATES = ['red_diff', 'dev_shots']
DELTA = 5


def select_shapes(data, baseline, tids, cov_names):
    """Compare common/half-specific shapes on the same training likelihood.

    Team strengths are fixed at their joint common-shape fit. BIC uses whole
    goal intervals, not the arbitrary number of covariate exposure pieces.
    """
    import numpy as np
    from scipy.optimize import minimize
    from models.covariates.grid_search import nll_joint
    x0 = np.r_[baseline['gamma'], 1., np.zeros(len(cov_names))]
    rows = []
    for name, bounds in [('common', (1., 1.)), ('half', (.5, 2.))]:
        result = minimize(nll_joint, x0, args=(data, baseline, tids, cov_names), method='L-BFGS-B',
                          bounds=[(.1, 5.), bounds]+[(None, None)]*len(cov_names),
                          options={'maxiter': 2000, 'ftol': 1e-11})
        if not result.success:
            raise RuntimeError(f'{name} shape fit failed: {result.message}')
        k = len(cov_names) + (1 if name == 'common' else 2)
        rows.append(dict(shape=name, params=result.x.tolist(), nll=float(result.fun), k=k,
                         n_intervals=int(data['n_intervals']), bic=float(2*result.fun+k*np.log(data['n_intervals']))))
        x0 = result.x
    if rows[1]['nll'] > rows[0]['nll']+1e-5:
        raise RuntimeError('Nested half-specific fit has a worse likelihood')
    return min(rows, key=lambda row: row['bic']), rows


def fit_id(fit):
    return hashlib.sha256(json.dumps(fit, sort_keys=True).encode()).hexdigest()


def fit_path(season):
    return ROOT / 'models' / 'weibull' / 'fitted' / f'{season}.json'


def load_fit(season):
    fit = json.loads(fit_path(season).read_text())
    if fit['version'] != VERSION or fit['season'] != season or fit['stoppage'] != [MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE]:
        raise ValueError('Stale Weibull fit; run eval/test.py --fit-only')
    return fit


def save_fit(season, gamma, phi, betas, meta, no_cov, delta):
    fit = dict(version=VERSION, season=season, gamma=float(gamma), phi=float(phi),
               betas=betas, meta=meta, no_cov=list(no_cov), delta=delta,
               stoppage=[MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE])
    path = fit_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fit, indent=2) + '\n')
    return fit
