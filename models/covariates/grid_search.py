"""Time-varying covariate likelihood: joint gamma/beta estimation on sub-intervals.

gamma_1H, gamma_2H_ratio and betas are free; team etas (mu, home, att, def) are
fixed from the first estimation stage. The tables are produced by eval/refit.py.
"""
import numpy as np
from scipy.special import gamma as gamma_fn

# Subinterval tuple indices for separate home/away covariates
# Tuple layout: 0:match_id, 1:interval_id, 2:rel_start, 3:rel_end, 4:home_id,
# 5:away_id, 6:scorer_id, 7:censored, 8:score_diff, 9:goals_diff,
# 10:ft_touch_diff, 11:red_diff, 12:ft_pass_diff, 13:psxg_diff, 14:ng_psxg_diff,
# 15:period, 16:is_final, 17:game_time,
# 18:ft_touch_home, 19:ft_touch_away, 20:ft_pass_home, 21:ft_pass_away,
# 22:psxg_home, 23:psxg_away, 24:ng_psxg_home, 25:ng_psxg_away,
# 26:goals_home, 27:goals_away, 28:xg_home, 29:xg_away,
# 30:ng_xg_home, 31:ng_xg_away, 32:cum_goals_home, 33:cum_goals_away,
# 34:cum_psxg_home, 35:cum_psxg_away, 36:cum_ng_psxg_home, 37:cum_ng_psxg_away
COV_SEP_IDX = {
    'ft_touch_home': 18, 'ft_touch_away': 19,
    'ft_pass_home': 20, 'ft_pass_away': 21,
    'psxg_home': 22, 'psxg_away': 23,
    'ng_psxg_home': 24, 'ng_psxg_away': 25,
    'goals_home': 26, 'goals_away': 27,
    'xg_home': 28, 'xg_away': 29,
    'ng_xg_home': 30, 'ng_xg_away': 31,
    'red_diff': 11,
    'cum_goals_home': 32, 'cum_goals_away': 33,
    'cum_psxg_home': 34, 'cum_psxg_away': 35,
    'cum_ng_psxg_home': 36, 'cum_ng_psxg_away': 37,
}


def prep_data(subintervals: list, tids: list, cov_names: list) -> dict:
    """Prepare arrays for time-varying NLL with separate home/away covariates."""
    idx = {t: i for i, t in enumerate(tids)}
    valid = [(s, idx.get(s[4]), idx.get(s[5])) for s in subintervals
             if idx.get(s[4]) is not None and idx.get(s[5]) is not None]

    interval_ids = list(dict.fromkeys(s[1] for s, _, _ in valid))
    iv_to_idx = {iv: i for i, iv in enumerate(interval_ids)}

    data = {
        'rel_start': np.array([s[2] for s, _, _ in valid]),
        'rel_end': np.array([s[3] for s, _, _ in valid]),
        'hi': np.array([h for _, h, _ in valid]),
        'ai': np.array([a for _, _, a in valid]),
        'interval_idx': np.array([iv_to_idx[s[1]] for s, _, _ in valid]),
        'n_intervals': len(interval_ids),
        'cens': np.array([s[7] for s, _, _ in valid], dtype=bool),
        'is_final': np.array([s[16] for s, _, _ in valid], dtype=bool),
        'is_home_scorer': np.array([s[6] == s[4] if s[16] and not s[7] else False for s, _, _ in valid]),
        'is_2h': np.array([s[15] == 2 for s, _, _ in valid], dtype=bool),
    }
    for name in cov_names:
        if name in ('dev_ng_psxg', 'dev_psxg', 'dev_goals', 'dev_xg', 'dev_shots'):  # cum value minus population mean
            gt = np.array([float(s[17]) for s, _, _ in valid])
            ci = {'dev_ng_psxg': 36, 'dev_psxg': 34, 'dev_goals': 32, 'dev_xg': 28, 'dev_shots': 40}[name]
            cum_h = np.array([s[ci] for s, _, _ in valid])
            cum_a = np.array([s[ci + 1] for s, _, _ in valid])
            coefs = np.polyfit(np.concatenate([gt, gt]),  # pool home+away for population curve
                               np.concatenate([cum_h, cum_a]), 1)
            pop_mean = np.polyval(coefs, gt)
            data[f'{name}_home'] = cum_h - pop_mean
            data[f'{name}_away'] = cum_a - pop_mean
            data[f'_{name}_fit'] = tuple(coefs)  # store for simulation
        elif name in ('red_diff', 'score_diff'):
            idx_pos = COV_SEP_IDX['red_diff'] if name == 'red_diff' else 8
            data[name] = np.array([s[idx_pos] for s, _, _ in valid])
        elif name == 'game_time':  # symmetric: same value for both teams
            gt = np.array([float(s[17]) for s, _, _ in valid])
            data['game_time_home'] = gt
            data['game_time_away'] = gt
        elif name == 'game_time_c':  # mean-centered game_time
            gt = np.array([float(s[17]) for s, _, _ in valid])
            data['game_time_c_mean'] = gt.mean()  # store for simulation centering
            gt_c = gt - gt.mean()
            data['game_time_c_home'] = gt_c
            data['game_time_c_away'] = gt_c.copy()
        else:
            data[f'{name}_home'] = np.array([s[COV_SEP_IDX[f'{name}_home']] for s, _, _ in valid])
            data[f'{name}_away'] = np.array([s[COV_SEP_IDX[f'{name}_away']] for s, _, _ in valid])
    return data


def _nll_with_g(g, betas, data, baseline, tids, cov_names):
    """Core NLL given pre-computed shape parameter vector g."""
    mu, home = baseline['mu'], baseline['home']
    att = np.array([baseline['attack'][t] for t in tids])
    def_ = np.array([baseline['defense'][t] for t in tids])

    hi, ai = data['hi'], data['ai']
    t0, t1 = data['rel_start'], data['rel_end']
    lgf = np.log(gamma_fn(1 + 1 / g))

    cov_h, cov_a = np.zeros(len(t0)), np.zeros(len(t0))
    for i, name in enumerate(cov_names):
        b = betas[i]
        if name in ('red_diff', 'score_diff'):  # symmetric
            cov_h += b * data[name]
            cov_a -= b * data[name]
        else:
            cov_h += b * data[f'{name}_home']
            cov_a += b * data[f'{name}_away']

    eta_h = mu + home + att[hi] + def_[ai] + cov_h
    eta_a = mu + att[ai] + def_[hi] + cov_a
    alpha_h, alpha_a = np.exp(eta_h - lgf), np.exp(eta_a - lgf)

    t0_safe = np.maximum(t0, 1e-9)
    dH_h = (t1 / alpha_h) ** g - (t0_safe / alpha_h) ** g
    dH_a = (t1 / alpha_a) ** g - (t0_safe / alpha_a) ** g
    dH_h = np.where(t0 == 0, (t1 / alpha_h) ** g, dH_h)
    dH_a = np.where(t0 == 0, (t1 / alpha_a) ** g, dH_a)

    interval_idx, n_iv = data['interval_idx'], data['n_intervals']
    cum_H_h, cum_H_a = np.zeros(n_iv), np.zeros(n_iv)
    np.add.at(cum_H_h, interval_idx, dH_h)
    np.add.at(cum_H_a, interval_idx, dH_a)

    result = cum_H_h.sum() + cum_H_a.sum()

    final_uncens = data['is_final'] & ~data['cens']
    hs = final_uncens & data['is_home_scorer']
    aws = final_uncens & ~data['is_home_scorer']

    if hs.any():
        result -= (np.log(g[hs]) - g[hs] * np.log(alpha_h[hs]) + (g[hs] - 1) * np.log(t1[hs])).sum()
    if aws.any():
        result -= (np.log(g[aws]) - g[aws] * np.log(alpha_a[aws]) + (g[aws] - 1) * np.log(t1[aws])).sum()

    return result


def _nll_core(g1h, g2h_ratio, betas, data, baseline, tids, cov_names):
    """NLL with gamma_1H + gamma_2H ratio. Delegates to _nll_with_g."""
    g = np.where(data['is_2h'], g1h * g2h_ratio, g1h)
    return _nll_with_g(g, betas, data, baseline, tids, cov_names)


def nll_joint(params: np.ndarray, data: dict, baseline: dict, tids: list,
              cov_names: list) -> float:
    """Joint NLL: params = [gamma_1H, gamma_2H_ratio, beta_1, ..., beta_k]."""
    return _nll_core(params[0], params[1], params[2:], data, baseline, tids, cov_names)
