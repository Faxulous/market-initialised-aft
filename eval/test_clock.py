"""Small regression checks for the continuous Weibull clock and calibration."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.utils import get_elapsed
from models.covariates.grid_search import prep_data, nll_joint
from models.strategies.extended import build_cov_lookup, CalibratedCovariateStrategy
from models.weibull import core
from models.weibull.calibrate import kickoff_covs, simulate_match
from models.weibull.clock import match_clock
from models.weibull.team_params.covariates import extract_subintervals_with_covariates
from models.weibull.team_params.estimate import exposure_intervals, _prep_data_with_period, weibull_nll_single_gamma_phi, common_nll, _prep_data, fit_team_intervals

SEED = 8


def fixture():
    records = [(0, 'FirstHalf', 1, False), (20, 'FirstHalf', 1, True),
               (48, 'FirstHalf', 2, False), (45, 'SecondHalf', 1, False),
               (60, 'SecondHalf', 2, True), (100, 'SecondHalf', 2, False)]
    ws = pd.DataFrame(records, columns=['minute', 'period', 'team_id', 'is_goal'])
    ws = ws.assign(ws_match_id=1, second=0, is_own_goal=False, start_x=0., card_type='')
    ws['team_name'] = ws.team_id.map({1: 'Home', 2: 'Away'})
    ws['is_home'] = ws.team_id == 1
    ws['event_type'] = np.where(ws.is_goal, 'Goal', 'Pass')
    shots = pd.DataFrame([dict(match_id='a', minute=47, period=1, team='Home',
                               outcome='SavedShot', xG=.7, PSxG=.4)])
    align = pd.DataFrame([dict(ws_game_id=1, fbref_match_id='a', ws_home_team='Home',
                               ws_away_team='Away', fbref_home_team='Home')])
    return ws, shots, align


class ClockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)

    def test_joint_shape_gradient_and_censoring(self):
        from scipy.optimize._numdiff import approx_derivative
        iv = [('a', 12., 1, 2, 1, False, 0), ('b', 24., 2, 3, 3, False, -1),
              ('c', 40., 3, 1, None, True, 1)]
        data = _prep_data(iv, [1, 2, 3])
        data['weights'] = np.array([1., .7, .3])
        x = np.array([4., -.15, -.2, .1, .1, -.1, 1.3])
        value, grad = common_nll(x, data, 3)
        numeric = approx_derivative(lambda p: common_nll(p, data, 3)[0], x).ravel()
        np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-7)
        # Independent density x opponent-survival calculation, with a censored tail.
        g, att, dff = x[-1], [-.2, .1, .1], [.1, -.1, 0.]
        expected = 0.
        for row, weight in zip(iv, data['weights']):
            _, t, h, a, scorer, cens, _ = row
            lh = math.exp(g*(math.lgamma(1+1/g)-x[0]-x[1]-att[h-1]-dff[a-1]))
            la = math.exp(g*(math.lgamma(1+1/g)-x[0]-att[a-1]-dff[h-1]))
            loss = (lh+la)*t**g
            if not cens:
                loss -= math.log(g*(lh if scorer == h else la)*t**(g-1))
            expected += weight*loss
        self.assertAlmostEqual(value, expected, places=10)

    def test_joint_fit_recovers_common_shape_without_score_labels(self):
        rng, iv = np.random.default_rng(SEED), []
        g, scale = 1.3, 60/math.gamma(1+1/1.3)
        for mid in range(1000):
            h, a = rng.choice([1, 2, 3, 4], 2, replace=False)
            left, score = 95., 0
            while left > 0:
                waits = scale*rng.weibull(g, 2)
                t = min(left, float(waits.min()))
                cens = t == left
                scorer = None if cens else (h if waits[0] < waits[1] else a)
                iv.append((str(mid), t, int(h), int(a), scorer, cens, score))
                left -= t
                score += 0 if cens else (1 if scorer == h else -1)
        fitted = fit_team_intervals(iv)
        self.assertLess(abs(fitted['gamma']-g), .1)
        self.assertLess(abs(fitted['mu']-math.log(60)), .1)
        relabelled = fit_team_intervals([r[:-1]+(100,) for r in iv])
        self.assertEqual(fitted, relabelled)

    def test_monotone_clock_and_goalless_elapsed(self):
        ws, _, _ = fixture()
        m, ht, end = match_clock(ws)
        self.assertEqual((ht, end), (48., 103.))
        self.assertEqual(m.time.tolist(), [0., 20., 48., 48., 63., 103.])
        empty = pd.DataFrame()
        self.assertEqual(get_elapsed(empty, 45, 2, h1_end=48), 48.)
        goals = pd.DataFrame({'minute': [47, 46], 'period': [1, 2]})
        self.assertEqual(get_elapsed(goals, 45, 2, h1_end=48), 1.)
        self.assertEqual(get_elapsed(goals, 47, 2, h1_end=48), 1.)
        self.assertEqual(get_elapsed(goals, 46, 1, h1_end=45), 46.)

    def test_exposure_and_joint_likelihood_agree(self):
        ws, shots, al = fixture()
        iv = exposure_intervals(ws)
        self.assertTrue(any(r[7] == 2 and r[8] == 28 for r in iv))
        self.assertAlmostEqual(sum(r[1]-r[8] for r in iv), 103.)
        base = dict(mu=math.log(60), home=0., attack={1: 0., 2: 0.}, defense={1: 0., 2: 0.})
        expected = weibull_nll_single_gamma_phi([1.26, .934], _prep_data_with_period(iv, [1, 2], exposure=True), base, [1, 2])
        for delta in [1, 5, 100]:
            sub = extract_subintervals_with_covariates(ws, shots, al, delta=delta)
            data = prep_data(sub, [1, 2], [])
            self.assertAlmostEqual(nll_joint(np.array([1.26, .934]), data, base, [1, 2], []), expected, places=9)
            self.assertAlmostEqual(sum(r[3]-r[2] for r in sub), 103.)
        reset = exposure_intervals(ws, reset_at_ht=True)
        self.assertNotEqual(iv, reset)

    def test_period_covariates_and_late_stoppage(self):
        ws, shots, al = fixture()
        cov = build_cov_lookup(ws, shots, al, period_keys=True)
        self.assertEqual(cov[(1, 46, 1)]['xg_home'], 0.)
        self.assertEqual(cov[(1, 46, 2)]['xg_home'], .7)
        self.assertIn((1, 100, 2), cov)
        strategy = CalibratedCovariateStrategy(cov_data=cov)
        with self.assertRaises(KeyError):
            strategy._get_covariates(1, 101, 2)
        with self.assertRaises(KeyError):
            core._compute_cov_effects({'red_diff': .1}, {})

    def test_residual_wait_avoids_cancellation(self):
        u = torch.tensor([1.-2**-24, .5])
        age = torch.tensor([100., 0.])
        wait = core.sample_conditional_weibull(u, torch.tensor(.02), 1.26, age)
        x = -math.log(float(u[0]))/.02
        expected = 100*math.expm1(math.log1p(x/100**1.26)/1.26)
        self.assertGreater(float(wait[0]), 0.)
        self.assertAlmostEqual(float(wait[0])/expected, 1., places=6)
        self.assertAlmostEqual(float(wait[1]), (-math.log(.5)/.02)**(1/1.26), places=4)

    def test_simulation_retains_age_across_boundary(self):
        ages = []
        def no_goal(u, lam, g, s):
            ages.append((float(g), float(s[0])))
            return torch.full_like(u, 1000.)
        with patch.object(core, 'sample_conditional_weibull', side_effect=no_goal):
            core.simulate_goals(4., 4., 44, 55.31, {}, {}, 1.26, .934,
                                elapsed=24, n=1, period=1, h1_left=4.1)
        self.assertAlmostEqual(ages[2][1], 28.1, places=5)
        self.assertAlmostEqual(ages[2][0], 1.26*.934)

    def test_kickoff_calibration_matches_prediction(self):
        ws, shots, al = fixture()
        fit = dict(betas={'red_diff': -.2, 'dev_xg': -.1}, meta={'dev_xg_fit': [.02, -.05]})
        cov = build_cov_lookup(ws, shots, al, fit_meta=fit['meta'], period_keys=True)
        expected = simulate_match(4., 4.2, 1.26, .934, 5000, betas=fit['betas'], covs=kickoff_covs(fit))
        strategy = CalibratedCovariateStrategy(gamma=1.26, phi=.934, betas=fit['betas'],
                     cov_data=cov, fit_meta=fit['meta'], calibrated_etas={(1, 2): {'eta_h_cal': 4., 'eta_a_cal': 4.2}})
        torch.manual_seed(SEED)
        from models.covariates.stoppage import MEAN_H1_STOPPAGE, MEAN_H2_STOPPAGE
        actual = strategy.predict_probs(None, None, 0, 90+MEAN_H1_STOPPAGE+MEAN_H2_STOPPAGE, None, 5000, match_id=1,
                    minute=0, period=1, h1_left=45+MEAN_H1_STOPPAGE, elapsed=0, home_id=1, away_id=2)
        for key in ['home', 'draw', 'away']:
            self.assertAlmostEqual(actual[key], expected[f'p_{key}'], places=6)
        state = torch.random.get_rng_state().clone()
        simulate_match(4., 4.2, 1.26, .934, 100, betas=fit['betas'], covs=kickoff_covs(fit))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_simulated_survival_matches_piecewise_formula(self):
        gamma, phi, eta, n = 1.26, .934, math.log(60), 50000
        gh, ga = core.simulate_goals(eta, eta, 44, 19.1, {}, {}, gamma, phi,
                                     elapsed=24, n=n, period=1, h1_left=4.1)
        def hazard(g, a, b):
            return math.exp(-g*(eta-math.lgamma(1+1/g))) * (b**g-a**g)
        expected = math.exp(-2*(hazard(gamma, 24, 28.1) + hazard(gamma*phi, 28.1, 43.1)))
        self.assertLess(abs(float(((gh+ga) == 0).float().mean())-expected), .009)

    def test_parallel_calibration_matches_serial(self):
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        from models.weibull.calibrate import calibrate_one
        fit = dict(gamma=1.26, phi=.934, betas={}, meta={})
        target = simulate_match(4., 4.2, 1.26, .934, 1000, betas={}, covs={})
        m = dict(date='2024-10-01', home_name='H', away_name='A',
                 **{k: target[k] for k in ['p_home', 'p_draw', 'p_away']},
                 ou_targets={t: target[f'p_over_{t}'] for t in [.5, 1.5, 2.5, 3.5, 4.5]})
        job = (1, 2, m, 4.05, 4.15, fit, 1000)
        serial = calibrate_one(job)
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
            parallel = list(pool.map(calibrate_one, [job, job]))
        self.assertEqual(parallel, [serial, serial])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ClockTests))
    Console().print(f'{result.testsRun} continuous-clock checks; success={result.wasSuccessful()}')
    raise SystemExit(not result.wasSuccessful())
