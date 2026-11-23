"""Maia et al. (2025) Cox process strategies — fitted and calibrated variants."""
from typing import Dict
import pandas as pd
from ..weibull import Strategy
from ..maia.model import MaiaParamBaseline, compute_outcome_probs, N_SIMS


def _state(cov_lookup, match_id, minute, kw):
    """In-play state from the harness and the period-keyed covariate lookup:
    (period, goals_h, goals_a, rc_h, rc_a, dev_shots_h, dev_shots_a). Shots are zero for the no-covariate fit."""
    period = kw["period"]
    c = cov_lookup[(int(match_id), int(minute), period)]   # missing state is an error, never a default
    dev = (float(c["dev_shots_home"]), float(c["dev_shots_away"])) if "dev_shots_home" in c else (0.0, 0.0)
    return period, int(kw["home_goals"]), int(kw["away_goals"]), int(c["red_home"]), int(c["red_away"]), *dev


class MaiaStrategy(Strategy):
    """Maia Cox process with expanding-window fitted team params.

    Pre-match: theta1 = alpha_h * beta_a, theta2 = alpha_a * beta_h
    In-play: condition on observed score + red cards + shots, simulate forward.
    """
    name = "maia"

    def __init__(self, maia_params: MaiaParamBaseline, match_dates: dict, cov_lookup: dict):
        super().__init__()
        self.maia_params = maia_params
        self.match_dates = match_dates
        self.cov_lookup = cov_lookup   # {(match_id, minute, period): covariates}

    def predict_probs(self, home, away, sd: int, rem: float, dev,
                      n_sims: int = N_SIMS, match_id=None, minute: int = None,
                      **kw) -> Dict[str, float]:
        mid = int(match_id) if match_id is not None else None
        if mid is None or mid not in self.match_dates:
            return None
        team_alpha, team_beta, params = self.maia_params.get_params_for_date(
            self.match_dates[mid])

        home_id, away_id = kw.get("home_id"), kw.get("away_id")
        if home_id not in team_alpha or away_id not in team_alpha:
            return None

        period, goals_h, goals_a, rc_h, rc_a, dev_h, dev_a = _state(self.cov_lookup, match_id, minute, kw)

        theta1 = team_alpha[home_id] * team_beta[away_id]
        theta2 = team_alpha[away_id] * team_beta[home_id]

        ph, pd_, pa = compute_outcome_probs(
            theta1, theta2, params, from_minute=minute,
            start_x=goals_h, start_y=goals_a, rc_h=rc_h, rc_a=rc_a,
            shots_dev_h=dev_h, shots_dev_a=dev_a, n_sims=n_sims, period=period)
        return {"home": ph, "draw": pd_, "away": pa}


class MaiaCalibratedStrategy(Strategy):
    """Maia Cox process with Betfair-calibrated pre-match theta1, theta2.

    Pre-match: per-match (theta1, theta2) optimised to match Betfair 1X2 + O/U.
    In-play: condition on observed score + red cards + shots, simulate forward.
    Isolates in-play model quality from pre-match estimation quality.
    """
    name = "maia_calibd"

    def __init__(self, maia_params: MaiaParamBaseline, match_dates: dict, calibrated_thetas: dict,
                 cov_lookup: dict):
        """Args:
            maia_params: expanding-window fits; gamma/xi coefficients come from the snapshot dated before each match
            match_dates: {match_id: date_str} for date dispatch
            calibrated_thetas: {match_id: (theta1, theta2)}
            cov_lookup: {(match_id, minute, period): covariates incl. red cards and shots deviations}
        """
        super().__init__()
        self.maia_params, self.match_dates = maia_params, match_dates
        self.calibrated_thetas = calibrated_thetas
        self.cov_lookup = cov_lookup

    def predict_probs(self, home, away, sd: int, rem: float, dev,
                      n_sims: int = N_SIMS, match_id=None, minute: int = None,
                      **kw) -> Dict[str, float]:
        if match_id is None or int(match_id) not in self.calibrated_thetas:
            return None
        t1, t2 = self.calibrated_thetas[int(match_id)]
        _, _, params = self.maia_params.get_params_for_date(self.match_dates[int(match_id)])
        period, goals_h, goals_a, rc_h, rc_a, dev_h, dev_a = _state(self.cov_lookup, match_id, minute, kw)

        ph, pd_, pa = compute_outcome_probs(
            t1, t2, params, from_minute=minute,
            start_x=goals_h, start_y=goals_a, rc_h=rc_h, rc_a=rc_a,
            shots_dev_h=dev_h, shots_dev_a=dev_a, n_sims=n_sims, period=period)
        return {"home": ph, "draw": pd_, "away": pa}


def load_calibrated_thetas(season='24-25', label='calibd'):
    """Load calibrated thetas from CSV. Returns dict: match_id -> (t1_cal, t2_cal)."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, 'models', 'maia', 'calibrated',
                        f'maia_calibration_{season.replace("-", "_")}_{label}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run calibration first: python -m models.maia.calibrate\n  Missing: {path}")
    df = pd.read_csv(path)
    return {int(r['ws_game_id']): (r['t1_cal'], r['t2_cal']) for _, r in df.iterrows()}
