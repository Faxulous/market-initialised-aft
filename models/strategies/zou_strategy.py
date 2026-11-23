"""Zou et al. (2020) Bayesian birth process strategies — strict paper implementation."""
from typing import Dict
import pandas as pd
from ..weibull import Strategy
from ..zoubirthprocess.model import (
    predict_match, compute_outcome_probs, bayesian_update, shots_before, ZouParamBaseline,
)


def _state(minute, kw, cov_lookup, bp_params, match_id):
    """Harness state on Zou's clock (first-half stoppage is minute 45): T, goals, the shots deviations
    known before minutes 1..T (for the update) and the current deviations (held fixed going forward).
    cov_lookup None is the model without shots."""
    period = kw["period"]
    if period not in (1, 2):
        raise ValueError(f"period must be 1 or 2, got {period!r}")
    T, gh, ga = (min(minute, 45) if period == 1 else minute), int(kw["home_goals"]), int(kw["away_goals"])
    if cov_lookup is None:
        return T, gh, ga, None, None, 0.0, 0.0
    mid = int(match_id)
    psi_h, psi_a = shots_before(cov_lookup, -bp_params["dev_shots_fit"][1], mid, upto=min(max(T, 1), 90))
    psi_h, psi_a = psi_h + psi_h[-1:] * max(T - 90, 0), psi_a + psi_a[-1:] * max(T - 90, 0)   # minutes past 90 keep the last value
    c = cov_lookup[(mid, int(minute), period)]
    return T, gh, ga, psi_h, psi_a, float(c["dev_shots_home"]), float(c["dev_shots_away"])


class ZouBayesianStrategy(Strategy):
    """Strict Zou et al. (2020) with expanding-window params dispatched by date.

    Uses ZouParamBaseline to get (team_alpha, team_beta, bp_params) for each
    match based on the most recent fit before that match's date.
    """
    name = "zou"

    def __init__(self, zou_params: ZouParamBaseline, match_dates: dict, cov_lookup: dict = None):
        """Args:
            zou_params: ZouParamBaseline with pre-computed expanding-window fits
            match_dates: {match_id: date_str} for date dispatch
            cov_lookup: {(match_id, minute, period): covariates}; None for the fit without shots
        """
        super().__init__()
        self.zou_params = zou_params
        self.match_dates = match_dates  # {mid: "2025-02-14"}
        if cov_lookup is not None:   # the harness splits match-keyed attributes per worker; absent = no shots
            self.cov_lookup = cov_lookup

    def predict_probs(self, home, away, sd: int, rem: float, dev,
                      n_sims: int = 1000, match_id=None, minute: int = None,
                      **kw) -> Dict[str, float]:
        mid = int(match_id) if match_id is not None else None
        if mid is None or mid not in self.match_dates:
            return None
        team_alpha, team_beta, bp_params = self.zou_params.get_params_for_date(
            self.match_dates[mid])

        home_id = kw.get("home_id")
        away_id = kw.get("away_id")
        if home_id not in team_alpha or away_id not in team_alpha:
            return None

        minute, goals_h, goals_a, psi_h, psi_a, dev_h, dev_a = _state(minute, kw, getattr(self, "cov_lookup", None), bp_params, match_id)
        to_minute = minute + int(round(rem)) if rem else 90
        if minute < 90:
            to_minute = min(to_minute, 90)  # rho2 at min 90 handles stoppage per paper

        return predict_match(
            home_id, away_id, team_alpha, team_beta, bp_params,
            minute=minute, goals_h=goals_h, goals_a=goals_a, to_minute=to_minute,
            psi_h=psi_h, psi_a=psi_a, shots_dev_h=dev_h, shots_dev_a=dev_a,
        )


class ZouCalibratedStrategy(Strategy):
    """Zou birth process with Betfair-calibrated pre-match theta1, theta2.

    Pre-match: per-match (theta1, theta2) optimised to match Betfair 1X2 + O/U.
    In-play: applies Zou's Bayesian updating on these calibrated priors.
    Isolates in-play updating quality from pre-match model quality.
    """
    name = "zou_calibd"

    def __init__(self, zou_params: ZouParamBaseline, match_dates: dict, calibrated_thetas: dict,
                 cov_lookup: dict = None):
        """Args:
            zou_params: expanding-window fits; structural bp_params come from the snapshot dated before each match
            match_dates: {match_id: date_str} for date dispatch
            calibrated_thetas: {match_id: (theta1, theta2)} calibrated priors
            cov_lookup: {(match_id, minute, period): covariates}; None for the fit without shots
        """
        super().__init__()
        self.zou_params, self.match_dates = zou_params, match_dates
        self.calibrated_thetas = calibrated_thetas
        if cov_lookup is not None:
            self.cov_lookup = cov_lookup

    def predict_probs(self, home, away, sd: int, rem: float, dev,
                      n_sims: int = 1000, match_id=None, minute: int = None,
                      **kw) -> Dict[str, float]:
        if match_id is None or int(match_id) not in self.calibrated_thetas:
            return None
        t1_prior, t2_prior = self.calibrated_thetas[int(match_id)]
        _, _, bp_params = self.zou_params.get_params_for_date(self.match_dates[int(match_id)])
        minute, goals_h, goals_a, psi_h, psi_a, dev_h, dev_a = _state(minute, kw, getattr(self, "cov_lookup", None), bp_params, match_id)

        if minute == 0:
            t1, t2 = t1_prior, t2_prior
        else:
            t1, t2 = bayesian_update(
                t1_prior, t2_prior, goals_h, goals_a, bp_params, minute, psi_h, psi_a
            )

        to_minute = minute + int(round(rem)) if rem else 90
        if minute < 90:
            to_minute = min(to_minute, 90)  # rho2 at min 90 handles stoppage per paper
        ph, pd, pa = compute_outcome_probs(
            t1, t2, bp_params, from_minute=minute,
            start_x=goals_h, start_y=goals_a, to_minute=to_minute,
            shots_dev_h=dev_h, shots_dev_a=dev_a,
        )
        return {"home": ph, "draw": pd, "away": pa}


def load_calibrated_thetas(season='24-25', label='calibd'):
    """Load calibrated thetas from CSV. Returns dict: match_id -> (t1_cal, t2_cal)."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, 'models', 'zoubirthprocess', 'calibrated',
                        f'zou_calibration_{season.replace("-", "_")}_{label}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run calibration first: python -m models.zoubirthprocess.calibrate\n  Missing: {path}")
    df = pd.read_csv(path)
    return {int(r['ws_game_id']): (r['t1_cal'], r['t2_cal']) for _, r in df.iterrows()}
