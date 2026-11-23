"""Extended Weibull strategies with conditional sampling and covariates."""

from typing import Dict

import pandas as pd

from ..weibull import Strategy
from ..weibull.core import simulate_covariates_single_gamma_phi, simulate_single_gamma_phi


class SingleGammaPhiStrategy(Strategy):
    """Single gamma (no score-state) + phi multiplier for 2H."""

    name = "weibull_no_cov"

    def __init__(self, gamma: float = 1.0, phi: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.phi = phi

    def predict_probs(
        self,
        home,
        away,
        sd: int,
        rem: float,
        dev,
        n_sims: int = 1000,
        match_id=None,
        minute: int = None,
        **kw,
    ) -> Dict[str, float]:
        eta_h = (
            home.baseline_log_et.item()
            if hasattr(home.baseline_log_et, "item")
            else home.baseline_log_et
        )
        eta_a = (
            away.baseline_log_et.item()
            if hasattr(away.baseline_log_et, "item")
            else away.baseline_log_et
        )
        minute = minute if minute is not None else 0
        elapsed = kw["elapsed"]
        p_h, p_d, p_a = simulate_single_gamma_phi(
            eta_h, eta_a, sd, minute, rem, self.gamma, self.phi, elapsed, n=n_sims, period=kw["period"], h1_left=kw["h1_left"]
        )
        return {"home": p_h, "draw": p_d, "away": p_a}


class ConditionalCovariateStrategy(Strategy):
    """Weibull with conditional sampling + time-varying covariates + single gamma/phi.

    Key features:
    - Rolling window covariates (ft_touch, goals, red_diff by default)
    - Single gamma with phi multiplier in 2H (no score-state gamma)
    - Conditional Weibull sampling: P(T > t+s | T > s) for elapsed time s
    """

    name = "weibull"

    def __init__(
        self,
        betas: dict = None,
        cov_data: dict = None,
        gamma: float = 0.93,
        phi: float = 1.51,
        fit_meta: dict = None,
    ):
        super().__init__()
        self.betas = betas or {}
        self.cov_data = cov_data or {}
        self.gamma = gamma
        self.phi = phi
        self.fit_meta = fit_meta or {}  # e.g. {"game_time_center": 45.2}

    def _get_covariates(self, match_id, minute: int, period: int) -> dict:
        return self.cov_data[(int(match_id), int(minute), int(period))].copy()

    def predict_probs(
        self,
        home,
        away,
        sd: int,
        rem: float,
        dev,
        n_sims: int = 1000,
        match_id=None,
        minute: int = None,
        **kw,
    ) -> Dict[str, float]:
        eta_h = (
            home.baseline_log_et.item()
            if hasattr(home.baseline_log_et, "item")
            else home.baseline_log_et
        )
        eta_a = (
            away.baseline_log_et.item()
            if hasattr(away.baseline_log_et, "item")
            else away.baseline_log_et
        )
        minute = minute if minute is not None else 0
        elapsed = kw["elapsed"]
        covs = self._get_covariates(match_id, minute, kw["period"])
        covs["score_diff"] = sd
        covs["cum_goals_home"] = float(kw.get("home_goals", 0))
        covs["cum_goals_away"] = float(kw.get("away_goals", 0))
        self._inject_derived_covs(covs, minute)
        p_h, p_d, p_a = simulate_covariates_single_gamma_phi(
            eta_h,
            eta_a,
            sd,
            minute,
            rem,
            self.betas,
            covs,
            self.gamma,
            self.phi,
            elapsed,
            n=n_sims, period=kw["period"], h1_left=kw["h1_left"],
        )
        return {"home": p_h, "draw": p_d, "away": p_a}

    def _inject_derived_covs(self, covs: dict, minute: int):
        """Inject derived covariates (game_time_c, dev_goals) into covs dict."""
        if "game_time_center" in self.fit_meta:
            center = self.fit_meta["game_time_center"]
            covs["_game_time_center"] = center
            covs["game_time_c_home"] = float(minute) - center
            covs["game_time_c_away"] = float(minute) - center
        if "dev_goals_fit" in self.fit_meta:
            s, i = self.fit_meta["dev_goals_fit"]
            pop = s * minute + i
            covs["dev_goals_home"] = covs.get("cum_goals_home", 0) - pop
            covs["dev_goals_away"] = covs.get("cum_goals_away", 0) - pop


class CalibratedCovariateStrategy(ConditionalCovariateStrategy):
    """Weibull with market-calibrated etas instead of model etas.

    At minute 0, uses Betfair 1X2 + O/U odds to find calibrated (eta_h, eta_a).
    Then applies same covariates as the uncalibrated Weibull.
    """

    name = "weibull_calibd"

    def __init__(
        self, calibrated_etas: dict = None, team_id_map: dict = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.calibrated_etas = calibrated_etas or {}
        self.team_id_map = team_id_map or {}  # team_name -> team_id

    def predict_probs(
        self,
        home,
        away,
        sd: int,
        rem: float,
        dev,
        n_sims: int = 1000,
        match_id=None,
        minute: int = None,
        **kw,
    ) -> Dict[str, float]:
        cal = self.calibrated_etas[(kw['home_id'], kw['away_id'])]
        eta_h, eta_a = cal['eta_h_cal'], cal['eta_a_cal']

        minute = minute if minute is not None else 0
        elapsed = kw["elapsed"]
        covs = self._get_covariates(match_id, minute, kw["period"])
        covs["score_diff"] = sd
        covs["cum_goals_home"] = float(kw.get("home_goals", 0))
        covs["cum_goals_away"] = float(kw.get("away_goals", 0))
        self._inject_derived_covs(covs, minute)
        p_h, p_d, p_a = simulate_covariates_single_gamma_phi(
            eta_h,
            eta_a,
            sd,
            minute,
            rem,
            self.betas,
            covs,
            self.gamma,
            self.phi,
            elapsed,
            n=n_sims, period=kw["period"], h1_left=kw["h1_left"],
        )
        return {"home": p_h, "draw": p_d, "away": p_a}


def compute_dev_fit(ws_events, fb_shots, align, field):
    """Fit population-mean linear curve to a cumulative covariate vs minute (pooled home+away).

    field: cumulative column from get_covariates_at ('shots' for the shots deviation).
    Returns (slope, intercept): deviation = cum value - (slope*t + intercept).
    """
    import numpy as np
    from ..weibull.team_params.covariates import compute_match_covariates, get_covariates_at
    t_all, cum_all = [], []
    for ws_mid in ws_events["ws_match_id"].unique():
        covs = compute_match_covariates(ws_mid, ws_events, fb_shots, align)
        for period, rng in [(1, range(58)), (2, range(45, 96))]:
            for t in rng:
                r = get_covariates_at(covs, t, period=period)
                t_all.append(float(t)); cum_all.append(float(r[f"{field}_home"]))
                t_all.append(float(t)); cum_all.append(float(r[f"{field}_away"]))
    return tuple(np.polyfit(np.array(t_all), np.array(cum_all), 1))


def build_cov_lookup(
    ws_events: pd.DataFrame, fb_shots: pd.DataFrame, align: pd.DataFrame,
    fit_meta: dict = None, *, period_keys: bool, prematch: bool = False,
) -> dict:
    """Build covariates; period_keys adds period to keys for Weibull forecasts.

    False explicitly retains the legacy minute-only layout for other models.
    prematch empties the first-half minute-0 entry: evaluation forecasts at minute 0 are made
    before kick-off, so no event of the opening minute is seen (model fits leave it False).
    """
    from ..weibull.team_params.covariates import (
        compute_match_covariates,
        get_covariates_at,
    )

    PAIR_FIELDS = ["ft_touch", "ft_pass", "psxg", "ng_psxg", "goals", "xg", "ng_xg",
                   "cum_psxg", "cum_ng_psxg", "shots"]
    fit_meta = fit_meta or {}

    lookup = {}
    for ws_mid in ws_events["ws_match_id"].unique():
        covs = compute_match_covariates(ws_mid, ws_events, fb_shots, align)

        for period, rng in [(1, range(len(covs["h1"]))),
                            (2, range(45, len(covs["h2"])))]:
            for t in rng:
                r = get_covariates_at(covs, t, period=period)
                if prematch and period == 1 and t == 0:
                    r = r * 0   # pre-kick-off state
                entry = {"red_diff": int(r["red_away"]) - int(r["red_home"]),
                         "red_home": int(r["red_home"]), "red_away": int(r["red_away"]),
                         "game_time_home": float(t), "game_time_away": float(t)}
                for f in PAIR_FIELDS:
                    entry[f"{f}_home"] = float(r[f"{f}_home"])
                    entry[f"{f}_away"] = float(r[f"{f}_away"])
                entry["cum_psxg_diff"] = float(r["cum_psxg_home"]) - float(r["cum_psxg_away"])
                entry["cum_ng_psxg_diff"] = float(r["cum_ng_psxg_home"]) - float(r["cum_ng_psxg_away"])
                # Deviation covariates: cum value minus population mean at this minute
                if "dev_psxg_fit" in fit_meta:
                    s, i = fit_meta["dev_psxg_fit"]
                    pop = s * t + i
                    entry["dev_psxg_home"] = entry["cum_psxg_home"] - pop
                    entry["dev_psxg_away"] = entry["cum_psxg_away"] - pop
                if "dev_ng_psxg_fit" in fit_meta:
                    s, i = fit_meta["dev_ng_psxg_fit"]
                    pop = s * t + i
                    entry["dev_ng_psxg_home"] = entry["cum_ng_psxg_home"] - pop
                    entry["dev_ng_psxg_away"] = entry["cum_ng_psxg_away"] - pop
                if "dev_xg_fit" in fit_meta:
                    s, i = fit_meta["dev_xg_fit"]
                    pop = s * t + i
                    entry["dev_xg_home"] = entry["xg_home"] - pop  # xg fields match prep_data idx 28/29
                    entry["dev_xg_away"] = entry["xg_away"] - pop
                if "dev_shots_fit" in fit_meta:
                    s, i = fit_meta["dev_shots_fit"]
                    pop = s * t + i
                    entry["dev_shots_home"] = entry["shots_home"] - pop
                    entry["dev_shots_away"] = entry["shots_away"] - pop
                key = (int(ws_mid), t, period) if period_keys else (int(ws_mid), t)
                lookup[key] = entry

    return lookup
