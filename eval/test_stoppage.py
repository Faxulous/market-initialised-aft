"""Compare weibull_no_cov with mean vs true match duration. Run: python eval/test_stoppage.py"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.utils import (
    EVAL_START_DATE,
    TRAIN_START_DATE,
    build_goals_lookup,
    build_team_id_map,
    get_elapsed,
    get_score_at,
    load_betfair_data,
)
from eval.test import load_data, fit_gamma_phi_only
from models import TeamParamBaseline, get_strategy
from models.covariates.stoppage import MEAN_H1_STOPPAGE, compute_remaining

N_SIMS = 5000
SEED = 8


@dataclass
class TeamState:
    baseline_log_et: float


def get_true_duration(ws_events: pd.DataFrame, ws_mid: int) -> tuple[float, float]:
    """Get actual match duration (max minute per half) for a match."""
    m = ws_events[ws_events["ws_match_id"] == ws_mid]
    h1 = m[m["period"] == "FirstHalf"]["minute"].max()
    h2 = m[m["period"] == "SecondHalf"]["minute"].max()
    if pd.isna(h1) or pd.isna(h2):
        raise ValueError(f"Match {ws_mid}: no events for a half, true duration unknown")
    return float(h1), float(h2)


def compute_remaining_true(
    minute: int, period: int, h1_end: float, h2_end: float
) -> float:
    """Compute remaining time using true match duration."""
    if period == 1:
        return max(0.1, h1_end + h2_end - 45 - minute)
    return max(0.1, h2_end - minute)


def rps(p_h, p_d, p_a, actual: int) -> float:
    """Ranked probability score. actual: 0=home, 1=draw, 2=away."""
    a = [1 if actual == 0 else 0, 1 if actual == 1 else 0, 1 if actual == 2 else 0]
    cum_p = [p_h, p_h + p_d, p_h + p_d + p_a]
    cum_a = [a[0], a[0] + a[1], a[0] + a[1] + a[2]]
    return 0.5 * sum((cum_p[i] - cum_a[i]) ** 2 for i in range(2))


def log_loss(p_h, p_d, p_a, actual: int) -> float:
    """Log loss. actual: 0=home, 1=draw, 2=away."""
    p = [p_h, p_d, p_a][actual]
    return -np.log(max(p, 1e-10))


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    root = os.path.dirname(os.path.dirname(__file__))
    season = "24-25"

    # Load all seasons (needed for gamma/phi fitting)
    ws, _, _ = load_data(["21-22", "22-23", "23-24", "24-25"])
    align = pd.read_parquet(os.path.join(root, "data", season, "match_alignment.parquet"))

    goals_lookup = build_goals_lookup(
        pd.read_parquet(os.path.join(root, "data", season, "whoscored_events.parquet")), align
    )
    team_id_map = build_team_id_map(
        pd.read_parquet(os.path.join(root, "data", season, "whoscored_events.parquet"))
    )
    _, outcome_lookup, _ = load_betfair_data(season, root)

    params_dir = os.path.join(
        root, "models", "weibull", "team_params", f"pl_{season.replace('-', '_')}"
    )
    team_params = TeamParamBaseline(params_dir)

    # Select common/half-specific shapes on training data
    gamma, phi = fit_gamma_phi_only(ws, TRAIN_START_DATE, EVAL_START_DATE)
    print(f"Fitted gamma={gamma:.4f}, phi={phi:.4f}")
    strategy = get_strategy("weibull_no_cov", gamma=gamma, phi=phi)

    # Filter to eval period
    eval_start = pd.to_datetime(EVAL_START_DATE)
    align["date"] = pd.to_datetime(align["date"])
    eval_matches = align[align["date"] >= eval_start].copy()
    print(f"Eval matches: {len(eval_matches)}")

    # Compute in-sample mean stoppage (before eval period)
    train_matches = align[align["date"] < eval_start]
    h1_train, h2_train = [], []
    for ws_mid in train_matches["ws_game_id"].unique():
        h1_end, h2_end = get_true_duration(ws, ws_mid)
        h1_train.append(h1_end - 45)
        h2_train.append(h2_end - 90)
    print(f"In-sample mean H1 stoppage: {np.mean(h1_train):.2f} min (n={len(h1_train)})")
    print(f"In-sample mean H2 stoppage: {np.mean(h2_train):.2f} min")

    # Precompute true durations for eval matches
    true_durations = {}
    for ws_mid in eval_matches["ws_game_id"].unique():
        true_durations[int(ws_mid)] = get_true_duration(ws, ws_mid)

    results = {"mean": [], "true": []}
    first_half = list(range(0, 45, 5))
    second_half = list(range(45, 91, 5))
    all_minutes = [(m, 1) for m in first_half] + [(m, 2) for m in second_half]
    print(f"Eval minutes: 1H {first_half}, 2H {second_half}")

    for minute, period in tqdm(all_minutes, desc="Minutes"):
        for _, m in eval_matches.iterrows():
            ws_mid = int(m["ws_game_id"])
            if ws_mid not in outcome_lookup:
                raise KeyError(f"Match {ws_mid}: no Betfair outcome")

            home_name, away_name = m["ws_home_team"], m["ws_away_team"]
            home_id, away_id = team_id_map[home_name], team_id_map[away_name]

            goals = goals_lookup.get(ws_mid, pd.DataFrame())  # goalless matches have no entry
            if period == 1 and minute == 0:
                goals = goals.iloc[:0]   # minute 0 is forecast before kick-off, as in eval/test.py
            home_goals, away_goals = get_score_at(goals, minute, period)
            sd = home_goals - away_goals

            date = m["date"]
            home_p = team_params.get_params(home_id, away_id, is_home=True, date=date)
            away_p = team_params.get_params(away_id, home_id, is_home=False, date=date)

            home_state = TeamState(baseline_log_et=home_p["baseline_log_et"])
            away_state = TeamState(baseline_log_et=away_p["baseline_log_et"])
            h1_end, h2_end = true_durations[ws_mid]
            elapsed = get_elapsed(goals, minute, period, h1_end=max(45., h1_end) if period == 2 else 45.)
            actual = outcome_lookup[ws_mid]

            # Mean stoppage
            rem_mean = compute_remaining(minute, period)
            if rem_mean > 0:
                probs = strategy.predict_probs(
                    home_state, away_state, sd, rem_mean, None, N_SIMS,
                    minute=minute, elapsed=elapsed, period=period,
                    h1_left=min(rem_mean, max(.1, 45+MEAN_H1_STOPPAGE-minute)) if period == 1 else 0.,
                )
                results["mean"].append(
                    {
                        "minute": minute,
                        "period": period,
                        "rps": rps(probs["home"], probs["draw"], probs["away"], actual),
                        "logloss": log_loss(
                            probs["home"], probs["draw"], probs["away"], actual
                        ),
                        "correct": int(
                            np.argmax([probs["home"], probs["draw"], probs["away"]])
                            == actual
                        ),
                    }
                )

            # True duration
            h1_end, h2_end = true_durations[ws_mid]
            rem_true = compute_remaining_true(minute, period, h1_end, h2_end)
            if rem_true > 0:
                probs = strategy.predict_probs(
                    home_state, away_state, sd, rem_true, None, N_SIMS,
                    minute=minute, elapsed=elapsed, period=period,
                    h1_left=min(rem_true, max(0., h1_end-minute)) if period == 1 else 0.,
                )
                results["true"].append(
                    {
                        "minute": minute,
                        "period": period,
                        "rps": rps(probs["home"], probs["draw"], probs["away"], actual),
                        "logloss": log_loss(
                            probs["home"], probs["draw"], probs["away"], actual
                        ),
                        "correct": int(
                            np.argmax([probs["home"], probs["draw"], probs["away"]])
                            == actual
                        ),
                    }
                )

    # Summary: mean stoppage against the oracle that knows the true match duration
    out = pd.DataFrame({mode: pd.DataFrame(results[mode])[["rps", "logloss", "correct"]].mean() for mode in ["mean", "true"]}).T
    out.rename(columns={"correct": "accuracy"}).rename_axis("stoppage").to_csv(os.path.join(os.path.dirname(__file__), "acc", "stoppage.csv"))
    print(out.round(4))


if __name__ == "__main__":
    main()
