"""Shared utilities for evaluation pipeline (event-based, no graphs)."""
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from models.market import load_prematch_1x2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# (first team-parameter snapshot, first evaluation date); evaluation runs to the end of the season
SEASON_CFG = {"24-25": ("2024-09-21", "2024-11-09")}
TRAIN_START_DATE, EVAL_START_DATE = SEASON_CFG["24-25"]
ALL_SEASONS = ["21-22", "22-23", "23-24", "24-25"]
BENCHMARKS = ["mkt_poisson", "mkt_poisson_t", "mkt_naive"]
# Market-only baselines: kappa = kick-off Betfair prices, t = time-varying (empirical) scoring rate
BENCH_NAMES = {"mkt_poisson": r"Poisson$_\kappa$", "mkt_poisson_t": r"Poisson$_{\kappa,t}$", "mkt_naive": r"Naive$_\kappa$"}  # mathtext/LaTeX
PROB_ATOL = 1.5e-4  # Three probabilities stored to 4 dp can accumulate rounding error.


SPIKE_JUMP = 0.05  # a print this far from BOTH neighbouring prints (same side) is a transient spike


def spike_mask(p: pd.DataFrame, match: pd.Series, order: pd.Series) -> pd.Series:
    """True for prints whose implied probability for some outcome sits >= SPIKE_JUMP away from both the
    previous and the next print of the same match, on the same side: a one-minute excursion of the
    last-traded price that reverts. Defined on the price path alone. p: implied probabilities (one
    column per outcome) for in-play prints; match, order: match id and wall-clock minute, same index."""
    idx = pd.MultiIndex.from_arrays([match, order]).argsort()
    q, g = p.iloc[idx], match.iloc[idx]
    up, dn = q - q.groupby(g.values).shift(1), q - q.groupby(g.values).shift(-1)
    hit = ((up.abs() >= SPIKE_JUMP) & (dn.abs() >= SPIKE_JUMP) & (np.sign(up) == np.sign(dn))).any(axis=1)
    return hit.reindex(p.index)


def clock(period, minute):
    """Chronological minute key: first-half stoppage minute m maps to 44 + (m - 44)/100, so it sorts
    after minute 44 and before the second half's minute 45 (the two halves share minute labels)."""
    period, minute = np.asarray(period), np.asarray(minute, dtype=float)
    return np.where((period == 1) & (minute >= 45), 44 + (minute - 44) / 100, minute)


def load_predictions(path, benchmarks=False):
    """Load predictions; optionally append saved baselines on an identical grid."""
    df = pd.read_parquet(path)
    if set(df["season"]) != set(SEASON_CFG):
        raise ValueError(f"{path} holds seasons {sorted(set(df['season']))}, expected {sorted(SEASON_CFG)}; run eval/test.py --merge")
    if benchmarks:
        src = os.path.join(os.path.dirname(__file__), 'analysis', 'out',
                           'benchmark_preds.parquet')
        bench = pd.read_parquet(src)
        if set(bench.columns) != set(df.columns):
            raise ValueError('Benchmark and main prediction schemas differ')
        if set(bench['model']) != set(BENCHMARKS):
            raise ValueError('Expected all three market benchmarks')
        if set(df['model']) & set(BENCHMARKS):
            raise ValueError('Main predictions already contain benchmarks')
        df = pd.concat([df, bench], ignore_index=True)
        keys = ['season', 'match_id', 'eval_minute', 'period']
        cols = [c for c in df if c not in keys +
                ['model', 'pred_home', 'pred_draw', 'pred_away']]
        ref = df[df['model'] == 'betfair'].set_index(keys)[cols].sort_index()
        if ref.empty or not ref.index.is_unique:
            raise ValueError('Expected a nonempty, unique Betfair grid')
        for model, sub in df.groupby('model'):
            grid = sub.set_index(keys)[cols].sort_index()
            if not grid.equals(ref):
                raise ValueError(f'{model}: grid or match metadata differs from Betfair')
            probs = sub[['pred_home', 'pred_draw', 'pred_away']].to_numpy()
            if (not np.isfinite(probs).all() or (probs < 0).any() or
                    (probs > 1).any() or
                    not np.allclose(probs.sum(1), 1, atol=PROB_ATOL, rtol=0)):
                raise ValueError(f'{model}: invalid probabilities')
    return df


def seasons_through(eval_season: str) -> list:
    """All seasons up to and including eval_season (training pool + eval season)."""
    return ALL_SEASONS[: ALL_SEASONS.index(eval_season) + 1]


def snapshot_cuts(eval_dates: pd.Series, eval_season: str) -> list:
    """Cut dates for expanding-window team-parameter snapshots: every distinct match date of the eval
    season from its train_start on. A snapshot at cut d trains on all matches dated <= d and serves
    matches dated > d, so a postponed fixture never pulls later results into an earlier snapshot."""
    d = pd.to_datetime(eval_dates)
    return sorted(d[d >= pd.to_datetime(SEASON_CFG[eval_season][0])].unique())


def get_score_at(goals: pd.DataFrame, minute: int, period: int) -> Tuple[int, int]:
    """Get (home_goals, away_goals) at given minute/period (inclusive).

    Uses <= minute: model sees all goals scored at or before eval minute M.
    BetfairStrategy uses a +2 minute shift (BF_SHIFT) so the Betfair odds
    at M+2 have definitely absorbed any event at M, preventing timing artifacts.
    For 2H minute 45, includes all 1H goals (including stoppage time).
    """
    if len(goals) == 0:
        return 0, 0

    if period == 1:
        prior = goals[(goals['period'] == 1) & (goals['minute'] <= minute)]
    else:  # period == 2
        # All 1H goals + 2H goals up to and including minute
        h1 = goals[goals['period'] == 1]
        h2 = goals[(goals['period'] == 2) & (goals['minute'] <= minute)]
        prior = pd.concat([h1, h2])
    
    h = (prior['is_home'] == True).sum()
    a = (prior['is_home'] == False).sum()
    return int(h), int(a)


def get_elapsed(goals, minute, period, *, h1_end):
    """Playing minutes since the last goal; the half-time break adds no exposure."""
    from models.weibull.clock import playing_time
    now = float(playing_time(minute, period, h1_end))
    if goals.empty:
        return now
    prior = goals[((goals.period == period) & (goals.minute <= minute)) |
                  ((period == 2) & (goals.period == 1))]
    if prior.empty:
        return now
    last = playing_time(prior.minute.to_numpy(), prior.period.to_numpy(), h1_end).max()
    if last > now:
        raise ValueError("A future goal entered the elapsed-time calculation")
    return float(now-last)


def build_goals_lookup(ws_events: pd.DataFrame, align: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Build lookup: ws_match_id -> DataFrame of goals with minute, period, is_home.
    
    Uses WhoScored events (not FBRef shots) to include own goals.
    """
    goals = ws_events[ws_events['event_type'] == 'Goal'].copy()
    
    # Map ws_match_id -> home team
    home_map = align.set_index('ws_game_id')['ws_home_team'].to_dict()
    
    lookup = {}
    for ws_mid, grp in goals.groupby('ws_match_id'):
        if ws_mid not in home_map:
            continue
        home_team = home_map[ws_mid]
        
        df = grp[['minute', 'period', 'team_name', 'is_own_goal']].copy()
        # Convert period: FirstHalf -> 1, SecondHalf -> 2
        df['period'] = df['period'].map({'FirstHalf': 1, 'SecondHalf': 2})
        # Own goal: team credited is OPPOSITE of team_name
        df['is_home'] = df.apply(lambda r: (r['team_name'] == home_team) != r['is_own_goal'], axis=1)
        df = df[['minute', 'period', 'is_home']].sort_values('minute')
        lookup[int(ws_mid)] = df
    
    return lookup


def build_team_id_map(ws_events: pd.DataFrame) -> Dict[str, int]:
    """Build team_name -> team_id mapping from WhoScored events."""
    return dict(ws_events.groupby('team_name')['team_id'].first())


def load_betfair_data(season: str, root: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Load Betfair odds with implied probs and outcomes.
    
    Returns: (odds_lookup, outcome_lookup, match_info) or (None, None, None) if not found.
    """
    odds_path = os.path.join(root, "data", season, "betfair_odds.parquet")
    align_path = os.path.join(root, "data", season, "match_alignment.parquet")
    sched_path = os.path.join(root, "data", season, "fbref_schedule.parquet")

    if not os.path.exists(odds_path) or not os.path.exists(align_path):
        return None, None, None

    odds = pd.read_parquet(odds_path)
    align = pd.read_parquet(align_path)
    
    # exact fixture-id join: odds files are schedule-aligned upstream (cup ties dropped, rescheduled
    # kick-offs renamed), so any unmatched id is a data error, not a date mismatch
    stray = set(odds["fbref_match_id"]) - set(align["fbref_match_id"])
    if stray:
        raise ValueError(f"{season}: {len(stray)} odds fixtures not in match_alignment: {sorted(stray)[:5]}")
    odds = odds.merge(align[["fbref_match_id", "ws_game_id"]], on="fbref_match_id", how="inner")
    odds = odds[odds[["odds_home", "odds_draw", "odds_away"]].notna().all(axis=1)]

    # Implied probabilities: normalized 1/odds (exchange data has ~0% margin, Shin unnecessary)
    inv = odds[["odds_home", "odds_draw", "odds_away"]].rdiv(1)
    normed = inv.div(inv.sum(axis=1), axis=0)
    odds["prob_home"], odds["prob_draw"], odds["prob_away"] = normed["odds_home"], normed["odds_draw"], normed["odds_away"]

    # Transient last-traded spikes are implausible, non-executable prints: drop them, so the lookup
    # carries the preceding price forward (eval/analysis/sanity.py check 8e reports the count)
    ip = odds["period"] != "PreMatch"
    odds = odds.drop(odds.index[ip][spike_mask(odds.loc[ip, ["prob_home", "prob_draw", "prob_away"]],
                                               odds.loc[ip, "fbref_match_id"], odds.loc[ip, "minute"]).values])

    odds = odds.sort_values("timestamp", na_position="first")
    # Half-time rows carry match_minute -1: continue the first-half clock through the break
    # (wall-clock minute minus the match's last first-half offset), so a +2 shift from a
    # first-half stoppage minute lands on break prices, never on second-half prices.
    fh_last = odds[odds["period"] == "FirstHalf"].sort_values("minute").groupby("fbref_match_id").last()
    ht = odds["period"] == "halftime"
    off = odds.loc[ht, "fbref_match_id"].map(fh_last["minute"] - fh_last["match_minute"])
    if off.isna().any():
        raise ValueError(f"{season}: half-time prices without first-half prices in {off.isna().sum()} rows")
    odds.loc[ht, "match_minute"] = odds.loc[ht, "minute"] - off
    # Filter: keep in-play (match_minute >= 0) and pre-match (match_minute < -1)
    odds = odds[(odds["match_minute"] >= 0) | (odds["match_minute"] < -1)]
    odds = odds[odds["match_minute"].notna()]

    # Fix broken matches where match_minute == minute (wall-clock, no HT gap).
    # Detect: 2H minute column starts at 45 with no gap from 1H end.
    # Fix: use timestamps to find real 2H kick-off, re-derive match_minute as game clock.
    for ws_id, grp in odds.groupby("ws_game_id"):
        fh_rows = grp[grp["period"] == "FirstHalf"]
        sh_rows = grp[grp["period"] == "SecondHalf"]
        if len(fh_rows) == 0 or len(sh_rows) == 0:
            continue
        ht_gap = sh_rows["minute"].min() - fh_rows["minute"].max()
        if ht_gap > 2:  # normal match with HT gap in minute column
            continue
        # Broken: minute == match_minute, need to fix 2H match_minute.
        # Use timestamps: find gap > 5min between consecutive rows = halftime.
        sh_sorted = sh_rows.sort_values("minute")
        ts = pd.to_datetime(sh_sorted["timestamp"])
        diffs = ts.diff().dt.total_seconds().fillna(0)
        big_gap_idx = diffs[diffs > 300].index  # >5 min gap = HT break in timestamps
        if len(big_gap_idx) > 0:
            # Rows after the gap are actual 2H; rows before are still HT
            first_real_2h_minute = int(sh_sorted.loc[big_gap_idx[0], "minute"])
        else:
            # No big timestamp gap; use first 2H row with real odds movement.
            # Offset = first_2h_minute - 45 (from 1H end to 2H start in minute col)
            # Approximate: 2H starts ~18 min after 1H end in wall-clock
            first_real_2h_minute = int(fh_rows["minute"].max()) + 18
        offset = first_real_2h_minute - 45  # wall-clock offset to subtract
        mask = (odds["ws_game_id"] == ws_id) & (odds["period"].isin(["SecondHalf", "unknown"]))
        odds.loc[mask, "match_minute"] = odds.loc[mask, "minute"] - offset

    # Build lookup with (match_id, match_minute, period) key
    odds_lookup = {}

    # period=1: FirstHalf incl. stoppage time, then the half-time break; unknown only below 45
    fh = odds[odds["period"].isin(["FirstHalf", "halftime"]) |
              ((odds["period"] == "unknown") & (odds["match_minute"] < 45))]
    fh = fh.drop_duplicates(subset=["ws_game_id", "match_minute"], keep="last")
    for r in fh.itertuples():
        odds_lookup[(int(r.ws_game_id), int(r.match_minute), 1)] = {"home": r.prob_home, "draw": r.prob_draw, "away": r.prob_away}

    # SecondHalf (period=2): match_minute >= 45
    sh = odds[odds["period"].isin(["SecondHalf", "unknown"]) & (odds["match_minute"] >= 45)]
    sh = sh.drop_duplicates(subset=["ws_game_id", "match_minute"], keep="last")
    for r in sh.itertuples():
        odds_lookup[(int(r.ws_game_id), int(r.match_minute), 2)] = {"home": r.prob_home, "draw": r.prob_draw, "away": r.prob_away}

    # Pre-kick-off price under minute -1 (BetfairStrategy serves it at minute 0 while the score is 0-0)
    for gid, q in load_prematch_1x2(season).items():
        odds_lookup[(gid, -1, 1)] = dict(zip(["home", "draw", "away"], q))

    # Build outcome lookup from schedule
    outcome_lookup = {}
    if os.path.exists(sched_path):
        sched = pd.read_parquet(sched_path)
        sched = sched.merge(align[["fbref_match_id", "ws_game_id"]], left_on="match_id", right_on="fbref_match_id", how="inner")

        for _, r in sched.iterrows():
            if pd.notna(r["score"]):
                parts = str(r["score"]).replace("–", "-").replace("—", "-").split("-")
                if len(parts) == 2:
                    try:
                        hs, aws = int(parts[0].strip()), int(parts[1].strip())
                        outcome_lookup[int(r["ws_game_id"])] = 0 if hs > aws else (1 if hs == aws else 2)
                    except ValueError:
                        pass

    # Build match info lookup
    match_info = {int(r["ws_game_id"]): {"home_team": r["ws_home_team"], "away_team": r["ws_away_team"],
                                          "date": str(r["date"])[:10]} for _, r in align.iterrows()}

    return odds_lookup, outcome_lookup, match_info
