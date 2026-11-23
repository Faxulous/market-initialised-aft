"""Betfair price reaction to goals, relative to the WhoScored goal minute.

Supports the BF_SHIFT = 2 used for the Betfair benchmark (models/strategies/betfair.py), on the
2024/25 matches played before the first evaluation date. Shares are of all goals. A goal with no
detectable price move within the window leaves no reaction to wait for, so it counts as absorbed.

Run: python eval/analysis/goal_latency.py -> eval/analysis/out/goal_latency.csv
"""
import os, sys, numpy as np, pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.utils import EVAL_START_DATE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEASON = '24-25'
PROB_THRESH = 0.05   # Min prob shift to count as "reacted"
WINDOW = 3           # Look ±3 minutes around goal
OUT = os.path.join(ROOT, 'eval', 'analysis', 'out', 'goal_latency.csv')
console = Console()


def find_bf_reaction(bf_match, goal_min, is_home_goal):
    """Find first minute in [goal_min - WINDOW, goal_min + WINDOW] where Betfair reacts.

    Returns latency (reaction_minute - goal_minute), or None if no reaction detected.
    """
    window = bf_match[(bf_match['match_minute'] >= goal_min - WINDOW) &
                      (bf_match['match_minute'] <= goal_min + WINDOW)].sort_values('match_minute')
    if len(window) < 2:
        return None

    baseline = window.iloc[0]['prob_home']
    sign = 1 if is_home_goal else -1  # Home goal -> prob_home rises

    for _, row in window.iterrows():
        shift = (row['prob_home'] - baseline) * sign
        if shift >= PROB_THRESH:
            return int(row['match_minute']) - goal_min
    return None


def main():
    odds, ws, align = (pd.read_parquet(os.path.join(ROOT, 'data', SEASON, f'{n}.parquet'))
                       for n in ('betfair_odds', 'whoscored_events', 'match_alignment'))
    date = pd.to_datetime(align['date'])
    ids = set(align.loc[date < pd.to_datetime(EVAL_START_DATE), 'ws_game_id'])   # every 2024/25 match before the evaluation period
    goals = ws[(ws['is_goal'] == True) & ws['ws_match_id'].isin(ids)]
    ws_to_fbref = dict(zip(align['ws_game_id'], align['fbref_match_id']))
    ws_to_home = dict(zip(align['ws_game_id'], align['ws_home_team']))
    bf = {mid: g.sort_values('match_minute') for mid, g in odds.groupby('fbref_match_id')}
    lat = [find_bf_reaction(bf[ws_to_fbref[g.ws_match_id]], int(g.minute),
                            (g.team_name == ws_to_home[g.ws_match_id]) != bool(g.is_own_goal))  # own goal benefits the other side
           for g in goals.itertuples()]
    lat = np.array([-np.inf if x is None else x for x in lat])   # no detectable price move: nothing to absorb
    rows = [dict(matches=len(ids), goals=len(goals), reacted=int(np.isfinite(lat).sum()), offset=k, absorbed=int((lat <= k).sum()),
                 share=float((lat <= k).mean()), share_reacted=float((lat[np.isfinite(lat)] <= k).mean())) for k in range(-WINDOW, WINDOW + 1)]
    pd.DataFrame(rows).to_csv(OUT, index=False)
    table = Table(title=f"Betfair reaction to goals: {rows[0]['reacted']} of {len(goals)} goals with a detectable reaction, {len(ids)} matches")
    for c in ("Minutes after goal", "Goals absorbed", "Share of all goals", "Share of goals with a detectable reaction"):
        table.add_column(c, justify="right")
    for r in rows:
        table.add_row(str(r['offset']), str(r['absorbed']), f"{r['share']:.1%}", f"{r['share_reacted']:.1%}")
    console.print(table)


if __name__ == '__main__':
    main()
