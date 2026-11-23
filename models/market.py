"""Pre-kick-off Betfair targets for market initialisation (kappa): 1X2 + over/under. Fails loudly."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OU_THRESH = [0.5, 1.5, 2.5, 3.5, 4.5]
ODDS = ['odds_home', 'odds_draw', 'odds_away']
N_SIMS_CAL = 10000   # paths per objective evaluation in simulation-based calibration (= forecast paths, eval/test.py)
SEED_CAL = 8         # common random numbers: every objective evaluation of a match reuses one stream


def _read(season, name):
    return pd.read_parquet(os.path.join(ROOT, 'data', season, f'{name}.parquet'))


def load_prematch_1x2(season: str) -> dict:
    """Last Betfair 1X2 price BEFORE kick-off (no in-play information), normalised, keyed by ws_game_id.
    21-22..23-24 have no period column (pre-match = minute < 0); 24-25 labels 'PreMatch' at minutes
    -30..-1."""
    bo, al = _read(season, 'betfair_odds'), _read(season, 'match_alignment')
    pre = bo[(bo['period'] == 'PreMatch') if 'period' in bo.columns else (bo['minute'] < 0)]
    pre = pre[pre[ODDS].notna().all(axis=1)].sort_values('minute').groupby('fbref_match_id').tail(1)
    pre = pre.merge(al[['fbref_match_id', 'ws_game_id']], on='fbref_match_id', how='inner')
    inv = pre[ODDS].rdiv(1)
    return dict(zip(pre['ws_game_id'].astype(int), inv.div(inv.sum(axis=1), axis=0).values))


def load_ou_targets(season: str) -> dict:
    """Overround-free P(total goals > t) per match, keyed by ws_game_id. Rows of one match carry
    complementary thresholds and are unioned; two prices for one threshold raise."""
    ou = _read(season, 'over_under').merge(_read(season, 'match_alignment')[['fbref_match_id', 'ws_game_id']],
                                           left_on='match_id', right_on='fbref_match_id', how='inner')
    out = {}
    for _, r in ou.iterrows():
        d = out.setdefault(int(r['ws_game_id']), {})
        for t in OU_THRESH:
            o, u = r[f'over_{t}'], r[f'under_{t}']
            if pd.notna(o) and pd.notna(u) and o > 1 and u > 1:
                p = (1 / o) / (1 / o + 1 / u)          # normalise the pair, removing overround
                if t in d and abs(d[t] - p) > 1e-12:
                    raise ValueError(f"conflicting over/under price: match {r['ws_game_id']}, threshold {t}")
                d[t] = p
    return out


def load_market_data(season: str, match_ids) -> dict:
    """Targets for every match in match_ids, keyed by (home_id, away_id). Raises if any match lacks a
    pre-kick-off 1X2 price, an over/under price or a team id: no match is ever dropped."""
    al = _read(season, 'match_alignment').set_index('ws_game_id')
    team_id = _read(season, 'whoscored_events').groupby('team_name')['team_id'].first()
    x12, ou = load_prematch_1x2(season), load_ou_targets(season)
    bad = {'pre-kick-off 1X2': [m for m in match_ids if m not in x12], 'alignment': [m for m in match_ids if m not in al.index],
           'over/under': [m for m in match_ids if not ou.get(m)]}   # .get: an absent match is reported here, not defaulted
    if any(bad.values()):
        raise ValueError(f"{season}: matches without market targets: { {k: v for k, v in bad.items() if v} }")
    out = {}
    for m in match_ids:
        r = al.loc[m]
        out[int(team_id[r['ws_home_team']]), int(team_id[r['ws_away_team']])] = {
            'ws_game_id': int(m), 'home_name': r['ws_home_team'], 'away_name': r['ws_away_team'],
            'date': str(r['date'])[:10], 'ou_targets': ou[m],
            **dict(zip(['p_home', 'p_draw', 'p_away'], x12[m]))}
    if len(out) != len(set(match_ids)):
        raise ValueError(f"{season}: duplicate (home, away) fixture among {len(set(match_ids))} matches")
    return out


def eval_match_ids(season: str) -> list:
    """Evaluation fixtures of one season, from the merged predictions file (ids only, never prices)."""
    p = pd.read_parquet(os.path.join(ROOT, 'eval', 'predictions.parquet'), columns=['season', 'match_id'])
    ids = sorted(int(m) for m in p.loc[p['season'] == season, 'match_id'].unique())
    if not ids:
        raise ValueError(f"no evaluation matches for {season} in eval/predictions.parquet")
    return ids
