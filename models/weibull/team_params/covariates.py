"""Covariate extraction for Weibull AFT extension with dynamic in-play covariates."""
import numpy as np, pandas as pd

WARM_START_2H = True  # If True, 2H cumulative fields include 1H totals


def compute_match_covariates(ws_mid: int, ws_events: pd.DataFrame, fb_shots: pd.DataFrame,
                              align: pd.DataFrame) -> dict:
    """Compute minute-level cumulative covariates for a single match.

    Returns dict with keys 'h1' and 'h2', each a DataFrame indexed by minute (0-90+).
    """
    row = align[align['ws_game_id'] == ws_mid]
    if len(row) == 0:
        raise ValueError(f"Match {ws_mid} not found in alignment")
    row = row.iloc[0]
    fb_mid, home_ws = row['fbref_match_id'], row['ws_home_team']
    home_fb = row['fbref_home_team']

    m_ws = ws_events[ws_events['ws_match_id'] == ws_mid]
    m_fb = fb_shots[fb_shots['match_id'] == fb_mid]
    on_target = m_fb['outcome'].isin(['Goal', 'Saved'])  # FBref defines PSxG for these only; off-target PSxG is null by convention
    bad = m_fb[m_fb[['minute', 'period', 'xG']].isna().any(axis=1) | (on_target & m_fb['PSxG'].isna())]
    if len(bad):
        raise ValueError(f"{fb_mid}: {len(bad)} shots with null minute/period/xG or on-target PSxG")

    # Get max minutes for each half
    h1_ws = m_ws[m_ws['period'] == 'FirstHalf']
    h2_ws = m_ws[m_ws['period'] == 'SecondHalf']
    max_h1 = int(h1_ws['minute'].max()) if len(h1_ws) > 0 else 45
    max_h2 = int(max(h2_ws['minute'].max() if len(h2_ws) > 0 else 90, 90))

    def build_arrays(events_ws, events_fb, max_min, home_ws, home_fb):
        """Build per-minute covariate arrays for a single half."""
        goals_h, goals_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        xg_h, xg_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        ng_xg_h, ng_xg_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        psxg_h, psxg_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        ng_psxg_h, ng_psxg_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        ft_touch_h, ft_touch_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        ft_pass_h, ft_pass_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        red_h, red_a = np.zeros(max_min + 1), np.zeros(max_min + 1)
        shots_h, shots_a = np.zeros(max_min + 1), np.zeros(max_min + 1)

        # Goals, xG, PSxG from fbref
        for _, s in events_fb.iterrows():
            t = int(s['minute'])
            if 0 <= t <= max_min:
                is_home = s['team'] == home_fb
                is_goal = s['outcome'] == 'Goal'
                (shots_h if is_home else shots_a)[t] += 1  # every attempt, on or off target
                if is_goal:
                    (goals_h if is_home else goals_a)[t] += 1
                xg_val = float(s['xG'])
                psxg_val = float(s['PSxG']) if pd.notna(s['PSxG']) else 0.0  # null = off target (validated above)
                (xg_h if is_home else xg_a)[t] += xg_val
                (psxg_h if is_home else psxg_a)[t] += psxg_val
                if not is_goal:
                    (ng_xg_h if is_home else ng_xg_a)[t] += xg_val
                    (ng_psxg_h if is_home else ng_psxg_a)[t] += psxg_val

        # WhoScored touches and passes
        touch_types = ['BallTouch', 'Pass', 'TakeOn', 'Tackle', 'Interception', 'Clearance', 'Aerial', 'Foul']
        ft_mask = events_ws['event_type'].isin(touch_types) & (events_ws['start_x'] >= 66.7)
        for _, e in events_ws[ft_mask].iterrows():
            t = int(e['minute'])
            if 0 <= t <= max_min:
                (ft_touch_h if e['team_name'] == home_ws else ft_touch_a)[t] += 1

        ft_pass_mask = (events_ws['event_type'] == 'Pass') & (events_ws['start_x'] >= 66.7)
        for _, e in events_ws[ft_pass_mask].iterrows():
            t = int(e['minute'])
            if 0 <= t <= max_min:
                (ft_pass_h if e['team_name'] == home_ws else ft_pass_a)[t] += 1

        # Red cards
        for _, e in events_ws[events_ws['card_type'].isin(['Red', 'SecondYellow'])].iterrows():
            t = int(e['minute'])
            if 0 <= t <= max_min:
                (red_h if e['team_name'] == home_ws else red_a)[t] += 1

        return {
            'goals_h': goals_h, 'goals_a': goals_a,
            'xg_h': xg_h, 'xg_a': xg_a, 'ng_xg_h': ng_xg_h, 'ng_xg_a': ng_xg_a,
            'psxg_h': psxg_h, 'psxg_a': psxg_a,
            'ng_psxg_h': ng_psxg_h, 'ng_psxg_a': ng_psxg_a,
            'ft_touch_h': ft_touch_h, 'ft_touch_a': ft_touch_a,
            'ft_pass_h': ft_pass_h, 'ft_pass_a': ft_pass_a,
            'red_h': red_h, 'red_a': red_a,
            'shots_h': shots_h, 'shots_a': shots_a
        }

    # Build arrays for each half
    h1_fb = m_fb[m_fb['period'] == 1]
    h2_fb = m_fb[m_fb['period'] == 2]
    arr_h1 = build_arrays(h1_ws, h1_fb, max_h1, home_ws, home_fb)
    arr_h2 = build_arrays(h2_ws, h2_fb, max_h2, home_ws, home_fb)

    def cumsum_lag(arr):
        """Cumulative sum excluding current minute (strictly backward-looking)."""
        return np.cumsum(arr) - arr

    def make_df(arr, max_min):
        cl = cumsum_lag  # shorthand
        return pd.DataFrame({
            'minute': range(max_min + 1),
            'goals_home': cl(arr['goals_h']).astype(int),
            'goals_away': cl(arr['goals_a']).astype(int),
            'xg_home': cl(arr['xg_h']), 'xg_away': cl(arr['xg_a']),
            'ng_xg_home': cl(arr['ng_xg_h']), 'ng_xg_away': cl(arr['ng_xg_a']),
            'psxg_home': cl(arr['psxg_h']), 'psxg_away': cl(arr['psxg_a']),
            'ng_psxg_home': cl(arr['ng_psxg_h']),
            'ng_psxg_away': cl(arr['ng_psxg_a']),
            'ft_touch_home': cl(arr['ft_touch_h']).astype(int),
            'ft_touch_away': cl(arr['ft_touch_a']).astype(int),
            'ft_pass_home': cl(arr['ft_pass_h']).astype(int),
            'ft_pass_away': cl(arr['ft_pass_a']).astype(int),
            'red_home': np.cumsum(arr['red_h']).astype(int),
            'red_away': np.cumsum(arr['red_a']).astype(int),
            'cum_psxg_home': cl(arr['psxg_h']),
            'cum_psxg_away': cl(arr['psxg_a']),
            'cum_ng_psxg_home': cl(arr['ng_psxg_h']),
            'cum_ng_psxg_away': cl(arr['ng_psxg_a']),
            'shots_home': cl(arr['shots_h']), 'shots_away': cl(arr['shots_a']),
        })

    result = {'h1': make_df(arr_h1, max_h1), 'h2': make_df(arr_h2, max_h2)}

    # Store raw 1H arrays for 2H cumulative carry-over
    if WARM_START_2H:
        result['h1_raw'] = arr_h1

    return result


def get_covariates_at(covs: dict, minute: int, period: int) -> pd.Series:
    """Get cumulative covariates at a specific (minute, period).

    Args:
        covs: Output from compute_match_covariates()
        minute: Game-time minute (0-45 for 1H, 45-90+ for 2H)
        period: 1 or 2
    """
    df = covs['h1'] if period == 1 else covs['h2']
    t = min(minute, len(df) - 1)
    row = df.iloc[t].copy()

    # Add entire 1H totals to 2H cumulative fields
    if WARM_START_2H and period == 2 and 'h1_raw' in covs:
        h1_raw = covs['h1_raw']
        for k in ('goals', 'xg', 'ng_xg', 'psxg', 'ng_psxg', 'ft_touch', 'ft_pass', 'shots'):
            row[f'{k}_home'] += h1_raw[f'{k}_h'].sum()
            row[f'{k}_away'] += h1_raw[f'{k}_a'].sum()
        row['red_home'] += int(h1_raw['red_h'].sum())
        row['red_away'] += int(h1_raw['red_a'].sum())
        row['cum_psxg_home'] += h1_raw['psxg_h'].sum()
        row['cum_psxg_away'] += h1_raw['psxg_a'].sum()
        row['cum_ng_psxg_home'] += h1_raw['ng_psxg_h'].sum()
        row['cum_ng_psxg_away'] += h1_raw['ng_psxg_a'].sum()

    return row


def extract_subintervals_with_covariates(ws_events: pd.DataFrame, fb_shots: pd.DataFrame,
                                          align: pd.DataFrame, mcol: str = "ws_match_id",
                                          delta: int = 5, verbose: bool = False) -> list:
    """Extract sub-intervals with time-varying covariates for modulated renewal process.

    Instead of one row per goal interval, returns multiple rows (one per delta-minute chunk).
    Each sub-interval has covariates measured at its START. Only final sub-interval is uncensored.

    Args:
        delta: Sub-interval width in minutes (e.g., 5 = update covariates every 5 mins)

    Returns list of tuples (indices 0-31, plus 32-39):
        0: match_id, 1: interval_id, 2: rel_start (relative to interval), 3: rel_end,
        4: home_id, 5: away_id, 6: scorer_id, 7: censored,
        8: score_diff, 9: goals_diff, 10: ft_touch_diff, 11: red_diff,
        12: ft_pass_diff, 13: psxg_diff, 14: ng_psxg_diff, 15: period, 16: is_final,
        17: game_time, 18: ft_touch_home, 19: ft_touch_away, 20: ft_pass_home,
        21: ft_pass_away, 22: psxg_home, 23: psxg_away, 24: ng_psxg_home, 25: ng_psxg_away,
        26: goals_home, 27: goals_away,
        28: xg_home, 29: xg_away, 30: ng_xg_home, 31: ng_xg_away,
        32-33: cum_goals_home/away, 34-37: cum_psxg/cum_ng_psxg (match-level),
        38: intv_ng_psxg_home, 39: intv_ng_psxg_away (interval-level, resets at each goal)
        40: shots_home, 41: shots_away
    """
    name_to_id = dict(ws_events.groupby('team_name')['team_id'].first())
    align_idx = align.set_index('ws_game_id')
    from models.weibull.clock import match_clock, training_events
    ws_events = training_events(ws_events, mcol)
    if delta <= 0:
        raise ValueError('delta must be positive')

    subintervals = []
    for i, ws_mid in enumerate(ws_events[mcol].unique()):
        if verbose and i % 50 == 0:
            print(f"Processing match {i+1}/{len(ws_events[mcol].unique())}")

        if ws_mid not in align_idx.index:
            raise KeyError(f"match {ws_mid} has events but no alignment row")
        arow = align_idx.loc[ws_mid]
        home_ws, home_fb = arow['ws_home_team'], arow['fbref_home_team']
        fb_mid = arow['fbref_match_id']

        home_id, away_id = int(name_to_id[home_ws]), int(name_to_id[arow['ws_away_team']])

        covs = compute_match_covariates(ws_mid, ws_events, fb_shots, align)
        m_ws, ht, dur = match_clock(ws_events[ws_events[mcol] == ws_mid])
        goals = [(float(g.time), away_id if g.team_id == home_id else home_id)
                 if g.is_own_goal else (float(g.time), int(g.team_id))
                 for g in m_ws[m_ws.is_goal == True].itertuples()]
        prev, h, a, interval_idx = 0., 0, 0, 0

        def emit_subintervals(interval_start, interval_end, scorer_id, is_censored,
                              score_diff, cum_h, cum_a):
            """Split [interval_start, interval_end) into sub-intervals of width delta."""
            nonlocal interval_idx
            interval_id = f"{ws_mid}_{interval_idx}"
            interval_idx += 1
            subs = []
            game_t = interval_start
            rel_t = 0.0
            interval_dur = interval_end - interval_start

            # Interval-level baseline: cum ng_psxg at interval start (for "since last goal")
            base_period = 1 if interval_start < ht else 2
            raw = interval_start if base_period == 1 else interval_start - ht + 45
            r0 = get_covariates_at(covs, int(raw), base_period)
            base_ng_h = float(r0['cum_ng_psxg_home'])
            base_ng_a = float(r0['cum_ng_psxg_away'])

            while rel_t < interval_dur:
                boundary = ht - interval_start if game_t < ht else interval_dur
                rel_end = min(rel_t + delta, interval_dur, boundary)
                is_final = abs(rel_end - interval_dur) < 0.01
                sub_period = 1 if game_t < ht else 2
                raw = game_t if sub_period == 1 else game_t - ht + 45
                r = get_covariates_at(covs, int(raw), sub_period)
                sub_censored = True if not is_final else is_censored
                subs.append((
                    str(ws_mid), interval_id, rel_t, rel_end,
                    home_id, away_id,
                    scorer_id if (is_final and not is_censored) else None,
                    sub_censored,
                    score_diff,
                    int(r['goals_home'] - r['goals_away']),
                    int(r['ft_touch_home'] - r['ft_touch_away']),
                    int(r['red_away'] - r['red_home']),
                    int(r['ft_pass_home'] - r['ft_pass_away']),
                    float(r['psxg_home'] - r['psxg_away']),
                    float(r['ng_psxg_home'] - r['ng_psxg_away']),
                    sub_period,
                    1 if is_final else 0,
                    raw,
                    int(r['ft_touch_home']), int(r['ft_touch_away']),
                    int(r['ft_pass_home']), int(r['ft_pass_away']),
                    float(r['psxg_home']), float(r['psxg_away']),
                    float(r['ng_psxg_home']), float(r['ng_psxg_away']),
                    int(r['goals_home']), int(r['goals_away']),
                    float(r['xg_home']), float(r['xg_away']),
                    float(r['ng_xg_home']), float(r['ng_xg_away']),
                    cum_h, cum_a,
                    float(r['cum_psxg_home']), float(r['cum_psxg_away']),
                    float(r['cum_ng_psxg_home']), float(r['cum_ng_psxg_away']),
                    float(r['cum_ng_psxg_home']) - base_ng_h,  # 38: intv_ng_psxg_home
                    float(r['cum_ng_psxg_away']) - base_ng_a,  # 39: intv_ng_psxg_away
                    float(r['shots_home']), float(r['shots_away']),  # 40-41: cum shots
                ))
                rel_t = rel_end
                game_t = interval_start + rel_t
            return subs

        # Exposure cuts retain one interval id and elapsed-since-goal origin.
        for gmin, gtid in goals:
            if gmin > prev:
                subintervals.extend(emit_subintervals(prev, gmin, gtid, False, h-a, h, a))
            prev = gmin
            h, a = (h+1, a) if gtid == home_id else (h, a+1)
        if dur > prev:
            subintervals.extend(emit_subintervals(prev, dur, None, True, h-a, h, a))

    return subintervals
