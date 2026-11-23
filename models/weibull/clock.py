"""Playing-time clock: retain first-half stoppage, exclude the half-time break."""
import numpy as np

EXCLUDED_MATCHES = {1549615}  # Newcastle–Tottenham 2021-10-17: truncated feed (sanity review).


def training_events(events, mcol='ws_match_id'):
    """Explicit reviewed exclusion; unexpected incomplete fixtures still fail."""
    return events[~events[mcol].isin(EXCLUDED_MATCHES)]


def playing_time(minute, period, h1_end):
    """Map period-local minutes to elapsed playing minutes; h1_end is observed."""
    if not np.isin(period, [1, 2]).all() or not np.isfinite(h1_end) or h1_end < 45:
        raise ValueError("Expected period 1/2 and a finite first-half end >= 45")
    return np.asarray(minute) + np.where(np.asarray(period) == 2, h1_end - 45, 0)


def match_clock(events):
    """Return regulation events on a monotone clock and observed half/full ends."""
    m = events[events['period'].isin(['FirstHalf', 'SecondHalf'])].copy()
    ends = m.groupby('period')['minute'].max()
    if not np.isfinite(ends.loc[['FirstHalf', 'SecondHalf']]).all():
        raise ValueError('Missing half-end time')
    if (m.loc[m.period == 'SecondHalf', 'minute'] < 45).any():
        raise ValueError('Second-half minute precedes its kick-off')
    h1, h2 = max(45., float(ends['FirstHalf'])), max(90., float(ends['SecondHalf']))
    m['half'] = m['period'].map({'FirstHalf': 1, 'SecondHalf': 2})
    m['time'] = playing_time(m['minute'].to_numpy(), m['half'].to_numpy(), h1)
    return m.sort_values(['half', 'minute', 'second']), h1, h1 + h2 - 45
