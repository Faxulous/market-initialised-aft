"""Market initialisation on made-up prices (Section 3.7 of the paper).

For one match, fit the two log-expected scoring times (eta_H, eta_A) so that the
Weibull model's kick-off probabilities match 1X2 and over/under prices. The odds
below are invented for illustration: no exchange or bookmaker data are used.

Run from the repository root:
    python examples/market_initialisation.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.weibull.calibrate import find_calibrated_etas, simulate_match  # noqa: E402

# CONFIG
GAMMA = 1.17            # common Weibull shape (paper, Section 3.1)
PHI = 1.0               # second-half shape multiplier; 1 keeps one shape for the whole match
MU, HOME = 4.21, -0.18  # intercept and home advantage, log-minutes scale (paper, Section 3.1)
N_SIMS = 10_000         # simulated paths per objective evaluation, as in the paper

# Invented decimal odds for one match with a clear home favourite.
ODDS_1X2 = {'p_home': 1.70, 'p_draw': 4.40, 'p_away': 4.90}
ODDS_OU = {0.5: (1.03, 23.0), 1.5: (1.25, 4.50), 2.5: (1.92, 2.00),
           3.5: (3.65, 1.34), 4.5: (8.60, 1.10)}  # goals threshold: (over, under)


def implied(odds):
    """Implied probabilities with the overround removed."""
    inv = {k: 1 / o for k, o in odds.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def main():
    target_1x2 = implied(ODDS_1X2)
    target_ou = {t: (1 / o) / (1 / o + 1 / u) for t, (o, u) in ODDS_OU.items()}

    # Start from two average teams: equal scoring ability plus home advantage.
    eta_h0, eta_a0 = MU + HOME, MU
    before = simulate_match(eta_h0, eta_a0, GAMMA, PHI, N_SIMS, betas={}, covs={})
    eta_h, eta_a, sse = find_calibrated_etas(target_1x2, target_ou, eta_h0, eta_a0,
                                             GAMMA, PHI, N_SIMS, betas={}, covs={})
    after = simulate_match(eta_h, eta_a, GAMMA, PHI, N_SIMS, betas={}, covs={})

    rows = [('home win', 'p_home'), ('draw', 'p_draw'), ('away win', 'p_away')]
    rows += [(f'over {t} goals', f'p_over_{t}') for t in target_ou]
    target = {**target_1x2, **{f'p_over_{t}': p for t, p in target_ou.items()}}
    print(f"{'':16s}{'prices':>8s}{'before':>8s}{'after':>8s}")
    for label, key in rows:
        print(f'{label:16s}{target[key]:8.3f}{before[key]:8.3f}{after[key]:8.3f}')
    print(f'\nsquared error against the prices: {sse:.5f}')
    for side, e0, e1 in [('home', eta_h0, eta_h), ('away', eta_a0, eta_a)]:
        print(f'{side} eta {e0:.3f} -> {e1:.3f} (shift {e1 - e0:+.3f}); '
              f'expected minutes to score {math.exp(e0):.1f} -> {math.exp(e1):.1f}')


if __name__ == '__main__':
    main()
