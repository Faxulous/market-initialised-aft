"""Betting simulation using model predictions vs Betfair odds.

Tests fixed-payout staking and the Kelly criterion on 1X2 outcomes; both back every +EV outcome.
Commission is applied on net winnings per match (Betfair exchange model).

Run: python eval/bet.py (no arguments).
Compares every model, including the market-implied benchmarks. Writes eval/bet/summary.csv.
"""
import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.utils import load_predictions, clock

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_KELLY = 1.0  # Uncapped Kelly
COMMISSION = 0.02  # Betfair commission on net winnings per match (2% for high-volume)
MIN_BET_MINUTE = 0  # Include all evaluation minutes
MIN_EDGE = 0.0  # Minimum probability edge to bet
MATCHES_PER_YEAR = 380  # Full EPL season: Sharpe annualisation
EVAL_DIR = os.path.dirname(__file__)
OUT = os.path.join(EVAL_DIR, 'bet')

def _c(val, fmt):  # Colorize value: green if +, red if -, no style if 0
    s = f"{val:{fmt}}"
    return f"[green]{s}[/]" if val > 0 else (f"[red]{s}[/]" if val < 0 else s)


def kelly_fraction(prob, odds, max_kelly=MAX_KELLY):
    """Calculate Kelly stake fraction. Returns 0 if no edge."""
    if prob <= 0 or prob >= 1 or odds <= 1:
        return 0
    b = odds - 1  # Net odds
    f = (b * prob - (1 - prob)) / b  # Kelly formula
    return max(0, min(f, max_kelly))


def simulate_betting(model_preds, bf_preds, min_edge=0.0):
    """Simulate betting for one minute. Stores gross PnL (no commission).

    Returns list of per-match results with gross PnL per bet.
    """
    merged = model_preds.merge(
        bf_preds[['match_id', 'pred_home', 'pred_draw', 'pred_away']].rename(
            columns={'pred_home': 'bf_home', 'pred_draw': 'bf_draw', 'pred_away': 'bf_away'}
        ), on='match_id', how='inner'
    )
    if len(merged) == 0:
        return None
    q = merged[['bf_home', 'bf_draw', 'bf_away']].to_numpy()
    if not (q > 0).all():  # Odds are 1/q unclipped; q >= 1 (slippage-adjusted callers) gives odds <= 1, never backed
        raise ValueError('Non-positive Betfair probability: no quoted price')

    results = []
    for _, row in merged.iterrows():
        model_probs = np.array([row['pred_home'], row['pred_draw'], row['pred_away']])
        bf_probs = np.array([row['bf_home'], row['bf_draw'], row['bf_away']])
        odds = 1 / bf_probs  # Market price as quoted: no clip, so no bet settles at odds the market did not offer
        actual = int(row['actual'])  # 0=H, 1=D, 2=A
        edges = model_probs - bf_probs

        # Both strategies back every +EV outcome and differ only in stake
        pay_bets, kelly_bets = [], []
        for i in range(3):
            ev = model_probs[i] * odds[i] - 1
            if ev > 0 and edges[i] >= min_edge:
                won = (i == actual)
                # Fixed payout: stake 1/odds, so every winning bet is paid out one unit
                for bets, stake in ((pay_bets, 1 / odds[i]), (kelly_bets, kelly_fraction(model_probs[i], odds[i]))):
                    if stake > 0:
                        bets.append({
                            'outcome': i, 'stake': stake, 'odds': odds[i], 'won': won,
                            'gross_pnl': stake * (odds[i] - 1) if won else -stake,  # No commission here
                            'edge': edges[i], 'ev': ev,
                        })

        results.append({
            'match_id': row['match_id'], 'date': row.get('date'),
            'home_team': row.get('home_team'), 'away_team': row.get('away_team'),
            'actual': actual, 'model_probs': model_probs, 'bf_probs': bf_probs,
            'odds': odds, 'edges': edges,
            'pay_bets': pay_bets, 'kelly_bets': kelly_bets,
        })
    return results


def apply_match_commission(results_by_minute, commission=COMMISSION):
    """Apply commission on net winnings per match across all minutes.

    Groups all bets by match_id, sums gross PnL, charges commission only on
    positive net profit per match. Returns summary dict with per-match Sharpe ratios.
    """
    mids = {r['match_id'] for results in results_by_minute.values() for r in results}
    out = {'n_matches': len(mids)}
    for k in ('pay', 'kelly'):
        gross, n, wins, stake = defaultdict(float), 0, 0, 0.0  # gross: match_id -> gross PnL
        for results in results_by_minute.values():
            for r in results:
                for b in r[f'{k}_bets']:
                    gross[r['match_id']] += b['gross_pnl']
                    n, wins, stake = n + 1, wins + b['won'], stake + b['stake']
        comm = {m: max(0, g) * commission for m, g in gross.items()}
        net = sum(gross.values()) - sum(comm.values())
        # Sharpe per match (bets within a match are dependent): mean/sd of net match P&L, x sqrt(matches/season)
        pnl = np.array([gross[m] - comm[m] if m in gross else 0. for m in mids])  # 0 = no bet placed in that match
        out.update({f'{k}_bets': n, f'{k}_wins': wins, f'{k}_gross': sum(gross.values()),
                    f'{k}_commission': sum(comm.values()), f'{k}_net': net, f'{k}_stake': stake,
                    f'{k}_roi': net / stake * 100 if stake > 0 else 0,
                    f'{k}_sharpe': pnl.mean() / pnl.std() * np.sqrt(MATCHES_PER_YEAR)})
    return out


def _net_stream(results_by_minute, key, commission):
    """One strategy's per-bet PnL sorted by (date, minute), net of commission: each match's
    commission on its net winnings is charged at that match's last bet (settlement)."""
    bets = sorted(((r.get('date') or '', minute, r['match_id'], b['gross_pnl'])
                   for minute, results in results_by_minute.items() for r in results for b in r[key]),
                  key=lambda x: x[:2])
    pnl, gross, last = np.array([b[3] for b in bets]), defaultdict(float), {}
    for n, (_, _, mid, g) in enumerate(bets):
        gross[mid] += g
        last[mid] = n
    for mid, n in last.items():
        pnl[n] -= max(0, gross[mid]) * commission
    return pnl


def evaluate(df, model):
    """Run the existing per-minute simulation for one model."""
    results = {}
    if (df.loc[df['model'] == 'betfair', ['pred_home', 'pred_draw', 'pred_away']] >= 1).any().any():
        raise ValueError('Betfair probability >= 1: no tradable price')
    for minute, sub in df.groupby('clock'):   # (period, minute) via the chronological clock key
        bets = simulate_betting(sub[sub['model'] == model],
                                sub[sub['model'] == 'betfair'], MIN_EDGE)
        if bets is None:
            raise ValueError(f'{model}: no matched predictions at minute {minute}')
        results[minute] = bets
    return results


def main():
    df = load_predictions(os.path.join(EVAL_DIR, 'predictions.parquet'),
                          benchmarks=True)
    df = df[df['eval_minute'] >= MIN_BET_MINUTE].assign(clock=lambda d: clock(d['period'], d['eval_minute']))
    if df.empty:
        raise ValueError('No predictions in the configured evaluation window')
    models = sorted(set(df['model']) - {'betfair'})
    console.print(f"[bold]All-model betting comparison[/] | {df['match_id'].nunique()} matches | "
                  f"Minutes {df['eval_minute'].min()}--{df['eval_minute'].max()} | "
                  f"Edge {MIN_EDGE:.1%} | Commission {COMMISSION:.1%}")
    rows = []
    for model in models:
        with console.status(f'Evaluating {model}...'):
            rows.append({'model': model, **apply_match_commission(evaluate(df, model), COMMISSION)})
    summary = pd.DataFrame(rows)
    table = Table(title=f'{COMMISSION:.0%} commission')
    for col in ['Model', 'Payout PnL', 'Payout ROI', 'Kelly PnL', 'Kelly ROI']:
        table.add_column(col, justify='left' if col == 'Model' else 'right')
    for r in summary.itertuples():
        table.add_row(r.model, _c(r.pay_net, '.2f'), _c(r.pay_roi, '.1f') + '%',
                      _c(r.kelly_net, '.2f'), _c(r.kelly_roi, '.1f') + '%')
    console.print(table)
    os.makedirs(OUT, exist_ok=True)
    summary.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    console.print('[dim]Saved: eval/bet/summary.csv[/]')


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Usage: python eval/bet.py (settings are in CONFIG)')
    main()
