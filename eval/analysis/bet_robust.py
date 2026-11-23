"""Betting robustness and calibration diagnostics behind Section 5, Section 6 and Appendix C.

Reuses eval/bet.py's simulation verbatim (simulate_betting, which applies kelly_fraction
internally, plus COMMISSION) at bet.py's evaluation points (every minute, first-half
stoppage included), so the numbers reconcile with the paper's betting table.

Produces the four paper figures (reliability for the headline model, reliability for every
model family, betting fragility, ROI breakdown), the appendix calibration table, and the
CSVs backing every number quoted in the text.

Usage: python eval/analysis/bet_robust.py            (full)
       python eval/analysis/bet_robust.py --smoke    (B=200)
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'eval'))
from bet import simulate_betting, _net_stream, COMMISSION  # noqa: E402
from eval.utils import BENCH_NAMES, clock  # noqa: E402

console = Console(width=170)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEED = 8
MODEL = 'weibull_calibd'          # paper's headline model
MARKET = 'betfair'
BASELINES = ['mkt_poisson', 'mkt_poisson_t', 'mkt_naive']   # eval/analysis/benchmarks.py
N_BINS = 10                       # equal-width reliability bins
N_BOOT = 10_000                   # match-clustered bootstrap resamples
DROP_TOP = [1, 2, 3, 5, 6, 10]    # leave-out-best-matches sensitivity
MIN_BUCKETS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90), (90, 999)]
ODDS_BANDS = [(0, 1.5), (1.5, 3), (3, 6), (6, 15), (15, np.inf)]
NAME = 'bet_robust'
OUT = os.path.join(REPO, 'eval', 'analysis', 'out')
OUT_PDF = os.path.join(OUT, 'pdf')
OUTCOMES = ['H', 'D', 'A']
DISP = {MODEL: r'Weibull$_{\kappa,\psi}$', MARKET: 'Betfair', **BENCH_NAMES}
FAMILIES = [('Weibull', [('weibull_no_cov', 'Weibull'), ('weibull', r'Weibull$_{\psi}$'),
                         ('weibull_calibd', r'Weibull$_{\kappa,\psi}$')]),
            ('Cox process', [('maia', r'Maia$_{\psi}$'), ('maia_calibd', r'Maia$_{\kappa,\psi}$')]),
            ('Birth process', [('zou', r'Zou$_{\psi}$'), ('zou_calibd', r'Zou$_{\kappa,\psi}$')]),
            ('Market-implied', [(k, BENCH_NAMES[k]) for k in BASELINES])]

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'mathtext.fontset': 'dejavusans', 'font.size': 9,
                     'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8})


def save(fig, suffix):
    os.makedirs(OUT_PDF, exist_ok=True)
    fig.savefig(os.path.join(OUT_PDF, f'{NAME}_{suffix}.pdf'), dpi=300)
    plt.close(fig)
    console.print(f"[green]saved[/] out/pdf/{NAME}_{suffix}.pdf")


def bucket_label(m):  # minute -> bucket string
    for lo, hi in MIN_BUCKETS:
        if lo <= m < hi:
            return '90+' if lo == 90 else f'{lo}-{hi - 1}'
    raise ValueError(f'minute {m} outside buckets')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CALIBRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def calib(p, y, n_bins=N_BINS):
    """Bin pooled (p, y) pairs in equal-width bins; return per-bin frame and diagnostics.

    Brier decomposition (Murphy): BS = reliability - resolution + uncertainty,
    computed on the binned data (exact only up to within-bin variation).
    """
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, edges[1:-1])  # 0 .. n_bins-1
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({'bin': b, 'lo': edges[b], 'hi': edges[b + 1], 'n': n,
                     'p_mean': p[m].mean(), 'obs_freq': y[m].mean()})
    bins = pd.DataFrame(rows)
    N, obar = len(p), y.mean()
    w = bins['n'].values / N
    gap = np.abs(bins['p_mean'].values - bins['obs_freq'].values)
    rel = float((w * (bins['p_mean'].values - bins['obs_freq'].values) ** 2).sum())
    res = float((w * (bins['obs_freq'].values - obar) ** 2).sum())
    return bins, {'n_points': N, 'ece': float((w * gap).sum()), 'mce': float(gap.max()),
                  'brier': float(((p - y) ** 2).mean()), 'reliability': rel,
                  'resolution': res, 'uncertainty': float(obar * (1 - obar)),
                  'rel_minus_res_plus_unc': rel - res + obar * (1 - obar)}


def pooled(sub):
    """Flatten a prediction frame to pooled binary (p, y) pairs over H/D/A."""
    p = sub[['pred_home', 'pred_draw', 'pred_away']].values.ravel()
    y = (sub['actual'].values[:, None] == np.arange(3)[None, :]).astype(float).ravel()
    return p, y


def cal_slope(p, y):  # logistic recalibration y ~ a + b logit(p); b < 1 means overconfident
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    r = minimize(lambda w: -np.mean(y * (w[0] + w[1] * x) - np.logaddexp(0, w[0] + w[1] * x)), [0, 1])
    if not r.success:
        raise RuntimeError(f'calibration slope fit failed: {r.message}')
    return float(r.x[1])


def run_calibration(preds, models):
    """Per-model reliability bins and diagnostics on the pooled H/D/A grid, Betfair first."""
    out, n = {}, None
    for m in models:
        sub = preds[preds['model'] == m]
        if n is None:
            n = len(sub)
        elif len(sub) != n:
            raise ValueError(f'{m}: prediction count differs from {models[0]}')
        p, y = pooled(sub)
        b, st = calib(p, y)
        out[m] = (b, {'model': m, **st, 'sd': float(p.std()), 'slope': cal_slope(p, y)})
    return out


def write_ece_table(df):  # LaTeX rows for the appendix calibration table
    names = {MARKET: 'Betfair', **{m: lab for _, mods in FAMILIES for m, lab in mods}}
    tex = {'Weibull': 'Weibull', r'Weibull$_{\psi}$': r'$\wbp$', r'Weibull$_{\kappa,\psi}$': r'$\wbpc$',
           r'Maia$_{\psi}$': r'$\maiap$', r'Maia$_{\kappa,\psi}$': r'$\maiac$', r'Zou$_{\psi}$': r'$\zoup$',
           r'Zou$_{\kappa,\psi}$': r'$\zouc$', r'Poisson$_\kappa$': r'$\poik$',
           r'Poisson$_{\kappa,t}$': r'$\poikt$', r'Naive$_\kappa$': r'$\naivek$', 'Betfair': 'Betfair'}
    lines = [f"{tex[names[r.model]]} & {r.brier:.4f} & {r.reliability:.5f} & {r.resolution:.4f} & "
             f"{r.ece:.3f} & {r.slope:.2f} \\\\" for r in df.itertuples()]
    with open(os.path.join(OUT, f'{NAME}_ece_table.tex'), 'w') as f:
        f.write('\n'.join(lines[1:] + ['\\addlinespace', lines[0]]) + '\n')   # Betfair last


def plot_reliability_all(cal):  # one panel per model family, Betfair as reference
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.8), sharex=True, sharey=True)
    bb, sb = cal[MARKET]
    for i, (a, (fam, mods)) in enumerate(zip(ax.ravel(), FAMILIES)):
        a.plot([0, 1], [0, 1], ls=':', lw=0.8, color='black', alpha=0.6)
        a.plot(bb['p_mean'], bb['obs_freq'], color='#bdbdbd', lw=2.5, label=f"Betfair ({sb['ece']:.3f})")
        for (m, lab), ls, mk in zip(mods, ['-', '--', '-.'], ['o', 's', '^']):
            b, st = cal[m]
            a.plot(b['p_mean'], b['obs_freq'], ls=ls, marker=mk, ms=3.5, lw=1.1, color='#1b9e77',
                   mec='black', mew=0.4, label=f"{lab} ({st['ece']:.3f})")
        a.legend(frameon=False, loc='upper left', fontsize=7); a.grid(alpha=0.2)
        a.set(xlim=(0, 1), ylim=(0, 1))
        a.text(-0.14, 1.03, 'ABCD'[i], transform=a.transAxes, fontsize=10, fontweight='bold')
        a.text(0.97, 0.04, fam, transform=a.transAxes, ha='right', fontsize=8)
    for a in ax[1]: a.set_xlabel('Forecast probability')
    for a in ax[:, 0]: a.set_ylabel('Observed frequency')
    fig.tight_layout(); save(fig, 'paper_reliability_all')


COLS = {'payout': '#7570b3', 'kelly': '#d95f02'}   # same colours as eval/bet.py bet_pnl
LABS = {'payout': 'Fixed payout', 'kelly': 'Kelly'}


def plot_paper(cal, pay, kelly, brk, mpnl, rbm):
    fig, a = plt.subplots(figsize=(3.6, 3.4))   # reliability, ECE quoted in Sections 5 and 6
    a.plot([0, 1], [0, 1], ls='--', lw=0.8, color='black', alpha=0.6)
    for mdl, col, mk in [(MODEL, '#1b9e77', 'o'), (MARKET, 'black', 's')]:
        b, st = cal[mdl]
        a.plot(b['p_mean'], b['obs_freq'], marker=mk, ms=4, lw=1.2, color=col, mec='black',
               mew=0.5, label=f"{DISP[mdl]} (ECE {st['ece']:.3f})")
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel='Forecast probability', ylabel='Observed frequency')
    a.legend(frameon=False, loc='upper left'); a.grid(alpha=0.2)
    fig.tight_layout(); save(fig, 'paper_reliability')

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))   # edges + net ROI by subgroup
    ax = ax.ravel()
    ax[0].hist(pay['edge'], bins=np.arange(0, 1.0001, 0.02), color='#bdbdbd', edgecolor='black', lw=0.4)
    ax[0].set(yscale='log', xlabel='Edge (model minus market probability)', ylabel='Bets')
    groups = [('score_state', ['<= -2', '-1', '0', '1', '>= 2'], ['≤−2', '−1', '0', '1', '≥2'],
               'Home goal difference'),
              ('odds_band', ['0-1.5', '1.5-3', '3-6', '6-15', '>15'],
               ['<1.5', '1.5–3', '3–6', '6–15', '>15'], 'Decimal odds'),
              ('minute_bucket', [bucket_label(lo) for lo, _ in MIN_BUCKETS],
               ['0–14', '15–29', '30–44', '45–59', '60–74', '75–89', '90+'], 'Match minute')]
    for a, (dim, order, ticks, xl) in zip(ax[1:], groups):
        x = np.arange(len(order))
        for j, s in enumerate(['payout', 'kelly']):
            sub = brk[(brk['strategy'] == s) & (brk['dimension'] == dim)].set_index('level').loc[order]
            a.bar(x + (j - 0.5) * 0.38, sub['net_roi'], 0.38, color=COLS[s], edgecolor='black',
                  lw=0.5, label=LABS[s])
        a.axhline(0, color='black', lw=0.8)
        a.set_xticks(x); a.set_xticklabels(ticks)
        a.set(xlabel=xl, ylabel='Net ROI (%)')
    ax[1].legend(frameon=False)
    for i, a in enumerate(ax):
        a.grid(alpha=0.2)
        a.text(-0.17, 1.03, 'ABCD'[i], transform=a.transAxes, fontsize=10, fontweight='bold')
    fig.tight_layout(); save(fig, 'paper_breakdown')

    # fragility: A cumulative profit, B ROI without the top-k matches, C per-match histogram, D matches ranked by profit
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax = ax.ravel()
    for s, b, key in [('payout', pay, 'pay_bets'), ('kelly', kelly, 'kelly_bets')]:
        cum = np.cumsum(_net_stream(rbm, key, COMMISSION))   # identical to eval/bet.py bet_pnl
        if not np.isclose(cum[-1], mpnl[s].sum()):
            raise ValueError(f'{s}: cumulative net {cum[-1]:.2f} != match net {mpnl[s].sum():.2f}')
        ax[0].plot(cum, lw=1.2, color=COLS[s], label=LABS[s])
        net = mpnl[s]
        stk, top, ks = b.groupby('match_id')['stake'].sum().loc[net.index], net.sort_values(ascending=False).index, range(21)
        roi = [(net.sum() - net[top[:k]].sum()) / (stk.sum() - stk[top[:k]].sum()) * 100 for k in ks]
        ax[1].plot(ks, roi, marker='o', ms=3, lw=1.2, color=COLS[s], mec='black', mew=0.4)
        ax[2].hist(net, bins=np.arange(-60, 61, 4), color=COLS[s], edgecolor='black', lw=0.4, alpha=0.6)
        v = np.sort(net.values)[::-1]
        x = np.arange(1, len(v) + 1)   # 280 bars: per-bar edges would hide the fill, so outline the profile
        ax[3].bar(x, v, 1.0, color=COLS[s], edgecolor='none', alpha=0.75 if s == 'payout' else 0.9)
        ax[3].step(x, v, where='mid', color='black', lw=0.5)
    ax[0].set(xlabel='Bet number', ylabel='Cumulative net profit (units)'); ax[0].legend(frameon=False)
    ax[1].set(xlabel='Most profitable matches removed', ylabel='Net ROI (%)', xticks=range(0, 21, 5))
    ax[2].set(xlabel='Net profit per match (units)', ylabel='Matches')
    ax[3].set(xlabel='Match (ranked by net profit)', ylabel='Net profit per match (units)')
    for i, a in enumerate(ax):
        a.axhline(0, color='black', ls='--', lw=0.8, alpha=0.5); a.grid(alpha=0.2)
        a.text(-0.17, 1.03, 'ABCD'[i], transform=a.transAxes, fontsize=10, fontweight='bold')
    fig.tight_layout(); save(fig, 'paper_fragile')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. BET EXTRACTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_sim(df, model, minutes):
    """eval/bet.py's per-minute simulation loop, model vs the Betfair benchmark."""
    rbm = {}
    for m in minutes:
        r = simulate_betting(df[(df.model == model) & (df.clock == m)],
                             df[(df.model == MARKET) & (df.clock == m)])
        if r is None:
            raise ValueError(f'no merge at minute {m} for {model}')
        rbm[m] = r
    return rbm


def flatten(results_by_minute):
    """Flatten simulate_betting output into fixed-payout/kelly bet frames."""
    pay, kelly = [], []
    for minute, results in results_by_minute.items():
        for r in results:
            base = {'match_id': r['match_id'], 'minute': minute, 'date': r['date'],
                    'actual': r['actual']}
            for key, out in (('pay_bets', pay), ('kelly_bets', kelly)):
                out += [{**base, 'outcome': OUTCOMES[b['outcome']], 'stake': b['stake'],
                         'odds': b['odds'], 'edge': b['edge'], 'ev': b['ev'],
                         'won': bool(b['won']), 'gross_pnl': b['gross_pnl']} for b in r[key]]
    return pd.DataFrame(pay), pd.DataFrame(kelly)


def net_roi(bets, commission=COMMISSION):
    """Subgroup net PnL/ROI: commission recomputed on this subgroup's per-match net."""
    if len(bets) == 0:
        raise ValueError('net_roi called on an empty subgroup — no silent defaults')
    g = bets.groupby('match_id')['gross_pnl'].sum()
    comm = float(np.maximum(0, g).sum() * commission)
    gross, staked = float(bets['gross_pnl'].sum()), float(bets['stake'].sum())
    return {'bets': len(bets), 'staked': staked, 'gross': gross, 'comm': comm,
            'net': gross - comm, 'gross_roi': gross / staked * 100,
            'net_roi': (gross - comm) / staked * 100, 'win_rate': bets['won'].mean()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MATCH-CLUSTERED BOOTSTRAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def boot_roi(bets, all_matches, label, n_boot, seed=SEED):
    """Resample the matches with replacement; percentile CI for net ROI."""
    g = bets.groupby('match_id').agg(gross=('gross_pnl', 'sum'), stake=('stake', 'sum'))
    g = g.reindex(all_matches).fillna(0.0)                    # matches with no bets contribute 0
    net = (g['gross'] - np.maximum(0, g['gross']) * COMMISSION).values
    stake = g['stake'].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(net), size=(n_boot, len(net)))
    ns, ss = net[idx].sum(1), stake[idx].sum(1)
    roi = np.where(ss > 0, ns / np.where(ss > 0, ss, 1) * 100, np.nan)
    lo, hi = np.nanpercentile(roi, [2.5, 97.5])
    return {'strategy': label, 'bets': len(bets), 'staked': stake.sum(),
            'net': net.sum(), 'roi_pct': net.sum() / stake.sum() * 100,
            'ci_lo': float(lo), 'ci_hi': float(hi), 'zero_in_ci': bool(lo <= 0 <= hi),
            'p_boot_le0': float(np.nanmean(roi <= 0)), 'boot_sd': float(np.nanstd(roi))}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run(df_all, bench, n_boot):
    """Full diagnostic pipeline; returns {csv key: frame}."""
    out = {}
    # every evaluation point, first-half stoppage included, keyed by the chronological clock
    df_pt = df_all.assign(clock=clock(df_all['period'], df_all['eval_minute']))
    console.print(f"[cyan]rows per model: {len(df_pt[df_pt.model == MODEL])}, "
                  f"matches: {df_pt.match_id.nunique()}[/]")

    # ---- 1. calibration (Figures bet_reliability and reliability_all, appendix table) ----
    cal = run_calibration(pd.concat([df_all, bench], ignore_index=True),
                          [MARKET] + [m for _, mods in FAMILIES for m, _ in mods])
    ece = pd.DataFrame([st for _, st in cal.values()])
    out['ece_all'] = ece
    out['calibration_bins'] = pd.concat(
        [cal[m][0].assign(model=m) for m in [MODEL, MARKET]], ignore_index=True)
    write_ece_table(ece)
    t = Table(title='Calibration, pooled H/D/A, equal-width bins')
    for c in ['Model', 'N', 'ECE', 'MCE', 'Brier', 'Rel', 'Res', 'Unc', 'SD', 'Slope']:
        t.add_column(c, justify='right' if c != 'Model' else 'left')
    for _, r in ece.iterrows():
        t.add_row(r['model'], f"{r['n_points']:,}", f"{r['ece']:.4f}", f"{r['mce']:.4f}",
                  f"{r['brier']:.4f}", f"{r['reliability']:.5f}", f"{r['resolution']:.4f}",
                  f"{r['uncertainty']:.4f}", f"{r['sd']:.4f}", f"{r['slope']:.2f}")
    console.print(t)
    plot_reliability_all(cal)

    # ---- 2. simulation (identical to eval/bet.py defaults) ----
    minutes = sorted(df_pt['clock'].unique())
    rbm = run_sim(df_pt, MODEL, minutes)
    meta = df_pt[df_pt.model == MODEL][['match_id', 'clock', 'score_diff']].rename(
        columns={'clock': 'minute'})

    def attach(d):  # add score_diff + minute bucket, failing loudly on any miss
        n0 = len(d)
        d = d.merge(meta, on=['match_id', 'minute'], how='left', validate='m:1')
        if d['score_diff'].isna().any() or len(d) != n0:
            raise ValueError('score_diff merge failed')
        d['mbucket'] = d['minute'].map(bucket_label)
        return d

    pay, kelly = (attach(x) for x in flatten(rbm))
    matches = np.sort(df_pt['match_id'].unique())

    # ---- 3. match-clustered bootstrap CIs for net ROI (Section 6) ----
    bt = pd.DataFrame([boot_roi(pay, matches, 'Payout (all bets)', n_boot),
                       boot_roi(kelly, matches, 'Kelly (all bets)', n_boot)])
    out['bootstrap'] = bt
    t = Table(title=f'Match-clustered bootstrap ({n_boot:,} resamples of {len(matches)} matches, seed {SEED})')
    for c in ['Strategy', 'Bets', 'Staked', 'Net', 'ROI %', '95% CI', '0 in CI', 'P(ROI<=0)']:
        t.add_column(c, justify='right' if c != 'Strategy' else 'left')
    for _, r in bt.iterrows():
        t.add_row(r['strategy'], f"{r['bets']:,}", f"{r['staked']:.0f}", f"{r['net']:.2f}",
                  f"{r['roi_pct']:.2f}", f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]",
                  '[red]yes[/]' if r['zero_in_ci'] else '[green]no[/]', f"{r['p_boot_le0']:.4f}")
    console.print(t)

    # ---- 4. profit concentration across matches (Figure bet_fragile) ----
    conc_rows, mpnl = [], {}
    for lab, b in [('payout', pay), ('kelly', kelly)]:
        g = b.groupby('match_id')['gross_pnl'].sum()
        net = g - np.maximum(0, g) * COMMISSION
        mpnl[lab] = net
        s = net.sort_values(ascending=False)
        row = {'strategy': lab, 'n_matches': len(net), 'total_net': net.sum(),
               'pct_matches_profitable': float((net > 0).mean() * 100),
               'median_match_net': float(net.median()),
               'best_match_net': float(s.iloc[0]), 'worst_match_net': float(s.iloc[-1])}
        for k in DROP_TOP:  # in-sample leave-out-best sensitivity (descriptive, biased down)
            row[f'roi_excl_top{k}_pct'] = net_roi(b[~b['match_id'].isin(s.index[:k])])['net_roi']
        conc_rows.append(row)
    conc = pd.DataFrame(conc_rows)
    out['concentration'] = conc
    out['match_pnl'] = pd.DataFrame({'kelly_net': mpnl['kelly'], 'pay_net': mpnl['payout']}).reset_index()
    t = Table(title='Profit concentration (match-level net PnL)')
    for c in ['Strategy', 'Total net', '% matches +', 'Median', 'Best', 'Worst'] + \
             [f'ROI ex-top{k}' for k in DROP_TOP]:
        t.add_column(c, justify='right' if c != 'Strategy' else 'left')
    for _, r in conc.iterrows():
        t.add_row(r['strategy'], f"{r['total_net']:.2f}", f"{r['pct_matches_profitable']:.1f}",
                  f"{r['median_match_net']:.3f}", f"{r['best_match_net']:.2f}",
                  f"{r['worst_match_net']:.2f}",
                  *[f"{r[f'roi_excl_top{k}_pct']:.2f}" for k in DROP_TOP])
    console.print(t)

    # ---- 5. ROI breakdown by state, price and minute (Figure bet_breakdown) ----
    def score_state(sd):
        return '<= -2' if sd <= -2 else ('-1' if sd == -1 else ('0' if sd == 0 else ('1' if sd == 1 else '>= 2')))

    def odds_band(o):
        for lo, hi in ODDS_BANDS:
            if lo <= o < hi:
                return f'{lo}-{hi}' if np.isfinite(hi) else '>15'
        raise ValueError(o)

    brk = []
    for lab, b in [('payout', pay), ('kelly', kelly)]:
        b = b.copy()
        b['sstate'] = b['score_diff'].map(score_state)
        b['oband'] = b['odds'].map(odds_band)
        for dim, col, order in [('minute_bucket', 'mbucket', [bucket_label(lo) for lo, _ in MIN_BUCKETS]),
                                ('score_state', 'sstate', ['<= -2', '-1', '0', '1', '>= 2']),
                                ('odds_band', 'oband', ['0-1.5', '1.5-3', '3-6', '6-15', '>15'])]:
            for lvl in order:
                brk.append({'strategy': lab, 'dimension': dim, 'level': lvl,
                            **net_roi(b[b[col] == lvl])})
    brk = pd.DataFrame(brk)
    out['breakdown'] = brk
    for lab in ['payout', 'kelly']:
        t = Table(title=f'ROI breakdown — {lab} staking (net ROI recomputes commission within subgroup)')
        for c in ['Dimension', 'Level', 'Bets', 'Staked', 'Win%', 'Gross', 'Net', 'Gross ROI %', 'Net ROI %']:
            t.add_column(c, justify='right' if c not in ('Dimension', 'Level') else 'left')
        for _, r in brk[brk['strategy'] == lab].iterrows():
            t.add_row(r['dimension'], r['level'], f"{r['bets']:,}", f"{r['staked']:.0f}",
                      f"{r['win_rate']:.2f}", f"{r['gross']:.2f}", f"{r['net']:.2f}",
                      f"{r['gross_roi']:.2f}", f"{r['net_roi']:.2f}")
        console.print(t)

    plot_paper(cal, pay, kelly, brk, mpnl, rbm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='B=200')
    args = ap.parse_args()
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    df_all = pd.read_parquet(os.path.join(REPO, 'eval', 'predictions.parquet'))
    bench = pd.read_parquet(os.path.join(OUT, 'benchmark_preds.parquet'))
    for key, df in run(df_all, bench, 200 if args.smoke else N_BOOT).items():
        df.to_csv(os.path.join(OUT, f'{NAME}_{key}.csv'), index=False)
    console.print(f"\n[bold green]CSVs written to {OUT}[/]")


if __name__ == '__main__':
    main()
