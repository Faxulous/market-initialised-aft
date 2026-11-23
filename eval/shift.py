"""Market-initialisation shifts for the first evaluation gameweek: figure and per-team table.

Run: python eval/shift.py -> eval/acc/pdf/calibration_shift.pdf, eval/acc/shift.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console

# Journal figure style: sans-serif lettering, ~8pt
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'dejavusans',
    'font.size':         9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / 'models' / 'weibull' / 'calibrated' / 'match_calibration_24_25_weibull_calibd.csv'
OUT = ROOT / 'eval' / 'acc'
CAL_GW_END = "2024-11-10"  # GW11, first eval gameweek
NAMES = {"Man Utd": "Manchester United", "Man City": "Manchester City"}  # match the paper prose
console = Console()


def plot_calibration_shift(cal_df: pd.DataFrame, output_path: str):
    """Paired diverging bar chart: delta-eta per team per match.
    
    Paper palette (purple/orange), direct labels, ordered by total absolute shift.
    Following Wilke (Fundamentals of Data Visualization) principles.
    """
    COL_HOME, COL_AWAY = '#7570b3', '#d95f02'  # same purple/orange as the betting figures

    df = cal_df.copy()
    df[['home_name', 'away_name']] = df[['home_name', 'away_name']].replace(NAMES)
    df['dh'] = df['eta_h_cal'] - df['eta_h_model']
    df['da'] = df['eta_a_cal'] - df['eta_a_model']
    df['abs_total'] = df['dh'].abs() + df['da'].abs()
    df = df.sort_values('abs_total').reset_index(drop=True)  # smallest at top

    n = len(df)
    y = np.arange(n)
    h = 0.32
    pad = 0.04  # text padding from bar end

    fig, ax = plt.subplots(figsize=(8, 0.8 * n + 1))

    ax.barh(y - h/2, df['dh'], height=h, color=COL_HOME,
            edgecolor='black', linewidth=0.4, zorder=3)
    ax.barh(y + h/2, df['da'], height=h, color=COL_AWAY,
            edgecolor='black', linewidth=0.4, zorder=3)

    # Direct labels: team name at zero-line side, value at bar tip
    xlim = max(df['dh'].abs().max(), df['da'].abs().max()) * 1.35
    for i, (_, r) in enumerate(df.iterrows()):
        for name, delta, ypos in [
            (r.home_name, r.dh, i - h/2),
            (r.away_name, r.da, i + h/2),
        ]:
            # Team name anchored at zero-line
            ha_name = 'right' if delta >= 0 else 'left'
            x_name = -pad if delta >= 0 else pad
            ax.text(x_name, ypos, name, ha=ha_name, va='center', fontsize=10)
            # Numeric value at bar tip
            ha_val = 'left' if delta >= 0 else 'right'
            x_val = delta + (pad if delta >= 0 else -pad)
            ax.text(x_val, ypos, f'{delta:+.2f}', ha=ha_val, va='center',
                    fontsize=10, color='#333333')

    ax.axvline(0, color='black', lw=0.8, zorder=4)
    ax.set_xlim(-xlim, xlim)
    ax.set_yticks([])
    ax.set_xlabel(r'Shift in $\eta$ (log-expected scoring time)', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', color='#dddddd', linewidth=0.5, zorder=0)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Minimal legend: stacked coloured text, centred above plot
    ax.text(0.5, 1.05, 'Home', color=COL_HOME, fontsize=10, fontweight='bold',
            transform=ax.transAxes, ha='center')
    ax.text(0.5, 1.01, 'Away', color=COL_AWAY, fontsize=10, fontweight='bold',
            transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    gw = pd.read_csv(CAL).query('date <= @CAL_GW_END').reset_index(drop=True)
    plot_calibration_shift(gw, OUT / 'pdf' / 'calibration_shift.pdf')
    rows = pd.concat([gw[['date', f'{s}_name', f'eta_{s[0]}_model', f'eta_{s[0]}_cal']].set_axis(['date', 'team', 'eta_model', 'eta_cal'], axis=1).assign(side=s)
                      for s in ('home', 'away')]).replace({'team': NAMES})
    rows['shift'] = rows.eta_cal - rows.eta_model
    rows['minutes'] = np.exp(rows.eta_cal) - np.exp(rows.eta_model)   # change in expected time to score
    rows.sort_values('shift', key=abs, ascending=False).to_csv(OUT / 'shift.csv', index=False)
    console.print(f"{len(gw)} matches | max |shift| {rows['shift'].abs().max():.2f} | {(rows['shift'] > 0).sum()} of {len(rows)} positive")


if __name__ == '__main__':
    main()
