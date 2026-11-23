"""Unified evaluation script for football outcome prediction.

Run: python eval/acc.py [--sf N]. --sf prints the tables to N significant figures
(default: 3 dp accuracy, 4 dp RPS/log-loss, as in the paper). Includes all models and market baselines.
Outputs under eval/acc/: summary.csv and pdf/logloss.pdf
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Journal figure style: sans-serif lettering, ~8pt
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset':  'dejavusans',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
})

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval.utils import BENCHMARKS, BENCH_NAMES, load_predictions

console = Console()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVAL_DIR = os.path.dirname(__file__)
PRED_PATH = os.path.join(EVAL_DIR, 'predictions.parquet')
OUT = os.path.join(EVAL_DIR, 'acc')
OUT_PDF = os.path.join(OUT, 'pdf')
# Paper display names and colours
PAPER_NAMES = {  # Match paper: kappa = market-initialised, psi = shots covariate
    "weibull_no_cov": "Weibull",
    "weibull":        r"Weibull$_\psi$",
    "weibull_calibd": r"Weibull$_{\kappa,\psi}$",
    "zou":            r"Zou$_\psi$",            # Zou variants include the shots covariate, as the Maia ones
    "zou_calibd":     r"Zou$_{\kappa,\psi}$",
    "maia":           r"Maia$_\psi$",            # Maia variants always include the shots covariate
    "maia_calibd":    r"Maia$_{\kappa,\psi}$",
    "betfair":        "Betfair",
    **BENCH_NAMES,
}
PAPER_COLORS = {
    "weibull_no_cov": "#d95f02", "weibull": "#1b9e77", "weibull_calibd": "#7570b3",
    "zou": "#66a61e", "zou_calibd": "#a6d854", "maia": "#e6ab02", "maia_calibd": "#fee08b",
    "betfair": "#e7298a",
    "mkt_poisson": "#444444", "mkt_poisson_t": "#888888", "mkt_naive": "#a65628",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rps(p_h: np.ndarray, p_d: np.ndarray, p_a: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Rank Probability Score. Lower = better, range [0, 1]."""
    c1, c2 = p_h, p_h + p_d
    e1, e2 = (actual == 0).astype(float), (actual <= 1).astype(float)
    return 0.5 * ((c1 - e1) ** 2 + (c2 - e2) ** 2)


def logloss(p_h: np.ndarray, p_d: np.ndarray, p_a: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Log-loss. Lower = better."""
    p_act = np.where(actual == 0, p_h, np.where(actual == 1, p_d, p_a))
    return -np.log(np.clip(p_act, 1e-15, 1 - 1e-15))


def accuracy(p_h: np.ndarray, p_d: np.ndarray, p_a: np.ndarray, actual: np.ndarray) -> float:
    """Classification accuracy."""
    pred = np.argmax(np.stack([p_h, p_d, p_a], axis=1), axis=1)
    return (pred == actual).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def filter_consistent(df: pd.DataFrame, model: str) -> set:
    """Get match IDs with predictions at >=80% of eval minutes."""
    sub = df[df["model"] == model]
    n_mins = sub["eval_minute"].nunique()
    counts = sub.groupby("match_id")["eval_minute"].nunique()
    return set(counts[counts >= int(n_mins * 0.8)].index)


def primary_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to primary timeline: period 1 for minutes <45, period 2 for >=45.
    Excludes H1 stoppage (period 1, minute >=45) to avoid overlap with 2H."""
    if "period" not in df.columns:
        return df
    return df[~((df["period"] == 1) & (df["eval_minute"] >= 45))]


def get_models(df: pd.DataFrame) -> list:
    return sorted(df["model"].unique().tolist())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_model(df: pd.DataFrame, model: str) -> dict:
    """Evaluate model. Returns dict with n, accuracy, rps, logloss."""
    sub = df[df["model"] == model]
    if len(sub) == 0:
        return {"n": 0, "accuracy": 0, "rps": 0, "logloss": 0}
    p_h, p_d, p_a = sub["pred_home"].values, sub["pred_draw"].values, sub["pred_away"].values
    act = sub["actual"].values
    return {
        "n": len(sub), "accuracy": accuracy(p_h, p_d, p_a, act),
        "rps": rps(p_h, p_d, p_a, act).mean(), "logloss": logloss(p_h, p_d, p_a, act).mean(),
    }


def eval_by_minute_period(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Evaluate by (minute, period) on primary timeline only."""
    sub = primary_timeline(df[df["model"] == model])
    if "period" not in sub.columns:
        sub = sub.copy()
        sub["period"] = np.where(sub["eval_minute"] < 45, 1, 2)

    consistent = filter_consistent(df, model)
    sub = sub[sub["match_id"].isin(consistent)]

    results = []
    for (minute, period), grp in sub.groupby(["eval_minute", "period"]):
        p_h, p_d, p_a = grp["pred_home"].values, grp["pred_draw"].values, grp["pred_away"].values
        act = grp["actual"].values
        results.append({
            "minute": minute, "period": period, "n": len(grp),
            "accuracy": accuracy(p_h, p_d, p_a, act), "rps": rps(p_h, p_d, p_a, act).mean(),
            "logloss": logloss(p_h, p_d, p_a, act).mean(),
        })
    return pd.DataFrame(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABLE PRINTING (RICH)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _print_metrics_table(df: pd.DataFrame, models: list, title: str, sf: int | None = None):
    """Print Acc/RPS/LogLoss table. Green = best excl. betfair, bold = also beats betfair."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col, kw in [("Model", {"style": "bold"}), ("N", {}), ("Acc", {}), ("RPS", {}), ("LogLoss", {})]:
        table.add_column(col, justify="right" if col != "Model" else "left", **kw)

    results = {m: eval_model(df, m) for m in models}
    non_bf = [m for m in models if m != "betfair" and results[m]["n"] > 0]
    bf = results.get("betfair", {"n": 0})
    best = {
        "accuracy": max((results[m]["accuracy"] for m in non_bf), default=None),
        "rps": min((results[m]["rps"] for m in non_bf), default=None),
        "logloss": min((results[m]["logloss"] for m in non_bf), default=None),
    }

    def fmt(val, metric, prec):
        s = f"{val:#.{sf}g}" if sf else f"{val:.{prec}f}"   # '#' keeps trailing zeros at sf sig. figs
        if best[metric] is None: return s
        is_best = abs(val - best[metric]) < 1e-9
        beats_bf = bf["n"] > 0 and ((metric == "accuracy" and val > bf[metric]) or
                                     (metric != "accuracy" and val < bf[metric]))
        if is_best and beats_bf: return f"[bold green]{s}[/]"
        return f"[green]{s}[/]" if is_best else s

    for m in models:
        r = results[m]
        if r["n"] > 0:
            table.add_row(m, str(r["n"]), fmt(r["accuracy"], "accuracy", 3),
                          fmt(r["rps"], "rps", 4), fmt(r["logloss"], "logloss", 4))

    # Reference baselines from Robberechts et al. (2021), KDD
    # 100 equally-spaced frames, 8 seasons, top-5 leagues
    if title == "Summary":
        table.add_section()
        table.add_row("[dim]† Robberechts (offline)[/]", "[dim]—[/]", "[dim]—[/]",
                      "[dim]0.1338[/]", "[dim]—[/]")
        table.add_row("[dim]† Robberechts (real-time)[/]", "[dim]—[/]", "[dim]—[/]",
                      "[dim]0.1380[/]", "[dim]—[/]")

    console.print(table)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_logloss_combined(df, models, ref="betfair", name="logloss"):
    """Paper-ready: 2x2 grid split by half (1H | 2H), absolute top, diff bottom."""
    results = {m: eval_by_minute_period(df, m).sort_values("minute") for m in models}
    ref_r = results[ref]
    styles = {m: "-" for m in models}
    styles.update(zip(BENCHMARKS, ["--", "-.", ":"]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), height_ratios=[1, 2],
                              gridspec_kw={"hspace": 0.05, "wspace": 0.02,
                                           "width_ratios": [45, 50]})
    h1_range, h2_range = (0, 44), (45, 95)  # Minute ranges per half

    for col, (lo, hi) in enumerate([(h1_range), (h2_range)]):
        ax_abs, ax_diff = axes[0, col], axes[1, col]
        mask_fn = lambda r: r[(r["minute"] >= lo) & (r["minute"] <= hi)]

        # Top: absolute log-loss
        for m in models:
            r = mask_fn(results[m])
            ax_abs.plot(r["minute"], r["logloss"], ls=styles[m], color=PAPER_COLORS[m],
                        label=PAPER_NAMES[m], lw=1.2)
        ax_abs.grid(True, alpha=0.2)
        ax_abs.set_xlim(lo - 1, hi + 1)

        # Bottom: diff vs Betfair
        for m in [m for m in models if m != ref]:
            merged = results[m].merge(ref_r[["minute", "logloss"]], on="minute", suffixes=("", "_ref"))
            merged = merged[(merged["minute"] >= lo) & (merged["minute"] <= hi)]
            diff = merged["logloss"] - merged["logloss_ref"]
            ax_diff.plot(merged["minute"], diff, ls=styles[m], color=PAPER_COLORS[m],
                         label=PAPER_NAMES[m], lw=1.2)
        ax_diff.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax_diff.set_xlim(lo - 1, hi + 1)
        ax_diff.grid(True, alpha=0.2)
        ax_diff.set_xlabel("First half" if col == 0 else "Second half")

    # Shared y-limits within each row
    for row in range(2):
        ylims = [axes[row, c].get_ylim() for c in range(2)]
        shared = (min(y[0] for y in ylims), max(y[1] for y in ylims))
        for c in range(2):
            axes[row, c].set_ylim(shared)

    # Labels and legend
    axes[0, 0].set_ylabel("Log-loss")
    axes[1, 0].set_ylabel("$\\Delta$ Log-loss vs Betfair")
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    axes[0, 1].tick_params(labelleft=False)
    axes[1, 1].tick_params(labelleft=False)
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False)
    if len(models) > 5:
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    else:
        axes[0, 1].legend(loc="upper right", fontsize=7, ncol=1)

    fig.subplots_adjust(left=0.08, right=0.97,
                        top=0.81 if len(models) > 8 else 0.87 if len(models) > 5 else 0.97,
                        bottom=0.09)
    os.makedirs(OUT_PDF, exist_ok=True)
    plt.savefig(os.path.join(OUT_PDF, name + ".pdf"), dpi=300)
    console.print(f"[dim]Saved: acc/pdf/{name}.pdf[/]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(sf: int | None = None):
    df = load_predictions(PRED_PATH, benchmarks=True)
    models = get_models(df)
    console.print(Panel(f"[bold]{len(models) - 1} models + Betfair[/] | "
                        f"{df['match_id'].nunique()} matches | {len(df):,} predictions\n"
                        f"Models: {', '.join(models)}",
                        title="Data Loaded", border_style="blue"))
    rows = []
    for view, data in [('minute0', df[df['eval_minute'] == 0]), ('all', df)]:
        results = _print_metrics_table(data, models, view, sf)
        rows.extend({'view': view, 'model': m, **r} for m, r in results.items())
    plot_logloss_combined(df, models)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    console.print('[dim]Saved: acc/summary.csv[/]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sf', type=int, help='significant figures in the printed tables (default: paper precision)')
    main(ap.parse_args().sf)
