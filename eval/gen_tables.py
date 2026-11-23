"""Generate LaTeX tables for the paper. Run: python eval/gen_tables.py"""
import argparse
import pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.utils import BENCHMARKS, BENCH_NAMES, load_predictions

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--benchmarks", action="store_true", help="Include saved market baselines")
args = p.parse_args()
df = load_predictions(os.path.join(os.path.dirname(__file__), "predictions.parquet"),
                      args.benchmarks)
p_act = np.where(df["actual"] == 0, df["pred_home"], np.where(df["actual"] == 1, df["pred_draw"], df["pred_away"]))
df["ll"] = -np.log(np.clip(p_act, 1e-15, 1 - 1e-15))
df["rps"] = 0.5 * (
    (df["pred_home"] - (df["actual"] == 0).astype(float)) ** 2
    + (df["pred_home"] + df["pred_draw"] - (df["actual"] <= 1).astype(float)) ** 2
)
# Primary timeline: period 1 for <45, period 2 for >=45 (excludes H1 stoppage overlap)
df_pt = df[~((df["period"] == 1) & (df["eval_minute"] >= 45))]
N = f"$N = {df['match_id'].nunique()}$ matches"

# Pre-match table: W+, W+_cal, Betfair (matches paper tab:prematch)
PRE_CODES = ["weibull", "weibull_calibd", "betfair"]
PRE_NAMES = ["Weibull+", "Weibull+*", "Betfair"]
# Main tables: all models
CODES = ["weibull_no_cov", "weibull", "weibull_calibd", "zou", "zou_calibd", "maia", "maia_calibd", "betfair"]
NAMES = ["Weibull", "Weibull+", "Weibull+*", "Zou", "Zou*", "Maia", "Maia*", "Betfair"]
if args.benchmarks:
    PRE_CODES[-1:-1] = BENCHMARKS
    CODES[-1:-1] = BENCHMARKS
    PRE_NAMES[-1:-1] = NAMES[-1:-1] = [BENCH_NAMES[b] for b in BENCHMARKS]


def bold_best(vals, fmt, lower_better=True):
    best = min(vals) if lower_better else max(vals)
    return [f"\\textbf{{{v:{fmt}}}}" if abs(v - best) < 1e-6 else f"{v:{fmt}}" for v in vals]


def model_metrics(sub):
    """Return (accuracy, rps, logloss) for a subset."""
    preds = np.argmax(np.stack([sub["pred_home"].values, sub["pred_draw"].values, sub["pred_away"].values], axis=1), axis=1)
    return (sub["actual"].values == preds).mean(), sub["rps"].mean(), sub["ll"].mean()


def print_table(codes, names, data, caption, label):
    """Print a summary table (Accuracy, RPS, Log-loss) for given models and data."""
    rows = [model_metrics(data[data["model"] == c]) for c in codes]
    accs, rpss, lls = zip(*rows)
    acc_f = bold_best(accs, ".3f", lower_better=False)
    rps_f = bold_best(rpss, ".4f")
    ll_f = bold_best(lls, ".4f")
    print(f"\\begin{{table}}[tb]")
    print(f"\\centering\\small")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print(f"\\begin{{tabular}}{{lrrr}}")
    print(f"\\toprule")
    print(f"Model & Accuracy & RPS & Log-loss \\\\")
    print(f"\\midrule")
    for name, a, r, l in zip(names, acc_f, rps_f, ll_f):
        print(f"{name} & {a} & {r} & {l} \\\\")
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")
    print(f"\\end{{table}}")
    print()


# ===== PRE-MATCH TABLE =====
print("% ===== PRE-MATCH TABLE =====")
print_table(PRE_CODES, PRE_NAMES, df[df["eval_minute"] == 0],
            f"Pre-match prediction performance (minute 0). {N}.",
            "tab:prematch")

# ===== SUMMARY TABLE =====
print("% ===== SUMMARY TABLE =====")
print_table(CODES, NAMES, df,
            "Aggregate predictive performance across all evaluation minutes. "
            f"{N}, evaluated every minute (0--90 + stoppage).",
            "tab:summary")

# ===== LOG-LOSS BY MINUTE TABLE =====
print("% ===== LOG-LOSS BY MINUTE TABLE =====")
minutes = [int(m) for m in sorted(df_pt["eval_minute"].unique()) if m % 10 == 0]
pivot = df_pt[df_pt["model"].isin(CODES)].groupby(["eval_minute", "model"])["ll"].mean().unstack()

print("\\begin{table}[tb]")
print("\\centering\\footnotesize")
print("\\caption{Log-loss at 10-minute intervals. Bold indicates best model at each minute.}")
print("\\label{tab:logloss_minute}")
print("\\begin{tabular}{l" + "r" * len(minutes) + "}")
print("\\toprule")
print("Model & " + " & ".join(str(m) for m in minutes) + " \\\\")
print("\\midrule")

best_at = {m: pivot.loc[m, CODES].min() for m in minutes}  # Pre-compute best per minute
for code, name in zip(CODES, NAMES):
    cells = [f"\\textbf{{{pivot.loc[m, code]:.3f}}}" if abs(pivot.loc[m, code] - best_at[m]) < 1e-6
             else f"{pivot.loc[m, code]:.3f}" for m in minutes]
    print(f"{name} & " + " & ".join(cells) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
