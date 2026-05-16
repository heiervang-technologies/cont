"""
Autoresearch progress analysis and plotting.

Reads results.tsv and autoresearch.toml, generates a progress plot.

Usage:
    python analysis.py              # plot from results.tsv
    python analysis.py results.tsv  # plot from a specific file
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="autoresearch.toml"):
    """Parse autoresearch.toml without requiring toml library."""
    config = {"name": "score", "direction": "minimize"}
    p = Path(path)
    if not p.exists():
        return config
    for line in p.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().strip('"')
        value = value.strip().strip('"')
        if key in ("name", "direction"):
            config[key] = value
    return config

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tsv_path = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
    config = load_config()
    metric_name = config["name"]
    minimize = config["direction"] == "minimize"

    # Load data
    df = pd.read_csv(tsv_path, sep="\t")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["status"] = df["status"].str.strip().str.upper()

    # Print summary
    counts = df["status"].value_counts()
    n_keep = counts.get("KEEP", 0)
    n_discard = counts.get("DISCARD", 0)
    n_crash = counts.get("CRASH", 0)
    print(f"Total experiments: {len(df)}")
    print(f"  Kept: {n_keep}  Discarded: {n_discard}  Crashed: {n_crash}")

    n_decided = n_keep + n_discard
    if n_decided > 0:
        print(f"  Keep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")

    # Filter out crashes for plotting
    valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
    if valid.empty:
        print("No valid experiments to plot.")
        return

    baseline_score = valid.loc[0, "score"]

    # Kept and discarded subsets
    kept_v = valid[valid["status"] == "KEEP"]
    disc_v = valid[valid["status"] == "DISCARD"]

    # Only plot points near the interesting region
    if minimize:
        near = valid[valid["score"] <= baseline_score * 1.001 + 0.0005]
    else:
        near = valid[valid["score"] >= baseline_score * 0.999 - 0.0005]
    disc_near = near[near["status"] == "DISCARD"]

    # Running best line
    kept_mask = valid["status"] == "KEEP"
    kept_idx = valid.index[kept_mask]
    kept_scores = valid.loc[kept_mask, "score"]
    running_best = kept_scores.cummin() if minimize else kept_scores.cummax()

    best_score = running_best.iloc[-1] if len(running_best) > 0 else baseline_score

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.scatter(disc_near.index, disc_near["score"],
               c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

    ax.scatter(kept_v.index, kept_v["score"],
               c="#2ecc71", s=50, zorder=4, label="Kept",
               edgecolors="black", linewidths=0.5)

    if len(kept_idx) > 0:
        ax.step(kept_idx, running_best, where="post", color="#27ae60",
                linewidth=2, alpha=0.7, zorder=3, label="Running best")

    # Label kept experiments
    for idx, score in zip(kept_idx, kept_scores):
        desc = str(valid.loc[idx, "description"]).strip()
        if len(desc) > 45:
            desc = desc[:42] + "..."
        ax.annotate(desc, (idx, score),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8.0, color="#1a7a3a", alpha=0.9,
                    rotation=30, ha="left", va="bottom")

    direction_label = "lower is better" if minimize else "higher is better"
    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(f"{metric_name} ({direction_label})", fontsize=12)
    ax.set_title(f"Autoresearch Progress: {len(df)} Experiments, {n_keep} Kept Improvements",
                 fontsize=14)
    ax.legend(loc="upper right" if minimize else "lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Y-axis limits
    margin = abs(baseline_score - best_score) * 0.15 if baseline_score != best_score else 0.01
    if minimize:
        ax.set_ylim(best_score - margin, baseline_score + margin)
    else:
        ax.set_ylim(baseline_score - margin, best_score + margin)

    plt.tight_layout()
    plt.savefig("progress.png", dpi=150, bbox_inches="tight")
    print("Saved progress.png")


if __name__ == "__main__":
    main()
