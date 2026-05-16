"""Generate gsm8k_comparison.png — the headline visual for the falsification.

Four conditions side-by-side on GSM8K-50:
    0-shot cold | K=5 fine-tune | 5-shot ICL | K=5 ft + 5-shot ICL

This is the chart that tells the actual story of the session: our fine-tune
is dominated by ICL at the same K, and is neutral when composed with ICL.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gsm8k_comparison.png"

conditions = [
    "0-shot cold\n(no training,\nno demos)",
    "K=5 fine-tune\n(0-shot eval)",
    "5-shot ICL\n(no training,\n5 demos in prompt)",
    "K=5 fine-tune\n+ 5-shot ICL",
]
pass_rates = [32.0, 44.0, 96.0, 96.0]
labels = ["16/50", "22/50", "48/50", "48/50"]
colors = ["#888888", "#4a90c2", "#5cb85c", "#7d6dc0"]

fig, ax = plt.subplots(figsize=(12, 7.5))

bars = ax.bar(conditions, pass_rates, color=colors, edgecolor="black", linewidth=0.8, width=0.65)

for bar, lbl, pct in zip(bars, labels, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{pct:.0f}%\n{lbl}", ha="center", va="bottom", fontsize=12, fontweight="bold")

# Delta annotations placed below the bars (won't collide)
ax.text(0.5, 24, "Δ +12 pp", ha="center", fontsize=11,
        color="#2c5f88", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#dde9f4", ec="#4a90c2", lw=1))
ax.annotate("", xy=(1, 22), xytext=(0, 22),
            arrowprops=dict(arrowstyle="-|>", color="#4a90c2", lw=1.5))

ax.text(1.5, 14, "Δ +64 pp (5-shot ICL beats fine-tune by 52pp)", ha="center", fontsize=11,
        color="#1d6a1d", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#dceedc", ec="#5cb85c", lw=1))
ax.annotate("", xy=(2, 12), xytext=(0, 12),
            arrowprops=dict(arrowstyle="-|>", color="#5cb85c", lw=1.5))

ax.text(2.5, 6, "Δ 0 pp (composition adds nothing)", ha="center", fontsize=11,
        color="#503e7c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#e4dff0", ec="#7d6dc0", lw=1))
ax.annotate("", xy=(3, 4), xytext=(2, 4),
            arrowprops=dict(arrowstyle="-|>", color="#7d6dc0", lw=1.5))

ax.set_ylabel("GSM8K pass rate (%, n=50 held-out)", fontsize=12)
ax.set_title("autoresearch/may16 — GSM8K sample-efficiency comparison\n"
             "Qwen3-8B-bnb-4bit · identical eval config (temp=0, max_tokens=3000, enable_thinking=true)",
             fontsize=12, pad=15)
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=10)

# Footer annotation
fig.text(0.5, 0.02,
         "Verdict: 5-shot ICL is a strictly stronger sample-efficient baseline at this K. "
         "Session goal 'novel SOTA in sample-efficient learning' — NOT MET. "
         "What was built: a continual-learning recipe with persistent params + no inference-time prompt overhead.",
         ha="center", fontsize=9.5, style="italic", color="#444444", wrap=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"wrote {OUT}", flush=True)
