"""Build the multi-gap figure with taln, difflib, and the LLM as lines.

Panels a (BOAT) and b (BIO-BOAT): held-out reconstruction by gap condition for
taln, difflib (punctuation-aware tokenization), and the LLM, all evaluated on
the identical document-clustered extension sample, with document-clustered
95% CIs for the LLM and method rates from the frozen Stage 2 records.

Data source: data/revision_2026/llm_baseline/heldout_multigap/llm_baseline_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_SUMMARY = (
    REPO_ROOT
    / "data"
    / "revision_2026"
    / "llm_baseline"
    / "heldout_multigap"
    / "llm_baseline_summary.json"
)
OUTPUT = REPO_ROOT / "docs" / "shared" / "figures" / "multigap_llm.pdf"

GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
BLUE = "#1f77b4"

CONDITIONS = [
    ("1", "1"), ("1", "2"), ("1", "3"),
    ("2", "1"), ("2", "2"), ("2", "3"),
    ("3", "1"), ("3", "2"), ("3", "3"),
]
DATASETS = [("boat", "BOAT"), ("bioboat", "BIO-BOAT")]


def main() -> None:
    plt.rcParams.update({"font.size": 15})
    summary = json.loads(LLM_SUMMARY.read_text())

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    positions = np.arange(len(CONDITIONS))
    for ax, (dataset, title) in zip(axs, DATASETS):
        blocks = [
            summary["strata"][f"stage2:{dataset}:gaps{g}_width{w}"]
            for g, w in CONDITIONS
        ]

        for method, label, color in (
            ("taln", "taln", GREEN),
            ("difflib", "Difflib", ORANGE),
        ):
            rates, err_low, err_high = [], [], []
            for block in blocks:
                r = block["methods"][method]["reconstruction"]
                rate = r["rate"] * 100
                rates.append(rate)
                err_low.append(rate - r["ci95"][0] * 100)
                err_high.append(r["ci95"][1] * 100 - rate)
            ax.errorbar(
                positions,
                rates,
                yerr=[err_low, err_high],
                marker="o",
                markersize=5,
                color=color,
                capsize=2,
                linewidth=1.5,
                label=label,
            )

        rates, err_low, err_high = [], [], []
        for block in blocks:
            r = block["llm"]["reconstruction"]
            rate = r["rate"] * 100
            rates.append(rate)
            err_low.append(rate - r["ci95"][0] * 100)
            err_high.append(r["ci95"][1] * 100 - rate)
        ax.errorbar(
            positions,
            rates,
            yerr=[err_low, err_high],
            marker="s",
            markersize=5,
            color=BLUE,
            capsize=2,
            linewidth=1.5,
            label="LLM",
        )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"{g}\u00d7{w}" for g, w in CONDITIONS], fontsize=11)
        ax.set_xlabel("Gaps \u00d7 gap width")
        ax.set_title(title, fontsize=15)
        ax.set_ylim(65, 102)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("Reconstruction (%)")
    axs[0].legend(fontsize=12, frameon=False, loc="lower left")

    for ax, letter in zip(axs, "ab"):
        ax.text(
            -0.12,
            1.10,
            f"{letter}.",
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
