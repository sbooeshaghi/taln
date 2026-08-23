"""Build supplementary S1/S2 grids from frozen Stage 1 held-out outputs.

S1 (reconstruction_accuracy.pdf) and S2 (localization_accuracy.pdf): full
method-by-tokenizer grids on held-out BOAT and BIO-BOAT, contiguous and
one-gap conditions, document-clustered 95% CIs.

Data source: data/revision_2026/stage1/heldout_test/tokenizer_by_method_table.tsv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    REPO_ROOT
    / "data"
    / "revision_2026"
    / "stage1"
    / "heldout_test"
    / "tokenizer_by_method_table.tsv"
)
FIGDIR = REPO_ROOT / "docs" / "shared" / "figures"

TOKENIZER_ORDER = [
    ("whitespace", "Word"),
    ("boundary_stripped_whitespace", "Word\n(stripped)"),
    ("punctuation", "Punctuation"),
    ("cl100k_base", "cl100k_base"),
    ("scibert", "SciBERT"),
    ("pubmedbert", "PubMedBERT"),
]
METHODS = [
    ("exact", "Exact", "#d62728"),
    ("taln", "taln", "#2ca02c"),
    ("lcs", "LCS", "#1f77b4"),
    ("semi_global_exact", "Semi-global", "#9467bd"),
    ("difflib", "Difflib", "#ff7f0e"),
]
PANELS = [
    ("boat", "contiguous", "BOAT contiguous"),
    ("bioboat", "contiguous", "BIO-BOAT contiguous"),
    ("boat", "one_gap", "BOAT one-gap"),
    ("bioboat", "one_gap", "BIO-BOAT one-gap"),
]


def build(metric: str, output_name: str) -> None:
    rate_key = "support_rate" if metric == "support" else "localization_rate"
    low_key = f"{rate_key.rsplit('_', 1)[0]}_ci_lower"
    high_key = f"{rate_key.rsplit('_', 1)[0]}_ci_upper"

    rows = [
        row
        for row in csv.DictReader(TABLE.open(), delimiter="\t")
        if row["stratum"] == "all"
    ]
    by_key = {
        (row["dataset"], row["condition"], row["tokenizer"], row["method"]): row
        for row in rows
    }

    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    width = 0.15
    positions = np.arange(len(TOKENIZER_ORDER))
    for ax, (dataset, condition, title) in zip(axs.flat, PANELS):
        for offset, (method, label, color) in enumerate(METHODS):
            rates, err_low, err_high = [], [], []
            for tokenizer, _ in TOKENIZER_ORDER:
                # Exact string matching does not depend on the tokenizer; its
                # single frozen row is shown at every tokenizer position.
                lookup = "character_exact" if method == "exact" else tokenizer
                row = by_key.get((dataset, condition, lookup, method))
                if row is None:
                    rates.append(0.0)
                    err_low.append(0.0)
                    err_high.append(0.0)
                    continue
                rate = float(row[rate_key]) * 100
                rates.append(rate)
                err_low.append(max(0.0, rate - float(row[low_key]) * 100))
                err_high.append(max(0.0, float(row[high_key]) * 100 - rate))
            ax.bar(
                positions + (offset - 2) * width,
                rates,
                width,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                yerr=[err_low, err_high],
                capsize=1.5,
                error_kw={"linewidth": 0.8},
                label=label,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [label for _, label in TOKENIZER_ORDER],
            fontsize=10,
            rotation=25,
            ha="right",
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 102)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel(
            "Reconstruction (%)" if metric == "support" else "Localization (%)"
        )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=12,
        frameon=False,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.02),
    )
    for ax, letter in zip(axs.flat, "abcd"):
        ax.text(
            -0.08,
            1.08,
            f"{letter}.",
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout()
    output = FIGDIR / output_name
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    plt.rcParams.update({"font.size": 14})
    build("support", "reconstruction_accuracy.pdf")
    build("localization", "localization_accuracy.pdf")


if __name__ == "__main__":
    main()
