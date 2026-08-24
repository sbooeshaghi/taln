"""Build the main tokenizer-comparison figure from frozen Stage 1 outputs.

Layout:
  Top row: contiguous and non-contiguous alignment schematics.
  Bottom row: held-out one-gap recovery and localization by tokenizer for
  BOAT (a) and BIO-BOAT (b), taln, document-clustered 95% CIs.

Data source: data/revision_2026/stage1/heldout_test/tokenizer_by_method_table.tsv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = (
    REPO_ROOT
    / "data"
    / "revision_2026"
    / "stage1"
    / "heldout_test"
    / "tokenizer_by_method_table.tsv"
)
OUTPUT = REPO_ROOT / "docs" / "shared" / "figures" / "tokenizer_comparison.pdf"

GREEN = "#2ca02c"
SKIP = "#f5c96a"

TOKENIZER_ORDER = [
    ("whitespace", "Word"),
    ("boundary_stripped_whitespace", "Word\n(stripped)"),
    ("punctuation", "Punctuation"),
    ("cl100k_base", "cl100k_base"),
    ("scibert", "SciBERT"),
    ("pubmedbert", "PubMedBERT"),
]
HIGHLIGHT = "punctuation"
DATASETS = [("boat", "BOAT"), ("bioboat", "BIO-BOAT")]


def draw_box(ax, x, y, label, face="white"):
    box = FancyBboxPatch(
        (x - 0.3, y - 0.28),
        0.6,
        0.56,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=face,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=11)


def draw_schematic(ax, title, skipped_index=None):
    """Target row mapped to source row; optionally skip one source token."""
    n_target = 5
    n_source = n_target + (1 if skipped_index is not None else 0)
    target_y, source_y = 1.6, 0.0

    ax.plot([-0.5, n_target - 0.5], [target_y, target_y], color="black", lw=1.5, zorder=0)
    ax.plot([-0.5, n_source - 0.5], [source_y, source_y], color="black", lw=1.5, zorder=0)

    source_positions = list(range(n_source))
    for i in range(n_target):
        draw_box(ax, i, target_y, str(i))
    for j in source_positions:
        face = SKIP if skipped_index is not None and j == skipped_index else "white"
        draw_box(ax, j, source_y, f"j+{j}" if j else "j", face=face)

    target_index = 0
    for j in source_positions:
        if skipped_index is not None and j == skipped_index:
            continue
        if target_index >= n_target:
            break
        ax.plot(
            [target_index, j],
            [target_y - 0.3, source_y + 0.3],
            linestyle="--",
            color="black",
            lw=1,
        )
        target_index += 1

    for x in (-0.85, n_source - 0.15):
        ax.text(x, source_y, r"$\bullet\!\bullet\!\bullet$", ha="center", va="center", fontsize=9)
    ax.text(-2.4, target_y, "Target", ha="left", va="center", fontsize=13)
    ax.text(-2.4, source_y, "Source", ha="left", va="center", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlim(-2.6, n_source + 0.6)
    ax.set_ylim(-0.7, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    plt.rcParams.update({"font.size": 15})

    rows = [
        row
        for row in csv.DictReader(TABLE.open(), delimiter="\t")
        if row["method"] == "taln"
        and row["condition"] == "one_gap"
        and row["stratum"] == "all"
    ]
    by_key = {(row["dataset"], row["tokenizer"]): row for row in rows}

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12, 7.6),
        gridspec_kw={"height_ratios": [1.0, 1.6]},
    )

    draw_schematic(axs[0, 0], "Contiguous")
    draw_schematic(axs[0, 1], "Non-contiguous", skipped_index=3)

    width = 0.38
    for ax, (dataset, title) in zip(axs[1], DATASETS):
        recon, recon_err = [], [[], []]
        loc, loc_err = [], [[], []]
        recon_colors, loc_colors = [], []
        for tokenizer, _ in TOKENIZER_ORDER:
            row = by_key[(dataset, tokenizer)]
            r = float(row["support_rate"]) * 100
            recon.append(r)
            recon_err[0].append(r - float(row["support_ci_lower"]) * 100)
            recon_err[1].append(float(row["support_ci_upper"]) * 100 - r)
            l = float(row["localization_rate"]) * 100
            loc.append(l)
            loc_err[0].append(l - float(row["localization_ci_lower"]) * 100)
            loc_err[1].append(float(row["localization_ci_upper"]) * 100 - l)
            color = GREEN if tokenizer == HIGHLIGHT else "white"
            recon_colors.append(color)
            loc_colors.append(color)
        positions = np.arange(len(TOKENIZER_ORDER))
        ax.bar(
            positions - width / 2,
            recon,
            width,
            facecolor=recon_colors,
            edgecolor="black",
            yerr=recon_err,
            capsize=2,
            error_kw={"linewidth": 1},
            label="Recovery",
        )
        ax.bar(
            positions + width / 2,
            loc,
            width,
            facecolor=loc_colors,
            edgecolor="black",
            hatch="///",
            yerr=loc_err,
            capsize=2,
            error_kw={"linewidth": 1},
            label="Localization",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [label for _, label in TOKENIZER_ORDER],
            fontsize=11,
            rotation=25,
            ha="right",
        )
        ax.set_title(title, fontsize=15)
        ax.set_ylim(0, 102)
        ax.spines[["top", "right"]].set_visible(False)
    axs[1, 0].set_ylabel("Accuracy (%)")
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=12,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.03),
    )

    for ax, letter in zip(axs[1], "ab"):
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
