"""Build the three-panel selection-assessment figure from held-out outputs.

Panels:
  a. Distribution of unique source matches per fully supported target.
  b. Top-1 localization per frozen selection rule with 95% CIs.
  c. Top-1 localization of the selected rule as a function of enumeration cap.

Data source: data/revision_2026/selection/heldout_test/ (records + summary),
pooled primary non-contiguous strata, punctuation-aware tokenizer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_DIR = REPO_ROOT / "data" / "revision_2026" / "selection" / "heldout_test"
OUTPUT = REPO_ROOT / "docs" / "shared" / "figures" / "selection_assessment.pdf"

GREEN = "#2ca02c"
GRAY = "#7f7f7f"

RULE_ORDER = [
    ("input_order", "Enumeration\norder"),
    ("earliest_interval", "Earliest\nposition"),
    ("shortest_interval", "Shortest\nmatch"),
    ("fewest_gap_characters", "Fewest gap\ncharacters"),
    ("matched_token_density", "Highest\ndensity"),
]
SELECTED_RULE = "shortest_interval"


def main() -> None:
    plt.rcParams.update({"font.size": 15})

    summary = json.loads(
        (SELECTION_DIR / "selection_heldout_summary.json").read_text()
    )
    pooled = summary["pooled_primary_noncontiguous"]["punctuation"]

    unique_counts = Counter()
    for line in (SELECTION_DIR / "selection_records.jsonl").open():
        record = json.loads(line)
        if not record["primary_noncontiguous"]:
            continue
        result = record["results"]["punctuation"]
        if not result["full_lexical_support"]:
            continue
        unique_counts[result["unique_interval_count"]] += 1

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    # a. unique source matches per target
    ax = axs[0]
    bins = [1, 2, 3, 4, 5, 6, 11, 10**9]
    labels = ["1", "2", "3", "4", "5", "6-10", ">10"]
    heights = [
        sum(count for value, count in unique_counts.items() if low <= value < high)
        for low, high in zip(bins[:-1], bins[1:])
    ]
    ax.bar(range(len(labels)), heights, facecolor="white", edgecolor="black")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10))
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Unique source matches")
    ax.set_ylabel("Targets")

    # b. top-1 localization per rule
    ax = axs[1]
    rates, err_low, err_high, colors = [], [], [], []
    for rule_id, _ in RULE_ORDER:
        block = pooled["rules"][rule_id]["top1"]
        rate = block["rate"] * 100
        low, high = (value * 100 for value in block["ci95"])
        rates.append(rate)
        err_low.append(rate - low)
        err_high.append(high - rate)
        colors.append(GREEN if rule_id == SELECTED_RULE else "white")
    positions = range(len(RULE_ORDER))
    ax.bar(
        positions,
        rates,
        facecolor=colors,
        edgecolor="black",
        yerr=[err_low, err_high],
        capsize=3,
        error_kw={"linewidth": 1},
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [label.replace("\n", " ") for _, label in RULE_ORDER],
        fontsize=11,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Top-1 localization (%)")
    ax.set_ylim(0, 102)

    # c. cap sweep for the selected rule
    ax = axs[2]
    caps = [1, 10, 100, 1000, 10000]
    denominator = pooled["any_rank_localization"]["n"]
    cap_rates = [
        pooled["cap_sweep"][str(cap)]["retained_top1_by_rule"][SELECTED_RULE]
        / denominator
        * 100
        for cap in caps
    ]
    full_rate = pooled["rules"][SELECTED_RULE]["top1"]["rate"] * 100
    ax.plot(
        caps + [10**5],
        cap_rates + [full_rate],
        marker="o",
        color="black",
        clip_on=False,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Enumeration cap (alignments)")
    ax.set_ylabel("Top-1 localization (%)")
    ax.set_ylim(70, 101)

    for ax, letter in zip(axs, "abc"):
        ax.text(
            -0.24,
            1.12,
            f"{letter}.",
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
        )
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(f"wrote {OUTPUT}")
    print("panel a heights:", dict(zip(labels, heights)))
    print("panel b rates:", [round(rate, 2) for rate in rates])
    print("panel c rates:", [round(rate, 2) for rate in cap_rates + [full_rate]])


if __name__ == "__main__":
    main()
