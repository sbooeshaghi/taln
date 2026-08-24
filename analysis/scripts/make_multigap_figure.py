"""Build the held-out Stage 2 multi-gap stress-test figure.

The figure deliberately separates full-target lexical support from candidate
burden. It reads only the aggregate held-out summary and writes standalone
TikZ, so no notebook state enters the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "data/revision_2026/stage2/heldout_test/multigap_benchmark_summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs/revision_2026/figures/stage2_multigap_stress_test.tex"
)

DATASETS = (("boat", "BOAT"), ("bioboat", "BIO-BOAT"))
WIDTH_COLORS = {1: "widthOne", 2: "widthTwo", 3: "widthThree"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("complete") or summary.get("splits") != ["heldout_test"]:
        raise ValueError("Figure requires a complete heldout_test-only summary")
    return summary


def group_key(dataset: str, gap_count: int, gap_width: int, method: str) -> str:
    return "::".join(
        (
            "heldout_test",
            dataset,
            "primary_noncontiguous",
            str(gap_count),
            str(gap_width),
            "punctuation",
            method,
        )
    )


def support_series(
    summary: dict[str, Any], dataset: str, method: str, gap_width: int
) -> list[float]:
    return [
        summary["groups"][group_key(dataset, gap_count, gap_width, method)][
            "full_lexical_support"
        ]["estimate"]
        for gap_count in (1, 2, 3)
    ]


def candidate_series(
    summary: dict[str, Any], dataset: str, gap_width: int
) -> list[float]:
    return [
        summary["groups"][group_key(dataset, gap_count, gap_width, "taln")][
            "candidate_burden"
        ]["p95_candidate_count"]
        for gap_count in (1, 2, 3)
    ]


def point(
    index: int,
    value: float,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float]:
    xx = x + width * index / 2
    yy = y + height * (value - y_min) / (y_max - y_min)
    return xx, yy


def draw_axes(
    lines: list[str],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    y_ticks: list[tuple[float, str]],
    y_label: str,
    title: str,
) -> None:
    lines.append(rf"\node[font=\small\bfseries] at ({x + width / 2:.2f},{y + height + 0.32:.2f}) {{{title}}};")
    lines.append(rf"\draw[black!55] ({x:.2f},{y:.2f}) -- ({x + width:.2f},{y:.2f});")
    lines.append(rf"\draw[black!55] ({x:.2f},{y:.2f}) -- ({x:.2f},{y + height:.2f});")
    for value, label in y_ticks:
        yy = y + height * (value - y_min) / (y_max - y_min)
        lines.append(rf"\draw[black!12] ({x:.2f},{yy:.2f}) -- ({x + width:.2f},{yy:.2f});")
        lines.append(rf"\node[anchor=east,font=\scriptsize] at ({x - 0.09:.2f},{yy:.2f}) {{{label}}};")
    for index, label in enumerate(("1", "2", "3")):
        xx = x + width * index / 2
        lines.append(rf"\draw[black!55] ({xx:.2f},{y:.2f}) -- ({xx:.2f},{y - 0.06:.2f});")
        lines.append(rf"\node[anchor=north,font=\scriptsize] at ({xx:.2f},{y - 0.10:.2f}) {{{label}}};")
    lines.append(rf"\node[anchor=north,font=\scriptsize] at ({x + width / 2:.2f},{y - 0.43:.2f}) {{Number of gaps}};")
    lines.append(
        rf"\node[rotate=90,font=\scriptsize] at ({x - 0.72:.2f},{y + height / 2:.2f}) {{{y_label}}};"
    )


def draw_series(
    lines: list[str],
    values: list[float],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
    color: str,
    dashed: bool = False,
    marker: str = "*",
) -> None:
    points = [
        point(
            index,
            value,
            x=x,
            y=y,
            width=width,
            height=height,
            y_min=y_min,
            y_max=y_max,
        )
        for index, value in enumerate(values)
    ]
    style = "densely dashed," if dashed else ""
    coords = " -- ".join(f"({xx:.2f},{yy:.2f})" for xx, yy in points)
    lines.append(rf"\draw[{style}color={color},line width=0.85pt] {coords};")
    for xx, yy in points:
        if marker == "square":
            lines.append(
                rf"\filldraw[fill=white,draw={color},line width=0.75pt] "
                rf"({xx - 0.055:.3f},{yy - 0.055:.3f}) rectangle "
                rf"({xx + 0.055:.3f},{yy + 0.055:.3f});"
            )
        else:
            lines.append(rf"\fill[{color}] ({xx:.2f},{yy:.2f}) circle (0.055);")


def legend_item(
    lines: list[str], *, x: float, y: float, color: str, label: str, dashed: bool = False
) -> None:
    style = "densely dashed," if dashed else ""
    lines.append(rf"\draw[{style}color={color},line width=0.85pt] ({x:.2f},{y:.2f}) -- ({x + 0.34:.2f},{y:.2f});")
    lines.append(rf"\fill[{color}] ({x + 0.17:.2f},{y:.2f}) circle (0.045);")
    lines.append(rf"\node[anchor=west,font=\scriptsize] at ({x + 0.43:.2f},{y:.2f}) {{{label}}};")


def build_figure(summary: dict[str, Any]) -> str:
    lines = [
        r"\documentclass[tikz,border=4pt]{standalone}",
        r"\usepackage{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\definecolor{talnGreen}{HTML}{2F7D4A}",
        r"\definecolor{widthOne}{HTML}{3973AC}",
        r"\definecolor{widthTwo}{HTML}{D07A28}",
        r"\definecolor{widthThree}{HTML}{B44949}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        r"\node[anchor=west,font=\bfseries] at (0.00,10.15) {a. Full-target lexical support};",
    ]

    legend_item(lines, x=0.95, y=9.78, color="talnGreen", label=r"\texttt{taln}", dashed=True)
    legend_item(lines, x=3.05, y=9.78, color="widthOne", label=r"\texttt{difflib}, width 1")
    legend_item(lines, x=6.18, y=9.78, color="widthTwo", label=r"\texttt{difflib}, width 2")
    legend_item(lines, x=9.30, y=9.78, color="widthThree", label=r"\texttt{difflib}, width 3")

    panel_x = (1.00, 7.20)
    for (dataset, label), x in zip(DATASETS, panel_x, strict=True):
        draw_axes(
            lines,
            x=x,
            y=6.45,
            width=4.85,
            height=2.55,
            y_min=0.75,
            y_max=1.01,
            y_ticks=[(0.8, "80"), (0.9, "90"), (1.0, "100")],
            y_label="Support (percent)",
            title=label,
        )
        taln_values = support_series(summary, dataset, "taln", 1)
        for gap_width in (2, 3):
            if support_series(summary, dataset, "taln", gap_width) != taln_values:
                raise ValueError(f"taln support unexpectedly varies by width for {dataset}")
        draw_series(
            lines,
            taln_values,
            x=x,
            y=6.45,
            width=4.85,
            height=2.55,
            y_min=0.75,
            y_max=1.01,
            color="talnGreen",
            dashed=True,
        )
        for gap_width in (1, 2, 3):
            draw_series(
                lines,
                support_series(summary, dataset, "difflib", gap_width),
                x=x,
                y=6.45,
                width=4.85,
                height=2.55,
                y_min=0.75,
                y_max=1.01,
                color=WIDTH_COLORS[gap_width],
            )

    lines.append(r"\node[anchor=west,font=\bfseries] at (0.00,5.42) {b. Candidate burden for \texttt{taln}};")
    legend_item(lines, x=2.55, y=5.03, color="widthOne", label="Width 1")
    legend_item(lines, x=5.00, y=5.03, color="widthTwo", label="Width 2")
    legend_item(lines, x=7.45, y=5.03, color="widthThree", label="Width 3")

    for (dataset, label), x in zip(DATASETS, panel_x, strict=True):
        draw_axes(
            lines,
            x=x,
            y=1.65,
            width=4.85,
            height=2.55,
            y_min=0.0,
            y_max=40.0,
            y_ticks=[(0.0, "0"), (20.0, "20"), (40.0, "40")],
            y_label="95th-percentile candidates",
            title=label,
        )
        for gap_width in (1, 2, 3):
            draw_series(
                lines,
                candidate_series(summary, dataset, gap_width),
                x=x,
                y=1.65,
                width=4.85,
                height=2.55,
                y_min=0.0,
                y_max=40.0,
                color=WIDTH_COLORS[gap_width],
                marker="square",
            )

    lines.extend(
        [
            r"\node[anchor=west,font=\tiny,text=black!70] at (0.05,0.45) {Held-out test; punctuation tokenizer. Width is the number of whitespace chunks removed per gap.};",
            r"\node[anchor=west,font=\tiny,text=black!70] at (0.05,0.15) {Candidate burden is computed among full-support \texttt{taln} tasks; no held-out task reached the 100,000-candidate cap.};",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_figure(summary), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
