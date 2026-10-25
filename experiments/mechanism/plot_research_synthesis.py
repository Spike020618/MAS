"""Create dependency-free SVG figures for the mechanism transmission chain."""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

NO = "independent__no_mechanism@0.2"
AEIC = "independent__static_aeic@0.2"
COLORS = {NO: "#7B8794", AEIC: "#2D7F7A"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def label(x, y, value, size=14, anchor="start", weight="normal", color="#17324D"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{color}">{html.escape(str(value))}</text>')


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, Helvetica, sans-serif">', *body, '</g>', '</svg>']
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def aggregate_actions(rows):
    totals = defaultdict(lambda: [0.0, 0])
    for row in rows:
        if row["condition"] in (NO, AEIC):
            totals[row["condition"]][0] += row["high_effort_rate"] * row["n"]
            totals[row["condition"]][1] += row["n"]
    return {key: value / n for key, (value, n) in totals.items()}


def synthesis_figure(base: Path, output: Path) -> None:
    chain = read(base / "transmission_chain.json")
    cost = read(base / "cost_effectiveness.json")
    flow = read(base / "error_flow.json")["counts"]
    effort = aggregate_actions(chain["actions"])
    body = [label(800, 38, "From public process rules to collective decision quality", 22, "middle", "bold")]
    panel_x = [35, 430, 825, 1220]
    for x in panel_x:
        body.append(f'<rect x="{x}" y="65" width="345" height="335" rx="5" fill="#F8FAFB" stroke="#DCE3E8"/>')
    body += [label(207, 95, "A. Behavioral response", 16, "middle", "bold")]
    values = [(NO, effort[NO] * 100, "No mechanism"), (AEIC, effort[AEIC] * 100, "Process contract")]
    for i, (condition, value, name) in enumerate(values):
        x = 100 + i * 145; height = value * 4.2; y = 335 - height
        body += [f'<rect x="{x}" y="{y:.1f}" width="75" height="{height:.1f}" fill="{COLORS[condition]}"/>',
                 label(x+37, y-9, f"{value:.2f}%", 15, "middle", "bold"), label(x+37, 365, name, 12, "middle")]
    body += [label(602, 95, "B. Task-level outcome", 16, "middle", "bold")]
    domains = [("MMLU", "MMLU-Hard"), ("Strategy", "StrategyQA"), ("Truthful", "TruthfulQA")]
    for i, (short, domain) in enumerate(domains):
        x = 470 + i * 103
        left = cost["by_dataset"][domain]["endpoints"]["correct"]["left_mean"] * 100
        right = cost["by_dataset"][domain]["endpoints"]["correct"]["right_mean"] * 100
        for j, (value, color) in enumerate(((left, COLORS[NO]), (right, COLORS[AEIC]))):
            height = (value - 45) * 4; bx = x + j * 31; by = 340 - height
            body.append(f'<rect x="{bx}" y="{by:.1f}" width="27" height="{height:.1f}" fill="{color}"/>')
        body += [label(x+29, 365, short, 11, "middle"), label(x+44, 325-(right-45)*4, f"{right:.1f}", 10, "middle")]
    body += [label(450, 385, "Final accuracy (axis starts at 45%)", 10, color="#657786")]
    body += [label(997, 95, "C. Error transition", 16, "middle", "bold")]
    transitions = [("Wrong → correct", flow["wrong_correct"], "#2D7F7A"), ("Correct → wrong", flow["correct_wrong"], "#B85450")]
    for i, (name, value, color) in enumerate(transitions):
        x = 875 + i * 145; height = value * 2.7; y = 335 - height
        body += [f'<rect x="{x}" y="{y}" width="80" height="{height}" fill="{color}"/>',
                 label(x+40, y-9, value, 16, "middle", "bold"), label(x+40, 365, name, 11, "middle")]
    body += [label(997, 135, "Net correction: +64 questions", 13, "middle", "bold")]
    body += [label(1392, 95, "D. Cost–benefit", 16, "middle", "bold")]
    metrics = [("+14.22 pp", "accuracy"), ("+2,182", "tokens / question"),
               ("15,343", "tokens / net correction"), ("5.64", "calls / net correction")]
    for i, (value, name) in enumerate(metrics):
        y = 145 + i * 65
        body += [label(1392, y, value, 22, "middle", "bold"), label(1392, y+21, name, 11, "middle", color="#657786")]
    write_svg(output, 1600, 430, body)


def trajectory_figure(base: Path, output: Path) -> None:
    rows = read(base / "round_trajectory.json")
    domains = ["MMLU-Hard", "StrategyQA", "TruthfulQA"]
    body = [label(750, 38, "Collective accuracy across deliberation rounds", 22, "middle", "bold")]
    for panel, domain in enumerate(domains):
        ox = 45 + panel * 490; oy = 85; width = 420; height = 280
        body += [label(ox+width/2, 70, domain, 16, "middle", "bold"),
                 f'<rect x="{ox}" y="{oy}" width="{width}" height="{height}" fill="#F8FAFB" stroke="#DCE3E8"/>']
        for value in (50, 60, 70, 80, 90, 100):
            y = oy + height - (value-50)/50*height
            body += [f'<line x1="{ox}" y1="{y}" x2="{ox+width}" y2="{y}" stroke="#DCE3E8"/>',
                     label(ox-8, y+4, value, 10, "end", color="#657786")]
        for condition in (NO, AEIC):
            selected = sorted((r for r in rows if r["dataset"] == domain and r["condition"] == condition), key=lambda r: r["round"])
            points = []
            for row in selected:
                x = ox + 70 + (row["round"]-1)*140
                y = oy + height - (row["majority_accuracy"]*100-50)/50*height
                points.append((x, y))
            point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            body.append(f'<polyline points="{point_string}" fill="none" stroke="{COLORS[condition]}" stroke-width="4"/>')
            for x, y in points:
                body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{COLORS[condition]}"/>')
        for round_id in (1, 2, 3):
            body.append(label(ox+70+(round_id-1)*140, oy+height+22, f"Round {round_id}", 11, "middle"))
    body += [f'<line x1="570" y1="405" x2="610" y2="405" stroke="{COLORS[NO]}" stroke-width="4"/>',
             label(620, 410, "No mechanism", 12),
             f'<line x1="770" y1="405" x2="810" y2="405" stroke="{COLORS[AEIC]}" stroke-width="4"/>',
             label(820, 410, "Process contract", 12)]
    write_svg(output, 1500, 440, body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    synthesis_figure(args.input_dir, args.output_dir / "mechanism_transmission_chain.svg")
    trajectory_figure(args.input_dir, args.output_dir / "round_accuracy_trajectory.svg")
    print(args.output_dir)


if __name__ == "__main__":
    main()
