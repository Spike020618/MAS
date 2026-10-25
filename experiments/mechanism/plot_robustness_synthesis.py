"""Render the frozen extension experiments as one dependency-free SVG."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


TEAL = "#2D7F7A"; GREY = "#7B8794"; ORANGE = "#D47A2C"; NAVY = "#17324D"


def txt(x, y, value, size=13, anchor="start", weight="normal", color=NAVY):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{color}">{html.escape(str(value))}</text>')


def panel(body, x, title):
    body += [f'<rect x="{x}" y="65" width="365" height="360" rx="5" fill="#F8FAFB" stroke="#DCE3E8"/>',
             txt(x+182, 94, title, 16, "middle", "bold")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))["suites"]
    body = [txt(800, 38, "Robustness, heterogeneity, and boundary evidence", 22, "middle", "bold")]
    xs = [25, 420, 815, 1210]
    for x, title in zip(xs, ("A. Incentive response", "B. Information environments",
                             "C. Perturbation stability", "D. Model boundary")):
        panel(body, x, title)

    boundary = data["boundary_v11_dev"]["contrasts"]
    items = [("Reward 0.05→0.50", boundary[0]["high_effort"]["delta"]*100, boundary[0]["decision_quality"]["delta"]*100),
             ("Cost 0.5→1.5", boundary[1]["high_effort"]["delta"]*100, boundary[1]["decision_quality"]["delta"]*100)]
    for i, (name, effort, acc) in enumerate(items):
        y = 150 + i*120
        body += [txt(50, y, name, 13, weight="bold"),
                 txt(50, y+35, f"Action response  {effort:+.2f} pp", 18, weight="bold", color=TEAL if effort>0 else ORANGE),
                 txt(50, y+65, f"Accuracy  {acc:+.2f} pp", 14),
                 txt(50, y+88, "Accuracy contrast not significant after Holm", 10, color="#657786")]

    info = data["information_v11_dev"]["contrasts"]
    for i, row in enumerate(info):
        y = 135 + i*62
        name = row["name"].split("_")[0].capitalize()
        effort = row["high_effort"]["delta"]*100; acc = row["decision_quality"]["delta"]*100
        body += [txt(445, y, name, 12, weight="bold"),
                 f'<rect x="535" y="{y-14}" width="{effort*3.2:.1f}" height="16" fill="{TEAL}"/>',
                 txt(540+effort*3.2, y, f"{effort:+.1f} pp effort", 10),
                 txt(445, y+22, f"Accuracy {acc:+.1f} pp", 11, color="#657786")]

    suites = [("Panel 3–5", "missing_agent_v11_dev"), ("AEIC fields", "signal_ablation_v11_dev"),
              ("History", "history_ablation_v11_dev"), ("Extra seeds", "seed_robustness_v11_dev")]
    for i, (name, suite) in enumerate(suites):
        conds = data[suite]["conditions"]
        accs = [v["accuracy"]*100 for v in conds.values()]
        y = 140 + i*64
        body += [txt(840, y, name, 12, weight="bold"),
                 txt(965, y, f"{min(accs):.1f}–{max(accs):.1f}%", 17, weight="bold", color=TEAL),
                 txt(965, y+20, "observed accuracy range", 10, color="#657786")]
    body += [txt(840, 395, "Field ablations do not establish component necessity.", 10, color="#657786")]

    glm = data["heldout_glm_v11_dev"]
    vals = list(glm["conditions"].values())
    labels = ["No mechanism", "Process contract"]
    for i, (name, row) in enumerate(zip(labels, vals)):
        x = 1270+i*145; value=row["accuracy"]*100; h=(value-50)*6; y=350-h
        body += [f'<rect x="{x}" y="{y}" width="75" height="{h}" fill="{GREY if i==0 else ORANGE}"/>',
                 txt(x+37, y-9, f"{value:.1f}%", 15, "middle", "bold"), txt(x+37, 375, name, 11, "middle")]
    contrast=glm["contrasts"][0]
    body += [txt(1392, 125, f"Effort Δ {contrast['high_effort']['delta']*100:+.2f} pp", 14, "middle", "bold"),
             txt(1392, 148, f"Accuracy Δ {contrast['decision_quality']['delta']*100:+.2f} pp", 14, "middle"),
             txt(1392, 171, "No significant transfer", 11, "middle", color="#657786")]

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="455" viewBox="0 0 1600 455">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, Helvetica, sans-serif">', *body, '</g>', '</svg>']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg)+"\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
