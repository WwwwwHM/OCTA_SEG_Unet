import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class AttentionResult:
    rank: int
    attention: str
    dice_mean: float
    dice_std: float
    iou_mean: float
    iou_std: float
    acc_mean: float
    acc_std: float
    detail_path: str


LINE_PATTERN = re.compile(
    r"#(?P<rank>\d+)\s+(?P<att>\w+)\s+\|\s+"
    r"Dice:\s+(?P<dice_mean>\d+\.\d+)\s+±\s+(?P<dice_std>\d+\.\d+)\s+\|\s+"
    r"IoU:\s+(?P<iou_mean>\d+\.\d+)\s+±\s+(?P<iou_std>\d+\.\d+)\s+\|\s+"
    r"ACC:\s+(?P<acc_mean>\d+\.\d+)\s+±\s+(?P<acc_std>\d+\.\d+)\s+\|\s+"
    r"Detail:\s+(?P<detail>.+)$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot attention benchmark results for thesis-ready figures")
    parser.add_argument(
        "--ranking-file",
        type=str,
        default="eval_result/attention_benchmark_rank_20260227_1732.txt",
        help="Path to attention benchmark ranking txt file",
    )
    parser.add_argument("--output-dir", type=str, default="eval_result/figures")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_ranking_file(file_path: str) -> List[AttentionResult]:
    rows: List[AttentionResult] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            match = LINE_PATTERN.match(line)
            if not match:
                continue
            rows.append(
                AttentionResult(
                    rank=int(match.group("rank")),
                    attention=match.group("att"),
                    dice_mean=float(match.group("dice_mean")),
                    dice_std=float(match.group("dice_std")),
                    iou_mean=float(match.group("iou_mean")),
                    iou_std=float(match.group("iou_std")),
                    acc_mean=float(match.group("acc_mean")),
                    acc_std=float(match.group("acc_std")),
                    detail_path=match.group("detail"),
                )
            )
    if not rows:
        raise ValueError(f"No ranking rows parsed from: {file_path}")
    return rows


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_metric_bar_figure(rows: List[AttentionResult], out_dir: str, dpi: int):
    by_name = sorted(rows, key=lambda x: x.attention)
    names = [x.attention.upper() for x in by_name]

    metrics: List[Tuple[str, List[float], List[float], str]] = [
        ("Dice", [x.dice_mean for x in by_name], [x.dice_std for x in by_name], "#4C78A8"),
        ("IoU", [x.iou_mean for x in by_name], [x.iou_std for x in by_name], "#59A14F"),
        ("ACC", [x.acc_mean for x in by_name], [x.acc_std for x in by_name], "#F28E2B"),
    ]

    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    for ax, (metric_name, means, stds, color) in zip(axes, metrics):
        bars = ax.bar(names, means, yerr=stds, capsize=4, color=color, edgecolor="black", linewidth=0.8)
        ymin = min(means) - 0.003
        ymax = max(means) + 0.003
        if ymin < 0:
            ymin = 0.0
        if ymax > 1:
            ymax = 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0006,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Attention Modules Benchmark (Mean ± Std)", fontsize=13)

    png_path = os.path.join(out_dir, "attention_metrics_bar.png")
    pdf_path = os.path.join(out_dir, "attention_metrics_bar.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_dice_ranking_figure(rows: List[AttentionResult], out_dir: str, dpi: int):
    by_rank = sorted(rows, key=lambda x: x.rank)
    names = [x.attention.upper() for x in by_rank]
    means = [x.dice_mean for x in by_rank]
    stds = [x.dice_std for x in by_rank]

    colors = ["#2ca02c"] + ["#9ecae1"] * (len(names) - 1)

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Dice Ranking by Attention Module", fontsize=12)
    ax.set_ylabel("Dice")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(min(means) - 0.003, max(means) + 0.003)

    for idx, (bar, value) in enumerate(zip(bars, means), start=1):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0006,
            f"#{idx}\n{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    png_path = os.path.join(out_dir, "attention_dice_ranking.png")
    pdf_path = os.path.join(out_dir, "attention_dice_ranking.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_csv(rows: List[AttentionResult], out_dir: str):
    path = os.path.join(out_dir, "attention_benchmark_table.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Rank,Attention,Dice_mean,Dice_std,IoU_mean,IoU_std,ACC_mean,ACC_std,Detail\n")
        for r in sorted(rows, key=lambda x: x.rank):
            f.write(
                f"{r.rank},{r.attention},{r.dice_mean:.4f},{r.dice_std:.4f},"
                f"{r.iou_mean:.4f},{r.iou_std:.4f},{r.acc_mean:.4f},{r.acc_std:.4f},{r.detail_path}\n"
            )
    return path


def best_values(rows: List[AttentionResult]) -> Dict[str, float]:
    return {
        "dice": max(x.dice_mean for x in rows),
        "iou": max(x.iou_mean for x in rows),
        "acc": max(x.acc_mean for x in rows),
    }


def fmt_metric(mean: float, std: float, best: float) -> str:
    text = f"{mean:.4f} $\\pm$ {std:.4f}"
    if abs(mean - best) < 1e-12:
        return f"\\textbf{{{text}}}"
    return text


def write_latex_table(rows: List[AttentionResult], out_dir: str):
    path = os.path.join(out_dir, "attention_benchmark_table.tex")
    best = best_values(rows)
    sorted_rows = sorted(rows, key=lambda x: x.rank)

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Attention modules benchmark on OCTA segmentation (3 seeds).}",
        "  \\label{tab:attention_benchmark}",
        "  \\begin{tabular}{c l c c c}",
        "    \\hline",
        "    Rank & Attention & Dice ($\\uparrow$) & IoU ($\\uparrow$) & ACC ($\\uparrow$) \\\\",
        "    \\hline",
    ]

    for r in sorted_rows:
        lines.append(
            "    "
            f"{r.rank} & {r.attention.upper()} & "
            f"{fmt_metric(r.dice_mean, r.dice_std, best['dice'])} & "
            f"{fmt_metric(r.iou_mean, r.iou_std, best['iou'])} & "
            f"{fmt_metric(r.acc_mean, r.acc_std, best['acc'])} \\\\",
        )

    lines.extend([
        "    \\hline",
        "  \\end{tabular}",
        "\\end{table}",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    rows = parse_ranking_file(args.ranking_file)
    bar_png, bar_pdf = save_metric_bar_figure(rows, args.output_dir, args.dpi)
    rank_png, rank_pdf = save_dice_ranking_figure(rows, args.output_dir, args.dpi)
    csv_path = write_csv(rows, args.output_dir)
    tex_path = write_latex_table(rows, args.output_dir)

    print("Generated files:")
    print(bar_png)
    print(bar_pdf)
    print(rank_png)
    print(rank_pdf)
    print(csv_path)
    print(tex_path)


if __name__ == "__main__":
    main()
