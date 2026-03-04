import argparse
import datetime
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt


LOSSES = ["bce", "dice", "bce_dice"]
AVG_LOSS_PATTERN = re.compile(r"Epoch\s+(\d+)/(\d+),\s+Avg Loss:\s+([0-9]*\.?[0-9]+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple loss functions and compare loss curves")
    parser.add_argument("--losses", type=str, default=",".join(LOSSES), help="Comma-separated losses")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--attention", type=str, default="eca", choices=["none", "eca", "gated", "spatial", "pde"])
    parser.add_argument("--image-dir", type=str, default="dataset/train/images")
    parser.add_argument("--mask-dir", type=str, default="dataset/train/masks")
    parser.add_argument("--output-dir", type=str, default="loss_result")
    parser.add_argument("--disable-plot", action="store_true")
    return parser.parse_args()


def run_and_collect(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    epoch_to_loss = {}
    output_lines = []

    try:
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
            match = AVG_LOSS_PATTERN.search(line)
            if match:
                epoch = int(match.group(1))
                avg_loss = float(match.group(3))
                epoch_to_loss[epoch] = avg_loss
    finally:
        process.wait()

    return process.returncode, epoch_to_loss, "".join(output_lines)


def build_main_command(args, loss_name):
    return [
        sys.executable,
        "main.py",
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--attention", args.attention,
        "--loss", loss_name,
        "--image-dir", args.image_dir,
        "--mask-dir", args.mask_dir,
        "--disable-plot",
    ]


def plot_curves(curves, output_path):
    plt.figure(figsize=(9, 6))
    for loss_name, points in curves.items():
        epochs = sorted(points.keys())
        values = [points[e] for e in epochs]
        plt.plot(epochs, values, marker="o", linewidth=1.5, markersize=3, label=loss_name)

    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Loss Function Benchmark")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def save_summary(summary_path, args, curves):
    final_rank = []
    for loss_name, points in curves.items():
        if not points:
            continue
        last_epoch = max(points.keys())
        final_rank.append((loss_name, points[last_epoch], last_epoch))

    final_rank.sort(key=lambda x: x[1])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Loss function benchmark summary\n")
        f.write(f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Seed: {args.seed}, Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n")
        f.write(f"Attention: {args.attention}\n")
        f.write(f"Losses: {','.join(curves.keys())}\n\n")
        f.write("Ranking by final Avg Loss (lower is better):\n")

        for idx, (loss_name, final_loss, epoch) in enumerate(final_rank, start=1):
            f.write(f"#{idx} {loss_name} | Final Avg Loss: {final_loss:.6f} @ epoch {epoch}\n")


def main():
    args = parse_args()
    loss_list = [x.strip() for x in args.losses.split(",") if x.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    all_curves = {}
    for loss_name in loss_list:
        print(f"\n===== Running loss benchmark: {loss_name} =====")
        command = build_main_command(args, loss_name)
        code, epoch_to_loss, _ = run_and_collect(command)

        if code != 0:
            raise RuntimeError(f"Training failed for loss={loss_name}")

        if not epoch_to_loss:
            raise RuntimeError(f"No avg loss parsed for loss={loss_name}; please check main.py logs")

        all_curves[loss_name] = epoch_to_loss

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    curve_path = os.path.join(args.output_dir, f"loss_benchmark_curve_{now}.png")
    summary_path = os.path.join(args.output_dir, f"loss_benchmark_summary_{now}.txt")

    plot_curves(all_curves, curve_path)
    save_summary(summary_path, args, all_curves)

    print(f"Loss benchmark curve saved to: {curve_path}")
    print(f"Loss benchmark summary saved to: {summary_path}")

    if args.disable_plot:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
