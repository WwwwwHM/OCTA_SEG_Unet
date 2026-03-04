import argparse
import datetime
import os
import statistics
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed training and evaluation")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--attention",
        type=str,
        default="eca",
        choices=["none", "eca", "gated", "spatial", "pde"],
        help="Attention module to use in main.py",
    )
    parser.add_argument("--train-image-dir", type=str, default="dataset/train/images")
    parser.add_argument("--train-mask-dir", type=str, default="dataset/train/masks")
    parser.add_argument("--test-image-dir", type=str, default="dataset/test/images")
    parser.add_argument("--test-mask-dir", type=str, default="dataset/test/masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--summary-dir", type=str, default="eval_result")
    return parser.parse_args()


def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    try:
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
    finally:
        process.wait()

    combined_output = "".join(output_lines)
    return process.returncode, combined_output, ""


def parse_value(output, key):
    prefix = f"{key}="
    for line in output.splitlines()[::-1]:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def parse_metric(output, metric_name):
    target = f"{metric_name}:"
    for line in output.splitlines():
        if target in line:
            value = line.split(":", 1)[1].strip()
            return float(value)
    return None


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.summary_dir, exist_ok=True)

    all_results = []

    for seed in seeds:
        print(f"\n===== Running seed {seed} =====")
        train_cmd = [
            sys.executable,
            "main.py",
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--attention", args.attention,
            "--image-dir", args.train_image_dir,
            "--mask-dir", args.train_mask_dir,
            "--disable-plot",
        ]
        train_code, train_out, train_err = run_command(train_cmd)
        if train_code != 0:
            print(train_out)
            print(train_err)
            raise RuntimeError(f"Training failed for seed {seed}")

        model_path = parse_value(train_out, "MODEL_PATH")
        if not model_path:
            print(train_out)
            raise RuntimeError(f"Cannot parse MODEL_PATH for seed {seed}")

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--model-path", model_path,
            "--image-dir", args.test_image_dir,
            "--mask-dir", args.test_mask_dir,
            "--threshold", str(args.threshold),
        ]
        eval_code, eval_out, eval_err = run_command(eval_cmd)
        if eval_code != 0:
            print(eval_out)
            print(eval_err)
            raise RuntimeError(f"Evaluation failed for seed {seed}")

        dice = parse_metric(eval_out, "Mean Dice Coefficient")
        iou = parse_metric(eval_out, "Mean IoU")
        acc = parse_metric(eval_out, "Mean ACC")

        if dice is None or iou is None or acc is None:
            print(eval_out)
            raise RuntimeError(f"Cannot parse evaluation metrics for seed {seed}")

        all_results.append({
            "seed": seed,
            "model_path": model_path,
            "dice": dice,
            "iou": iou,
            "acc": acc,
        })

        print(f"Seed {seed} | Dice: {dice:.4f} | IoU: {iou:.4f} | ACC: {acc:.4f}")

    dice_values = [r["dice"] for r in all_results]
    iou_values = [r["iou"] for r in all_results]
    acc_values = [r["acc"] for r in all_results]

    dice_mean, dice_std = mean_std(dice_values)
    iou_mean, iou_std = mean_std(iou_values)
    acc_mean, acc_std = mean_std(acc_values)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = os.path.join(args.summary_dir, f"multiseed_summary_{now}.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Multi-seed experiment summary\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n")
        f.write(f"Attention: {args.attention}\n")
        f.write(f"Threshold: {args.threshold}\n\n")

        for r in all_results:
            f.write(
                f"Seed {r['seed']} | Dice: {r['dice']:.4f} | IoU: {r['iou']:.4f} | ACC: {r['acc']:.4f} | Model: {r['model_path']}\n"
            )

        f.write("\n")
        f.write(f"Dice Mean ± Std: {dice_mean:.4f} ± {dice_std:.4f}\n")
        f.write(f"IoU Mean ± Std: {iou_mean:.4f} ± {iou_std:.4f}\n")
        f.write(f"ACC Mean ± Std: {acc_mean:.4f} ± {acc_std:.4f}\n")

    print("\n===== Multi-seed summary =====")
    print(f"Dice Mean ± Std: {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"IoU Mean ± Std: {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"ACC Mean ± Std: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
