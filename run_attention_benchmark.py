import argparse
import datetime
import json
import os
import subprocess
import sys


ATTENTIONS = ["none", "eca", "gated", "spatial", "pde"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark multiple attention modules with multi-seed runs")
    parser.add_argument("--attentions", type=str, default=",".join(ATTENTIONS), help="Comma-separated attentions")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-image-dir", type=str, default="dataset/train/images")
    parser.add_argument("--train-mask-dir", type=str, default="dataset/train/masks")
    parser.add_argument("--test-image-dir", type=str, default="dataset/test/images")
    parser.add_argument("--test-mask-dir", type=str, default="dataset/test/masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--summary-dir", type=str, default="eval_result")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--checkpoint-file", type=str, default="", help="Optional checkpoint file path")
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


def parse_metric_with_std(text, metric_name):
    key = f"{metric_name} Mean ± Std:"
    for line in text.splitlines():
        if key in line:
            right = line.split(":", 1)[1].strip()
            parts = right.split("±")
            if len(parts) != 2:
                return None, None
            mean = float(parts[0].strip())
            std = float(parts[1].strip())
            return mean, std
    return None, None


def parse_summary_path(text):
    key = "Summary saved to:"
    for line in text.splitlines()[::-1]:
        if key in line:
            return line.split(key, 1)[1].strip()
    return ""


def build_multiseed_command(args, attention):
    return [
        sys.executable,
        "run_multiseed.py",
        "--seeds", args.seeds,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--attention", attention,
        "--train-image-dir", args.train_image_dir,
        "--train-mask-dir", args.train_mask_dir,
        "--test-image-dir", args.test_image_dir,
        "--test-mask-dir", args.test_mask_dir,
        "--threshold", str(args.threshold),
        "--summary-dir", args.summary_dir,
    ]


def save_checkpoint(checkpoint_path, state):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_ranking(rank_path, args, attentions, all_results):
    sorted_results = sorted(all_results, key=lambda x: x["dice_mean"], reverse=True)
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write("Attention benchmark ranking (sorted by Dice mean)\n")
        f.write(f"Attentions: {attentions}\n")
        f.write(f"Seeds: {args.seeds}\n")
        f.write(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.learning_rate}\n")
        f.write(f"Threshold: {args.threshold}\n\n")

        for idx, item in enumerate(sorted_results, start=1):
            f.write(
                f"#{idx} {item['attention']} | "
                f"Dice: {item['dice_mean']:.4f} ± {item['dice_std']:.4f} | "
                f"IoU: {item['iou_mean']:.4f} ± {item['iou_std']:.4f} | "
                f"ACC: {item['acc_mean']:.4f} ± {item['acc_std']:.4f} | "
                f"Detail: {item['summary_path']}\n"
            )
    return sorted_results


def main():
    args = parse_args()
    attentions = [a.strip() for a in args.attentions.split(",") if a.strip()]

    os.makedirs(args.summary_dir, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    rank_path = os.path.join(args.summary_dir, f"attention_benchmark_rank_{now}.txt")
    checkpoint_path = args.checkpoint_file or os.path.join(args.summary_dir, "attention_benchmark_checkpoint.json")

    all_results = []
    completed_attentions = set()
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            all_results = checkpoint.get("results", [])
            completed_attentions = {item.get("attention") for item in all_results if item.get("attention")}
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Already completed: {sorted(completed_attentions)}")

    for attention in attentions:
        if attention in completed_attentions:
            print(f"\n===== Skip attention (already done): {attention} =====")
            continue

        print(f"\n===== Benchmark attention: {attention} =====")
        command = build_multiseed_command(args, attention)
        code, out, err = run_command(command)
        if code != 0:
            print(out)
            print(err)
            raise RuntimeError(f"run_multiseed failed for attention={attention}")

        dice_mean, dice_std = parse_metric_with_std(out, "Dice")
        iou_mean, iou_std = parse_metric_with_std(out, "IoU")
        acc_mean, acc_std = parse_metric_with_std(out, "ACC")
        summary_path = parse_summary_path(out)

        if None in [dice_mean, dice_std, iou_mean, iou_std, acc_mean, acc_std]:
            print(out)
            raise RuntimeError(f"Cannot parse metrics from run_multiseed output for attention={attention}")

        all_results.append(
            {
                "attention": attention,
                "dice_mean": dice_mean,
                "dice_std": dice_std,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "summary_path": summary_path,
            }
        )
        completed_attentions.add(attention)

        save_checkpoint(
            checkpoint_path,
            {
                "attentions": attentions,
                "results": all_results,
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
        )
        write_ranking(rank_path, args, attentions, all_results)

        print(
            f"{attention} | Dice: {dice_mean:.4f} ± {dice_std:.4f} | "
            f"IoU: {iou_mean:.4f} ± {iou_std:.4f} | ACC: {acc_mean:.4f} ± {acc_std:.4f}"
        )

    sorted_results = write_ranking(rank_path, args, attentions, all_results)

    print("\n===== Attention Ranking =====")
    for idx, item in enumerate(sorted_results, start=1):
        print(
            f"#{idx} {item['attention']} | "
            f"Dice: {item['dice_mean']:.4f} ± {item['dice_std']:.4f} | "
            f"IoU: {item['iou_mean']:.4f} ± {item['iou_std']:.4f}"
        )
    print(f"Ranking saved to: {rank_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
