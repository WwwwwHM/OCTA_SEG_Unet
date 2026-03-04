import torch
from torch.utils.data import DataLoader
from unet import UNet_Transformer
from dataset import SegmentationDataset
import argparse
import torchvision.transforms as transforms
import datetime
import os
import re

model_config = {
    "use_residual": False,
    "use_gated_attention": False,
    "use_eca_attention": True,
    "use_spatial_attention": False,
    "use_pde_attention": False,
    "use_aspp": False,
    "use_edge_branch": False,
    "return_edge": False,
}


def adapt_legacy_state_dict_keys(state_dict):
    adapted = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("attentions."):
            new_key = new_key.replace("attentions.", "gated_attentions.", 1)
        adapted[new_key] = value
    return adapted


def infer_model_config_from_state_dict(state_dict, base_config):
    inferred = dict(base_config)

    # 残差块：新结构有 downs.*.conv1，旧结构是 downs.*.0
    if any(k.startswith("downs.0.conv1") for k in state_dict.keys()):
        inferred["use_residual"] = True
    elif any(k.startswith("downs.0.0") for k in state_dict.keys()):
        inferred["use_residual"] = False

    inferred["use_gated_attention"] = any(
        k.startswith("gated_attentions.") or k.startswith("attentions.")
        for k in state_dict.keys()
    )
    inferred["use_eca_attention"] = any(k.startswith("eca_attentions.") for k in state_dict.keys())
    inferred["use_spatial_attention"] = any(k.startswith("spatial_attentions.") for k in state_dict.keys())
    inferred["use_pde_attention"] = any(k.startswith("pde_attentions.") for k in state_dict.keys())
    inferred["use_aspp"] = any(k.startswith("aspp.") for k in state_dict.keys())
    inferred["use_edge_branch"] = any(k.startswith("edge_branch.") for k in state_dict.keys())

    # 评估默认只用主分割输出
    inferred["return_edge"] = False
    return inferred


def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|\s]+', '_', name).strip('_')


def build_eval_result_path(model_path, image_dir, output_dir="eval_result"):
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = sanitize_filename(os.path.splitext(os.path.basename(model_path))[0])
    dataset_tag = sanitize_filename(os.path.basename(os.path.normpath(image_dir)) or "dataset")
    file_name = f"{model_name}_{dataset_tag}_{now}.txt"
    return os.path.join(output_dir, file_name)

def dice_coef(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def iou_coef(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-8)

def acc_coef(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > threshold).float()
    correct = (pred == target).sum().item()
    total = pred.numel()
    return correct / total

def evaluate(model_path, image_dir, mask_dir, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动解析模型路径：优先原路径，其次 model_result/ 下同名文件
    if not os.path.exists(model_path):
        alt_model_path = os.path.join("model_result", os.path.basename(model_path))
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    raw_state_dict = torch.load(model_path, map_location=device)
    raw_state_dict = adapt_legacy_state_dict_keys(raw_state_dict)
    runtime_config = infer_model_config_from_state_dict(raw_state_dict, model_config)

    model = UNet_Transformer(in_channels=3, out_channels=1, **runtime_config).to(device)
    model.load_state_dict(raw_state_dict, strict=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    dices = []
    ious = []
    accs = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 只用主分割输出
            dice = dice_coef(outputs, masks, threshold=threshold)
            iou = iou_coef(outputs, masks, threshold=threshold)
            acc = acc_coef(outputs, masks, threshold=threshold)
            dices.append(dice.item())
            ious.append(iou)
            accs.append(acc)
    mean_dice = sum(dices) / len(dices)
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean ACC: {mean_acc:.4f}")

    result_filename = build_eval_result_path(model_path, image_dir)
    with open(result_filename, "w", encoding="utf-8") as f:
        f.write(f"Mean Dice Coefficient: {mean_dice:.4f}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Mean ACC: {mean_acc:.4f}\n")
    print(f"Evaluation result saved to: {result_filename}")
    return {
        "dice": mean_dice,
        "iou": mean_iou,
        "acc": mean_acc,
        "result_file": result_filename,
    }
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained UNet_Transformer model")
    parser.add_argument("--model-path", type=str, default="model_result/unet+transformer+ECA_20260227_1431.pth")
    parser.add_argument("--image-dir", type=str, default="dataset/test/images")
    parser.add_argument("--mask-dir", type=str, default="dataset/test/masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args.model_path, args.image_dir, args.mask_dir, threshold=args.threshold)