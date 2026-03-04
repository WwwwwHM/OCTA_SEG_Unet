import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet_Transformer
import argparse
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import datetime
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet_Transformer for OCTA segmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--attention",
        type=str,
        default="eca",
        choices=["none", "eca", "gated", "spatial", "pde"],
        help="Attention module to enable",
    )
    parser.add_argument("--image-dir", type=str, default="dataset/train/images", help="Training image directory")
    parser.add_argument("--mask-dir", type=str, default="dataset/train/masks", help="Training mask directory")
    parser.add_argument(
        "--loss",
        type=str,
        default="bce",
        choices=["bce", "dice", "bce_dice"],
        help="Loss function: bce | dice | bce_dice",
    )
    parser.add_argument("--disable-plot", action="store_true", help="Disable plt.show() for non-interactive runs")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 参数设置
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
image_dir = args.image_dir
mask_dir = args.mask_dir
set_seed(args.seed)

attention_flags = {
    "use_gated_attention": args.attention == "gated",
    "use_eca_attention": args.attention == "eca",
    "use_spatial_attention": args.attention == "spatial",
    "use_pde_attention": args.attention == "pde",
}

model_config = {
    "use_residual": False,
    **attention_flags,
    "use_aspp": False,
    "use_edge_branch": False,
    "return_edge": False,
}

use_edge_supervision = model_config["use_edge_branch"] and model_config["return_edge"]
edge_loss_weight = 0.3


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(loss_name):
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "dice":
        return DiceLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def build_experiment_tag(config):
    tag_parts = ["unet", "transformer"]
    mapping = [
        ("use_residual", "Res"),
        ("use_gated_attention", "GatedAT"),
        ("use_eca_attention", "ECA"),
        ("use_spatial_attention", "SpatialAT"),
        ("use_pde_attention", "PDE"),
        ("use_aspp", "ASPP"),
        ("use_edge_branch", "edgeBranch"),
    ]
    for key, name in mapping:
        if config.get(key, False):
            tag_parts.append(name)
    if config.get("return_edge", False):
        tag_parts.append("edgeOut")
    return "+".join(tag_parts)


experiment_tag = build_experiment_tag(model_config)
experiment_tag = f"{experiment_tag}_{args.loss}_s{args.seed}"
print(f"Experiment Tag: {experiment_tag}")
print(f"Loss Function: {args.loss}")

# 数据增强
class JointTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # 随机旋转
        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        # 缩放
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)
        # 转为Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

# 修改Dataset以支持联合变换
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, joint_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        return image, mask

# 边缘检测：将掩码转换为边缘图
def mask_to_edge(mask_tensor):
    mask_np = mask_tensor.squeeze().cpu().numpy()
    edge_np = cv2.Canny((mask_np*255).astype(np.uint8), 50, 150)
    edge_tensor = torch.from_numpy(edge_np / 255.0).float().to(mask_tensor.device)
    return edge_tensor

# 数据集与 DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
joint_transform = JointTransform((256, 256))
dataset = SegmentationDataset(image_dir, mask_dir, joint_transform=joint_transform)
loader_generator = torch.Generator()
loader_generator.manual_seed(args.seed)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)

# 模型、损失函数、优化器
model = UNet_Transformer(in_channels=3, out_channels=1, **model_config).to(device)
criterion = build_loss(args.loss)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # 每20个epoch学习率减半

# 计算指标
def compute_metrics(preds, masks, threshold=0.5):
    # 二值化
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    masks = (masks > threshold).float()
    # IoU
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = (preds + masks).sum(dim=(1,2,3)) - intersection
    iou = (intersection / (union + 1e-8)).mean().item()
    # Dice
    dice = (2 * intersection / (preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-8)).mean().item()
    # 准确度
    correct = (preds == masks).sum().item()
    total = preds.numel()
    acc = correct / total
    return iou, dice, acc

# 训练循环
losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_edge_loss = 0
    epoch_iou = 0
    epoch_dice = 0
    epoch_acc = 0
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)

        if isinstance(outputs, tuple):
            seg_out, edge_out = outputs
        else:
            seg_out = outputs
            edge_out = None

        loss = criterion(seg_out, masks)

        if use_edge_supervision and edge_out is not None:
            edge_labels = torch.stack([mask_to_edge(m) for m in masks]).unsqueeze(1)
            if edge_out.shape != edge_labels.shape:
                edge_out = nn.functional.interpolate(
                    edge_out,
                    size=edge_labels.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            loss_edge = criterion(edge_out, edge_labels)
            loss = loss + edge_loss_weight * loss_edge
            epoch_edge_loss += loss_edge.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iou, dice, acc = compute_metrics(seg_out, masks)
        epoch_iou += iou
        epoch_dice += dice
        epoch_acc += acc

        if use_edge_supervision and edge_out is not None:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, EdgeLoss: {loss_edge.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {acc:.4f}")
        else:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {acc:.4f}")

    avg_loss = epoch_loss / len(dataloader)
    avg_iou = epoch_iou / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)
    avg_acc = epoch_acc / len(dataloader)
    losses.append(avg_loss)

    if use_edge_supervision:
        avg_edge_loss = epoch_edge_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg EdgeLoss: {avg_edge_loss:.4f}, Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}, Avg Acc: {avg_acc:.4f}")
    else:
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}, Avg Acc: {avg_acc:.4f}")

    scheduler.step()

# 保存模型
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
os.makedirs("model_result", exist_ok=True)
model_save_path = f"model_result/{experiment_tag}_{now}.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")
print(f"MODEL_PATH={model_save_path}")

# 绘制训练损失曲线
os.makedirs("loss_result", exist_ok=True)
plt.figure()
plt.plot(range(1, epochs+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Loss Curve ({args.loss})')
loss_curve_path = f'loss_result/loss_curve_{experiment_tag}_{now}.png'
plt.savefig(loss_curve_path)
print(f"Loss curve saved to: {loss_curve_path}")
if args.disable_plot:
    plt.close()
else:
    plt.show()