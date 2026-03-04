# OCTA_seg_Unet+

基于 U-Net Transformer 变体的 OCTA 血管分割项目，支持训练、评估、推理，以及注意力模块与损失函数的对比实验。

## 1. 项目功能
- 训练分割模型：`main.py`
- 模型评估（Dice / IoU / ACC）：`evaluate.py`
- 批量图像预测：`predict.py`
- 损失函数对比实验：`run_loss_benchmark.py`
- 注意力模块对比实验：`run_attention_benchmark.py`
- 多随机种子稳定性实验：`run_multiseed.py`

## 2. 目录结构（核心）
```text
OCTA_seg_Unet+/
├─ main.py
├─ unet.py
├─ dataset.py
├─ evaluate.py
├─ predict.py
├─ run_loss_benchmark.py
├─ run_attention_benchmark.py
├─ run_multiseed.py
├─ requirements.txt
└─ dataset/
   ├─ train/
   │  ├─ images/
   │  └─ masks/
   └─ test/
      ├─ images/
      └─ masks/
```

> 说明：模型权重（`.pth`）、训练产物、评估日志、预测结果和数据大文件默认已在 `.gitignore` 中忽略。

## 3. 环境准备
建议 Python 3.9+，并在虚拟环境中安装依赖：

```bash
pip install -r requirements.txt
```

## 4. 训练
默认训练命令：

```bash
python main.py
```

常用参数：
- `--epochs`：训练轮数，默认 `200`
- `--batch-size`：批大小，默认 `8`
- `--learning-rate`：学习率，默认 `1e-4`
- `--seed`：随机种子，默认 `42`
- `--attention`：`none|eca|gated|spatial|pde`（默认 `eca`）
- `--loss`：`bce|dice|bce_dice`（默认 `bce`）
- `--image-dir` / `--mask-dir`：训练数据路径

示例：

```bash
python main.py --epochs 100 --batch-size 8 --attention eca --loss bce_dice --seed 42
```

## 5. 评估
```bash
python evaluate.py --model-path model_result/your_model.pth --image-dir dataset/test/images --mask-dir dataset/test/masks --threshold 0.5
```

评估输出：
- 终端打印 `Dice / IoU / ACC`
- 文本结果保存到 `eval_result/`（自动带时间戳）

## 6. 推理
```bash
python predict.py
```

脚本会对输入目录中的图像批量预测，并将结果保存到 `predict_result/<模型名_时间戳>/`。

## 7. 对比实验
### 7.1 损失函数对比
```bash
python run_loss_benchmark.py --losses bce,dice,bce_dice --epochs 50 --seed 42 --attention eca
```

输出示例：
- `loss_result/loss_benchmark_curve_*.png`
- `loss_result/loss_benchmark_summary_*.txt`

### 7.2 注意力模块对比
```bash
python run_attention_benchmark.py
```

输出结果保存在 `eval_result/`。

### 7.3 多种子稳定性测试
```bash
python run_multiseed.py
```

输出结果保存在 `eval_result/`。

## 8. 复现建议
- 固定 `--seed` 以减少随机波动
- 对比实验时保持相同训练轮数、学习率和数据划分
- 若显存不足，优先降低 `--batch-size`

## 9. 常见问题
- **找不到模型文件**：优先检查 `--model-path`，评估/推理脚本也会尝试在 `model_result/` 下查找同名文件。
- **找不到输入图片目录**：`predict.py` 会尝试回退到 `predictset`、`dataset/test/images` 或 `dataset/images`。