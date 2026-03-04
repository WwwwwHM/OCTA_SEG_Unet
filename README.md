# OCTA_seg_Unet+

## 项目简介
本项目基于U-Net实现医学图像分割，支持训练、推理和评估。

## 文件结构
- `unet.py`：U-Net模型结构
- `dataset.py`：数据集读取
- `main.py`：训练主程序
- `predict.py`：推理脚本
- `evaluate.py`：评估脚本
- `requirements.txt`：依赖列表

## 快速开始
1. 安装依赖  
   ```
   pip install -r requirements.txt
   ```
2. 训练模型  
   ```
   python main.py
   ```
   可选参数示例（损失函数对比常用）：
   ```
   python main.py --loss bce
   python main.py --loss dice
   python main.py --loss bce_dice
   ```
3. 推理  
   ```
   python predict.py
   ```
4. 评估  
   ```
   python evaluate.py
   ```

## 损失函数曲线对比
项目默认损失函数为 `BCEWithLogitsLoss`（`--loss bce`）。

可一键运行多损失函数训练并生成同图对比曲线：
```
python run_loss_benchmark.py --losses bce,dice,bce_dice --epochs 50 --seed 42 --attention eca
```
输出：
- `loss_result/loss_benchmark_curve_*.png`
- `loss_result/loss_benchmark_summary_*.txt`