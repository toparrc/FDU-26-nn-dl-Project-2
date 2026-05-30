# Project 2：CIFAR-10 图像分类与 Batch Normalization

本仓库是“Neural Network and Deep Learning”课程 Project 2 的代码与报告材料，主要包含两个部分：

- Task 1: 在 CIFAR-10 上设计、训练并优化 CNN 分类模型。
- Task 2: 比较 VGG-A with / without Batch Normalization，并从 loss landscape 角度分析 BN 对优化过程的影响。

## 目录结构

```text
.
├── report.ipynb                         # 项目报告
├── project_2_2026.pdf                   # 作业说明
├── codes/
│   ├── CIFAR10_CNN/                     # Task 1：CIFAR-10 CNN 实验
│   │   ├── model.py                     # CNN / Plain CNN / Residual CNN 模型
│   │   ├── data.py                      # CIFAR-10 dataloader
│   │   ├── train.py                     # 单次训练脚本
│   │   ├── run_task1_experiments.py     # 对比实验脚本
│   │   ├── plots.py                     # 绘图工具函数
│   │   ├── results/                     # 保存曲线、summary 和混淆矩阵
│   │   └── checkpoints/                 # 保存模型 checkpoint
│   └── VGG_BatchNorm/                   # Task 2：VGG-A 与 BatchNorm 实验
│       ├── models/vgg.py                # VGG-A 和 VGG-A-BN 模型
│       ├── data/loaders.py              # VGG 实验使用的 CIFAR-10 dataloader
│       ├── training_framework.py        # 通用训练与评估框架
│       ├── VGG_BN_Comparison.py         # VGG-A 与 VGG-A-BN 性能比较
│       ├── VGG_Loss_Landscape.py        # loss landscape 实验
│       └── results/                     # 保存训练记录、图像和 checkpoint
└── pic/                                 # 作业说明和报告使用的图片
```

## 环境依赖

实验代码基于 Python 和 PyTorch 实现，主要依赖如下：

```text
torch
torchvision
numpy
matplotlib
scikit-learn
```

运行脚本时，CIFAR-10 会通过 `torchvision.datasets.CIFAR10` 自动下载。

## Task 1：CIFAR-10 CNN

Task 1 的主要模型实现在 `codes/CIFAR10_CNN/model.py` 中：

- `CIFAR10CNN`：四个卷积 stage 的基础 CNN。
- `CIFAR10CNN3`：较浅的三 stage CNN。
- `CIFAR10CNN4`：可配置激活函数的四 stage CNN。
- `CIFAR10ResidualCNN`：带 residual block 的 CNN。

训练基础 CNN 和 Residual CNN：

```bash
python codes/CIFAR10_CNN/train.py --model both
```

运行 Task 1 的全部对比实验：

```bash
python codes/CIFAR10_CNN/run_task1_experiments.py --group all
```

也可以只运行其中一组对比实验：

```bash
python codes/CIFAR10_CNN/run_task1_experiments.py --group regularization
python codes/CIFAR10_CNN/run_task1_experiments.py --group activation
python codes/CIFAR10_CNN/run_task1_experiments.py --group depth
python codes/CIFAR10_CNN/run_task1_experiments.py --group optimizer
```

Task 1 的主要输出保存在：

```text
codes/CIFAR10_CNN/results/
codes/CIFAR10_CNN/checkpoints/
```

如果不重新训练，也可以从百度网盘下载已经训练好的 checkpoint 后进行验证：

```text
文件名：PJ2-checkpoint.zip
百度网盘链接：https://pan.baidu.com/s/1aSYma4uqEmXt0JxLKdvV0w
提取码：vze2
```

下载后请将 checkpoint 放到对应目录，例如：

```text
codes/CIFAR10_CNN/checkpoints/
codes/VGG_BatchNorm/results/bn_comparison/models/
codes/VGG_BatchNorm/results/bn_loss_landscape/models/
```

当前报告中的最佳 Task 1 结果：

| 模型 | 测试准确率 | 测试错误率 |
|---|---:|---:|
| Residual CNN | 91.76% | 8.24% |

## Task 2：VGG-A 与 Batch Normalization

训练并比较 VGG-A with / without Batch Normalization：

```bash
python codes/VGG_BatchNorm/VGG_BN_Comparison.py
```

运行 BN loss landscape 实验：

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py
```

如果只想根据已有结果重新画图，可以使用：

```bash
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py --plot-only
```

Task 2 的主要输出保存在：

```text
codes/VGG_BatchNorm/results/bn_comparison/
codes/VGG_BatchNorm/results/bn_loss_landscape/
```

当前 VGG-A 对比结果：

| 模型 | 最佳验证准确率 | 测试准确率 |
|---|---:|---:|
| VGG-A without BN | 86.00% | 84.81% |
| VGG-A with BN | 89.24% | 88.33% |

## 报告

最终报告位于：

```text
report.ipynb
```

报告中包含模型结构、训练设置、对比实验表格、验证曲线、混淆矩阵分析，以及 BN loss landscape 可视化结果。
