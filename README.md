# ENGS: A Novel Deep Learning Model for Wheat Leaf Disease Identification

> 官方 PyTorch 实现 | 论文标题：*ENGS: A Novel Deep Learning Model for Wheat Leaf Disease Identification*（投稿中）

本文提出一种基于改进 EfficientNet-B0 并融合广义均值池化（GeM）与选择性核模块（SKBlock）的小麦叶片病害识别模型 ENGS，实现对白粉病、叶锈病、斑枯病、黄叶斑病及健康叶片的高精度分类，为田间病害智能诊断提供高效轻量化解决方案。

## 1. 研究背景与模型定位

小麦是中国主要粮食作物之一，其病害严重影响产量与质量。传统病害检测方法效率低、适应性差，而现有轻量化模型在多尺度病斑特征提取与复杂背景鲁棒性方面存在不足。

本文在精简后的 EfficientNet-B0 骨干上，将 SKBlock 插入每个 MBConv 块的投影层之后，并将全局平均池化替换为带可学习指数 p 的 GeM 池化，增强模型对多尺度病斑的识别能力与特征区分度。完整模型（Improved EfficientNet-B0 + SKBlock + GeM）参数量仅 3.42M，在自建小麦叶片病害数据集上取得了高精度分类结果（详见第 5 节实验表格）。

## 2. 核心创新点

- **轻量选择性核模块（SKBlock）**
  采用 1×1 与 3×3 双分支深度卷积，配合纹理感知空间注意力分支与全局通道门（压缩比 8/6/4 分阶段设置），自适应捕捉不同尺度的病斑特征。与标准 SKNet 模块（3×3/5×5）相比，参数量和 FLOPs 更低（见论文 Table 1）。

- **自适应广义均值池化（GeM）**
  池化指数 p 为可学习参数（初始化为 3，通过反向传播训练优化），在全局平滑与局部显著特征保留之间自适应权衡，提升对白粉病与锈病等视觉相似病害的区分能力。

- **轻量化设计**
  SKBlock 与 GeM 的引入仅增加 0.31M 参数与 0.02G FLOPs，完整模型约 3.42M 参数，兼顾精度与边缘部署需求。

## 3. 实验数据集

### 3.1 数据集概况

数据集采集于河南省小麦种植基地，包含 5 个类别（Healthy、Leaf rust、Powdery mildew、Spot blotch、Yellow leaf blotch），共 7493 张图像（原始田间图像 + Pix2Pix 生成图像，合成图像仅用于训练集），图像统一预处理为 224×224 分辨率。

| 类别 | 训练集 | 验证集 | 测试集 | 合计 |
|---|---|---|---|---|
| Healthy | 1032 | 344 | 347 | 1723 |
| Leaf rust | 786 | 263 | 267 | 1316 |
| Powdery mildew | 925 | 313 | 315 | 1553 |
| Spot blotch | 890 | 299 | 300 | 1489 |
| Yellow leaf blotch | 842 | 283 | 287 | 1412 |
| 总计 | 4475 | 1502 | 1516 | 7493 |

### 3.2 数据集结构

```text
dataset/
└─ wheat/
   ├─ train/
   │  ├─ Healthy/
   │  ├─ Leaf rust/
   │  ├─ Powdery mildew/
   │  ├─ Spot blotch/
   │  └─ Yellow leaf blotch/
   ├─ val/        # 结构同 train
   └─ test/       # 结构同 train
```

## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境（Python 3.9，PyTorch 2.5.1，示例为 CUDA 12.1 版本）：

```bash
conda create -n wheat-efficientnet python=3.9
conda activate wheat-efficientnet
pip install torch==2.5.1 torchvision==0.20.1
pip install numpy matplotlib pillow pandas tqdm
pip install scikit-learn thop tensorboard statsmodels
```

### 4.2 硬件要求

- GPU：NVIDIA RTX 4060 Laptop GPU（8GB），单组 40 轮训练约 1–1.5 小时
- CPU：可支持推理，训练速度较慢

## 5. 实验结果

### 5.1 核心指标对比

| 模型 | Params (M) | FLOPs (G) | Latency (ms/img) | Accuracy |
|---|---|---|---|---|
| EfficientNet（5 类头） | 4.014 | 0.386 | 1.01 | 91.53%* |
| Improved EfficientNet（基线） | 3.109 | 0.363 | 0.63 | 93.30%* |
| + SKBlock | 3.417 | 0.383 | 0.97 | 95.98% |
| + GeM | 3.110 | 0.363 | 0.63 | 98.02% |
| + SKBlock + GeM（本文 ENGS） | 3.417 | 0.383 | 0.98 | 98.81%* |

> 注：带 * 的准确率为原实验在修订前的模型实现上取得的结果（论文报告值，Manuscript under review）。论文投稿后模型实现经过了修订，为保持代码与论文一致，目前正基于修订后的代码重新复现各配置，完成后将以复现实测值更新本表。不带 * 的为基于本仓库当前权重在测试集（1516 张）上的实测值。Latency 以 batch size 32 在 RTX 4060 Laptop GPU 上测得。

### 5.2 统计显著性检验

在测试集上对消融变体进行 McNemar 检验：+GeM 显著优于 +SKB（χ² = 14.785, p < 0.001），表明 GeM 模块带来的提升具有统计显著性。

## 6. 代码使用说明

### 6.1 模型训练

运行 train.py 启动训练，支持消融开关与训练配置，示例命令：

```bash
# 完整模型（SKBlock + GeM）
python train.py --epochs 40 --no_gem_attn_fusion --use_gem_dynamic_p

# 基线（无 SKBlock，固定 p）
python train.py --epochs 40 --no_skblock --no_skb_attention --no_top_skb --no_gem_attn_fusion --no_gem_dynamic_p
```

常用参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| --train_data_path | 训练集目录 | dataset/wheat/train |
| --val_data_path | 验证集目录 | dataset/wheat/val |
| --epochs | 训练轮数 | 40 |
| --batch_size | 批次大小 | 16 |
| --lr | 初始学习率 | 0.01 |
| --weights | 预训练权重路径（warm start） | 空 |
| --use_skblock / --no_skblock | 是否启用 SKBlock | 启用 |
| --use_gem_attn_fusion / --no_gem_attn_fusion | GeM 是否融合注意力 | 启用 |
| --use_gem_dynamic_p / --no_gem_dynamic_p | p 是否可学习 | 启用 |
| --save_root | 结果保存目录 | ./ablation_results |

训练输出：验证集准确率最高的权重保存至 `ablation_results/<实验配置>/weights/best.pth`，逐轮指标记录于同目录 `metrics.csv`，TensorBoard 日志位于 `runs/` 子目录。

批量运行多组消融实验可参考 `run_repro_3configs.py` 或 `run_all_ablations.py`。

### 6.2 模型预测

使用 predict.py 对测试集或单张图像进行预测。运行前请修改脚本顶部路径配置：

```python
WEIGHT_PATH = r"...\ablation_results\<实验配置>\weights\best.pth"
IMG_DIR = r"...\dataset\wheat\test"
```

预测结果保存至 `evaluation_results_*/` 目录。统计显著性检验见 `mcnemar_test.py`。

### 6.3 预训练权重

本仓库提供基于自建小麦数据集训练的最优权重（Git LFS 管理，位于 `ablation_results/` 各配置目录下）。如仓库内权重下载受限，也可通过百度网盘获取：

链接: https://pan.baidu.com/s/1OG8uLUL0_OQL-BaDWNEhmA

适用场景：仅针对小麦叶片的 Healthy、Leaf rust、Powdery mildew、Spot blotch、Yellow leaf blotch 五类分类。如需扩展其他小麦病害，建议基于此权重微调。

## 7. 项目文件结构

```text
EffcientNet-b0/
├── model.py                  # 模型定义（SKBlock、GeM、改进 EfficientNet-B0）
├── train.py                  # 训练脚本（支持消融开关）
├── predict.py                # 预测/评估脚本
├── mcnemar_test.py           # McNemar 显著性检验
├── run_repro_3configs.py     # 消融实验批量运行脚本
├── run_all_ablations.py      # 消融实验运行脚本
├── dataset/                  # 数据集（见 3.2）
├── ablation_results/         # 各消融配置的训练结果与权重
└── weights/                  # 权重文件（Git LFS）
```

## 8. 注意事项

- 当前模型针对五类小麦叶片病害，扩展类别需重新训练或微调。
- 输入图像统一为 224×224 分辨率，与训练预处理保持一致。
- 如需部署到边缘设备，可使用 TensorRT/ONNX 进行量化加速。

## 9. 引用与联系方式

### 9.1 引用方式

论文处于投稿阶段，正式发表后将更新 BibTeX 引用格式，当前可临时引用：

```bibtex
@article{engs2026,
  title={ENGS: A Novel Deep Learning Model for Wheat Leaf Disease Identification},
  author={Zhou, Ruiqian and Dai, Wenhao and Li, Shengfan and Wu, Longguo and Xu, Peng and Yang, Shengyuan and Xu, Laixiang},
  journal={Under review},
  year={2026}
}
```

### 9.2 联系方式

若遇到代码运行问题或学术交流需求，请联系：

- 邮箱：daiwenhao@huuc.edu.cn
- GitHub Issue：直接在本仓库提交 Issue
