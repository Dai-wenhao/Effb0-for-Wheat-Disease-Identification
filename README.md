# ENGS: An Novel Deep Learning Model for Wheat Leaf Disease Identification

>官方 PyTorch 实现 | 论文标题：《ENGS: An Novel Deep Learning Model for Wheat Leaf Disease Identification》


>本文提出一种基于改进 EfficientNet-B0 并融合广义均值池化（GeM）与选择性核注意力（SKBlock）的小麦叶片病害识别模型，实现了对白粉病、叶锈病、斑枯病、黄叶斑病及健康叶片的高精度分类，为田间病害智能诊断提供高效轻量化解决方案。


**#1. 研究背景与模型定位**
小麦是中国主要粮食作物之一，其病害严重影响产量与质量。传统病害检测方法效率低、适应性差，而现有轻量化模型在多尺度病斑特征提取与复杂背景鲁棒性方面存在不足。

本文提出一种改进的 EfficientNet-B0 模型，通过引入动态选择性卷积模块（SKBlock）与自适应广义均值池化（GeM），增强模型对多尺度病斑的识别能力与特征区分度，在自建小麦叶片病害数据集上达到 98.81% 的分类准确率，参数量仅为 32.18M，兼顾高精度与轻量化。


**#2. 核心创新点**
- 动态选择性卷积模块（SKBlock）：
  针对小麦病斑尺度差异（0.1–2 cm²），设计双分支卷积（3×3 与 5×5）与病斑区域注意力机制，增强对微小病斑的特征响应，较标准 SKBlock 提升 12%。

- 自适应广义均值池化（GeM）：
  通过可学习参数 p（初始化为 3，优化至 2.9±0.1）动态调整池化行为，保留病斑纹理细节，提升对白粉病与锈病早期症状的区分能力，较全局平均池化（GAP）准确率提升 3.5%。

- 分阶段轻量化计算策略：
  前3阶段禁用 SKBlock，采用深度可分离卷积降低 42% 计算量；后4阶段启用 SKBlock 并配合动态通道压缩（r=2–4），平衡模型精度与效率。



**#3. 实验数据集：WPLDD（自建小麦病害数据集）**
**##3.1 数据集概况**
数据集采集于河南省小麦种植基地，包含 5 个类别（健康、叶锈病、斑枯病、白粉病、黄叶斑病），共 7493 张图像（原始+GAN增强），图像统一预处理为 256×256 分辨率。

| 类别     | 训练集 | 验证集 | 测试集 | 合计  |
|----------|--------|--------|--------|-------|
| 叶锈病   | 786    | 263    | 267    | 1316  |
| 斑枯病   | 890    | 299    | 300    | 1489  |
| 健康叶片 | 1032   | 344    | 347    | 1723  |
| 白粉病   | 925    | 313    | 315    | 1553  |
| 黄叶斑病 | 842    | 283    | 287    | 1412  |
| 总计     | 4475   | 1502   | 1516   | 7493  |

## 3.2 数据集结构
```text
WPLDD/
├─ 枯萎病/
├─ 白粉病/
├─ 斑枯病/
├─ 叶锈病/
└─ 健康植株/
```

#4. 实验环境配置
##4.1 依赖安装
推荐使用Anaconda创建虚拟环境，确保依赖版本匹配（避免兼容性问题，尤其适配PyTorch 2.7.1）：
```
# 1. 创建并激活虚拟环境
conda create -n wheat-efficientnet python=3.9
conda activate wheat-efficientnet
pip install numpy matplotlib opencv-python pillow tqdm pandas

# 2. 安装PyTorch与TorchVision（需适配CUDA版本，示例为CUDA 12.1；CPU用户可替换为cpu版本）
pip install torch==2.5.1 torchvision==0.20.1

# 3. 安装其他依赖库（数据处理、可视化、模型工具等）
pip install numpy~=2.0.2 matplotlib~=3.9.4 opencv-python~=4.12.0.88
pip install pandas~=2.3.1 pillow~=11.2.1 torchviz~=0.0.3 xlwt~=1.3.0
pip install tqdm~=4.67.1 timm~=1.0.15
```
##4.2 硬件要求
GPU：NVIDIA RTX 4060（8GB+），训练约 2–3 小时
CPU：可支持推理，训练速度较慢

#5. 实验结果
##5.1 核心指标对比
HH-Former 与主流深度学习模型在小麦叶片病害分类任务上的性能对比如下，模型在精度、计算效率上均表现更优，尤其对相似病害（条锈病/叶锈病）的区分能力显著提升：

模型	分类准确率（Accuracy）	计算量（FLOPs）	参数量（M）
EfficientNet-B0（基线）	93.30%	3.36	0.38	0.01
+ SKBlock	97.21%	32.18	2.42	0.10
+ GeM	98.75%	3.36	0.38	0.01
+ SKB + GeM（本文）	98.81%	32.18	2.42	0.09

#6. 代码使用说明
##6.1 模型训练
运行 train.py 脚本启动训练，支持通过参数调整训练配置，示例命令：
```
python train.py \
  --data_dir ./WPLDD \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_dir ./weights \
  --device cuda:0
```
关键参数说明：
参数名	含义	默认值
--data_dir	WPLDD数据集根目录路径	./WPLDD
--epochs	训练轮数	100
--batch_size	批次大小（根据GPU显存调整，8/16/32）	16
--lr	初始学习率	1e-4
--save_dir	训练权重保存目录	./weights
--device	训练设备（cuda:0 或 cpu）	cuda:0
训练输出：
训练过程中，模型会自动保存验证集准确率最高的权重至 --save_dir 目录，文件名为 best-model.pth；
训练日志（损失值、准确率）会实时打印，并保存至 train_log.txt。
##6.2 模型预测
使用训练好的权重进行单张小麦叶片图像预测，运行 predict.py 脚本，示例命令：
```
python predict.py \
  --image_path ./test_leaf.jpg \
  --weight_path ./weights/best_model.pth \
  --device cuda:0
```
预测输出示例：
预测类别：小麦白粉病
置信度：0.988
##6.3 预训练权重
提供基于 WPLDD 数据集训练完成的最优权重，可直接用于预测或微调。除随项目仓库附带的权重外，也可通过百度网盘获取完整权重文件：

百度网盘分享： 链接: https://pan.baidu.com/s/1OG8uLUL0_OQL-BaDWNEhmA  本地权重文件：weights/best_hh_former.pth（若仓库内权重存在大小限制，可通过上述网盘链接获取完整版本）； 适用场景：仅针对小麦叶片的 “白粉病、枯萎病、叶锈病、斑枯病、健康叶片” 五类分类，若需扩展其他小麦病害，建议基于此权重微调（冻结浅层注意力模块，仅训练分类头与深层特征融合层，可减少 50% 以上训练数据量）。

#7. 项目文件结构
```
Wheat-EfficientNet-SKB-GeM/
├── WPLDD/                 # 数据集
├── models/                # 模型定义
│   ├── efficientnet_skb_gem.py
│   └── sk_block.py
├── dataset/               # 数据加载与增强
│   ├── data_loader.py
│   └── augmentation.py
├── train.py              # 训练脚本
├── predict.py            # 预测脚本
├── evaluate.py           # 评估脚本
└── README.md
```

#8. 注意事项
当前模型针对 WPLDD 五类病害，扩展类别需重新训练或微调。
输入图像建议分辨率 ≥256×256，低分辨率可能影响小病斑识别。
如需部署到边缘设备，可使用 TensorRT/ONNX 进行量化加速。

#9. 引用与联系方式
##9.1 引用方式
论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：
```
@article{improved_efficientnet_wheat,
  title={A Deep Learning Model for Wheat Leaf Disease Recognition Based on Improved EfficientNet-B0 with GeM and SKBlock Attention Mechanism},
  author={Xu, Laixiang and Dai, Wenhao and Bijani, Madineh and Ahmad, Mohammad Nazir and Liu, Jia and Zhao, Junmin},
  journal={Submitted},
  year={2026}
}
```
##9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：

邮箱：daiwenhao@huuc.edu.cn
GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。
