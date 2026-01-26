Improved EfficientNet-B0 + SKB + GeM for Wheat Disease Identification
官方 PyTorch 实现 | 论文标题：《A Deep Learning Model for Wheat Leaf Disease Recognition Based on Improved EfficientNet-B0 with GeM and SKBlock Attention Mechanism》

本文提出一种基于改进 EfficientNet-B0 并融合广义均值池化（GeM）与选择性核注意力（SKBlock）的小麦叶片病害识别模型，实现了对白粉病、叶锈病、斑枯病、黄叶斑病及健康叶片的高精度分类，为田间病害智能诊断提供高效轻量化解决方案。

1. 研究背景与模型定位
小麦是中国主要粮食作物之一，其病害严重影响产量与质量。传统病害检测方法效率低、适应性差，而现有轻量化模型在多尺度病斑特征提取与复杂背景鲁棒性方面存在不足。

本文提出一种改进的 EfficientNet-B0 模型，通过引入动态选择性卷积模块（SKBlock）与自适应广义均值池化（GeM），增强模型对多尺度病斑的识别能力与特征区分度，在自建小麦叶片病害数据集上达到 98.81% 的分类准确率，参数量仅为 32.18M，兼顾高精度与轻量化。

2. 核心创新点
动态选择性卷积模块（SKBlock）：
针对小麦病斑尺度差异（0.1–2 cm²），设计双分支卷积（3×3 与 5×5）与病斑区域注意力机制，增强对微小病斑的特征响应，较标准 SKBlock 提升 12%。

自适应广义均值池化（GeM）：
通过可学习参数 p（初始化为 3，优化至 2.9±0.1）动态调整池化行为，保留病斑纹理细节，提升对白粉病与锈病早期症状的区分能力，较全局平均池化（GAP）准确率提升 3.5%。

分阶段轻量化计算策略：
前3阶段禁用 SKBlock，采用深度可分离卷积降低 42% 计算量；后4阶段启用 SKBlock 并配合动态通道压缩（r=2–4），平衡模型精度与效率。

3. 实验数据集：WPLDD（自建小麦病害数据集）
3.1 数据集概况
数据集采集于河南省小麦种植基地，包含 5 个类别（健康、叶锈病、斑枯病、白粉病、黄叶斑病），共 7493 张图像（原始+GAN增强），图像统一预处理为 256×256 分辨率。

类别	训练集	验证集	测试集	合计
叶锈病	786	263	267	1316
斑枯病	890	299	300	1489
健康叶片	1032	344	347	1723
白粉病	925	313	315	1553
黄叶斑病	842	283	287	1412
总计	4475	1502	1516	7493
3.2 数据目录结构
text
WPLDD/
├── 健康/
├── 叶锈病/
├── 斑枯病/
├── 白粉病/
└── 黄叶斑病/
4. 实验环境配置
4.1 依赖安装
bash
conda create -n wheat-efficientnet python=3.9
conda activate wheat-efficientnet
pip install torch==2.5.1 torchvision==0.20.1
pip install numpy matplotlib opencv-python pillow tqdm pandas
4.2 硬件建议
GPU：NVIDIA RTX 4060（8GB+），训练约 2–3 小时

CPU：可支持推理，训练速度较慢

5. 实验结果
5.1 核心性能对比（WPLDD 数据集）
模型	准确率	参数量（M）	FLOPs（G）	推理速度（s/张）
EfficientNet-B0（基线）	93.30%	3.36	0.38	0.01
+ SKBlock	97.21%	32.18	2.42	0.10
+ GeM	98.75%	3.36	0.38	0.01
+ SKB + GeM（本文）	98.81%	32.18	2.42	0.09
5.2 消融实验与可视化
SKB 与 GeM 组合在各项指标（Accuracy、F1-Score、Precision、Sensitivity）上均优于单一模块。

热力图可视化显示，融合模块能更精准激活病斑区域，尤其是早期与微小病斑。

6. 代码使用说明
6.1 模型训练
bash
python train.py \
  --data_dir ./WPLDD \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --save_dir ./weights \
  --device cuda:0
6.2 单图预测
bash
python predict.py \
  --image_path ./test_leaf.jpg \
  --weight_path ./weights/best_model.pth \
  --device cuda:0
输出示例：

text
预测类别：小麦白粉病
置信度：0.988
6.3 预训练权重
我们提供在 WPLDD 上训练的最佳模型权重：

百度网盘链接（提取码：xxxx）

本地路径：weights/best_model.pth

7. 项目文件结构
text
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
8. 注意事项
当前模型针对 WPLDD 五类病害，扩展类别需重新训练或微调。

输入图像建议分辨率 ≥256×256，低分辨率可能影响小病斑识别。

如需部署到边缘设备，可使用 TensorRT/ONNX 进行量化加速。

9. 引用
若使用本模型或代码，请引用：

bibtex
@article{improved_efficientnet_wheat,
  title={A Deep Learning Model for Wheat Leaf Disease Recognition Based on Improved EfficientNet-B0 with GeM and SKBlock Attention Mechanism},
  author={Xu, Laixiang and Dai, Wenhao and Bijani, Madineh and Ahmad, Mohammad Nazir and Liu, Jia and Zhao, Junmin},
  journal={Submitted},
  year={2025}
}
10. 联系方式
如有问题或合作意向，请联系：

邮箱：daiwenhao@huuc.edu.cn

GitHub Issue：欢迎在本仓库提交问题
