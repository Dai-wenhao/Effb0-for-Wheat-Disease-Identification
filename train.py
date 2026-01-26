import os
import argparse
import csv
from collections import Counter
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np

# 导入外部依赖
from model import efficientnet_b0 as create_model  # 模型需返回(outputs, attn_loss)，且attn_loss为张量
from utils import train_one_epoch, evaluate

# 全局配置
torch.autograd.set_detect_anomaly(True)
plt.switch_backend('Agg')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ======================== 1. 损失函数：Focal Loss ========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", device='cuda:0'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha.to(device) if alpha is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")

    def forward(self, inputs, targets):
        targets = targets.long()
        ce = self.ce_loss(inputs, targets)
        log_pt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(log_pt)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        if self.alpha is not None:
            focal = -self.alpha[targets] * (1 - pt) ** self.gamma * log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            focal = -(1 - pt) ** self.gamma * log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        return focal.mean() if self.reduction == "mean" else focal.sum()


# ======================== 2. 注意力正则化：确保返回张量 ========================
def skb_attn_regularization(attn_map, lambda_reg=1e-5):
    """无论是否有注意力图，均返回PyTorch张量"""
    if attn_map is None:
        # 返回0张量（默认CPU，后续会转移到设备）
        return torch.tensor(0.0, dtype=torch.float32)

    # 确保损失计算在张量上进行
    batch_size = attn_map.shape[0]
    sparse_loss = torch.tensor(0.0, dtype=torch.float32, device=attn_map.device)  # 与attn_map同设备
    for i in range(batch_size):
        attn_single = torch.mean(attn_map[i], dim=0)
        sparse_loss += torch.norm(attn_single, p=1)  # L1正则化

    return lambda_reg * sparse_loss / batch_size


# ======================== 3. 数据集类 ========================
class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, is_val: bool = False):
        self.images_path = images_path
        self.images_class = images_class
        self.is_val = is_val

        # 基础预处理（所有图像必经步骤）
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 训练集增强（仅对PIL图像操作）
        if not is_val:
            self.major_aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        label = self.images_class[item]

        # 图像读取与异常处理
        try:
            img = Image.open(img_path).convert("RGB")
            if img.mode != "RGB":
                raise ValueError(f"图像 {img_path} 模式不是RGB，已强制转换")
        except Exception as e:
            print(f"警告：读取图像 {img_path} 失败，原因：{str(e)}，使用空白图像替代")
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))

        # 训练集增强
        if not self.is_val:
            img = self.major_aug(img)

        # 转为张量并归一化
        img_tensor = self.base_transform(img)
        return img_tensor, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


# ======================== 4. 数据准备 ========================
def get_image_paths_and_labels(data_dir: str):
    classes = []
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for cls_name in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(img_extensions)]
        if len(img_files) > 0:
            classes.append(cls_name)

    if len(classes) == 0:
        raise ValueError(f"数据目录 {data_dir} 中未找到有效类别或图像文件")

    classes.sort()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    image_paths = []
    labels = []
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(img_extensions):
                img_path = os.path.join(cls_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[cls_name])

    return image_paths, labels, classes


def prepare_data(args):
    # 加载图像路径和标签
    train_image_paths, train_labels, class_names = get_image_paths_and_labels(args.train_data_path)
    val_image_paths, val_labels, _ = get_image_paths_and_labels(args.val_data_path)

    # 创建数据集
    train_dataset = MyDataSet(images_path=train_image_paths, images_class=train_labels, is_val=False)
    val_dataset = MyDataSet(images_path=val_image_paths, images_class=val_labels, is_val=True)

    # 数据加载参数
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    print(f"\n数据加载配置：批次大小={batch_size}，Worker数={nw}")

    # 打印类别分布
    print("\n【训练集类别分布】")
    class_counts = Counter(train_labels)
    total_train = len(train_labels)
    for cls_idx in sorted(class_counts.keys()):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"未知类别{cls_idx}"
        count = class_counts[cls_idx]
        ratio = count / total_train * 100
        print(f"类别 {cls_name}（索引{cls_idx}）: {count} 样本（占比{ratio:.1f}%）")

    # 类别权重与加权采样
    num_classes = len(class_names)
    class_weights = torch.tensor(
        [total_train / (num_classes * class_counts.get(cls_idx, 1)) for cls_idx in range(num_classes)],
        dtype=torch.float
    )
    print(f"\n类别权重：{class_weights.tolist()}")

    sample_weights = class_weights[train_labels]
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=nw,
        drop_last=True,
        collate_fn=MyDataSet.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=nw,
        collate_fn=MyDataSet.collate_fn
    )

    return train_loader, val_loader, class_names, class_weights


# ======================== 5. 消融实验工具 ========================
def get_experiment_id(args):
    """生成包含关键参数的实验ID"""
    id_parts = []
    id_parts.append(f"skblock={args.use_skblock}")
    if args.use_skblock:
        id_parts.append(f"skb_attn={args.skb_use_attention}")
        id_parts.append(f"top_skb={args.use_top_skb}")
    id_parts.append(f"gem_fusion={args.use_gem_attn_fusion}")
    id_parts.append(f"gem_dynamic_p={args.use_gem_dynamic_p}")
    return "_".join(id_parts)


# ======================== 6. 训练循环 ========================
def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader,
                device, args, tb_writer, class_names, exp_save_path):
    best_acc = 0.0
    best_epoch = 0
    attn_reg_lambda = args.attn_reg_lambda
    attn_vis_dir = os.path.join(exp_save_path, "attn_vis")
    weights_dir = os.path.join(exp_save_path, "weights")
    os.makedirs(attn_vis_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # 初始化指标CSV
    metrics_file = os.path.join(exp_save_path, 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['epoch', 'train_loss', 'train_acc', 'train_weighted_f1',
                   'val_loss', 'val_acc', 'val_weighted_f1', 'lr']
        headers.extend([f"val_{cls}_f1" for cls in class_names])
        writer.writerow(headers)

    # 训练循环
    for epoch in range(args.epochs):
        # 训练阶段（确保attn_loss为张量）
        train_loss, train_acc, train_report, train_attn_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            class_names=class_names,
            attn_reg_func=skb_attn_regularization,
            attn_reg_lambda=attn_reg_lambda
        )

        # 验证阶段
        val_loss, val_acc, val_report = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            class_names=class_names,
            criterion=criterion
        )

        scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch,
                'best_acc': best_acc,
                'class_names': class_names
            }, os.path.join(weights_dir, "best.pth"))
            print(f"[epoch {epoch}] 保存最佳模型！验证准确率：{best_acc:.3f}")

        # 打印日志
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"\n[epoch {epoch}/{args.epochs - 1}] "
            f"训练总损失: {train_loss:.3f}, "
            f"注意力正则化损失: {train_attn_loss:.6f}, "
            f"训练准确率: {train_acc:.3f}, "
            f"验证损失: {val_loss:.3f}, "
            f"验证准确率: {val_acc:.3f}, "
            f"当前学习率: {current_lr:.6f}"
        )
        print(f"[epoch {epoch}] 训练加权F1: {train_report['weighted avg']['f1-score']:.3f}")
        print(f"[epoch {epoch}] 验证加权F1: {val_report['weighted avg']['f1-score']:.3f}")

        # TensorBoard记录
        tb_writer.add_scalar("train/total_loss", train_loss, epoch)
        tb_writer.add_scalar("train/attn_reg_loss", train_attn_loss, epoch)
        tb_writer.add_scalar("train/accuracy", train_acc, epoch)
        tb_writer.add_scalar("train/weighted_f1", train_report['weighted avg']['f1-score'], epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/accuracy", val_acc, epoch)
        tb_writer.add_scalar("val/weighted_f1", val_report['weighted avg']['f1-score'], epoch)
        tb_writer.add_scalar("lr/current", current_lr, epoch)

        # 写入CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                epoch, train_loss, train_acc, train_report['weighted avg']['f1-score'],
                val_loss, val_acc, val_report['weighted avg']['f1-score'], current_lr
            ]
            row.extend([val_report[cls]['f1-score'] for cls in class_names])
            writer.writerow(row)

        # 注意力图可视化（关闭SKB时自动跳过）
        if (epoch + 1) % args.vis_interval == 0:
            model.eval()
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    outputs, skb_attn = model(images)  # 模型返回(outputs, attn_loss)，此处skb_attn为注意力图
                    if skb_attn is not None and isinstance(skb_attn, torch.Tensor):
                        attn_vis = torch.mean(skb_attn[0], dim=0).detach().cpu().numpy()
                        vis_path = os.path.join(attn_vis_dir, f"epoch_{epoch}_batch_{batch_idx}_attn.png")
                        plt.imsave(vis_path, attn_vis, cmap='jet')
                        print(f"[epoch {epoch}] 保存注意力图到: {vis_path}")
                    break
            model.train()

    print(f"\n训练完成！最佳验证准确率: {best_acc:.3f}（对应epoch: {best_epoch}）")
    tb_writer.close()


# ======================== 7. 主函数 ========================
def main(args):
    # 生成实验ID与保存路径
    exp_id = get_experiment_id(args)
    exp_save_path = os.path.join(args.save_root, exp_id)
    os.makedirs(exp_save_path, exist_ok=True)
    print(f"实验ID: {exp_id}")
    print(f"实验结果保存路径: {exp_save_path}")

    # 设备初始化
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # TensorBoard初始化
    runs_dir = os.path.join(exp_save_path, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=runs_dir)
    print(f"TensorBoard日志目录: {runs_dir}")
    print(f"启动命令: tensorboard --logdir={runs_dir}")

    # 数据准备
    print("\n=== 开始数据准备 ===")
    train_loader, val_loader, class_names, class_weights = prepare_data(args)
    num_classes = len(class_names)
    print(f"=== 数据准备完成 ===")
    print(f"类别数量: {num_classes}，类别名称: {class_names}")

    # 模型初始化（传递消融参数）
    print("\n=== 初始化模型 ===")
    model = create_model(
        num_classes=num_classes,
        use_skblock=args.use_skblock,
        skb_use_attention=args.skb_use_attention,
        use_top_skb=args.use_top_skb,
        use_gem_attn_fusion=args.use_gem_attn_fusion,
        use_gem_dynamic_p=args.use_gem_dynamic_p
    ).to(device)
    print(f"模型结构: {model.__class__.__name__}")
    print(
        f"消融配置：use_skblock={args.use_skblock}, gem_fusion={args.use_gem_attn_fusion}, gem_dynamic_p={args.use_gem_dynamic_p}")

    # 加载预训练权重
    if args.weights != "":
        if os.path.exists(args.weights):
            try:
                weights_dict = torch.load(args.weights, map_location=device)
                model_state_dict = model.state_dict()
                valid_weights = {}
                for k, v in weights_dict.items():
                    if "model_state_dict" in weights_dict:
                        k = k.replace("model_state_dict.", "")
                        v = weights_dict["model_state_dict"][k]
                    if k in model_state_dict and model_state_dict[k].shape == v.shape:
                        valid_weights[k] = v
                model.load_state_dict(valid_weights, strict=False)
                print(f"成功加载 {len(valid_weights)}/{len(model_state_dict)} 个有效权重")
            except Exception as e:
                print(f"警告：加载权重 {args.weights} 失败，原因：{str(e)}，使用随机初始化")
        else:
            raise FileNotFoundError(f"预训练权重文件不存在: {args.weights}")

    # 冻结层配置
    if args.freeze_layers:
        print("\n=== 冻结骨干网络 ===")
        frozen_params = 0
        trainable_params = 0
        for name, para in model.named_parameters():
            if any(key in name for key in ["top_skb", "expand_skb", "dwconv_skb", "classifier", "avgpool"]):
                para.requires_grad_(True)
                trainable_params += para.numel()
            else:
                para.requires_grad_(False)
                frozen_params += para.numel()
        print(f"冻结参数数量: {frozen_params / 1e6:.2f} M")
        print(f"可训练参数数量: {trainable_params / 1e6:.2f} M")
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"全量训练，可训练参数数量: {total_params / 1e6:.2f} M")

    # 优化器配置（动态分组，避免空分组）
    print("\n=== 初始化优化器 ===")
    pg1 = []  # SKB相关层
    pg2 = []  # 其他层
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue
        if any(key in name for key in ["skb", "avgpool"]):
            pg1.append(para)
        else:
            pg2.append(para)

    # 动态构建参数组
    optimizer_params = []
    if pg1:
        optimizer_params.append({"params": pg1, "lr": args.lr * 2.0})
    if pg2:
        optimizer_params.append({"params": pg2, "lr": args.lr})

    optimizer = optim.SGD(
        optimizer_params,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    # 打印优化器配置
    if len(optimizer_params) == 2:
        print(f"优化器分组：SKB相关层（学习率{args.lr * 2.0:.6f}），其他层（学习率{args.lr:.6f}）")
    else:
        print(f"优化器分组：所有可训练层（学习率{args.lr:.6f}）")

    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
        last_epoch=-1
    )

    # 损失函数
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=args.focal_gamma,
        device=device
    )
    print(f"损失函数：Focal Loss（gamma={args.focal_gamma}）")

    # 启动训练
    print("\n=== 开始训练 ===")
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        args=args,
        tb_writer=tb_writer,
        class_names=class_names,
        exp_save_path=exp_save_path
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="小麦病害分类训练（支持消融实验）")
    # 数据配置
    parser.add_argument('--train_data_path', type=str,
                        default=r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\wheat\train',
                        help='训练集目录')
    parser.add_argument('--val_data_path', type=str,
                        default=r'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\wheat\val',
                        help='验证集目录')

    # 模型配置
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze_layers', type=bool, default=False)

    # 训练配置
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # 损失函数配置
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # 注意力配置
    parser.add_argument('--attn_reg_lambda', type=float, default=1e-5)
    parser.add_argument('--vis_interval', type=int, default=10)

    # 设备配置
    parser.add_argument('--device', type=str, default='cuda:0')

    # 消融实验参数（默认：关闭SKB+打开GeM）
    parser.add_argument('--use_skblock', action='store_true', default=True,
                        help='是否启用WheatSKBlock（默认关闭）')
    parser.add_argument('--no_skblock', action='store_false', dest='use_skblock',
                        help='关闭WheatSKBlock（消融实验）')

    parser.add_argument('--skb_use_attention', action='store_true', default=True,
                        help='是否启用WheatSKBlock内部注意力（默认关闭）')
    parser.add_argument('--no_skb_attention', action='store_false', dest='skb_use_attention',
                        help='关闭WheatSKBlock内部注意力（消融实验）')

    parser.add_argument('--use_top_skb', action='store_true', default=True,
                        help='是否启用顶层WheatSKBlock（默认关闭）')
    parser.add_argument('--no_top_skb', action='store_false', dest='use_top_skb',
                        help='关闭顶层WheatSKBlock（消融实验）')

    parser.add_argument('--use_gem_attn_fusion', action='store_true', default=True,
                        help='是否启用WheatGeM注意力融合（默认打开）')

    parser.add_argument('--use_gem_dynamic_p', action='store_true', default=True,
                        help='是否启用WheatGeM动态p值（默认打开）')

    # 实验保存根目录
    parser.add_argument('--save_root', type=str, default='./ablation_results',
                        help='消融实验结果根目录')

    args = parser.parse_args()

    # 参数验证
    for path in [args.train_data_path, args.val_data_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据目录不存在：{path}")

    main(args)








































































