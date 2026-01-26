import os
import sys
import json
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # 新增：用于生成分类报告

import torch
from torch import nn
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()  # 固定类别顺序
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    # 保存类别索引（路径修复）
    json_path = os.path.join("dataset/splits", "class_indices.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as json_file:
        json.dump(dict((val, key) for key, val in class_indices.items()), json_file, indent=4)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        if len(images) == 0:
            continue
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))

        # 分层抽样划分训练/验证集
        train_paths, val_paths = train_test_split(
            images, test_size=val_rate, random_state=0
        )

        train_images_path.extend(train_paths)
        train_images_label.extend([image_class] * len(train_paths))
        val_images_path.extend(val_paths)
        val_images_label.extend([image_class] * len(val_paths))

    print("{} images found in dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0 and len(val_images_path) > 0, "No training/validation images!"

    # 关键：返回类别名称列表（flower_class = class_names）
    return train_images_path, train_images_label, val_images_path, val_images_label, flower_class


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = 'D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


# 修改后的train_one_epoch函数
# utils.py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report


def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, total_epochs,
                    class_names, attn_reg_func=None, attn_reg_lambda=1e-5):
    """
    修复：添加 attn_reg_func 和 attn_reg_lambda 参数，用于SKB注意力正则化
    """
    model.train()
    loss_function = criterion
    total_loss = 0.0
    total_attn_loss = 0.0  # 记录注意力正则化损失
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # 使用tqdm显示进度条
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}/{total_epochs-1}", unit="batch") as pbar:
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 模型输出：必须返回(outputs, attn_loss)，且attn_loss为张量
            outputs, skb_attn = model(images)  # skb_attn为注意力图（可能为None）

            # 计算分类损失
            cls_loss = criterion(outputs, labels)

            # 计算注意力正则化损失（确保返回张量）
            attn_loss = attn_reg_func(skb_attn, attn_reg_lambda)
            # 强制转换为张量（终极保障）
            if not isinstance(attn_loss, torch.Tensor):
                attn_loss = torch.tensor(attn_loss, dtype=torch.float32, device=device)
            # 转移到与cls_loss同设备
            attn_loss = attn_loss.to(device)

            # 总损失 = 分类损失 + 注意力正则化损失
            loss = cls_loss + attn_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item() * labels.size(0)
            total_attn_loss += attn_loss.item() * labels.size(0)  # 此时attn_loss必为张量

            # 准确率计算
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{total_correct/total_samples:.4f}"})

    # 计算平均损失和准确率
    avg_loss = total_loss / total_samples
    avg_attn_loss = total_attn_loss / total_samples
    acc = total_correct / total_samples

    # 生成分类报告
    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return avg_loss, acc, report, avg_attn_loss


@torch.no_grad()  # 验证阶段不需要计算梯度，加速并节省内存
def evaluate(model, data_loader, device, class_names, criterion):
    model.eval()  # 切换模型到评估模式
    loss_function = criterion
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []  # 存储所有真实标签（y_true）
    all_preds = []   # 存储所有预测标签（y_pred）

    # 进度条显示
    with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            total_samples += labels.size(0)

            # 1. 前向传播：适配模型返回的 (分类输出, 注意力图) 元组，忽略注意力图
            outputs, _ = model(images)  # 只取分类输出，丢弃注意力图（验证阶段用不到）

            # 2. 计算验证损失
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # 按批次大小加权累积

            # 3. 计算准确率
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别（概率最大的类别）
            total_correct += (predicted == labels).sum().item()  # 统计正确预测数

            # 4. 收集标签和预测结果（用于后续计算 classification_report）
            # 注意：必须先转CPU再转numpy，否则会报错（GPU张量不能直接转numpy）
            all_labels.extend(labels.cpu().numpy())  # y_true：真实标签
            all_preds.extend(predicted.cpu().numpy()) # y_pred：预测标签

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_acc": f"{total_correct/total_samples:.4f}"})

    # 5. 计算平均验证损失和验证准确率
    avg_val_loss = total_loss / total_samples
    val_accuracy = total_correct / total_samples

    # 6. 生成分类报告（关键：确保参数顺序是 y_true, y_pred，且必传）
    # zero_division=0：避免某些类别无预测时出现除以零错误
    class_report = classification_report(
        y_true=all_labels,        # 第一个参数：真实标签（必需）
        y_pred=all_preds,         # 第二个参数：预测标签（必需）
        target_names=class_names, # 类别名称（可选，用于报告可读性）
        output_dict=True,         # 返回字典格式（方便后续取F1等指标）
        zero_division=0           # 无预测时默认0，避免报错
    )

    # 返回验证损失、验证准确率、分类报告
    return avg_val_loss, val_accuracy, class_report