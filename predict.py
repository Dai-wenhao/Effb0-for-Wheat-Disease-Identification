import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
# ========== 新增：导入计算核心指标所需模块 ==========
import time
from thop import profile  # 用于计算Params/FLOPs
# ==================================================

# 导入你的模型（确保路径正确）
from model import efficientnet_b0 as create_model

# ==================== 路径配置 ====================
WEIGHT_PATH = r"D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\ablation_results\skblock=True_skb_attn=True_top_skb=True_gem_fusion=True_gem_dynamic_p=True\weights\best.pth"
IMG_DIR = r"D:\pycharm\pycharmprojects\learn-pytorch\EffcientNet-b0\data\wheat\test"
SAVE_DIR = r"./evaluation_results"
SINGLE_IMG_RESULT_PATH = os.path.join(SAVE_DIR, "single_image_results.txt")
# 新增：保存模型核心指标的路径（可选）
MODEL_METRICS_PATH = os.path.join(SAVE_DIR, "model_core_metrics.txt")


# ==================================================


def load_model(weight_path, device):
    """加载模型：确保结构与训练时一致"""
    model = create_model(
        num_classes=5,
        use_skblock=True,  # 根据权重文件结构调整（与训练时一致）
        skb_use_attention=True,
        use_top_skb=True,
        use_gem_attn_fusion=True,
        use_gem_dynamic_p=True
    ).to(device)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")

    weights = torch.load(weight_path, map_location=device, weights_only=True)
    model_weights = weights["model_state_dict"] if "model_state_dict" in weights else weights

    # 非严格加载（忽略结构差异的层）
    model.load_state_dict(model_weights, strict=False)
    model.eval()
    print("模型加载完成（非严格匹配），已设置为eval模式")

    class_names = weights.get('class_names',
                              ["Healthy", "Leaf rust", "Powdery mildew", "Spot blight", "Yellow leaf blotch"])
    return model, class_names


def predict_single_image(img_path, model, device):
    """预测单张图片，返回预测索引和完整概率分布（用于提取true_class_prob）"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"警告：读取图片 {os.path.basename(img_path)} 失败：{str(e)}，跳过该图片")
        return None, None

    with torch.no_grad():
        tensor = transform(img).unsqueeze(0).to(device)
        outputs, _ = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        pred_idx = np.argmax(probs)

    return pred_idx, probs


def generate_confusion_matrix(y_true, y_pred, class_names, num_classes):
    """生成混淆矩阵（仅返回矩阵）"""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def calculate_specificity(cm, class_idx):
    """计算特异度（TN/(TN+FP)）"""
    tn = cm.sum() - cm[class_idx, :].sum() - cm[:, class_idx].sum() + cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - cm[class_idx, class_idx]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def calculate_class_accuracy_binary(cm, class_idx):
    """二分法准确率：(TP + TN) / 总样本数"""
    total_samples = cm.sum()
    if total_samples == 0:
        return 0.0

    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp
    tn = total_samples - tp - fn - fp

    return (tp + tn) / total_samples


def calculate_metric_correlation(metric_data):
    """计算指标间的皮尔逊相关系数矩阵并打印"""
    df = pd.DataFrame(metric_data)
    corr_matrix = df.corr(method='pearson')
    print("\n================ 评估指标相关性矩阵（皮尔逊系数） ================")
    print(corr_matrix)
    return corr_matrix


def calculate_model_core_metrics(model, device, input_shape=(1, 3, 224, 224)):
    """
    计算模型参数量、计算量、单张图片推理速度（优化精度）
    :param model: 加载好的模型（已to(device)）
    :param device: 运行设备（cuda/cpu）
    :param input_shape: 模型输入尺寸 (batch, C, H, W)
    :return: params_m（M）、flops_g（G）、avg_speed（s，保留3位小数）
    """
    print("\n" + "=" * 50)
    print("开始计算模型核心指标（Params/FLOPs/Speed）...")

    # 生成虚拟输入（匹配模型输入尺寸）
    dummy_input = torch.randn(input_shape).to(device)
    model.eval()

    # 1. 计算参数量（Params）和计算量（FLOPs）（保持原精度）
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        params_m = round(params / 1e6, 2)  # 仍保留2位小数（符合行业习惯）
        flops_g = round(flops / 1e9, 2)  # 仍保留2位小数

    # 2. 优化推理速度计算：增加迭代次数+保留3位小数
    warmup_iter = 20  # 增加预热迭代（减少GPU初始化波动）
    test_iter = 200  # 增加测试迭代（提升平均值稳定性）
    with torch.no_grad():
        # 预热（确保GPU/CPU进入稳定状态）
        for _ in range(warmup_iter):
            model(dummy_input)

        # 精确计时（用time.perf_counter()，比time.time()精度更高）
        start_time = time.perf_counter()
        for _ in range(test_iter):
            model(dummy_input)
        end_time = time.perf_counter()

    # 保留3位小数，避免快速推理显示为0.0 s
    avg_speed = round((end_time - start_time) / test_iter, 3)

    # 打印结果（Speed显示3位小数）
    print(f"✅ 参数量（Params）：{params_m} M")
    print(f"✅ 计算量（FLOPs）：{flops_g} G")
    print(f"✅ 单张图片推理速度（Speed）：{avg_speed} s")
    print("=" * 50 + "\n")

    return params_m, flops_g, avg_speed


# ==================================================


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"开始小麦病害分类测试，使用设备：{device}")

    # 1. 加载模型和类别
    model, class_names = load_model(WEIGHT_PATH, device)
    num_classes = len(class_names)

    # ========== 新增：调用核心指标计算函数 ==========
    # 输入尺寸与predict_single_image保持一致（224×224×3）
    params_m, flops_g, avg_speed = calculate_model_core_metrics(model, device, input_shape=(1, 3, 224, 224))
    # ==================================================

    # 2. 初始化统计变量（保留true_class_prob）
    class_total = [0] * num_classes
    class_correct = [0] * num_classes  # 新增：统计每个类别“正确预测数”
    single_img_results = []  # 包含：路径、真实类别、预测类别、真实类别概率、是否正确
    y_true, y_pred = [], []

    # 3. 遍历测试集
    if not os.path.isdir(IMG_DIR):
        raise NotADirectoryError(f"测试集文件夹不存在：{IMG_DIR}")
    print(f"\n当前测试集根目录：{IMG_DIR}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(IMG_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"类别文件夹不存在，跳过：{class_dir}")
            continue

        img_count = 0
        for filename in os.listdir(class_dir):
            img_ext = os.path.splitext(filename)[1].lower()
            if img_ext not in ['.jpg', '.jpeg', '.png']:
                continue

            img_path = os.path.join(class_dir, filename)
            true_label = class_idx
            pred_idx, all_probs = predict_single_image(img_path, model, device)
            if pred_idx is None or all_probs is None:
                continue

            pred_class_name = class_names[pred_idx]
            is_correct = (pred_idx == true_label)
            true_class_prob = all_probs[true_label]  # 保留真实类别的预测概率

            # 记录结果（保留true_class_prob）
            single_img_results.append({
                "img_path": img_path,
                "true_class": class_name,
                "pred_class": pred_class_name,
                "true_class_prob": true_class_prob,
                "is_correct": is_correct
            })

            # 更新统计
            y_true.append(true_label)
            y_pred.append(pred_idx)
            class_total[true_label] += 1
            if is_correct:  # 新增：仅当预测正确时，对应类别正确数+1
                class_correct[true_label] += 1
            img_count += 1

        print(f"已处理类别 {class_name}：{img_count} 张有效图片")

    # 4. 生成评估结果
    if y_true and y_pred:
        print(f"\n测试集统计：有效样本总数 {len(y_true)} 张")

        # 混淆矩阵
        cm = generate_confusion_matrix(y_true, y_pred, class_names, num_classes)

        # 各类别指标计算（去掉“（二分法）”标注）
        class_metrics = []
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        for i, cls in enumerate(class_names):
            accuracy = calculate_class_accuracy_binary(cm, i)  # 二分法逻辑但不标注
            precision = report[cls]['precision']
            sensitivity = report[cls]['recall']
            specificity = calculate_specificity(cm, i)
            f1 = report[cls]['f1-score']
            class_metrics.append({
                "类别": cls,
                "Accuracy": accuracy,  # 去掉“（二分法）”
                "Precision": precision,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "F1-score": f1
            })
            print(f"\n=== 类别 {cls} 指标 ===")
            print(f"Accuracy: {accuracy:.4f}")  # 去掉“（二分法）”
            print(f"Precision: {precision:.4f}")
            print(f"Sensitivity: {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"该类真实样本数: {class_total[i]}")
            print(f"该类正确预测数: {class_correct[i]}")  # 新增：打印正确预测数

        # 指标相关性矩阵（使用修改后的Accuracy字段）
        metric_data = []
        for metric in class_metrics:
            metric_data.append([
                metric["Accuracy"],  # 对应修改后的字段
                metric["Precision"],
                metric["Sensitivity"],
                metric["Specificity"],
                metric["F1-score"]
            ])
        calculate_metric_correlation(metric_data)

        # 整体准确率
        overall_acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"\n整体测试准确率：{overall_acc:.4f}（有效样本：{len(y_true)}）")

        # 保存单张图片结果（新增模型核心指标）
        os.makedirs(SAVE_DIR, exist_ok=True)
        with open(SINGLE_IMG_RESULT_PATH, 'w', encoding='utf-8') as f:
            f.write("小麦病害分类单张图片预测详情\n")
            f.write("=" * 180 + "\n")
            # 新增：写入模型核心指标和整体准确率
            f.write(f"模型核心指标：参数量={params_m} M | 计算量={flops_g} G | 单张推理速度={avg_speed} s\n")
            f.write(f"整体测试准确率：{overall_acc:.4f} | 有效测试样本数：{len(y_true)}\n")
            f.write("=" * 180 + "\n")
            f.write(f"{'图片路径':<80} | {'真实类别':<20} | {'预测类别':<20} | {'真实类别概率':<15} | {'是否正确'}\n")
            f.write("-" * 180 + "\n")
            for res in single_img_results:
                # 调整路径显示长度（避免换行），保留关键信息
                display_path = res['img_path'][-70:] if len(res['img_path']) > 70 else res['img_path']
                f.write(
                    f"{display_path:<80} | {res['true_class']:<20} | {res['pred_class']:<20} | {res['true_class_prob']:.6f}           | {'是' if res['is_correct'] else '否'}\n"
                )
        print(f"\n单张图片结果已保存至：{SINGLE_IMG_RESULT_PATH}")

        # ========== 新增：单独保存模型核心指标（便于后续整理表格） ==========
        with open(MODEL_METRICS_PATH, 'w', encoding='utf-8') as f:
            f.write("小麦病害分类模型核心指标\n")
            f.write("=" * 50 + "\n")
            f.write(f"参数量（Params）：{params_m} M\n")
            f.write(f"计算量（FLOPs）：{flops_g} G\n")
            f.write(f"单张图片推理速度（Speed）：{avg_speed} s\n")
            f.write(f"整体测试准确率（Accuracy）：{overall_acc:.4f}\n")
            f.write(f"测试样本总数：{len(y_true)}\n")
        print(f"模型核心指标已单独保存至：{MODEL_METRICS_PATH}")
        # ==================================================

    else:
        print("\n未找到有效图片，无法生成评估结果！")

    print("\n测试完成！")


if __name__ == "__main__":
    main()