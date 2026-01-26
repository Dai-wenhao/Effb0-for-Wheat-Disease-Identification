import math
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from thop import profile
import logging

logging.getLogger('thop').disabled = True


# 工具类和函数保持不变
class FLOPsCounter:
    _calculated = False
    _flops = None
    _params = None

    @classmethod
    def calculate(cls, model, input_tensor):
        if not cls._calculated:
            cls._flops, cls._params = profile(model, inputs=(input_tensor,))
            cls._calculated = True
        return cls._flops, cls._params


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# ======================== 1. WheatSKBlock（添加开关） ========================
class WheatSKBlock(nn.Module):
    def __init__(self, channels, stride=1, kernel_list=[1,3], reduction=8, texture_dim_ratio=0.125,
                 use_attention=True):  # 新增：注意力开关
        super().__init__()
        self.M = len(kernel_list)
        self.stride = stride
        self.channels = channels
        self.use_attention = use_attention  # 控制是否启用双门控注意力

        # 双分支卷积（保留，即使关闭注意力也需要基础特征提取）
        self.convs = nn.ModuleList()
        for k in kernel_list:
            padding = (k - 1) // 2
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k,
                              stride=stride, padding=padding,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=False)
                )
            )

        # 注意力模块（仅当use_attention=True时初始化）
        self.texture_attn = None
        self.fc = None
        self.softmax = None
        if self.use_attention:
            self.disease_texture_dim = max(int(channels * texture_dim_ratio), 4)
            self.texture_attn = nn.Sequential(
                nn.Conv2d(channels, self.disease_texture_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.disease_texture_dim),
                nn.SiLU(inplace=False),
                nn.Conv2d(self.disease_texture_dim, self.M, kernel_size=3, padding=1, bias=False),
                nn.Softmax(dim=1)
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            reduction_dim = max(channels // reduction, 4)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduction_dim),
                nn.BatchNorm1d(reduction_dim),
                nn.SiLU(inplace=False),
                nn.Linear(reduction_dim, channels * self.M)
            )
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feats_final, _ = self.forward_with_attn(x)
        return feats_final

    def forward_with_attn(self, x):
        batch_size = x.size(0)
        feats = [conv(x) for conv in self.convs]
        feats_stack = torch.stack(feats, dim=1)

        # 若关闭注意力，直接平均融合双分支
        if not self.use_attention:
            feats_final = feats_stack.mean(dim=1)  # 简单平均替代注意力加权
            return feats_final, None  # 无注意力图

        # 注意力逻辑（与原代码一致）
        texture_weight = self.texture_attn(x)
        if self.training is False and x.size(0) == 1:
            import matplotlib.pyplot as plt
            import os
            os.makedirs("./attn_vis/skb", exist_ok=True)
            attn_vis = texture_weight[0, 0].detach().cpu().numpy()
            plt.imsave(f'./attn_vis/skb/skblock_attn_ch{self.channels}.png', attn_vis, cmap='jet')

        ref_h, ref_w = feats_stack.shape[3], feats_stack.shape[4]
        if texture_weight.shape[2] != ref_h or texture_weight.shape[3] != ref_w:
            texture_weight = F.interpolate(
                texture_weight, size=(ref_h, ref_w), mode='bilinear', align_corners=False
            )

        texture_gate = torch.sigmoid(texture_weight.unsqueeze(2))
        attn_global = self.gap(x).view(batch_size, -1)
        attn_global = self.fc(attn_global).view(batch_size, self.M, self.channels)
        global_gate = torch.sigmoid(self.softmax(attn_global).unsqueeze(-1).unsqueeze(-1))

        feats_final = feats_stack * texture_gate * (1 + global_gate)
        feats_final = feats_final.sum(dim=1)

        return feats_final, texture_weight


# ======================== 2. WheatGeM（添加开关） ========================
# ======================== 2. WheatGeM（修复pow(None)错误） ========================
class WheatGeM(nn.Module):
    def __init__(self, in_channels, p_init=3.0, eps=1e-6, max_p=5.0, min_p=1.0,
                 use_attn_fusion=False,  # 是否融合SKB注意力
                 use_dynamic_p=False):   # 是否使用动态p值
        super(WheatGeM, self).__init__()
        # 修复：use_dynamic_p=True 时创建可训练参数，False 时为 None（正确逻辑）
        self.p = nn.Parameter(torch.tensor(p_init)) if use_dynamic_p else None
        self.fixed_p = p_init  # 静态p时的固定值（确保始终有默认值）
        self.eps = eps
        self.max_p = max_p
        self.min_p = min_p
        self.use_attn_fusion = use_attn_fusion
        self.use_dynamic_p = use_dynamic_p

        # 基础空间注意力（仅当使用注意力时初始化）
        self.spatial_attn = None
        if self.use_attn_fusion:
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )

        self.bn = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x, skb_attn=None):
        # 修复：统一处理 p 的赋值，确保不会为 None
        if self.use_dynamic_p:
            # 动态p：优先用可训练参数 self.p（已通过 __init__ 确保不为 None）
            p = self.p
        else:
            # 静态p：用固定值 self.fixed_p（数值，不会为 None）
            p = self.fixed_p

        # 若关闭注意力融合，直接使用普通GeM池化
        if not self.use_attn_fusion:
            pooled = F.avg_pool2d(
                x.clamp(min=self.eps).pow(p),  # 此时 p 要么是 Parameter，要么是数值
                kernel_size=(x.size(-2), x.size(-1))
            ).pow(1. / p)
            pooled = self.bn(pooled)
            return pooled, None

        # 注意力融合逻辑（修正 p_clamped 的赋值，确保不为 None）
        x_gray = torch.mean(x, dim=1, keepdim=True)
        gem_attn = self.spatial_attn(x_gray)

        if skb_attn is not None:
            skb_attn_single = torch.mean(skb_attn, dim=1, keepdim=True)
            skb_attn_aligned = F.interpolate(
                skb_attn_single, size=gem_attn.shape[2:], mode='bilinear', align_corners=False
            )
            gem_attn = (gem_attn + skb_attn_aligned) / 2

        # 动态p值计算（确保 p_clamped 是有效数值）
        if self.use_dynamic_p:
            lesion_ratio = torch.mean(gem_attn)
            p_adjusted = self.min_p + (self.max_p - self.min_p) * (1.0 - lesion_ratio)
            p_clamped = torch.clamp(p_adjusted, self.min_p, self.max_p)
        else:
            # 静态p：直接用 fixed_p，避免依赖 self.p
            p_clamped = self.fixed_p

        x_weighted = x * gem_attn
        pooled = F.avg_pool2d(
            x_weighted.clamp(min=self.eps).pow(p_clamped),  # p_clamped 已确保是有效数值
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1. / p_clamped)

        pooled = self.bn(pooled)
        return pooled, gem_attn


# ======================== 3. InvertedResidual（添加SKB开关） ========================
class InvertedResidualConfig:
    def __init__(self, kernel, input_c, out_c, expanded_ratio, stride, index, width_coefficient):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.stride = stride
        self.index = index

    @staticmethod
    def adjust_channels(channels, width_coefficient):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer, use_skblock=False,  # 新增：是否启用SKB
                 skb_kernel_list=[1,3], reduction=4, skb_use_attention=True):  # SKB内部注意力开关
        super().__init__()
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)
        self.use_skblock = use_skblock  # 控制是否使用SKB模块
        layers = OrderedDict()
        activation = nn.SiLU(inplace=False)

        if cnf.expanded_c != cnf.input_c:
            layers["expand"] = nn.Sequential(
                nn.Conv2d(cnf.input_c, cnf.expanded_c, 1, bias=False),
                norm_layer(cnf.expanded_c),
                activation
            )

        # 扩展层后SKB（仅当use_skblock=True时添加）
        if self.use_skblock:
            layers["expand_skb"] = WheatSKBlock(
                channels=cnf.expanded_c,
                stride=1,
                kernel_list=skb_kernel_list,
                reduction=reduction,
                texture_dim_ratio=0.125,
                use_attention=skb_use_attention  # 控制SKB内部注意力
            )
        # 不使用SKB时，扩展层后无额外操作

        # 深度卷积层（SKB或普通卷积）
        if self.use_skblock:
            layers["dwconv_skb"] = WheatSKBlock(
                channels=cnf.expanded_c,
                stride=cnf.stride,
                kernel_list=skb_kernel_list,
                reduction=reduction,
                texture_dim_ratio=0.125,
                use_attention=skb_use_attention
            )
        else:
            # 普通深度卷积替代SKB
            padding = (cnf.kernel - 1) // 2
            layers["dwconv"] = nn.Sequential(
                nn.Conv2d(cnf.expanded_c, cnf.expanded_c, kernel_size=cnf.kernel,
                          stride=cnf.stride, padding=padding,
                          groups=cnf.expanded_c, bias=False),
                norm_layer(cnf.expanded_c),
                activation
            )

        layers["project"] = nn.Sequential(
            nn.Conv2d(cnf.expanded_c, cnf.out_c, 1, bias=False),
            norm_layer(cnf.out_c)
        )

        self.block = nn.Sequential(layers)

    def forward(self, x):
        res = self.block(x)
        if self.use_res_connect:
            res_importance = torch.mean(res, dim=[1, 2, 3], keepdim=True)
            x_importance = torch.mean(x, dim=[1, 2, 3], keepdim=True)
            weight = torch.sigmoid(res_importance - x_importance)
            res = res * weight + x * (1 - weight)
        return res


# ======================== 4. 主模型（添加全局消融开关） ========================
class WheatDiseaseEfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, num_classes=1000,
                 dropout_rate=0.2, block=None, norm_layer=None,
                 # 消融实验开关
                 use_skblock=True,  # 是否启用所有SKB模块
                 skb_use_attention=True,  # SKB内部是否启用注意力
                 use_top_skb=True,  # 是否启用顶层SKB
                 use_gem_attn_fusion=False,  # WheatGeM是否融合SKB注意力
                 use_gem_dynamic_p=False):  # WheatGeM是否使用动态p值
        super().__init__()
        default_cnf = [
            [3, 32, 16, 1, 1, 1],
            [3, 16, 24, 4, 2, 2],
            [5, 24, 40, 4, 2, 2],
            [3, 40, 80, 6, 2, 3],
            [5, 80, 112, 6, 1, 3],
            [5, 112, 192, 6, 2, 4],
            [3, 192, 320, 6, 1, 1]
        ]

        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)
        bneck_conf = partial(InvertedResidualConfig, width_coefficient=width_coefficient)

        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            kernel, input_c, out_c, exp_ratio, stride, repeats = args
            for i in range(round_repeats(repeats)):
                if i > 0:
                    stride = 1
                    input_c = out_c
                index = f"{stage + 1}{chr(97 + i)}"
                inverted_residual_setting.append(bneck_conf(kernel, input_c, out_c, exp_ratio, stride, index))

        # Backbone
        layers = OrderedDict()
        layers["stem_conv"] = nn.Sequential(
            nn.Conv2d(3, adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(adjust_channels(32)),
            nn.SiLU()
        )

        stage_repeats = [round_repeats(args[-1]) for args in default_cnf]
        stage_cumsum = [sum(stage_repeats[:i + 1]) for i in range(len(stage_repeats))]
        for i, cnf in enumerate(inverted_residual_setting):
            if i < stage_cumsum[1]:
                reduction = 8
            elif i < stage_cumsum[4]:
                reduction = 6
            else:
                reduction = 4

            layers[cnf.index] = block(
                cnf, norm_layer,
                use_skblock=use_skblock,  # 控制是否启用SKB
                skb_kernel_list=[1,3],
                reduction=reduction,
                skb_use_attention=skb_use_attention  # 控制SKB内部注意力
            )
        self.features = nn.Sequential(layers)

        # 顶层特征处理（控制是否启用顶层SKB）
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        self.top_conv = nn.Sequential(
            nn.Conv2d(last_conv_input_c, last_conv_output_c, kernel_size=1, bias=False),
            norm_layer(last_conv_output_c),
            nn.SiLU()
        )

        self.top_skb = None
        if use_top_skb and use_skblock:  # 顶层SKB依赖全局SKB开关
            self.top_skb = WheatSKBlock(
                channels=last_conv_output_c,
                stride=1,
                kernel_list=[1,3],
                reduction=8,
                texture_dim_ratio=0.0625,
                use_attention=skb_use_attention
            )

        # 池化与分类器（WheatGeM开关）
        self.avgpool = WheatGeM(
            in_channels=last_conv_output_c,
            p_init=3.0,
            max_p=5.0,
            min_p=1.0,
            use_attn_fusion=use_gem_attn_fusion,
            use_dynamic_p=use_gem_dynamic_p
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(last_conv_output_c, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if 'dwconv' in getattr(m, '_name', ''):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.features(x)
        x = self.top_conv(x)

        # 顶层SKB（若关闭则直接使用top_conv输出）
        skb_attn = None
        if self.top_skb is not None:
            x_top, skb_attn = self.top_skb.forward_with_attn(x)
        else:
            x_top = x  # 无顶层SKB时，直接使用卷积特征

        x_pooled, _ = self.avgpool(x_top, skb_attn=skb_attn)
        x_flat = torch.flatten(x_pooled, 1)
        x_out = self.classifier(x_flat)

        return x_out, skb_attn


# ======================== 模型实例化（带消融参数） ========================
def efficientnet_b0(num_classes=5,
                    # 消融实验参数（默认启用所有机制）
                    use_skblock=False,
                    skb_use_attention=False,
                    use_top_skb=False,
                    use_gem_attn_fusion=False,
                    use_gem_dynamic_p=False):
    return WheatDiseaseEfficientNet(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=0.5,
        num_classes=num_classes,
        # 传入消融开关
        use_skblock=use_skblock,
        skb_use_attention=skb_use_attention,
        use_top_skb=use_top_skb,
        use_gem_attn_fusion=use_gem_attn_fusion,
        use_gem_dynamic_p=use_gem_dynamic_p
    )


# 测试代码
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备：{device}")

    # 测试消融模式（例如：关闭所有SKB）
    model = efficientnet_b0(
        num_classes=5,
        use_skblock=False,  # 关闭所有SKB模块
        use_gem_attn_fusion=False,
        use_gem_dynamic_p=False
    ).to(device)
    print(f"模型结构：{model.__class__.__name__}（消融模式：关闭SKB）")

    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    print(f"输入张量形状：{input_tensor.shape}")

    print("\n" + "=" * 60)
    output, skb_attn = model(input_tensor)
    print(f"分类输出形状：{output.shape}（期望：(2, 5)）")
    print(f"顶层SKB注意力图：{skb_attn.shape if skb_attn is not None else 'None'}（关闭SKB时应为None）")
    print("=" * 60)

    print("\n模型测试通过！")