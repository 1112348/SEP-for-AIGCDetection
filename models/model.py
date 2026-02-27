import warnings

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

from models.srm import all_normalized_hpf_list

# 忽略所有来自 torchvision.transforms.functional 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")


class GroupedDilatedFrequencyEnhancer(nn.Module):
    def __init__(self, in_channels=64, patch_channels=3, groups=8, lambda_val=0.2):
        """
        :param in_channels: 输入频域图的通道数（如 SRM 是 9）
        :param patch_channels: 要增强的 RGB 图像通道数（通常为 3）
        :param groups: 将频域图通道划分为几组
        :param lambda_val: 控制增强强度的 λ 值
        """
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.lambda_val = lambda_val
        self.groups = groups
        self.group_in_ch = in_channels // groups

        self.group_encoders = nn.ModuleList([
            self._make_encoder(self.group_in_ch, patch_channels)
            for _ in range(groups)
        ])

    def _make_encoder(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(32, out_ch, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),  # 输出 [B, C, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, patch_rgb, freq_map):
        """
        :param patch_rgb: [B, C, H, W] 输入图像（如 RGB patch）
        :param freq_map: [B, F, H, W] 频域图（如 SRM / DWT）
        :return: 增强后的 patch
        """
        B, C, H, W = patch_rgb.shape
        group_weights = []

        for i in range(self.groups):
            freq_sub = freq_map[:, i*self.group_in_ch : (i+1)*self.group_in_ch, :, :]
            group_weight = self.group_encoders[i](freq_sub)  # [B, C, 1, 1]
            group_weights.append(group_weight)

        # 通道维度拼接 → 再求平均（融合）
        weight = torch.stack(group_weights, dim=0).mean(dim=0)  # [B, C, 1, 1]

        # 残差增强（不破坏原图）
        enhanced = patch_rgb * (1 + self.lambda_val * weight)
        return enhanced


class MultiHeadBidirectionalCrossSpatialAttention_srm(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(MultiHeadBidirectionalCrossSpatialAttention_srm, self).__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.lambda_val = 0.2

        # QKV 卷积
        self.q_conv_h = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_conv_w = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_conv_w = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.q_conv_w = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_conv_h = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_conv_h = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # srm 引导 Q 卷积
        self.q_conv_h_srm = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.q_conv_w_srm = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        # 融合 projection
        self.projection = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def reshape_heads(self, x, dim):
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_dim, H, W)
        if dim == 'h':
            return x.squeeze(3).permute(0, 1, 3, 2)  # (B, heads, W, head_dim)
        else:
            return x.squeeze(4).permute(0, 1, 3, 2)  # (B, heads, H, head_dim)

    def forward(self, x, srm):
        B, C, H, W = x.shape

        # === H → W ===
        gap_h = x.mean(dim=2, keepdim=True)              # (B, C, 1, W)
        gap_h_srm = srm.mean(dim=2, keepdim=True)        # (B, C, 1, W)

        Q_h = self.q_conv_h(gap_h)
        Q_h_srm = self.q_conv_h_srm(gap_h_srm)
        Q_H = self.reshape_heads(Q_h * (1 + Q_h_srm), dim='h')  # 融合引导

        gap_w = x.mean(dim=3, keepdim=True)
        K_W = self.reshape_heads(self.k_conv_w(gap_w), dim='w')
        V_W = self.reshape_heads(self.v_conv_w(gap_w), dim='w')

        A_HW = torch.matmul(Q_H, K_W.transpose(-2, -1)) / self.head_dim ** 0.5
        A_HW = F.softmax(A_HW, dim=-1)
        Out_HW = torch.matmul(A_HW, V_W).permute(0, 1, 3, 2).contiguous().view(B, C, 1, W).expand(-1, -1, H, -1)

        # === W → H ===
        gap_w_srm = srm.mean(dim=3, keepdim=True)
        Q_w = self.q_conv_w(gap_w)
        Q_w_srm = self.q_conv_w_srm(gap_w_srm)
        Q_W = self.reshape_heads(Q_w * (1 + Q_w_srm), dim='w')  # 融合引导

        K_H = self.reshape_heads(self.k_conv_h(gap_h), dim='h')
        V_H = self.reshape_heads(self.v_conv_h(gap_h), dim='h')

        A_WH = torch.matmul(Q_W, K_H.transpose(-2, -1)) / self.head_dim ** 0.5
        A_WH = F.softmax(A_WH, dim=-1)
        Out_WH = torch.matmul(A_WH, V_H).permute(0, 1, 3, 2).contiguous().view(B, C, H, 1).expand(-1, -1, -1, W)

        # 融合输出
        fused = torch.cat([Out_HW, Out_WH], dim=1)
        weight = self.projection(fused)
        out = x + weight * x
        return out


class SEP(nn.Module):
    def __init__(self, freeze_layers=True):
        super(SEP, self).__init__()

        self.srm = SRMConv2d_simple()

        self.spalAtten = MultiHeadBidirectionalCrossSpatialAttention_srm(channels=3, num_heads=3)

        self.dwt = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # 256-128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # 128-64

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # 64-32

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),  # 32-16

        )

        self.dwt_piex = GroupedDilatedFrequencyEnhancer(in_channels=64, patch_channels=3, groups=16, lambda_val=0.2)

        self.disc = My_resnet50(pretrained=True)
        self.disc.fc = nn.Linear(2048, 2)

    def forward(self, patch, img_dwt):

        self.device = img_dwt.device

        img_dwt = self._preprocess_dwt(img_dwt)  # [B, 3, 256, 256]
        img_dwt = self.dwt(img_dwt)

        patch_32 = patch
        
        patch_big = F.interpolate(patch_32, size=(512, 512), mode='bilinear')
        
        patch_srm = self.srm(patch_32)
        patch_fusion = self.spalAtten(patch_32, patch_srm)
        patch_fusion = F.interpolate(patch_fusion, size=(512, 512), mode='bilinear')
        patch_fusion = self.dwt_piex(patch_fusion, img_dwt) + patch_big

        feature = patch_fusion

        x = feature
        x = self.disc(x)

        return x

    # def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
    #     """
    #     pip install pywavelets pytorch_wavelets
    #     """
    #     from pytorch_wavelets import DWTForward, DWTInverse
    #     DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
    #     Yl, Yh = DWT_filter(x)
    #     return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        """
        pip install pywavelets pytorch_wavelets
        """
        import torch
        from pytorch_wavelets import DWTForward, DWTInverse
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        Yl, Yh = DWT_filter(x)

        # ---- single-level DWT sub-bands ----
        I_LL = Yl  # (B, C, H/2, W/2)
        I_LH = Yh[0][:, :, 0, :, :]  # (B, C, H/2, W/2)
        I_HL = Yh[0][:, :, 1, :, :]  # (B, C, H/2, W/2)
        I_HH = Yh[0][:, :, 2, :, :]  # (B, C, H/2, W/2)

        # ---- 2x2 spatial concatenation (tiling) ----
        # I_fre = [[I_LL, I_LH],
        #          [I_HL, I_HH]]  -> (B, C, H, W)
        top = torch.cat([I_LL, I_LH], dim=-1)  # concat on width
        bottom = torch.cat([I_HL, I_HH], dim=-1)  # concat on width
        I_fre = torch.cat([top, bottom], dim=-2)  # concat on height

        # (optional safety) ensure exactly (H, W) in case of odd-sized inputs
        I_fre = transforms.Resize([x.shape[-2], x.shape[-1]])(I_fre)

        return I_fre


class SRMConv2d_simple(nn.Module):

    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)

    def forward(self, x):
        """
        x: imgs (Batch, H, W, 3)
        """
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # stack the filters
        filters = [[filter1],
                    [filter2],
                    [filter3]]
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)    # (3,3,5,5)
        return filters


# ============================================================
# ResNet 实现（仅保留 Bottleneck + ResNet-50）
# ============================================================

model_urls = {
    "resnet50": "http://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def My_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    return model
