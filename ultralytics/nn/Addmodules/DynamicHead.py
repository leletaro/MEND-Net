# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# DynamicHead adapter for the user's modified Ultralytics 8.4.3 tasks.py.
# YAML: [[P3, P4, P5], 1, DynamicHead, [nc]]
# tasks.py calls DynamicHead(nc, [C3, C4, C5]).
# This keeps the original three DyConv towers, DCNv2, spatial attention,
# and DYReLU, while using the v8.4.3 Detect output/loss protocol.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d
from ultralytics.nn.modules.head import Detect

__all__ = ["DynamicHead"]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    """Keep the original two-piece DYReLU; spatial branch is optional."""

    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True,
                 use_bias=True, use_spatial=False,
                 init_a=(1.0, 0.0), init_b=(0.0, 0.0)):
        super().__init__()
        self.oup = oup
        self.lambda_a = 2.0 * lambda_a
        self.use_bias = use_bias
        self.exp = (4 if use_bias else 2) if K2 else (2 if use_bias else 1)
        self.init_a, self.init_b = init_a, init_b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        squeeze = inp // reduction if reduction == 4 else _make_divisible(inp // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid(),
        )
        self.spa = (
            nn.Sequential(nn.Conv2d(inp, 1, kernel_size=1), nn.BatchNorm2d(1))
            if use_spatial else None
        )

    def forward(self, x):
        x_in, x_out = x if isinstance(x, (list, tuple)) else (x, x)
        b, c, h, w = x_in.shape
        y = self.fc(self.avg_pool(x_in).reshape(b, c)).reshape(
            b, self.oup * self.exp, 1, 1
        )

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            if self.use_bias:
                out = x_out * ((a1 - 0.5) * self.lambda_a + self.init_a[0]) + (b1 - 0.5 + self.init_b[0])
            else:
                out = torch.max(
                    x_out * ((a1 - 0.5) * self.lambda_a + self.init_a[0]),
                    x_out * ((b1 - 0.5) * self.lambda_a + self.init_a[1]),
                )
        else:
            out = x_out * ((y - 0.5) * self.lambda_a + self.init_a[0])

        if self.spa is not None:
            ys = self.spa(x_in).reshape(b, -1)
            ys = F.softmax(ys, dim=1).reshape(b, 1, h, w) * (h * w)
            out = out * (F.hardtanh(ys, 0, 3, inplace=True) / 3)
        return out


class Conv3x3Norm(nn.Module):
    """MMCV DCNv2; compatible CPU implementation for Ultralytics' stride probe.

    Ultralytics runs one dummy image through the whole detection model on CPU
    while building strides. On CPU we use torchvision's equivalent *modulated*
    deformable convolution rather than silently replacing DCNv2 with Conv2d.
    GPU training and inference use the original MMCV operator.
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn = nn.GroupNorm(
            num_groups=math.gcd(16, out_channels), num_channels=out_channels
        )
        self.stride = stride

    def forward(self, x, offset, mask):
        x = x.contiguous()
        offset = offset.contiguous()
        mask = mask.contiguous()
        if x.device.type == "cpu":
            try:
                from torchvision.ops import deform_conv2d
            except (ImportError, RuntimeError, OSError) as exc:
                raise RuntimeError(
                    "DynamicHead CPU initialization requires torchvision.ops.deform_conv2d; "
                    "install a torchvision build compatible with your PyTorch/CUDA build."
                ) from exc
            # TorchVision: modulated deformable convolution (mask included).
            y = deform_conv2d(
                x, offset, self.conv.weight, self.conv.bias,
                stride=(self.stride, self.stride), padding=(1, 1),
                dilation=(1, 1), mask=mask,
            )
        else:
            y = self.conv(x, offset=offset, mask=mask)
        return self.bn(y)


class DyConv(nn.Module):
    """One cross-scale DyHead fusion block, preserving level order 0=P3,1=P4,2=P5."""

    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super().__init__()
        self.DyConv = nn.ModuleList([
            conv_func(in_channels, out_channels, 1),  # coarser -> up
            conv_func(in_channels, out_channels, 1),  # current
            conv_func(in_channels, out_channels, 2),  # finer -> down
        ])
        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(out_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        # E10 alpha is a fixed, checkpoint-serialized Python attribute.
        # The environment variable also ensures Ultralytics re-builds use
        # exactly the same alpha during .train() model reconstruction.
        self.cross_scale_alpha = float(os.environ.get("MENDNET_DYHEAD_ALPHA", "1.0"))
        self.init_weights()

    def init_weights(self):
        # Preserve the original attention-layer initialization.
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _offset_args(offset_mask, target_size):
        # DCNv2 requires offsets/masks at the *output* spatial resolution.
        # One target-level offset field is reused for the neighboring branch,
        # resampled only when source/output resolutions differ.
        if tuple(offset_mask.shape[-2:]) != tuple(target_size):
            offset_mask = F.interpolate(
                offset_mask, size=target_size, mode="bilinear", align_corners=False
            )
        return {
            "offset": offset_mask[:, :18].contiguous(),
            "mask": offset_mask[:, 18:].sigmoid().contiguous(),
        }

    def forward(self, x):
        if not isinstance(x, dict) or set(x) != set(range(len(x))):
            raise ValueError(
                f"DyConv expects feature dictionary with keys 0..N-1, got "
                f"{list(x) if isinstance(x, dict) else type(x)}"
            )

        outputs = {}
        for level in range(len(x)):
            feature = x[level]
            h, w = feature.shape[-2:]
            offset_mask = self.offset(feature)

            fused = [self.DyConv[1](feature, **self._offset_args(offset_mask, (h, w)))]

            if level > 0:
                finer = x[level - 1]
                hf, wf = finer.shape[-2:]
                # k=3, pad=1, stride=2 => ceil(hf/2), ceil(wf/2)
                down_hw = ((hf + 1) // 2, (wf + 1) // 2)
                down = self.DyConv[2](
                    finer, **self._offset_args(offset_mask, down_hw)
                )
                if down.shape[-2:] != (h, w):
                    down = F.interpolate(down, size=(h, w), mode="nearest")
                fused.append(down)

            if level < len(x) - 1:
                coarser = x[level + 1]
                up = self.DyConv[0](
                    coarser,
                    **self._offset_args(offset_mask, coarser.shape[-2:]),
                )
                up = F.interpolate(up, size=(h, w), mode="nearest")
                fused.append(up)

            attentions = torch.stack([self.h_sigmoid(self.AttnConv(t)) for t in fused])
            feature_stack = torch.stack(fused)
            # [MEND_E10_SCALE_ALPHA_V1] current-scale branch remains fully active.
            # Adjacent-scale contributions are attenuated without changing
            # offsets, DCNv2, spatial weights, DYReLU or Detect.
            alpha = float(getattr(self, "cross_scale_alpha", 1.0))
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"DyConv cross_scale_alpha out of [0,1]: {alpha}")
            if alpha == 1.0:
                result = torch.mean(feature_stack * attentions, dim=0)
            else:
                weighted = feature_stack * attentions
                n_neighbors = len(fused) - 1
                result = (weighted[0] + alpha * weighted[1:].sum(dim=0)) / (1.0 + alpha * n_neighbors)
            outputs[level] = self.relu(result)
        return outputs


class DynamicHead(Detect):
    """Original DyHead feature enhancer + native Ultralytics 8.4.3 detection.

    IMPORTANT: signature (nc, ch) matches the user's CURRENT tasks.py:
        elif m is DynamicHead:
            args = [args[0], [ch[x] for x in f]]

    Training output: {'boxes': Tensor, 'scores': Tensor, 'feats': [P3,P4,P5]}
    Evaluation output: (predictions, same_dictionary)
    """

    def __init__(self, nc=80, ch=(), reg_max=16, end2end=False):
        if len(ch) != 3:
            raise ValueError(f"DynamicHead requires P3, P4, P5; got ch={ch}")
        if len(set(ch)) != 1:
            raise ValueError(
                "DynamicHead cross-scale DCNv2 expects equal channel counts "
                f"for P3/P4/P5, received ch={ch}. Add 1x1 projections if unequal."
            )
        # v8.4.3 native Detect creates cv2/cv3/DFL, exposes bias_init and
        # returns dict outputs compatible with v8DetectionLoss.
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.dyhead_tower = nn.Sequential(
            *[DyConv(ch[0], ch[0], conv_func=Conv3x3Norm) for _ in range(self.nl)]
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.nl:
            raise TypeError(
                f"DynamicHead expects list/tuple of {self.nl} feature tensors "
                "in P3, P4, P5 order."
            )
        x = {i: feature for i, feature in enumerate(x)}
        x = self.dyhead_tower(x)
        # Enumerate by index explicitly: avoid mismatched/insertion-ordered
        # dictionary keys and the old KeyError: 0 situation.
        enhanced = [x[i] for i in range(self.nl)]
        return super().forward(enhanced)
