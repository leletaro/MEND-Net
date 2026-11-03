import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["Detect_dyhead"]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        reduction=4,
        lambda_a=1.0,
        K2=True,
        use_bias=True,
        use_spatial=False,
        init_a=[1.0, 0.0],
        init_b=[0.0, 0.0],
    ):
        super().__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid(),
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x

        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)

        elif self.exp == 2:
            if self.use_bias:
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1
            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        else:  # self.exp == 1
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


class Conv3x3Norm(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x


class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super().__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))  # upsample branch
        self.DyConv.append(conv_func(in_channels, out_channels, 1))  # same level
        self.DyConv.append(conv_func(in_channels, out_channels, 2))  # downsample branch

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # x: dict{level: tensor}
        next_x = {}
        feature_names = list(x.keys())

        for level, name in enumerate(feature_names):
            feature = x[name]

            # offset/mask for current level
            offset_mask = self.offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args_cur = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args_cur)]

            # prev level
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args_cur))

            # next level: MUST compute offset/mask on next feature
            if level < len(x) - 1:
                input = x[feature_names[level + 1]]
                offset_mask_n = self.offset(input)
                offset_n = offset_mask_n[:, :18, :, :]
                mask_n = offset_mask_n[:, 18:, :, :].sigmoid()
                conv_args_n = dict(offset=offset_n, mask=mask_n)

                fea_n = self.DyConv[0](input, **conv_args_n)
                temp_fea.append(F.interpolate(fea_n, size=[feature.size(2), feature.size(3)], mode="nearest"))

            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution: (ch_in, ch_out, k, s, p, g, d, act)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DFL(nn.Module):
    """Distribution Focal Loss integral module."""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect_dyhead(nn.Module):
    """
    DyHead + YOLO Detect head.

    关键适配：
    - 训练/验证阶段返回 dict：{"boxes":..., "scores":..., "feats":...}，使 v8DetectionLoss 可正常索引 preds["boxes"]。
    - 推理阶段返回 (y, preds_dict) 与 Ultralytics 习惯兼容。
    """
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    end2end = False

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.end2end = False
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # ===== unify channels before DyHead =====
        self.c_mid = _make_divisible(min(ch), 16)  # keep divisible by 16 for GroupNorm
        self.reduce = nn.ModuleList([Conv(ci, self.c_mid, k=1, s=1) for ci in ch])
        self.expand = nn.ModuleList([Conv(self.c_mid, ci, k=1, s=1) for ci in ch])

        num_dy_layers = 1
        self.dyhead_tower = nn.Sequential(*[
            DyConv(self.c_mid, self.c_mid, conv_func=Conv3x3Norm)
            for _ in range(num_dy_layers)
        ])
        # ================================

        # debug 打印开关（默认关）
        self.debug = False

    def _pack_preds(self, feats_list, x_cat, box, cls):
        # 统一给 loss 的 dict 结构：避免 list->dict 不匹配
        return {
            "boxes": box,        # (bs, 4*reg_max, N)
            "scores": cls,       # (bs, nc, N) raw logits
            "cls": cls,          # alias
            "feats": feats_list, # list[(bs, no, h, w)] 用于 anchors/stride/shape
            "pred": x_cat,       # (bs, no, N) 可选
        }

    def forward(self, x):
        # x: list of backbone/neck feature maps [P3,P4,P5] etc.

        if self.debug:
            print("before dyhead:", [t.shape for t in x])

        # 1) reduce to unified channels
        x_mid = [self.reduce[i](x[i]) for i in range(self.nl)]

        # 2) dyhead on dict
        tensor_dict = {i: x_mid[i] for i in range(self.nl)}
        tensor_dict = self.dyhead_tower(tensor_dict)

        # 3) restore order explicitly
        x_mid = [tensor_dict[i] for i in range(self.nl)]

        # 4) expand back
        x = [self.expand[i](x_mid[i]) for i in range(self.nl)]

        if self.debug:
            print("after reduce:", [t.shape for t in x_mid])

        # ---- YOLO detect outputs ----
        shape = x[0].shape  # BCHW

        # per-level logits: (bs, no, h, w)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        bs = shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)  # (bs, no, N)

        # split box/cls logits
        if self.export and getattr(self, "format", None) in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        preds = self._pack_preds(x, x_cat, box, cls)

        # ✅ training/val: return dict for v8DetectionLoss
        if self.training:
            return preds

        # ✅ inference: build anchors if needed
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        # tflite/edgetpu normalization
        if self.export and getattr(self, "format", None) in ("tflite", "edgetpu"):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)  # (bs, 4+nc, N)
        return y if self.export else (y, preds)

    def bias_init(self):
        """
        Ultralytics 在 build stride 后会调用 m.bias_init()。
        你之前报错就是因为这个方法缺失。
        """
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls bias: follow Ultralytics default heuristic
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
