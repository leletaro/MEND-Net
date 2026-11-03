# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------
 
import torch
import torch.nn as nn
 
__all__ = ['C2fMLLABlock', 'C2fMLLAHybrid', 'C3k2_MLLABlock1', 'C3k2_MLLABlock2']
 
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
 
 
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
 
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
 
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
 
 
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x
 
 
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
 
    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base
 
    def generate_rotations(self, x):
        """2D rotary phases for BCHW-coordinate layout (B, H, W, C)."""
        if x.ndim != 4:
            raise ValueError(f"RoPE expects (B,H,W,C), got {tuple(x.shape)}")
        _, height, width, channels = x.shape
        if channels % 4 != 0:
            raise ValueError(f"2D RoPE needs channels divisible by 4, got {channels}")
        quarter = channels // 4
        inv_freq = self.base ** (-torch.arange(quarter, dtype=x.dtype, device=x.device) / quarter)
        row = torch.arange(height, dtype=x.dtype, device=x.device)
        col = torch.arange(width, dtype=x.dtype, device=x.device)
        row_phase = (row[:, None, None] * inv_freq[None, None, :]).expand(height, width, -1)
        col_phase = (col[None, :, None] * inv_freq[None, None, :]).expand(height, width, -1)
        angles = torch.cat((row_phase, col_phase), dim=-1)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(self, x):
        # The complex-valued rotary product is calculated in FP32 to work with
        # CUDA automatic mixed precision and to avoid half-precision phase drift.
        out_dtype = x.dtype
        x32 = x.float()
        rotations = self.generate_rotations(x32)
        x_complex = torch.view_as_complex(x32.reshape(*x32.shape[:-1], -1, 2).contiguous())
        rotation_complex = torch.view_as_complex(rotations.contiguous())
        return torch.view_as_real(x_complex * rotation_complex.unsqueeze(0)).flatten(-2).to(out_dtype)

class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
 
    def __init__(self, dim,  num_heads=4, qkv_bias=True, **kwargs):
 
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()
 
    def forward(self, x, h=None, w=None):
        """Linear attention for real H,W grid (including rectangular inputs)."""
        b, n, c = x.shape
        if h is None or w is None:
            h = w = int(n ** 0.5)
        if h * w != n:
            raise ValueError(f"MLLA attention: H*W={h*w} but N={n}")
        num_heads = self.num_heads
        head_dim = c // num_heads
 
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c
 
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
 
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z
 
        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
 
        return x
 
    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
 
 
class MLLABlock(nn.Module):
    r""" MLLA Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
 
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        num_heads = max(1, dim // 64)
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
 
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
 
        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
 
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> B(HW)C
        L = H * W
 
        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
 
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)
 
        # Linear Attention
        x = self.attn(x, H, W)
 
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
 
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape((B, C, H, W))
        return x
 
    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"
 
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
 
class Bottleneck(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
 
 
 
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
 
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
 

class C2fMLLABlock(C2f):
    """C2f whose internal bottleneck modules are replaced with MLLA blocks.

    Intended YAML format: [-1, repeats, C2fMLLABlock, [out_channels, shortcut]]
    parse_model injects c1, c2, n to match (c1,c2,n,shortcut,g,e).
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.ModuleList(MLLABlock(self.c) for _ in range(n))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
 
class C3kMLLABlock(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(MLLABlock(c_) for _ in range(n)))
 
class C3k2_MLLABlock1(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else MLLABlock(self.c,) for _ in range(n)
        )
        # 解析利用MLLABlock替换Bottneck
 
class C3k2_MLLABlock2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3kMLLABlock(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        # 解析利用MLLABlock替换C3k中的Bottneck
 


# =============================================================================
# P5 Hybrid: preserve original Ultralytics C2f bottlenecks + residual MLLA.
# YAML usage: [-1, 3, C2fMLLAHybrid, [1024, True]]
# parse_model MUST register this class in BOTH base_modules and repeat_modules.
# =============================================================================
from ultralytics.nn.modules.block import C2f as _UltralyticsC2f


class C2fMLLAHybrid(_UltralyticsC2f):
    """Pretrained C2f path with a *gated residual* MLLA enhancement.

    Keep native C2f names ``cv1``, ``m``, ``cv2`` unchanged, so matching
    pretrained YOLOv8 weights can be transferred into the original conv path.
    Only the final hidden branch after the native bottlenecks is enhanced:

        last_hybrid = last + tanh(gamma) * (MLLA(last) - last)

    gamma is initialized to zero: with identical C2f weights, this module's
    first forward is exactly the original C2f, up to floating-point rounding.
    It is not an assertion that training will improve mAP; verify empirically.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, gamma_init=0.0):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        if self.c < 4 or self.c % 4 != 0:
            raise ValueError(
                f"C2fMLLAHybrid requires hidden channels divisible by 4; "
                f"c1={c1}, c2={c2}, hidden={self.c}"
            )
        self.mlla = MLLABlock(dim=self.c)
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)], dtype=torch.float32))

    def _project_with_attention(self, parts):
        last = parts[-1]
        # MLLABlock is already residual; subtract its input so gamma=0
        # recovers the unmodified YOLOv8 C2f computation.
        parts[-1] = last + torch.tanh(self.gamma) * (self.mlla(last) - last)
        return self.cv2(torch.cat(parts, dim=1))

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, 1))
        parts.extend(block(parts[-1]) for block in self.m)
        return self._project_with_attention(parts)

    def forward_split(self, x):
        parts = list(self.cv1(x).split((self.c, self.c), dim=1))
        parts.extend(block(parts[-1]) for block in self.m)
        return self._project_with_attention(parts)


# 区别在于不同的位置在yaml文件中使用.
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)
 
    # Model
    model = C3k2_MLLABlock2(64, 64, 3, c3k=True)
 
    out = model(image)
    print(out.size())