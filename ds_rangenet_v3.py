"""
DS-RangeNet v3, paper-aligned and reviewer-response ready.

Default model:
  - 16-channel range-image input
  - 5-channel material/intensity stream
  - 11-channel geometry stream
  - shallow CBAM fusion
  - per-stream ASPP
  - pooled IGCA with pairwise intensity-curvature bias (ICB)
  - DSConv decoder and 9-class segmentation head

The file also includes controlled variants and analysis utilities required by
the review response: conventional cross-attention controls, IGCA without ICB,
unidirectional IGCA, DSConv-vs-standard-convolution controls, feature
complementarity metrics, and robustness corruptions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSES = [
    "background",
    "ground",
    "roof",
    "side_facade",
    "front_facade",
    "beam",
    "column",
    "window",
    "dynamic",
]
NUM_CLASSES = len(CLASSES)

INTENSITY_CHANNELS = [
    "range_norm",
    "intensity_mean",
    "intensity_boundary",
    "intensity_curvature",
    "intensity_std",
]
GEOMETRY_CHANNELS = [
    "normal_x",
    "normal_y",
    "normal_z",
    "x",
    "y",
    "z",
    "linearity",
    "planarity",
    "scattering",
    "eigen_entropy",
    "relative_elevation",
]

IN_INTENSITY = len(INTENSITY_CHANNELS)
IN_GEO = len(GEOMETRY_CHANNELS)
IN_TOTAL = IN_INTENSITY + IN_GEO
IN_RANGE = IN_INTENSITY  # compatibility name; this is now the 5-channel material stream
CURVATURE_CHANNEL = 3


@dataclass
class DSRangeNetConfig:
    num_classes: int = NUM_CLASSES
    base: int = 32
    conv_type: str = "ds"  # "ds" or "standard"
    fusion_mode: str = "full"
    # full, cbam_only, igca_only, no_attention,
    # igca_no_icb, igca_g2i_only, igca_i2g_only,
    # conventional_g2i, conventional_bidir
    igca_dim: int = 64
    igca_pool: int = 2
    gamma_icb: float = 1.0
    cbam_reduction: int = 8
    input_policy: str = "dual"  # dual, intensity_only, geometry_only


def _check_choice(name: str, value: str, choices: Iterable[str]):
    choices = tuple(choices)
    if value not in choices:
        raise ValueError(f"{name}={value!r} must be one of {choices}")


class StandardConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


def make_conv(conv_type: str, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
    _check_choice("conv_type", conv_type, ("ds", "standard"))
    cls = DSConv2d if conv_type == "ds" else StandardConv2d
    return cls(in_ch, out_ch, k, s, p)


class ResDSBlock(nn.Module):
    """Residual block from the corrected paper, with selectable convolution type."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, conv_type: str = "ds"):
        super().__init__()
        self.conv1 = make_conv(conv_type, in_ch, out_ch, s=stride)
        self.conv2 = make_conv(conv_type, out_ch, out_ch)
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            )
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResBlock(nn.Module):
    def __init__(self, ch: int, conv_type: str = "ds"):
        super().__init__()
        self.block = ResDSBlock(ch, ch, conv_type=conv_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx = self.mlp(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, 1, k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch: int, reduction: int = 8, spatial_k: int = 7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention(spatial_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class FeatureFusion(nn.Module):
    """Concat -> 1x1 reduction -> optional CBAM -> residual."""

    def __init__(
        self,
        in_ch_i: int,
        in_ch_g: int,
        out_ch: int,
        reduction: int = 8,
        use_cbam: bool = True,
    ):
        super().__init__()
        fused = in_ch_i + in_ch_g
        self.proj = nn.Sequential(
            nn.Conv2d(fused, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch, reduction) if use_cbam else nn.Identity()
        self.use_cbam = use_cbam

    def forward(self, i: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if i.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=i.shape[2:], mode="bilinear", align_corners=False)
        z = self.proj(torch.cat([i, g], dim=1))
        return self.cbam(z) + z if self.use_cbam else z


CBAMFusion = FeatureFusion


class ASPP(nn.Module):
    """Per-stream atrous spatial pyramid pooling."""

    def __init__(
        self,
        ch: int,
        dilations: Tuple[int, ...] = (1, 2, 4),
        conv_type: str = "ds",
    ):
        super().__init__()
        def atrous_branch(d: int):
            if conv_type == "standard":
                return nn.Sequential(
                    nn.Conv2d(ch, ch, 3, 1, d, dilation=d, bias=False),
                    nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
                    nn.ReLU(inplace=True),
                )
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, 1, d, dilation=d, groups=ch, bias=False),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
                nn.ReLU(inplace=True),
            )

        self.branches = nn.ModuleList(
            [atrous_branch(d) for d in dilations]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ch * len(dilations), ch, 1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([branch(x) for branch in self.branches], dim=1))


class StreamEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 32, conv_type: str = "ds"):
        super().__init__()
        c0, c1, c2, c3 = base, base * 2, base * 4, base * 8
        self.s0 = ResDSBlock(in_ch, c0, stride=1, conv_type=conv_type)
        self.s1 = ResDSBlock(c0, c1, stride=2, conv_type=conv_type)
        self.s2 = ResDSBlock(c1, c2, stride=2, conv_type=conv_type)
        self.s3 = ResDSBlock(c2, c3, stride=2, conv_type=conv_type)

    def forward(self, x: torch.Tensor):
        e0 = self.s0(x)
        e1 = self.s1(e0)
        e2 = self.s2(e1)
        e3 = self.s3(e2)
        return e0, e1, e2, e3


class IGCrossAttention(nn.Module):
    """Guiding-modality affinity + complementary value transport + optional ICB."""

    def __init__(
        self,
        ch: int,
        attn_dim: int = 64,
        pool: int = 2,
        gamma_icb: float = 1.0,
        use_icb: bool = True,
        branch_mode: str = "both",
        num_heads: Optional[int] = None,
        spatial_max: Optional[int] = None,
    ):
        super().__init__()
        _check_choice("branch_mode", branch_mode, ("both", "g2i", "i2g"))
        self.ch = ch
        self.attn_dim = attn_dim
        self.pool = pool
        self.gamma_icb = gamma_icb
        self.use_icb = use_icb
        self.branch_mode = branch_mode
        self.nh = 1 if num_heads is None else num_heads
        self.d_k = attn_dim
        self.spatial_max = spatial_max

        self.q_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.k_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.v_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.o_i = nn.Conv2d(attn_dim, ch, 1, bias=False)

        self.q_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.k_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.v_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.o_g = nn.Conv2d(attn_dim, ch, 1, bias=False)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.pool <= 1 else F.max_pool2d(x, self.pool, self.pool)

    @staticmethod
    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    def _icb(self, curvature: Optional[torch.Tensor], size: Tuple[int, int]):
        if not self.use_icb or curvature is None:
            return None
        c = F.interpolate(curvature, size=size, mode="bilinear", align_corners=False)
        c = c.flatten(2).transpose(1, 2).clamp(0.0, 1.0)
        return -self.gamma_icb * torch.abs(c - c.transpose(1, 2))

    def forward(self, fi: torch.Tensor, fg: torch.Tensor, curvature: Optional[torch.Tensor] = None):
        out_size = fi.shape[2:]
        fi_p, fg_p = self._pool(fi), self._pool(fg)
        pooled_size = fi_p.shape[2:]
        bias = self._icb(curvature, pooled_size)
        scale = math.sqrt(self.attn_dim)

        fused = fi + fg
        b, _, hp, wp = fi_p.shape

        if self.branch_mode in ("both", "g2i"):
            qg, kg = self._flat(self.q_g(fg_p)), self._flat(self.k_g(fg_p))
            vi = self._flat(self.v_i(fi_p))
            logits = torch.matmul(qg, kg.transpose(-2, -1)) / scale
            logits = logits + bias if bias is not None else logits
            delta = torch.matmul(F.softmax(logits, dim=-1), vi)
            delta = delta.transpose(1, 2).reshape(b, self.attn_dim, hp, wp)
            delta = self.o_i(delta)
            delta = F.interpolate(delta, size=out_size, mode="bilinear", align_corners=False)
            fused = fused + self.alpha * delta

        if self.branch_mode in ("both", "i2g"):
            qi, ki = self._flat(self.q_i(fi_p)), self._flat(self.k_i(fi_p))
            vg = self._flat(self.v_g(fg_p))
            logits = torch.matmul(qi, ki.transpose(-2, -1)) / scale
            logits = logits + bias if bias is not None else logits
            delta = torch.matmul(F.softmax(logits, dim=-1), vg)
            delta = delta.transpose(1, 2).reshape(b, self.attn_dim, hp, wp)
            delta = self.o_g(delta)
            delta = F.interpolate(delta, size=out_size, mode="bilinear", align_corners=False)
            fused = fused + self.beta * delta

        return self.refine(fused)


class ConventionalCrossAttention(nn.Module):
    """Control: conventional cross-modal QK attention with opposite-stream values."""

    def __init__(self, ch: int, attn_dim: int = 64, pool: int = 2, bidirectional: bool = True):
        super().__init__()
        self.ch = ch
        self.attn_dim = attn_dim
        self.pool = pool
        self.bidirectional = bidirectional
        self.q_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.k_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.v_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.o_i = nn.Conv2d(attn_dim, ch, 1, bias=False)
        self.q_i = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.k_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.v_g = nn.Conv2d(ch, attn_dim, 1, bias=False)
        self.o_g = nn.Conv2d(attn_dim, ch, 1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

    def _pool(self, x):
        return x if self.pool <= 1 else F.max_pool2d(x, self.pool, self.pool)

    @staticmethod
    def _flat(x):
        return x.flatten(2).transpose(1, 2)

    def _attend(self, q, k, v, out_proj, out_size):
        b, _, hp, wp = q.shape
        qq, kk, vv = self._flat(q), self._flat(k), self._flat(v)
        logits = torch.matmul(qq, kk.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        out = torch.matmul(F.softmax(logits, dim=-1), vv)
        out = out.transpose(1, 2).reshape(b, self.attn_dim, hp, wp)
        out = out_proj(out)
        return F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)

    def forward(self, fi: torch.Tensor, fg: torch.Tensor, curvature: Optional[torch.Tensor] = None):
        del curvature
        out_size = fi.shape[2:]
        fi_p, fg_p = self._pool(fi), self._pool(fg)
        fused = fi + fg
        delta_i = self._attend(self.q_g(fg_p), self.k_i(fi_p), self.v_i(fi_p), self.o_i, out_size)
        fused = fused + self.alpha * delta_i
        if self.bidirectional:
            delta_g = self._attend(self.q_i(fi_p), self.k_g(fg_p), self.v_g(fg_p), self.o_g, out_size)
            fused = fused + self.beta * delta_g
        return self.refine(fused)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, conv_type: str = "ds"):
        super().__init__()
        self.conv = nn.Sequential(
            make_conv(conv_type, in_ch + skip_ch, out_ch),
            ResBlock(out_ch, conv_type=conv_type),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DualStreamRangeNetV3(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base: int = 32,
        igca_dim: int = 64,
        igca_pool: int = 2,
        gamma_icb: float = 1.0,
        igca_heads: Optional[int] = None,
        igca_spatial_max: Optional[int] = None,
        conv_type: str = "ds",
        fusion_mode: str = "full",
        input_policy: str = "dual",
        config: Optional[DSRangeNetConfig] = None,
    ):
        super().__init__()
        if config is not None:
            num_classes = config.num_classes
            base = config.base
            conv_type = config.conv_type
            fusion_mode = config.fusion_mode
            igca_dim = config.igca_dim
            igca_pool = config.igca_pool
            gamma_icb = config.gamma_icb
            input_policy = config.input_policy

        del igca_heads, igca_spatial_max
        _check_choice("input_policy", input_policy, ("dual", "intensity_only", "geometry_only"))
        _check_choice(
            "fusion_mode",
            fusion_mode,
            (
                "full",
                "cbam_only",
                "igca_only",
                "no_attention",
                "igca_no_icb",
                "igca_g2i_only",
                "igca_i2g_only",
                "conventional_g2i",
                "conventional_bidir",
            ),
        )
        self.config = DSRangeNetConfig(
            num_classes=num_classes,
            base=base,
            conv_type=conv_type,
            fusion_mode=fusion_mode,
            igca_dim=igca_dim,
            igca_pool=igca_pool,
            gamma_icb=gamma_icb,
            input_policy=input_policy,
        )

        c0, c1, c2, c3 = base, base * 2, base * 4, base * 8
        shallow_cbam = fusion_mode not in ("igca_only", "no_attention")
        self.enc_intensity = StreamEncoder(IN_INTENSITY, base, conv_type=conv_type)
        self.enc_geo = StreamEncoder(IN_GEO, base, conv_type=conv_type)
        self.aspp_intensity = ASPP(c3, conv_type=conv_type)
        self.aspp_geo = ASPP(c3, conv_type=conv_type)

        self.fuse0 = FeatureFusion(c0, c0, c0, use_cbam=shallow_cbam)
        self.fuse1 = FeatureFusion(c1, c1, c1, use_cbam=shallow_cbam)
        self.fuse2 = FeatureFusion(c2, c2, c2, use_cbam=shallow_cbam)
        self.bottleneck_fuse = FeatureFusion(c3, c3, c3, use_cbam=fusion_mode == "cbam_only")

        if fusion_mode in ("full", "igca_only", "igca_no_icb", "igca_g2i_only", "igca_i2g_only"):
            self.attention = IGCrossAttention(
                c3,
                attn_dim=igca_dim,
                pool=igca_pool,
                gamma_icb=gamma_icb,
                use_icb=fusion_mode != "igca_no_icb",
                branch_mode={
                    "igca_g2i_only": "g2i",
                    "igca_i2g_only": "i2g",
                }.get(fusion_mode, "both"),
            )
        elif fusion_mode in ("conventional_g2i", "conventional_bidir"):
            self.attention = ConventionalCrossAttention(
                c3,
                attn_dim=igca_dim,
                pool=igca_pool,
                bidirectional=fusion_mode == "conventional_bidir",
            )
        else:
            self.attention = None

        self.up3 = UpBlock(c3, c2, c2, conv_type=conv_type)
        self.up2 = UpBlock(c2, c1, c1, conv_type=conv_type)
        self.up1 = UpBlock(c1, c0, c0, conv_type=conv_type)
        self.head = nn.Sequential(make_conv(conv_type, c0, c0), nn.Conv2d(c0, num_classes, 1))
        self._init_weights()

    @property
    def enc_range(self):
        return self.enc_intensity

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if isinstance(self.attention, IGCrossAttention):
            self.attention._init_weights()

    def _split_input(self, x: torch.Tensor):
        xi = x[:, :IN_INTENSITY]
        xg = x[:, IN_INTENSITY:]
        if self.config.input_policy == "intensity_only":
            xg = torch.zeros_like(xg)
        elif self.config.input_policy == "geometry_only":
            xi = torch.zeros_like(xi)
        curvature = x[:, CURVATURE_CHANNEL : CURVATURE_CHANNEL + 1]
        return xi, xg, curvature

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        xi, xg, curvature = self._split_input(x)
        e0, e1, e2, e3 = self.enc_intensity(xi)
        g0, g1, g2, g3 = self.enc_geo(xg)

        f0 = self.fuse0(e0, g0)
        f1 = self.fuse1(e1, g1)
        f2 = self.fuse2(e2, g2)

        e3_bar = self.aspp_intensity(e3)
        g3_bar = self.aspp_geo(g3)
        if self.attention is None:
            f3 = self.bottleneck_fuse(e3_bar, g3_bar)
        else:
            f3 = self.attention(e3_bar, g3_bar, curvature)

        d2 = self.up3(f3, f2)
        d1 = self.up2(d2, f1)
        d0 = self.up1(d1, f0)
        logits = self.head(d0)

        return {
            "input_intensity": xi,
            "input_geometry": xg,
            "enc_intensity": e3,
            "enc_geometry": g3,
            "aspp_intensity": e3_bar,
            "aspp_geometry": g3_bar,
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "decoder": d0,
            "logits": logits,
        }

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.forward_features(x)
        return feats if return_features else feats["logits"]

    def param_summary(self) -> dict:
        def cnt(m): return sum(p.numel() for p in m.parameters()) / 1e6
        return {
            "enc_intensity": cnt(self.enc_intensity),
            "enc_geo": cnt(self.enc_geo),
            "aspp": cnt(self.aspp_intensity) + cnt(self.aspp_geo),
            "fuse0-2": cnt(self.fuse0) + cnt(self.fuse1) + cnt(self.fuse2),
            "attention": 0.0 if self.attention is None else cnt(self.attention),
            "bottleneck_fuse": cnt(self.bottleneck_fuse),
            "decoder": cnt(self.up3) + cnt(self.up2) + cnt(self.up1),
            "head": cnt(self.head),
            "total": cnt(self),
        }


def build_model(variant: str = "full", **kwargs) -> DualStreamRangeNetV3:
    """Factory for paper/reviewer-control variants."""
    variant_map = {
        "full": {"fusion_mode": "full", "conv_type": "ds", "input_policy": "dual"},
        "cbam_only": {"fusion_mode": "cbam_only"},
        "igca_only": {"fusion_mode": "igca_only"},
        "no_attention": {"fusion_mode": "no_attention"},
        "igca_no_icb": {"fusion_mode": "igca_no_icb"},
        "igca_g2i_only": {"fusion_mode": "igca_g2i_only"},
        "igca_i2g_only": {"fusion_mode": "igca_i2g_only"},
        "conventional_g2i": {"fusion_mode": "conventional_g2i"},
        "conventional_bidir": {"fusion_mode": "conventional_bidir"},
        "standard_conv": {"conv_type": "standard", "fusion_mode": "full"},
        "intensity_only": {"input_policy": "intensity_only", "fusion_mode": "no_attention"},
        "geometry_only": {"input_policy": "geometry_only", "fusion_mode": "no_attention"},
    }
    if variant not in variant_map:
        raise ValueError(f"Unknown variant {variant!r}. Choices: {sorted(variant_map)}")
    opts = dict(variant_map[variant])
    opts.update(kwargs)
    return DualStreamRangeNetV3(**opts)


def flatten_features(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    elif x.dim() != 2:
        x = x.reshape(x.shape[0], -1)
    return x


def normalized_cross_covariance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = flatten_features(a).float()
    b = flatten_features(b).float()
    n = min(a.shape[0], b.shape[0])
    c = min(a.shape[1], b.shape[1])
    a = a[:n, :c] - a[:n, :c].mean(dim=0, keepdim=True)
    b = b[:n, :c] - b[:n, :c].mean(dim=0, keepdim=True)
    return torch.linalg.norm(a.t().matmul(b), ord="fro") / (
        torch.linalg.norm(a, ord="fro") * torch.linalg.norm(b, ord="fro") + eps
    )


def linear_cka(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = flatten_features(a).float()
    b = flatten_features(b).float()
    n = min(a.shape[0], b.shape[0])
    c = min(a.shape[1], b.shape[1])
    a = a[:n, :c] - a[:n, :c].mean(dim=0, keepdim=True)
    b = b[:n, :c] - b[:n, :c].mean(dim=0, keepdim=True)
    hsic = torch.linalg.norm(a.t().matmul(b), ord="fro") ** 2
    var_a = torch.linalg.norm(a.t().matmul(a), ord="fro")
    var_b = torch.linalg.norm(b.t().matmul(b), ord="fro")
    return hsic / (var_a * var_b + eps)


@torch.no_grad()
def complementarity_report(model: DualStreamRangeNetV3, x: torch.Tensor) -> Dict[str, Dict[str, float]]:
    feats = model(x, return_features=True)
    pairs = {
        "input_modalities": ("input_intensity", "input_geometry"),
        "independent_encoders": ("enc_intensity", "enc_geometry"),
        "after_aspp": ("aspp_intensity", "aspp_geometry"),
        "after_fusion": ("f3", "decoder"),
    }
    report = {}
    for name, (ka, kb) in pairs.items():
        report[name] = {
            "cka": float(linear_cka(feats[ka], feats[kb]).cpu()),
            "cross_cov": float(normalized_cross_covariance(feats[ka], feats[kb]).cpu()),
        }
    return report


def apply_corruption(
    x: torch.Tensor,
    kind: str,
    severity: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Range-image corruptions used for robustness evaluation."""
    _check_choice(kind, kind, ("range_noise", "intensity_noise", "point_dropout", "scanline_dropout", "block_occlusion"))
    y = x.clone()
    if kind == "range_noise":
        noise = torch.randn(
            y[:, 0:1].shape,
            device=y.device,
            dtype=y.dtype,
            generator=generator,
        ) * (0.05 * severity)
        y[:, 0:1] = y[:, 0:1] + noise
    elif kind == "intensity_noise":
        noise = torch.randn(
            y[:, 1:5].shape,
            device=y.device,
            dtype=y.dtype,
            generator=generator,
        ) * (0.2 * severity)
        y[:, 1:5] = y[:, 1:5] * (1.0 + noise)
    elif kind == "point_dropout":
        keep = torch.rand(y[:, 0:1].shape, device=y.device, generator=generator) > (0.3 * severity)
        y = y * keep
    elif kind == "scanline_dropout":
        b, _, h, _ = y.shape
        drop = max(1, int(h * 0.25 * severity))
        rows = torch.randperm(h, device=y.device, generator=generator)[:drop]
        y[:, :, rows, :] = 0
    elif kind == "block_occlusion":
        _, _, h, w = y.shape
        bh = max(1, min(h, int(16 * severity)))
        bw = max(1, min(w, int(64 * severity)))
        for _ in range(3):
            top = int(torch.randint(0, max(1, h - bh + 1), (1,), device=y.device, generator=generator))
            left = int(torch.randint(0, max(1, w - bw + 1), (1,), device=y.device, generator=generator))
            y[:, :, top : top + bh, left : left + bw] = 0
    return y


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce = F.cross_entropy(
            pred,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, ignore_index: int = 0):
        super().__init__()
        self.smooth = smooth
        self.ignore = ignore_index

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        dice = 0.0
        count = 0
        for c in range(pred.shape[1]):
            if c == self.ignore:
                continue
            mask = (target == c).float()
            p = pred_soft[:, c]
            inter = (p * mask).sum()
            union = p.sum() + mask.sum()
            dice += 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
            count += 1
        return dice / max(count, 1)


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.alpha * self.focal(pred, target) + (1 - self.alpha) * self.dice(pred, target)
