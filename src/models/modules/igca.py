"""IGCA — Intensity-Guided Cross Attention with Intensity-Curvature Bias (re-export).

The bottleneck fusion of DS-RangeNet v3. Geometry queries/keys drive an affinity
over the intensity stream (g2i) and vice-versa (i2g); the pairwise
Intensity-Curvature Bias (ICB) ``-gamma * |c_i - c_j|`` down-weights attention
between pixels whose surface curvature differs. ConventionalCrossAttention is the
control (standard cross-modal QKV, no ICB). Canonical definitions:
python/ds_rangenet_v3.py.
"""
from __future__ import annotations

from ..._canonical import ds_rangenet_v3 as _m

IGCrossAttention = _m.IGCrossAttention
ConventionalCrossAttention = _m.ConventionalCrossAttention

__all__ = ["IGCrossAttention", "ConventionalCrossAttention"]
