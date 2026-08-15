"""DS-RangeNet v3 source package.

Canonical, reviewer-facing layout. The model definition itself lives in the
original, tested implementation at ``python/ds_rangenet_v3.py``; the modules
under ``src.models`` re-export it so there is exactly one source of truth for
the architecture. New utilities (preprocessing, metrics, post-processing) are
implemented here.
"""
__all__ = ["models", "data", "postprocess", "metrics", "corruptions"]
