#!/usr/bin/env python3
"""Extract linear CKA and normalized cross-covariance between the two streams.

Reviewer 1 asks for the extraction layers, frame ids, token sampling, and
per-seed results behind the modality-complementarity claim. This script reads
``cka_config.yaml``, loads each seed's checkpoint, runs the configured frames
through the model, samples tokens reproducibly, and writes one CSV row per
(seed, layer_pair).

It reuses the ``linear_cka`` and ``normalized_cross_covariance`` functions that
already live in ``ds_rangenet_v3.py`` so the reported numbers come from the same
implementation used in the model file.

Usage:
    python extract_cka.py --config cka_config.yaml \
        --checkpoints-dir ../deployment/checkpoints \
        --frames-dir /path/to/ubpc9/val

Frames are expected as ``[C, H, W]`` float tensors saved with ``torch.save`` and
named ``frame_<id>.pt`` (16 channels). If ``--frames-dir`` is omitted the script
prints the resolved configuration and exits without writing numbers.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_PY = os.path.normpath(os.path.join(HERE, "..", "..", "python"))
if REPO_PY not in sys.path:
    sys.path.insert(0, REPO_PY)


def load_config(path: str) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        return _mini_yaml(path)


def _mini_yaml(path: str) -> dict:
    """Tiny fallback parser for the flat subset of YAML used in the config."""
    import ast
    cfg: dict = {}
    stack = [(-1, cfg)]
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            key, _, val = line.strip().partition(":")
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            val = val.strip()
            if val == "":
                child: dict = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                try:
                    parent[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    parent[key] = val
    return cfg


def sample_tokens(feat, n_tokens: int, seed: int, offset: int):
    """Reproducibly subsample flattened tokens from a [1,C,H,W] feature map."""
    import torch
    from ds_rangenet_v3 import flatten_features
    flat = flatten_features(feat)  # [H*W, C]
    if n_tokens is None or n_tokens < 0 or n_tokens >= flat.shape[0]:
        return flat
    g = torch.Generator().manual_seed(int(seed) + int(offset))
    idx = torch.randperm(flat.shape[0], generator=g)[:n_tokens]
    return flat[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "cka_config.yaml"))
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--frames-dir", default=None,
                    help="Directory with frame_<id>.pt tensors ([16,H,W]).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg["seeds"]
    layer_pairs = cfg["layer_pairs"]
    frame_ids = cfg["frame_ids"]
    ts = cfg["token_sampling"]
    out_csv = args.out or os.path.join(HERE, cfg.get("output_csv", "results/cka_per_seed.csv"))

    print(f"variant={cfg['model_variant']} seeds={seeds}")
    print(f"layer_pairs={list(layer_pairs)} frames={frame_ids}")
    print(f"tokens_per_frame={ts['tokens_per_frame']} strategy={ts['strategy']}")

    if not args.frames_dir or not args.checkpoints_dir:
        print("frames-dir/checkpoints-dir not given; dry run, no numbers written.")
        return

    import torch
    from ds_rangenet_v3 import build_model, linear_cka, normalized_cross_covariance

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "layer_pair", "n_frames", "tokens_per_frame", "linear_cka", "normalized_cross_cov"])
        for seed in seeds:
            model = build_model(cfg["model_variant"]).eval()
            ckpt = os.path.join(args.checkpoints_dir, f"{cfg['model_variant']}_seed{seed}.pth")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            # accumulate sampled tokens per side across frames
            acc = {name: {"a": [], "b": []} for name in layer_pairs}
            for fid in frame_ids:
                x = torch.load(os.path.join(args.frames_dir, f"frame_{fid}.pt")).unsqueeze(0).float()
                with torch.no_grad():
                    feats = model(x, return_features=True)
                for name, (ka, kb) in layer_pairs.items():
                    acc[name]["a"].append(sample_tokens(feats[ka], ts["tokens_per_frame"], seed, ts["rng_offset"]))
                    acc[name]["b"].append(sample_tokens(feats[kb], ts["tokens_per_frame"], seed, ts["rng_offset"]))
            for name in layer_pairs:
                a = torch.cat(acc[name]["a"], dim=0)
                b = torch.cat(acc[name]["b"], dim=0)
                cka = float(linear_cka(a, b))
                cc = float(normalized_cross_covariance(a, b))
                w.writerow([seed, name, len(frame_ids), ts["tokens_per_frame"], round(cka, 4), round(cc, 4)])
                print(f"  seed={seed} {name}: cka={cka:.4f} cross_cov={cc:.4f}")
    print(f"Wrote {out_csv}.")


if __name__ == "__main__":
    main()
