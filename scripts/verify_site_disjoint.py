#!/usr/bin/env python3
"""Verify that no physical building (site) leaks across train/val/test.

Checks two things:
  1. The primary split (train/val/test txt files) uses disjoint sites.
  2. Every LOEO fold YAML has pairwise-disjoint train/val/test sites and that
     the held-out test site is exactly one site.

A "site" is the S<N> prefix of each "S<N>/seq<M>" entry, i.e. the physical
building. Sequence-level disjointness is not enough for this benchmark — the
whole point of UBPC-9 is that no *building* appears in more than one split.

Exit code 0 == all checks pass; non-zero == leakage detected.

Usage:
    python scripts/verify_site_disjoint.py \
        --splits data/splits --loeo data/splits/loeo
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


def site_of(entry: str) -> str:
    """'S6/seq12' -> 'S6'. Robust to whitespace and comments."""
    return entry.strip().split("/", 1)[0]


def read_txt_sites(path: str) -> set[str]:
    sites: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sites.add(site_of(line))
    return sites


def check_primary(splits_dir: str) -> list[str]:
    errors: list[str] = []
    files = {
        "train": glob.glob(os.path.join(splits_dir, "train_*.txt")),
        "val": glob.glob(os.path.join(splits_dir, "val_*.txt")),
        "test": glob.glob(os.path.join(splits_dir, "test_*.txt")),
    }
    sites: dict[str, set[str]] = {}
    for role, matches in files.items():
        if not matches:
            errors.append(f"[primary] no {role} split file found in {splits_dir}")
            continue
        acc: set[str] = set()
        for m in matches:
            acc |= read_txt_sites(m)
        sites[role] = acc

    roles = [r for r in ("train", "val", "test") if r in sites]
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            a, b = roles[i], roles[j]
            overlap = sites[a] & sites[b]
            if overlap:
                errors.append(
                    f"[primary] SITE LEAKAGE between {a} and {b}: {sorted(overlap)}"
                )
    if not errors:
        summary = ", ".join(f"{r}={sorted(sites[r])}" for r in roles)
        print(f"[primary] OK — disjoint sites: {summary}")
    return errors


def check_loeo(loeo_dir: str) -> list[str]:
    errors: list[str] = []
    fold_files = sorted(glob.glob(os.path.join(loeo_dir, "fold_*.yaml")))
    if not fold_files:
        return [f"[loeo] no fold_*.yaml files found in {loeo_dir}"]

    for path in fold_files:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        name = os.path.basename(path)

        test_site = cfg.get("test_site")
        val_site = cfg.get("val_site")
        train_sites = set(cfg.get("train_sites", []))

        if not test_site:
            errors.append(f"[loeo:{name}] missing test_site")
            continue
        # exactly one held-out test site
        if test_site in train_sites:
            errors.append(f"[loeo:{name}] test site {test_site} also in train_sites")
        if val_site in train_sites:
            errors.append(f"[loeo:{name}] val site {val_site} also in train_sites")
        if val_site == test_site:
            errors.append(f"[loeo:{name}] val site equals test site ({test_site})")

        # cross-check sequence lists resolve to the declared sites
        seqs = cfg.get("sequences", {})
        if seqs:
            resolved = {role: {site_of(e) for e in seqs.get(role, [])}
                        for role in ("train", "val", "test")}
            if resolved.get("test") != {test_site}:
                errors.append(
                    f"[loeo:{name}] test sequences resolve to {sorted(resolved['test'])}, "
                    f"expected {{{test_site}}}"
                )
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    if a < b and resolved.get(a, set()) & resolved.get(b, set()):
                        errors.append(
                            f"[loeo:{name}] SITE LEAKAGE {a}/{b}: "
                            f"{sorted(resolved[a] & resolved[b])}"
                        )
        # union should cover 9 sites
        all_sites = train_sites | {val_site, test_site}
        if len(all_sites) != 9:
            errors.append(
                f"[loeo:{name}] fold covers {len(all_sites)} sites, expected 9: "
                f"{sorted(all_sites)}"
            )
        if not any(e.startswith(f"[loeo:{name}]") for e in errors):
            print(f"[loeo:{name}] OK — test={test_site}, val={val_site}, "
                  f"train={sorted(train_sites)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="data/splits",
                    help="directory containing train_*/val_*/test_*.txt")
    ap.add_argument("--loeo", default="data/splits/loeo",
                    help="directory containing fold_*.yaml")
    args = ap.parse_args()

    errors: list[str] = []
    errors += check_primary(args.splits)
    errors += check_loeo(args.loeo)

    if errors:
        print("\nFAILED — physical-site leakage or config errors detected:",
              file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print("\nAll splits are physically site-disjoint. No building crosses "
          "train/val/test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
