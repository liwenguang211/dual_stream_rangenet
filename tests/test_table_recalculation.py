"""Tables must be recomputable from raw CSVs, with no hardcoded paper numbers.

This is the anti-fabrication guard: it recomputes the headline aggregates
directly from results/raw/*.csv and checks they match the values the build
script produces, and that build_tables.py contains no hardcoded result numbers.
"""
import csv
import importlib.util
import os
import statistics
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "results", "raw")
TABLES = os.path.join(REPO, "results", "tables", "build_tables.py")


def _load_build_tables():
    spec = importlib.util.spec_from_file_location("build_tables", TABLES)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_tables"] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(row for row in f if not row.lstrip().startswith("#")))


def test_five_seed_means_recomputed_from_raw():
    rows = _read(os.path.join(RAW, "five_seed_results.csv"))
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(float(r["mIoU"]))
    # recompute and check against known reference means from the raw per-seed data
    means = {m: round(statistics.mean(v), 2) for m, v in by_model.items()}
    assert means["ds_rangenet_v3"] == pytest.approx(73.25, abs=0.02)
    assert means["single_stream_16ch"] == pytest.approx(68.60, abs=0.02)
    assert means["cenet_16ch"] == pytest.approx(65.95, abs=0.02)
    # every model must have exactly 5 seeds
    for m, v in by_model.items():
        assert len(v) == 5, f"{m} must have 5 seeds, has {len(v)}"


def test_build_tables_runs_and_matches_raw():
    mod = _load_build_tables()
    # the five-seed table the builder emits must contain the recomputed mean
    lines = mod.table_five_seed()
    joined = "\n".join(lines)
    assert "73.25" in joined, "DS-RangeNet 5-seed mean must appear, recomputed from raw"
    assert "68.60" in joined
    assert "65.95" in joined


def test_build_tables_has_no_hardcoded_results():
    # the script must not embed paper result numbers as literals: it should only
    # read CSVs and format. We forbid any float literal that looks like a mIoU
    # (two-decimal number in 40..100) in the source outside of comments.
    import re
    with open(TABLES) as f:
        code_lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    pat = re.compile(r"\b([4-9]\d|100)\.\d{1,2}\b")
    offenders = []
    for ln in code_lines:
        # ignore format-spec fragments like {:.2f}
        stripped = re.sub(r"\{[^}]*\}", "", ln)
        for m in pat.finditer(stripped):
            offenders.append((m.group(0), ln.strip()))
    assert not offenders, f"build_tables.py must not hardcode result numbers: {offenders}"


def test_missing_values_render_as_dash():
    mod = _load_build_tables()
    assert mod.fnum("FILL_ME") == "--"
    assert mod.fnum("") == "--"
    assert mod.fnum("73.25") == "73.25"
