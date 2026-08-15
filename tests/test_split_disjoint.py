"""Site-disjointness of the primary split and every LOEO fold.

No physical building may appear in more than one of {train, val, test}. This
mirrors scripts/verify_site_disjoint.py but as pytest assertions so CI fails
loudly on any leakage.
"""
import os

import pytest

yaml = pytest.importorskip("yaml")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "data", "splits")
LOEO = os.path.join(SPLITS, "loeo")

ALL_SITES = {f"S{i}" for i in range(1, 10)}


def _sites_in_txt(path):
    sites = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split("/")[0].split()[0]
            # accept "S3", "S3/seq06", "seq06 S3", etc.: find an S<digit> token
            for part in line.replace("/", " ").split():
                if part.startswith("S") and part[1:].isdigit():
                    sites.add(part)
                    break
    return sites


def test_primary_split_site_disjoint():
    train = _sites_in_txt(os.path.join(SPLITS, "train_S1_S2_S4_S5_S8_S9.txt"))
    val = _sites_in_txt(os.path.join(SPLITS, "val_S3.txt"))
    test = _sites_in_txt(os.path.join(SPLITS, "test_S6_S7.txt"))
    assert train and val and test, "each split must list at least one site"
    assert not (train & val), f"train/val overlap: {train & val}"
    assert not (train & test), f"train/test overlap: {train & test}"
    assert not (val & test), f"val/test overlap: {val & test}"


def test_loeo_folds_disjoint_and_complete():
    fold_files = sorted(
        os.path.join(LOEO, f) for f in os.listdir(LOEO) if f.endswith(".yaml")
    )
    assert len(fold_files) == 9, f"expected 9 LOEO folds, found {len(fold_files)}"
    seen_test = set()
    for fp in fold_files:
        with open(fp) as f:
            fold = yaml.safe_load(f)
        test_site = fold["test_site"]
        val_site = fold["val_site"]
        train_sites = set(fold["train_sites"])
        assert test_site not in train_sites, f"{fp}: test site in train"
        assert val_site not in train_sites, f"{fp}: val site in train"
        assert val_site != test_site, f"{fp}: val == test"
        covered = train_sites | {val_site, test_site}
        assert covered == ALL_SITES, f"{fp}: sites not fully covered: {ALL_SITES - covered}"
        seen_test.add(test_site)
    assert seen_test == ALL_SITES, f"every site must be held out once; missing {ALL_SITES - seen_test}"
