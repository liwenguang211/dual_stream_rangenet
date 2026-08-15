"""UBPC-9 nested site/sequence paths must resolve to the same frame."""
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.ubpc9_dataset import UBPC9Dataset


def test_nested_sequence_paths_are_preserved(tmp_path):
    sequence = tmp_path / "S1" / "seq01"
    (sequence / "velodyne").mkdir(parents=True)
    (sequence / "labels").mkdir()
    points = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    labels = np.array([4], dtype=np.int32)
    points.tofile(sequence / "velodyne" / "000001.bin")
    labels.tofile(sequence / "labels" / "000001.label")
    split = tmp_path / "split.txt"
    split.write_text("S1/seq01\n", encoding="utf-8")

    dataset = UBPC9Dataset(str(tmp_path), str(split))
    xyz, intensity = dataset._load_points(dataset.frames[0])
    loaded_labels = dataset._load_labels(dataset.frames[0])

    np.testing.assert_allclose(xyz, points[:, :3])
    np.testing.assert_allclose(intensity, points[:, 3])
    np.testing.assert_array_equal(loaded_labels, labels)
