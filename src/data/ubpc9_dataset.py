"""UBPC-9 torch Dataset built on the canonical splits and 16-channel features.

A sample is one LiDAR frame. The dataset reads a split file (primary txt or a
LOEO fold YAML), enumerates the frames of the listed sequences, and returns the
(16, H, W) input tensor plus the (H, W) label map. Preprocessed tensors are
loaded from ``data/processed`` when present; otherwise they are built on the fly
from the raw point cloud via :func:`build_16ch_input`.

Class ids follow data/class_mapping.yaml, whose order matches the model output.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    Dataset = object  # allows import for docs without torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .build_16ch_input import build_16ch_input


def _read_split(path: str) -> List[str]:
    """Return list of '<site>/<seq>' entries from a txt or LOEO yaml split."""
    if path.endswith((".yaml", ".yml")):
        if yaml is None:
            raise ImportError("PyYAML required to read LOEO fold files")
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        seqs = cfg.get("sequences", {})
        out: List[str] = []
        for role in ("train", "val", "test"):
            out += list(seqs.get(role, []))
        return out
    entries = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


class UBPC9Dataset(Dataset):
    def __init__(self, data_root: str, split_file: str,
                 split_role: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 height: int = 64, width: int = 512):
        """
        data_root:    directory containing raw sequences (data/UBPC9)
        split_file:   primary txt split or LOEO fold yaml
        split_role:   for LOEO yaml, restrict to 'train'/'val'/'test'
        processed_dir: optional dir with cached .npz tensors
        """
        self.data_root = data_root
        self.height = height
        self.width = width
        self.processed_dir = processed_dir

        if split_file.endswith((".yaml", ".yml")) and split_role:
            if yaml is None:
                raise ImportError("PyYAML required to read LOEO fold files")
            with open(split_file, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            self.sequences = list(cfg.get("sequences", {}).get(split_role, []))
        else:
            self.sequences = _read_split(split_file)

        self.frames = self._index_frames()

    def _index_frames(self) -> List[str]:
        """Enumerate frame ids under each sequence directory."""
        frames: List[str] = []
        for seq in self.sequences:
            seq_dir = os.path.join(self.data_root, seq, "velodyne")
            if not os.path.isdir(seq_dir):
                # sequence not downloaded; skip but keep the reference
                continue
            for fn in sorted(os.listdir(seq_dir)):
                if fn.endswith((".bin", ".npy")):
                    frames.append(os.path.join(seq, os.path.splitext(fn)[0]))
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def _load_points(self, frame: str):
        sequence, frame_id = os.path.split(frame)
        path = os.path.join(self.data_root, sequence, "velodyne", frame_id)
        raw = np.fromfile(path + ".bin", dtype=np.float32).reshape(-1, 4)
        return raw[:, :3], raw[:, 3]

    def _load_labels(self, frame: str) -> np.ndarray:
        sequence, frame_id = os.path.split(frame)
        path = os.path.join(self.data_root, sequence, "labels",
                            frame_id + ".label")
        return np.fromfile(path, dtype=np.int32)

    def __getitem__(self, idx: int):
        frame = self.frames[idx]
        if self.processed_dir:
            npz = os.path.join(self.processed_dir, frame + ".npz")
            if os.path.exists(npz):
                d = np.load(npz)
                return torch.from_numpy(d["tensor"]), torch.from_numpy(d["label"])
        points, intensity = self._load_points(frame)
        built = build_16ch_input(points, intensity, self.height, self.width)
        tensor = torch.from_numpy(built["tensor"])
        # label projection uses the same point_index mapping
        raw_labels = self._load_labels(frame)
        label_img = np.full((self.height, self.width), 255, np.int64)
        pi = built["point_index"]
        valid = pi >= 0
        label_img[valid] = raw_labels[pi[valid]]
        return tensor, torch.from_numpy(label_img)
