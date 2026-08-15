# Tests

Reproducibility and correctness guards. Run all with:

```bash
pip install pytest pyyaml numpy
pytest tests/ -v
```

| Test | Guards |
|------|--------|
| `test_split_disjoint.py` | No physical site crosses train/val/test; all 9 LOEO folds are disjoint and cover every site once |
| `test_16ch_preprocessing.py` | 16-channel layout, curvature at index 3, intensity(5)/geometry(11) partition |
| `test_corruption_determinism.py` | Corruptions are bit-exact for a fixed (seed, frame); 100% intensity-missing is seed-independent |
| `test_motion_distortion.py` | Motion distortion is genuine per-point timestamp interpolation (displacement grows with timestamp) |
| `test_metrics.py` | IoU/mIoU math, ignore-label handling, absent-class exclusion |
| `test_knn_protocol.py` | KNN default k=3; configs declare k=3 (primary/LOEO) and k=5 (cross-sensor); majority-vote cleanup |
| `test_table_recalculation.py` | Paper tables recompute from raw CSVs; `build_tables.py` embeds no hardcoded result numbers |

Tests use `pytest.importorskip` so a partial checkout (e.g. without `numpy` or
`pyyaml`) skips rather than errors. Nothing here requires a GPU, dataset, or
checkpoint download.
