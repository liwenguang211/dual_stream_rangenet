# UBPC-9 Dataset

UBPC-9 (**U**rban **B**uilding **P**oint **C**loud, **9** classes) is a MID-360
LiDAR semantic-segmentation benchmark for industrial-building perception. It
contains **12,000 frames** collected across **9 physically distinct sites**
(steel-structure factories, multi-level warehouses, underground parking lots),
labeled with **9 classes**.

The dataset itself is distributed separately (research use only, see LICENSE);
this directory ships only the metadata and the split definitions needed to
reproduce every paper table.

## Physical parameters (self-consistent)

| Quantity            | Value                                             |
|---------------------|---------------------------------------------------|
| Total frames        | 12,000                                            |
| LiDAR rate          | 10 Hz (Livox MID-360)                             |
| AGV speed           | 1.5 m/s                                           |
| Total trajectory    | 1,800 m                                           |
| Total duration      | 1,200 s                                           |
| Keyframe rule       | every 0.3 m travelled or 5° yaw change            |
| Projection          | spherical, 64 × 512 range image                   |

Derivation: 12,000 frames / 10 Hz = 1,200 s; 1,200 s × 1.5 m/s = 1,800 m.

## The 9 sites

See [`site_metadata.csv`](site_metadata.csv) for the machine-readable table.

| Site | Environment            | Sequences        | Frames | Primary split |
|------|------------------------|------------------|-------:|---------------|
| S1   | Steel-structure factory A | seq01–seq03   |  1900  | train         |
| S2   | Steel-structure factory B | seq04–seq05   |  1500  | train         |
| S3   | Multi-level warehouse A   | seq06–seq07   |  1300  | validation    |
| S4   | Multi-level warehouse B   | seq08–seq09   |  1200  | train         |
| S5   | Multi-level warehouse C   | seq10–seq11   |  1300  | train         |
| S6   | Underground parking lot A | seq12–seq13   |  1100  | test          |
| S7   | Underground parking lot B | seq14–seq15   |  1000  | test          |
| S8   | Steel-structure factory C | seq16–seq17   |  1400  | train         |
| S9   | Multi-level warehouse D   | seq18–seq20   |  1300  | train         |

## Splits

**Primary split** (site-disjoint):

- train = S1,S2,S4,S5,S8,S9 → 8,600 frames (71.67%) — [`splits/train_S1_S2_S4_S5_S8_S9.txt`](splits/train_S1_S2_S4_S5_S8_S9.txt)
- val   = S3 → 1,300 frames (10.83%) — [`splits/val_S3.txt`](splits/val_S3.txt)
- test  = S6,S7 → 2,100 frames (17.50%) — [`splits/test_S6_S7.txt`](splits/test_S6_S7.txt)

The two test sites are underground parking facilities in different building
complexes from every training site, so **no physical building contributes
sequences to more than one split**.

**LOEO (Leave-One-Environment-Out) 9-fold** — [`splits/loeo/fold_S1.yaml … fold_S9.yaml`](splits/loeo/):
each fold holds out one site as test, uses S3 as validation (S4 when S3 is the
test site), and trains on the remaining 7 sites. Train/val/test are pairwise
site-disjoint in every fold. Per-fold reported mIoU: 70.4, 70.6, 71.0, 71.2,
71.5, 71.8, 70.2, 70.8, 71.1 (macro-avg 70.96 ≈ reported 71.0).

## Classes

The 9 classes (channel order is authoritative, matches model output
`python/ds_rangenet_v3.py` and every per-class IoU column of
`reproducibility/seeds/results/per_seed_miou.csv`): `background, ground, roof,
side_facade, front_facade, beam, column, window, dynamic`. Raw-label → class
mapping is in [`class_mapping.yaml`](class_mapping.yaml).

## Directory layout expected after download

```
data/
├── README_UBPC9.md          # this file
├── site_metadata.csv        # 9 sites, frames, sequences, split
├── class_mapping.yaml       # raw label -> 9 UBPC-9 classes
├── splits/
│   ├── train_S1_S2_S4_S5_S8_S9.txt
│   ├── val_S3.txt
│   ├── test_S6_S7.txt
│   └── loeo/fold_S1.yaml … fold_S9.yaml
├── UBPC9/                    # (you download this) raw sequences seq01..seq20
│   └── <site>/<seq>/*.bin, labels/*.label
└── processed/               # (generated) 16-channel range images
```

## Verify integrity of the splits

Before training, confirm there is no physical-site leakage:

```bash
python scripts/verify_site_disjoint.py --splits data/splits --loeo data/splits/loeo
```

The single source of truth for all split definitions and reported metrics is
[`../loeo_supporting_materials/ubpc9_splits_v3.json`](../loeo_supporting_materials/ubpc9_splits_v3.json).
