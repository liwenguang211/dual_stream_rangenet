# RangeFormer Baseline

The paper evaluates a **Swin-T re-implementation** of RangeFormer, not an
unversioned Python package. Its implementation is included in
`src/models/rangeformer_adapter.py` and fixed by these dependencies:

- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- CUDA 11.8 for the reported GPU measurements

Install `requirements.txt` or `environment.yml`. The model does not download
code or pretrained weights at runtime.

## Controlled training

```bash
python scripts/train.py \
  --model configs/models/rangeformer.yaml \
  --train configs/train/default.yaml \
  --seed 0 \
  --out checkpoints
```

The run uses the common preprocessing and site-disjoint split. A fixed selector
extracts the five conventional channels (`x`, `y`, `z`, normalized range and
intensity mean) from the canonical tensor; engineered descriptors are not used.
The validation-best checkpoint is `checkpoints/rangeformer_seed0.pth`. Record its
SHA-256 in `configs/models/rangeformer.yaml` and `checkpoints/sha256sums.txt`.
No checkpoint is claimed as released until the real file and matching hash are
present.

## External implementation

An alternative implementation may be supplied explicitly as `backbone=` to
`build_rangeformer`. Such a run must record the upstream repository URL,
immutable Git commit, configuration and checkpoint hash, and must not be mixed
with the Swin-T results reported by the paper.
