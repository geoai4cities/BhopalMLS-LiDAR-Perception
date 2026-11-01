# Framework changes

The benchmark runs on two upstream code bases, vendored in full at the commits the paper used: `Pointcept/` (shared by PTv3, KPConvX, Sonata, Concerto and Utonia) and `LitePT/litept/` (the LitePT fork, used by the LitePT-L runs). This folder records exactly what differs from upstream so the difference is inspectable: one git patch per framework for the upstream files we modified, and a list of the files we added (their content is in the vendored trees). `rebuild_from_upstream.sh` re-creates the vendored trees from GitHub (maintainers only). `Pointcept/UPSTREAM_COMMIT` and `LitePT/litept/UPSTREAM_COMMIT` name the commits.

| framework | upstream | pinned commit | licence | vendored files |
|---|---|---|---|---|
| Pointcept | https://github.com/Pointcept/Pointcept | `cca1802c9083759f40a71ff740b7522c47dc112a` | MIT | 576 |
| LitePT | https://github.com/prs-eth/LitePT | `fb44846dcdb33deeb0739c53a0b338f1091aecbb` | MIT | 206 (sparsehash reduced to its headers; upstream had committed 29 MB of build output) |
| KPConvX (inside Pointcept) | https://github.com/apple/ml-kpconvx | files under `Pointcept/pointcept/models/kpconvx/` | MIT (Apple) | |

## Pointcept

New files (`pointcept_new_files.txt`):

- `pointcept/datasets/bhopal.py`
- `pointcept/datasets/kerala.py`
- `pointcept/datasets/toronto.py`
- `pointcept/datasets/paris_lille.py`
- `pointcept/models/point_transformer_v3/point_transformer_v3m3_utonia.py`
- `pointcept/datasets/__init__.py`
- `pointcept/models/__init__.py`
- `pointcept/models/point_transformer_v3/__init__.py`
- `pointcept/models/kpconvx/` (40 files: the KPConvX port with its C++ wrapper sources)

Patched upstream files (`pointcept.patch`):

- `pointcept/datasets/transform.py`
- `pointcept/datasets/utils.py`
- `pointcept/datasets/semantic_kitti.py`
- `pointcept/engines/hooks/evaluator.py`
- `pointcept/engines/hooks/misc.py`
- `pointcept/engines/test.py`
- `pointcept/models/point_group/point_group_v1m2_custom_criteria.py`
- `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
- `pointcept/models/point_transformer_v3/point_transformer_v3m2_sonata.py`
- `pointcept/models/utils/serialization/default.py`

What the patch does: `ZeroColor`/`ZeroNormal` transforms (geometry-only probes); `np.in1d` -> `np.isin` (NumPy 2); grid coordinates recomputed after CopyPaste mixing when a config carries `grid_size`; `learning_map` argument for the SemanticKITTI datasets (unified 12-class taxonomy); `CheckpointLoader` skips key remapping on resume; `InsSegEvaluator` frees GPU memory before evaluation; PointGroup returns an empty proposal tensor of the right shape when no cluster survives; two upstream fixes back-ported (flash-attention dtype, #594; z-order decode, #600).

## LitePT

New files (`litept_new_files.txt`):

- `datasets/bhopal.py`
- `datasets/__init__.py`
- `libs/pointrope/setup_bhopalmls.py`

Patched upstream files (`litept.patch`):

- `configs/_base_/default_runtime.py`
- `datasets/transform.py`
- `engines/hooks/evaluator.py`
- `engines/hooks/misc.py`
- `engines/test.py`
- `engines/train.py`
- `models/point_group/point_group_v1m2_custom_criteria.py`
- `tools/test.py`

What the patch does: gradient accumulation (`gradient_accumulation_steps`) in the trainer and the base config; `InsSegTester` and a `PreciseEvaluator` that forwards tester arguments (instance segmentation); `np.in1d` -> `np.isin`; the same empty-proposal guard in PointGroup. `libs/pointrope/setup_bhopalmls.py` builds the upstream PointROPE CUDA kernel for the GPUs of the machine (upstream hard-codes sm_90); without it LitePT uses its pure-PyTorch PointROPE, same results, slower.
