#!/usr/bin/env bash
################################################################################
# BhopalMLS benchmark: environment setup
#
# Creates bhopalmls_venv/ (Python 3.11), installs the pinned PyTorch stack the paper used and
# builds the CUDA extensions of the vendored frameworks (Pointcept/, LitePT/litept/).
#
#   ./setup_venv.sh                       # CUDA 11.8 wheels (what the paper ran), full build
#   ./setup_venv.sh --cuda 12.4           # CUDA 12.4 wheels
#   ./setup_venv.sh --with-kpconvx        # also build the KPConvX C++ wrappers
#   ./setup_venv.sh --with-pointrope      # also build LitePT's PointROPE CUDA kernel
#   ./setup_venv.sh --skip-compile        # no CUDA extensions (configs will not run)
#   ./setup_venv.sh --cpu                 # CPU-only wheels, for reading data and configs
#
# Afterwards:  source activate.sh
#
# Tested: Ubuntu 22.04, gcc 9.5, nvcc 11.8, driver 535, A100 40 GB and H200 143 GB,
# Python 3.11.12, torch 2.5.0+cu118, flash-attn 2.8.3, spconv-cu118 2.3.8.
################################################################################
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/bhopalmls_venv"
PY_VERSION="3.11"
CUDA_TAG=""            # cu118 | cu124 | cpu
SKIP_COMPILE=0
WITH_KPCONVX=0
WITH_POINTROPE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) case "$2" in 11.8|cu118) CUDA_TAG=cu118;; 12.*|cu124) CUDA_TAG=cu124;; *) echo "bad --cuda $2 (11.8 or 12.4)"; exit 1;; esac; shift 2;;
    --cpu) CUDA_TAG=cpu; shift;;
    --skip-compile) SKIP_COMPILE=1; shift;;
    --with-kpconvx) WITH_KPCONVX=1; shift;;
    --with-pointrope) WITH_POINTROPE=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1"; exit 1;;
  esac
done

say() { printf '\n\033[1;35m== %s\033[0m\n' "$*"; }
ok()  { printf '\033[0;32m[ok]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------------ 1. prerequisites
say "1/8 prerequisites"
if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found; installing it (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

if [[ -z "$CUDA_TAG" ]]; then
  if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then
    NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    V="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')"
    case "$V" in 11.*) CUDA_TAG=cu118;; 12.*) CUDA_TAG=cu124;; *) CUDA_TAG=cu118;; esac
    ok "nvcc $V -> torch wheels $CUDA_TAG"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_TAG=cu118; warn "no nvcc on PATH; using cu118 wheels, but the CUDA extensions need a toolkit (install CUDA 11.8 or 12.x and re-run)"
  else
    CUDA_TAG=cpu; warn "no GPU detected; CPU-only install"
  fi
fi
if [[ "$CUDA_TAG" != cpu && -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda /usr/local/cuda-11.8 /usr/local/cuda-12.4 /usr/local/cuda-12.8 /opt/cuda; do
    [[ -d "$c/include" ]] && { export CUDA_HOME="$c"; break; }
  done
  [[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH" && ok "CUDA_HOME=$CUDA_HOME"
fi
case "$CUDA_TAG" in
  cu118) TORCH_INDEX=https://download.pytorch.org/whl/cu118;;
  cu124) TORCH_INDEX=https://download.pytorch.org/whl/cu124;;
  cpu)   TORCH_INDEX=https://download.pytorch.org/whl/cpu;;
esac

# ------------------------------------------------------------------ 2. venv
say "2/8 virtual environment at $VENV"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python "$PY_VERSION" --seed
fi
# shellcheck disable=SC1091
set +u; source "$VENV/bin/activate"; set -u   # activate scripts are not written for `set -u`
ok "python $(python --version 2>&1)"
mkdir -p "$VENV/cache"/{torch,huggingface,torch_extensions,cuda,pip}
export XDG_CACHE_HOME="$VENV/cache" TORCH_HOME="$VENV/cache/torch" HF_HOME="$VENV/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$VENV/cache/torch_extensions" CUDA_CACHE_PATH="$VENV/cache/cuda" PIP_CACHE_DIR="$VENV/cache/pip"
PIP="uv pip install"

# ------------------------------------------------------------------ 3. PyTorch
say "3/8 PyTorch 2.5.0 ($CUDA_TAG)"
if [[ "$CUDA_TAG" == cpu ]]; then
  $PIP torch==2.5.0 torchvision==0.20.0 --index-url "$TORCH_INDEX"
else
  $PIP "torch==2.5.0+$CUDA_TAG" "torchvision==0.20.0+$CUDA_TAG" --index-url "$TORCH_INDEX"
fi
# the cu124 wheels bundle their own nvidia libs; put them first so a system CUDA cannot shadow them
NVLIBS="$(python -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(os.path.join(b,'*','lib')))))" 2>/dev/null || true)"
[[ -n "$NVLIBS" ]] && export LD_LIBRARY_PATH="$NVLIBS:${LD_LIBRARY_PATH:-}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# ------------------------------------------------------------------ 4. python packages
say "4/8 python packages"
$PIP "numpy==1.26.4" "scipy==1.17.0" "scikit-learn==1.8.0" "pandas==3.0.0" "h5py==3.15.1" "tqdm==4.67.1" \
     "pyyaml==6.0.3" "plyfile==1.1.3" "addict==2.4.0" "yapf==0.43.0" "einops==0.8.1" "termcolor==3.3.0" \
     "easydict==1.13" "tensorboardx==2.6.4" "timm==1.0.24" "peft==0.18.1" "colorhash==2.1.0" "torchmetrics==1.8.2" \
     "huggingface-hub==0.36.0" "psutil==7.2.1" "wandb==0.24.0" "open3d==0.19.0" "fast_pytorch_kmeans" \
     "transformers==4.50.3" "matplotlib==3.10.8" "shapely" "geopandas" "setuptools==80.10.1" wheel "ninja==1.13.0" packaging
$PIP "torch-geometric==2.7.0"
if [[ "$CUDA_TAG" == cpu ]]; then
  $PIP torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
else
  # the PyG index for torch 2.5.0 carries exactly the versions of requirements-lock.txt (2.1.2 / 1.6.3 / 0.6.18)
  $PIP torch-scatter torch-cluster torch-sparse -f "https://data.pyg.org/whl/torch-2.5.0+$CUDA_TAG.html"
  # spconv: the cu118 build is used with both CUDA 11.8 and 12.x wheels (the cu120 build crashed with a
  # floating-point exception on the H200 in our runs)
  $PIP "spconv-cu118==2.3.8"
fi

# ------------------------------------------------------------------ 5. flash attention
say "5/8 flash-attention 2.8.3 (compiles for 15 to 40 min; needed by PTv3, Sonata, Concerto, Utonia, LitePT)"
if [[ "$CUDA_TAG" == cpu ]]; then
  warn "skipped on CPU"
else
  MAX_JOBS="${MAX_JOBS:-8}" $PIP "flash-attn==2.8.3" --no-build-isolation || warn "flash-attn failed; configs with enable_flash=True will not run until it is installed"
fi

# ------------------------------------------------------------------ 6. frameworks
say "6/8 vendored frameworks"
for FWDIR in Pointcept LitePT/litept; do
  [[ -f "$REPO/$FWDIR/UPSTREAM_COMMIT" ]] || { echo "$FWDIR is missing; clone the full repository"; exit 1; }
  ok "$FWDIR @ $(sed -n 2p "$REPO/$FWDIR/UPSTREAM_COMMIT")"
done

# ------------------------------------------------------------------ 7. CUDA extensions
say "7/8 CUDA extensions"
if [[ "$SKIP_COMPILE" == 1 || "$CUDA_TAG" == cpu ]]; then
  warn "skipped"
else
  PC="$REPO/Pointcept"
  if ( cd "$PC/libs/pointops" && rm -rf build dist ./*.egg-info && ($PIP --no-build-isolation . || python setup.py install) ) >/dev/null 2>"$VENV/pointops_build.err"; then
    ok "pointops"
  else
    warn "pointops failed (log: $VENV/pointops_build.err); nothing runs without it"
  fi
  # pointgroup_ops needs the header-only sparsehash library; the headers are vendored with LitePT
  SPARSEHASH="$REPO/LitePT/litept/libs/sparsehash/src"
  [[ -f "$SPARSEHASH/google/dense_hash_map" ]] || warn "sparsehash headers not found at $SPARSEHASH"
  if ( cd "$PC/libs/pointgroup_ops" && rm -rf build dist ./*.egg-info && CPLUS_INCLUDE_PATH="$SPARSEHASH:${CPLUS_INCLUDE_PATH:-}" $PIP --no-build-isolation . ) >/dev/null 2>"$VENV/pointgroup_ops_build.err"; then
    ok "pointgroup_ops (instance segmentation)"
  else
    warn "pointgroup_ops failed (log: $VENV/pointgroup_ops_build.err); instance-segmentation configs will not run"
  fi
  if [[ "$WITH_KPCONVX" == 1 ]]; then
    for w in cpp_subsampling cpp_neighbors; do
      if ( cd "$PC/pointcept/models/kpconvx/cpp_wrappers/$w" && rm -rf build && python setup.py build_ext --inplace ) >/dev/null 2>"$VENV/kpconvx_${w}_build.err"; then
        ok "KPConvX $w"
      else
        warn "KPConvX $w failed (log: $VENV/kpconvx_${w}_build.err)"
      fi
    done
  fi
  if [[ "$WITH_POINTROPE" == 1 ]]; then
    if ( cd "$REPO/LitePT/litept/libs/pointrope" && MAX_JOBS="${MAX_JOBS:-4}" python setup_bhopalmls.py build_ext --inplace ) >/dev/null 2>"$VENV/pointrope_build.err"; then
      ok "PointROPE CUDA kernel"
    else
      warn "PointROPE kernel failed (log: $VENV/pointrope_build.err); LitePT falls back to its PyTorch implementation"
    fi
  fi
fi

# ------------------------------------------------------------------ 8. verify
say "8/8 verification"
export PYTHONPATH="$REPO/Pointcept:${PYTHONPATH:-}"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
for m in ("spconv", "flash_attn", "torch_scatter", "pointops", "pointgroup_ops", "h5py", "transformers", "peft"):
    try:
        importlib.import_module(m); print("[ok]", m)
    except Exception as e:
        print("[--]", m, "->", type(e).__name__, str(e)[:80])
try:
    from pointcept.datasets import BhopalSceneHDF5Dataset  # noqa
    from pointcept.models.point_transformer_v3 import point_transformer_v3m3_utonia  # noqa
    print("[ok] Pointcept + BhopalMLS additions")
except Exception as e:
    print("[--] Pointcept ->", type(e).__name__, e)
PY
ok "done. Next:  source ./activate.sh"
