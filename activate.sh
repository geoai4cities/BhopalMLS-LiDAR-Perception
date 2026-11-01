#!/usr/bin/env bash
# source activate.sh   (from anywhere)
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$_REPO/bhopalmls_venv/bin/activate" ]]; then
  echo "no environment yet: run ./setup_venv.sh first"; return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$_REPO/bhopalmls_venv/bin/activate"
export XDG_CACHE_HOME="$_REPO/bhopalmls_venv/cache" TORCH_HOME="$_REPO/bhopalmls_venv/cache/torch" HF_HOME="$_REPO/bhopalmls_venv/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$_REPO/bhopalmls_venv/cache/torch_extensions" CUDA_CACHE_PATH="$_REPO/bhopalmls_venv/cache/cuda"
_NVLIBS="$(python -c "import os,glob,nvidia; b=os.path.dirname(nvidia.__file__); print(':'.join(sorted(glob.glob(os.path.join(b,'*','lib')))))" 2>/dev/null || true)"
[[ -n "$_NVLIBS" ]] && export LD_LIBRARY_PATH="$_NVLIBS:${LD_LIBRARY_PATH:-}"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /usr/local/cuda /usr/local/cuda-11.8 /usr/local/cuda-12.4 /usr/local/cuda-12.8; do [[ -d "$c/include" ]] && export CUDA_HOME="$c" && break; done
fi
export BHOPALMLS_ROOT="$_REPO"
export PYTHONPATH="$_REPO/Pointcept:${PYTHONPATH:-}"
echo "BhopalMLS benchmark environment: $(python --version 2>&1), torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '?'), root $_REPO"
unset _REPO _NVLIBS
