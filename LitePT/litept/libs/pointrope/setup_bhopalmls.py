"""Build LitePT's PointROPE CUDA kernel for the GPUs of this machine.

Upstream libs/pointrope/setup.py hard-codes sm_90. This copy compiles the unmodified upstream
sources (kernels.cu, pointrope.cpp) for the architectures in $TORCH_CUDA_ARCH_LIST, or for the
GPUs visible to torch when the variable is unset.  Run from LitePT/litept/libs/pointrope:

    MAX_JOBS=4 python setup_bhopalmls.py build_ext --inplace

The .so lands in this directory; the launcher puts the directory on PYTHONPATH. Without it LitePT
falls back to its pure-PyTorch PointROPE (same results, slower).
"""
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _archs():
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        caps = [a.strip().replace(".", "") for a in env.replace(";", " ").split() if a.strip()]
    else:
        caps = sorted({"%d%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
    if not caps:
        raise SystemExit("no GPU visible; set TORCH_CUDA_ARCH_LIST, e.g. 8.0 (A100) or 9.0 (H100/H200)")
    flags = []
    for c in caps:
        flags += ["-gencode", f"arch=compute_{c},code=sm_{c}"]
    return flags


setup(
    name="pointrope",
    ext_modules=[CUDAExtension(name="pointrope", sources=["pointrope.cpp", "kernels.cu"],
                               extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"] + _archs()})],
    cmdclass={"build_ext": BuildExtension},
)
