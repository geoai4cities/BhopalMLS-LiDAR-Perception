# Environment

`setup_venv.sh` builds one virtual environment for every model in the benchmark (PTv3, LitePT-L, KPConvX-L, Sonata, Concerto, Utonia). It is what the paper's runs used. The two frameworks are vendored under the repository (`Pointcept/`, `LitePT/litept/`) at the commits those runs were made on, so the script downloads Python packages only.

```bash
./setup_venv.sh --with-kpconvx --with-pointrope   # one-off, 30 to 60 min (flash-attn compiles)
source activate.sh                                 # every session
```

| component | version | note |
|---|---|---|
| Python | 3.11 | created with `uv`, which also provides the interpreter if the system has none |
| PyTorch / torchvision | 2.5.0 / 0.20.0 | `cu118` wheels by default (what the paper ran); `--cuda 12.4` for cu124 |
| flash-attn | 2.8.3 | required by all transformer configs (`enable_flash=True`) |
| spconv | spconv-cu118 2.3.8 | used with both CUDA 11.8 and 12.x torch wheels; the cu120 build crashed on our H200 |
| torch-scatter / cluster / sparse | 2.1.2 / 1.6.3 / 0.6.18 | PyG wheels for torch 2.5.0 |
| torch-geometric | 2.7.0 | KPConvX voxel grid pooling |
| transformers | 4.50.3 | imported by the Concerto model code |
| numpy | 1.26.4 | the patched frameworks also run on NumPy 2 (`np.in1d` -> `np.isin` patches) |
| pointops, pointgroup_ops | built from `Pointcept/libs` | pointgroup_ops needs the header-only sparsehash library, vendored in `LitePT/litept/libs/sparsehash/src` |
| KPConvX C++ wrappers | `--with-kpconvx` | grid subsampling and neighbour search, built in place |
| PointROPE CUDA kernel | `--with-pointrope` | optional; LitePT falls back to its PyTorch implementation (same results, slower) |

`requirements-lock.txt` is the complete `pip freeze` of the environment the numbers were produced in, for reference.

The paper's runs used Ubuntu 22.04, gcc 9.5, CUDA toolkit 11.8 with driver 535, on NVIDIA A100 40 GB and H200 143 GB GPUs. Flash-attention needs an Ampere or newer GPU (compute capability 8.0+).

## Why one environment

KPConvX was trained in a separate environment with the identical package versions plus its two C++ wrappers; the wrappers do not conflict with anything else, so the release builds them into the same environment behind `--with-kpconvx`.

## Troubleshooting

- `libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4` on `import torch` (cu124 wheels): a system CUDA on `LD_LIBRARY_PATH` shadows the wheel's libraries. `activate.sh` prepends the bundled ones; make sure it was sourced.
- `Python.h: No such file` while compiling: the interpreter lacks headers. `uv venv --python 3.11` installs a uv-managed interpreter that has them; delete `bhopalmls_venv` and re-run.
- flash-attn build killed: set `MAX_JOBS=4` (memory) and re-run `setup_venv.sh`; it skips what is already installed.
- A CUDA extension failed: its build log is in `bhopalmls_venv/<name>_build.err`; the usual causes are a missing `CUDA_HOME` or an nvcc that does not match the torch wheels (11.8 wheels need a CUDA 11.x toolkit, 12.4 wheels a 12.x one).
- `AF_UNIX path too long` in DataLoader workers: do not point `TMPDIR` into a deep directory; the scripts leave it alone on purpose.
