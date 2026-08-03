# Installation

`hone` is built for Apple Silicon — macOS on M1/M2/M3/M4-series
chips with unified memory. The MLX backend runs on the Metal GPU
and is the default and supported path. A CUDA/Unsloth fallback
is available for NVIDIA hosts.

## Apple Silicon (MLX, recommended)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,mlx]'
```

Verify:

```bash
python -c "import hone; print(hone.__version__)"
python -m hone.run --help   # prints mlx_lm.lora's --help text
```

`HONE_DEVICE=gpu` (the default) requires Metal. If Metal is
unavailable, the launcher refuses to start with a clear error.

## CUDA / NVIDIA (Unsloth)

```bash
uv pip install -e '.[dev,cuda]'
hone train code --backend cuda
```

The CUDA path is a thin wrapper around the OpenBMB MiniCPM5
Unsloth recipe. It is not the default and is not tested in CI.

## Extras

| Extra  | Purpose                                        |
|--------|------------------------------------------------|
| `dev`  | pytest, mypy, ruff, hypothesis                  |
| `mlx`  | MLX and MLX-LM (Apple Silicon only)            |
| `cuda` | torch, transformers, trl, peft, unsloth        |

## Environment variables

| Variable      | Default | Effect                                  |
|---------------|---------|-----------------------------------------|
| `HONE_DEVICE` | `gpu`   | `gpu` or `cpu`; case-insensitive        |
