# JAX backend setup instructions

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Optional for GPU installs: NVIDIA GPU, current drivers, and CUDA-compatible system setup

## Install and sync dependencies

From the repository root:

```bash
cd /Users/conorheins/Documents/collective_motion_actinf/jax_backend
```

### CPU setup

```bash
uv sync --group cpu --group dev
```

### CUDA 12 GPU setup

```bash
uv sync --group cuda12 --group dev
```

Dependency versions are controlled by `pyproject.toml` and locked in `uv.lock`.

## Running a demo

Run demo scripts through `uv run` so they use the synced environment.

```bash
uv run python src/demo_nolearning.py --seed 2 --N 10 --dt 0.01 --T 20 --last_T_seconds 10
```

You can append `--save` to persist trajectory history in a local `.npz` file.

## Testing

```bash
uv run pytest -q
```

## Troubleshooting

### `TypeError: 'type' object is not subscriptable` in Haiku

This usually indicates Python 3.8 with newer package versions. Use Python 3.11+ and resync:

```bash
uv sync --group cpu --group dev --python 3.11
```

### `AttributeError: module 'jax.random' has no attribute 'KeyArray'`

This comes from incompatible `jax`/`jax-md` combinations (historically seen with older `jax-md` plus newer JAX). Use the lockfile-driven install to keep compatible versions together:

```bash
uv sync --group cpu --group dev --frozen
```

### Dependency drift after manual pip installs

If `pip install ...` is run outside `uv sync`, transitive dependencies can drift and break imports. Re-sync from lockfile:

```bash
uv sync --group cpu --group dev --frozen
```

### Matplotlib figure does not show

In headless environments (CI, remote shells), use file output (`--save`) or set:

```bash
export MPLBACKEND=Agg
```
