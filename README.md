# Collective motion through multi-agent active inference
This repository provides code for simulating emergent collective motion from groups of continuous-time and -space active inference agents. This code serves as companion for the paper ["Collective behavior from surprise minimization"](https://www.pnas.org/doi/10.1073/pnas.2320239121) (2024) by Conor Heins, Beren Millidge, Lancelot Da Costa, Richard Mann, Karl Friston, and Iain Couzin.

This codebase contains both a [JAX](https://github.com/google/jax) and a [Julia](https://julialang.org/) implementation of a multi-agent active inference algorithm for generating collective motion. The JAX implementation (in the `jax_backend` folder) is the recommended implementation, especially because it can automatically take advantage of GPU support on machines with NVIDIA-capable GPUs (see [the official instructions](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)).

- JAX backend installation/run instructions: [jax_backend/README_JAX.md](https://github.com/conorheins/collective_motion_actinf/blob/main/jax_backend/README_JAX.md)
- Julia installation/run instructions: [julia/README_Julia.md](https://github.com/conorheins/collective_motion_actinf/blob/main/julia/README_Julia.md)
