# Collective motion through multi-agent active inference -- Modified from Heins et al. 2024
This repository provides code for simulating emergent collective motion from groups of continuous-time and -space active inference agents. **This code has been adapted** and originates from the paper ["Collective behavior from surprise minimization"](https://www.pnas.org/doi/10.1073/pnas.2320239121) (2024) by Conor Heins, Beren Millidge, Lancelot Da Costa, Richard Mann, Karl Friston, and Iain Couzin.

All coding modifications were created for an undergraduate research experience supported by Cal-Bridge and the University of California Riverside Mentoring Summer Research Internship Program (MSRIP). The program ran from June - August 2026 with a final symposium presentation on August 15 [See poster].

The original codebase contained both a [JAX](https://github.com/google/jax) and a [Julia](https://julialang.org/) implementation of a multi-agent active inference algorithm for generating collective motion. \*\**All coding modifications for this project have been applied to the JAX implementation*, focusing on demo_nolearning (see [the official instructions](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)).

- JAX backend installation/run instructions: [jax_backend/README_JAX.md](https://github.com/conorheins/collective_motion_actinf/blob/main/jax_backend/README_JAX.md)
