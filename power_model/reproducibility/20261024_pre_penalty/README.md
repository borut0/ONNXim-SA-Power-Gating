# Pre-penalty GreenSA experiment checkpoint (2026-10-24)

This records the code and inputs used for the BookSim2/128-GB baseline before
changing performance-penalty modeling. The research branch's main repository
code is commit `b42429d86415092f43cce4ae2427d572deee6b0d`. The active
Docker workspace reported Git HEAD
`fb87ca44fc0d5e59cde5ae3d3f2dd2ab445eee67`, with local source edits;
the simulator source and tests were copied into the local snapshot named below.

The two submodules also had uncommitted changes, now preserved here as binary
Git patches. Their pinned base commits are:

| Submodule | Base commit | Patch |
| --- | --- | --- |
| BookSim2 | `1d69c9d9db190d130087471b373d12ed4605bf6e` | `booksim_dirty.patch` |
| Ramulator2 | `0d063011281a112da29e67cbd708308dd05e7e64` | `ramulator2_dirty.patch` |

Apply each patch within a clean checkout of its pinned submodule commit:

```sh
git -C extern/booksim apply ../../power_model/reproducibility/20261024_pre_penalty/booksim_dirty.patch
git -C extern/ramulator2 apply ../../power_model/reproducibility/20261024_pre_penalty/ramulator2_dirty.patch
```

The active Docker binary was `sha256:91539f35e0c9bb906f719faa2f6375d5dfae327334e6313e353e2a739f277f49`.
The configuration was `configs/systolic_ws_128x128_c4_booksim2_big_model.json`
with SHA-256 `00112af82c4f364d1b650deede53c35c670b1167f8134dab3a6c83b49e330b61`.
The language trace was `traces/input_32_8.csv` with SHA-256
`0522a662d69e723a5bcc9c8314cbf73ff38f4bbaa9cd92c3b04d86337790e52d`.
`model_sha256.txt` lists the 25 model files available inside Docker.

The container image reference was `onnxim`, image ID
`sha256:89d5d10ccd14255b0134e962b229ba558d92df216d5a24b91fd97b69437882d2`.
The container had no host bind mounts.

The local, Git-ignored snapshot is
`power_model/final_experiments/repro_snapshots/20261024_pre_penalty/`. It
contains the copied source, build tree, 25 model files, metadata, and the 12
completed workloads' raw logs and summaries. Their always-on, ReGate,
GreenSA, and idleness status markers all passed. The snapshot also contains
separately labeled partial copies of the large-model jobs that were still
running. Its `SHA256SUMS` file verified all copied files, and copied GPT-2
comparison and ResNet-18 idleness logs regenerated their stored summaries
byte for byte.

The large model inputs, raw result logs, and Docker image are **not** stored
in GitHub. This Git checkpoint preserves the compact source changes and input
fingerprints; the local snapshot preserves the data needed to compare a future
performance model with the current one.
