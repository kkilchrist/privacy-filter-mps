# Privacy Filter on Apple Silicon (MPS)

This fork of [openai/privacy-filter](https://github.com/openai/privacy-filter) runs OPF inference on the Apple silicon GPU via PyTorch MPS. On an M4 Pro it is **~65× faster than CPU** with identical output.

Everything here has been submitted upstream — [PR #36](https://github.com/openai/privacy-filter/pull/36), which builds on [PR #22](https://github.com/openai/privacy-filter/pull/22) by [@Berkkirik](https://github.com/Berkkirik). This fork exists so Mac users can run it today.

## Quick start

```bash
pip install -e .
opf "Alice was born on 1990-01-02."
# info: no CUDA device detected; using Apple Metal (MPS).
# Alice was born on <PRIVATE_DATE>.
```

No flags needed — `--device auto` is the default and resolves `cuda > mps > cpu`. The Python API does the same:

```python
from opf import OPF
redactor = OPF()          # device="auto" -> mps on Apple silicon
redactor.redact("Call me at +1 (415) 555-0132.")
```

Requirements: Apple silicon Mac, PyTorch with MPS support (bf16 needs macOS 14+; verified with torch 2.12.1).

## Performance

Measured on a MacBook Pro (M4 Pro, 20-core GPU, 48 GB), released 1.5B checkpoint, ~1.5k-token document:

| Configuration | Throughput | Output vs CPU |
|---|---|---|
| CPU | ~65 tok/s | — |
| MPS, per-token gather MoE | ~173 tok/s (2.6×) | identical spans |
| **MPS, grouped MoE (default)** | **~4,300 tok/s (~65×)** | **identical spans, identical eval F1** |

Throughput holds on long documents (~4,400 tok/s at 50k tokens).

**Your numbers will differ.** The workload is GPU-compute and memory-bandwidth bound, so the speedup scales with GPU core count: expect roughly half of this on a base M4 (10-core GPU) and up to ~2× on an M4 Max (40-core GPU). CPU baseline varies much less across chips, so the *ratio* moves with your GPU.

## What makes it fast

The stock non-Triton MoE path gathers each routed expert's full weight matrices per token (~20 MB/token) — fine for CPU's small batches, catastrophic on a GPU. This fork adds a **grouped pure-torch MoE path**: tokens are sorted and packed by expert (the same packing the CUDA Triton kernels use), scattered into a padded per-expert batch, and each projection runs as a single `bmm`. Each expert's weights are read once per layer regardless of token count.

## Tuning

| Knob | Default on MPS | Notes |
|---|---|---|
| `--device` / `OPF(device=...)` | `auto` → `mps` | `cpu`, `cuda`, `mps` force a backend |
| `--n-ctx` | 4096 | Larger windows do **not** improve throughput (windows are non-overlapping; the attention band is ±128 tokens) but transient memory scales with window size — 16k was measurably *slower* and used ~2.5× the memory; the checkpoint's 128k default OOMs on long inputs even with 48 GB. Leave it alone. |
| `OPF_MOE_GROUPED` | `1` | `0` falls back to the per-token gather path (~24× slower on MPS) |
| `OPF_MOE_TRITON` | `0` | CUDA-only; forcing it on MPS raises |
| `OPF_ATTN_LOW_PRECISION` | `0` | No measurable benefit on MPS at the default window size — attention is not the bottleneck |

## Verified

- CPU vs MPS predicted spans are identical on multi-thousand-token synthetic PII documents.
- `opf eval` precision/recall/F1 identical between `--device cpu` and `--device mps` on the sample dataset.
- MXFP4 checkpoint decode is bit-identical to the CPU `ldexp` path (rewritten as an `exp2` multiply because `ldexp` has no MPS kernel).
- Finetuning (`opf train`) on MPS is **untested** — the grouped path is differentiable, but no training run has been validated here.

## Alternatives

If you need minimum *energy* rather than maximum throughput, see the [ANE/CoreML port](https://github.com/videlalvaro/emilio/blob/feat/privacy-filter-ane/python/privacy/README.md) by @videlalvaro, which compiles each expert as a separate CoreML model to fit the Neural Engine's single-graph memory ceiling — ~19× less energy per sentence than CPU.

---

*Upstream documentation: [README.md](README.md)*
