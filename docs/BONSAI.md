# Ternary Bonsai 2 27B

[Back to README](../README.md)

Prism Ternary Bonsai 2 27B is the model the `qwen35` path was ported for: a
dense 26.90 B trunk of 64 layers, 48 gated delta-net layers and 16
gated-attention layers at interval 4, a dense SwiGLU FFN, no MoE, no n-gram
embeddings and no MTP block. Every matmul weight is PQ2_0: 128 weights per
block, one FP16 scale and 32 ternary code bytes, 2.125 bits per weight.

The Prism exporter stores those weights already rotated into the block-Hadamard
basis the quantization likes and expects the runtime to rotate the activation
instead. This GGUF declares `prism.hadamard` with a 1024-wide block basis,
three sign vectors and the gated-delta-net head permutation, and ds4 applies
exactly that in the CPU reference and in the CUDA graph. Norms and the ssm
scalars are F32 and the two ssm gates are BF16, as the exporter writes them.

## Download and run

```sh
make cuda-generic
./download_model.sh bonsai-pq2
./ds4 --cuda -p "The capital of France is"
```

`bonsai-pq2` fetches one **6.71 GiB** GGUF from
[prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
verifies the published SHA-256, and links `ds4flash.gguf` to it. The loader
validates every matmul tensor of this family as PQ2_0
(`tensor_expect_pq2_0_layout` in `ds4.c`) and exits otherwise, so the
repository's other files (F16, Q2_0, Q2_g64, PTQ1_0 and the dspark pair) are
not DwarfStar models.

PQ2_0 kernels exist only in the CUDA path (`cuda/mmq/`): this model runs on
CUDA, with the in-process CPU implementation as its correctness reference.
Metal and ROCm have no PQ2_0 kernel yet. There is no vision path for the family
either: `--vision` expects a Qwen3-VL mmproj, so the Q8_0 mmproj published
next to these weights serves llama.cpp.

## Context and memory

The graph allocates its fp16 k/v cache up front at about 64 KiB per token (16
attention layers, 4 kv heads, 256 dimensions, k and v), so the context is fixed
at startup and changing `--ctx` means a restart. The GGUF declares a
262144-token context; the card decides how much of it is usable. Measured with
the CUDA backend on an RTX 4070 SUPER (12 GB, desktop session resident):

| `--ctx` | Process VRAM | Outcome |
| --- | ---: | --- |
| 49152 | 10628 MiB | safe |
| 57344 | 11140 MiB | comfortable, about 900 MiB of headroom |
| 61440 | 11396 MiB | works, about 530 MiB free during prefill |
| 65536 | 11652 MiB | first request fails: `ds4_mmq_pq2_0_rows_f32: launch failed: out of memory` |

The weights are not the constraint: the model itself is 6.7 GiB and fits a
12 GB card with room to spare, while the k/v cache is what grows with context.
A client's own prompt is the other constraint: an agent turn carrying a large
tool schema prefills tens of thousands of tokens (about 40000 in the open-grok
setup measured here), so a context below about 49152 cannot hold one at all.

## Serving

```sh
./ds4-server --cuda -m ds4flash.gguf --ctx 57344 --port 8899
```

The server serves `prism-bonsai-2-27b` under the Qwen ChatML-with-reasoning
syntax this family shares with Qwen3.8. Prefill runs in 512-token chunks. The
model answers with a thinking block first, delivered as `reasoning_content`, so
short requests need room to finish: a one-word answer took 15 completion tokens
after a 59-token prompt.

`run-bonsai.sh` (decode, with `compare` and `session` diffed against the CPU
reference, plus `bench` and `status`) and `serve-bonsai.sh` (start, stop,
status, smoke, and the open-grok model block for this host) wrap both entry
points from the repository root.

## Validation

```sh
make pq2-0-test             # PQ2_0 block format against the Prism reference
make test-qwen35-cuda       # CUDA kernels against the in-process CPU reference
DS4_TEST_MODEL=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf make test-qwen35-session
make bonsai-ref-check       # greedy decode on the CPU reference
make bonsai-fold-selftest   # fold round-trips and the gated-delta-net permutation
make test-download-model    # the bonsai-pq2 target, offline, with fixtures
```

The first two need no model weights: they cover the row lookup, the decode
matvec and the prefill tile against the reference dequantizer, so the test
never re-implements the block format. The session test needs the GGUF and
covers create/sync/eval, prefix reuse, rewind replay and invalidate rebuild.
`bonsai-ref-check` and `bonsai-fold-selftest` read `DS4_BONSAI_MODEL`, which
defaults to a local model path.

## Notes and limits

- PQ2_0 only, as described above.
- CUDA and the CPU reference; no Metal or ROCm kernel for PQ2_0 yet.
- No vision: the encoder shipped with these weights is the llama.cpp one.
- The `--first-token-test` diagnostic decodes one token per forward; the
  session and server paths use the chunked prefill.
