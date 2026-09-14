# Laguna S 2.1 CRACK — CUDA support

This branch (`laguna-crack`) adds CUDA support for the community CRACK
Laguna S 2.1 exports (Hugging Face `dealignai/Laguna-S-2.1-CRACK-GGUF`),
which the upstream `laguna-s2.1` branch rejected because they use a third
quantization layout. The Q4_K_M export is the file verified end to end:

    Laguna-S-2.1-CRACK-Q4_K_M.gguf   72,748,281,728 bytes (67.75 GiB)

Layout the branch accepts (verified from the GGUF header): Q4_K embedding
and dense/routed gate/up, Q8_0 attention and shared experts, routed down
in a per-layer Q4_K/Q6_K mix, Q6_K output head, 1M-context YaRN rope
(factor 128, attn factor 1.485). The Q6_K export (90.66 GiB) uses the
same hybrid with all-Q6_K experts and is also accepted, but only the
Q4_K_M file is verified end to end so far.

## Quick start

Build, then generate:

    make cuda-spark
    ./ds4 --cuda -m ~/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf \
      -c 32768 -p "Write a haiku about mountains." -n 32

The weights (67.75 GiB) take a minute or two to load; then a line like
`ds4: Laguna GPU graph: ctx=32768, KV ... GiB` appears, followed by the
generated text.

`make cuda-spark` targets GB10 / DGX Spark (it omits an explicit
`nvcc -arch`). Use `make cuda-generic` for a native-arch build, or
`make cuda CUDA_ARCH=sm_120` for an explicit arch.

## Steps

### 1. Get the model and build

    ./download_model.sh laguna-crack-q4
    make cuda-spark

The downloader also has `laguna-crack-q6` (91 GiB) and `laguna-crack-q2`
(42 GiB).

### 2. Kernel self-test

    make test-laguna-crack-kernels

Expected: `27/27` checks pass, `0 failures`. This exercises the Q4_K/Q6_K
matmul, embedding and routed-MoE kernels against CPU references on
synthetic weights, including the large-batch cases that cover the
grid-dimension limit for long prefill.

### 3. Generation smoke test

    ./ds4 --cuda -m ~/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf \
      -c 32768 -p "Write a haiku about mountains." -n 32

What to check:
- Read the text, do not just check the exit code. A quant-index
  misalignment shows up as degenerate repetition, not a crash.
- The load prints a line like `ds4: Laguna GPU graph: ctx=..., KV ...
  GiB, scratch ...` — that confirms the graph came up.
- If it fails on memory, drop `-c`. The weights need 67.75 GiB resident;
  context is cheap because 36 of 48 layers keep a 512-token sliding
  window (about 12 GiB of KV at 262144 tokens, about 48 GiB at 1M).

### 4. Serve it (OpenAI-compatible API)

    ./start-laguna-crack-ds4.sh start

Then:

    curl -s http://127.0.0.1:8002/v1/models
    curl -s http://127.0.0.1:8002/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"hello"}]}'

The script logs to `/tmp/laguna-crack-ds4.log`; control it with
`./start-laguna-crack-ds4.sh stop|status`, and dry-run the memory plan
with `./start-laguna-crack-ds4.sh plan`.
Environment knobs: `LAGUNA_CRACK_MODEL`, `LAGUNA_CTX` (default 262144),
`LAGUNA_PORT` (default 8002), `LAGUNA_BUDGET_GIB` (default 115),
`LAGUNA_DFLASH` (optional DFlash draft GGUF).

Notes:
- The engine rejects several flags for Laguna: `--prefill-chunk`,
  `--power` below 100, `--ssd-streaming`, and MTP/DSpark. The only
  flag-level speed lever is DFlash speculation.
- DFlash needs the Q8_0 draft: `./download_model.sh laguna-dflash`
  (laguna-s-2.1-DFlash-Q8_0.gguf, about 1.1 GiB). Pass it as
  `LAGUNA_DFLASH=<path> ./start-laguna-crack-ds4.sh start`.
- Context ceiling: the export declares a 1M context and the engine
  adopts it, but 1M needs about 122 GiB resident (48.07 GiB KV + 67.75
  GiB weights + about 5.9 GiB scratch), so it does not fit a 128 GB
  unified-memory machine. Roughly 786432 is the practical ceiling;
  262144 is the default.

## What is verified

- `make test-laguna-crack-kernels`: 27/27, bit-exact against CPU
  references (dense Q4_K/Q6_K matmul, Q4_K/Q6_K embeddings, three routed
  MoE layouts, three large-batch cases).
- `make q4k-dot-test`: 4/4.
- A full GB10 (sm_121) build links all five binaries.
- End-to-end generation on the real 67.75 GiB file: coherent output at
  about 41 t/s prefill and 22 t/s decode at ctx 32768.
- `--inspect` on the real file exits 0 with the expected tensor types
  (f32 / q8_0 / q4_k / q6_k).

Not verified: the Q6_K export end to end (its kernels are covered), and
DFlash speculation with this file.
