# Laguna S 2.1 CRACK — try it on the DGX Spark

This branch (laguna-crack) adds CUDA support for the community CRACK
Laguna S 2.1 exports (huggingface `dealignai/Laguna-S-2.1-CRACK-GGUF`).
The model file targeted here is:

    /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf   (72,748,281,728 bytes, 67.75 GiB)

Layout the branch accepts (verified from the GGUF header): Q4_K embedding
and dense/routed gate/up, Q8_0 attention and shared experts, routed down
in a per-layer Q4_K/Q6_K mix, Q6_K output head, 1M-context YaRN rope
(factor 128, attn factor 1.485). The Q6_K export (90.66 GiB) uses the
same hybrid with all-Q6_K experts and is also accepted, but only the
Q4_K_M file is verified end to end so far.

## Quick start (one line)

Everything in one paste — builds if needed, then generates a haiku:

    cd ~/ds4-laguna-crack && make cuda-spark && ./ds4 --cuda -m /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf -c 32768 -p "Write a haiku about mountains." -n 32

After the first build, the short form:

    cd ~/ds4-laguna-crack && ./ds4 --cuda -m /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf -c 32768 -p "Write a haiku about mountains." -n 32

The weights (67.75 GiB) take a minute or two to load; then a line like
`ds4: Laguna GPU graph: ctx=32768, KV ... GiB` appears, followed by the
generated text.

## Steps

Run everything on the DGX Spark, from the tree copied to `~/ds4-laguna-crack`.

### 1. Build

    cd ~/ds4-laguna-crack
    make cuda-spark

Expected: `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`, `ds4-eval` are
built. This target omits an explicit `nvcc -arch`, which is the fastest
path on GB10.

### 2. Kernel self-test (optional but quick)

    make test-laguna-crack-kernels

Expected: `18/18` checks pass, `0 failures`. This exercises the new
Q4_K/Q6_K matmul, embedding and routed-MoE kernels against CPU
references on synthetic weights.

### 3. Generation smoke test

    ./ds4 --cuda -m /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf \
      -c 32768 -p "Write a haiku about mountains." -n 32

What to check:
- Read the text, do not just check the exit code. A quant-index
  misalignment shows up as degenerate repetition, not a crash.
- The load prints a line like `ds4: Laguna GPU graph: ctx=..., KV ...
  GiB, scratch ...` — that line confirms the graph came up.
- If it fails on memory, drop `-c` to 8192 first. The weights need
  67.75 GiB resident; context is cheap because 36 of 48 layers keep a
  512-token sliding window (about 12 GiB of KV even at 262144 tokens).

### 4. Serve it (OpenAI-compatible API)

    ./start-laguna-crack-ds4.sh start

Then:

    curl -s http://127.0.0.1:8002/v1/models
    curl -s http://127.0.0.1:8002/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"hello"}]}'

The script logs to `/tmp/laguna-crack-ds4.log`; control it with
`./start-laguna-crack-ds4.sh stop|status`.
Environment knobs: `LAGUNA_CRACK_MODEL`, `LAGUNA_CTX` (default
262144), `LAGUNA_PORT` (default 8002), `LAGUNA_DFLASH` (optional DFlash
draft GGUF).

Notes:
- Do NOT use `~/start-ds4-laguna.sh`: that script carries llama.cpp
  flags and does not apply to the ds4 engine.
- DFlash (optional speculative decoding) needs the Q8_0 draft from
  `./download_model.sh laguna-dflash`; the Myric Q4_K draft file used
  with llama.cpp is not the format this engine expects. It is a speed
  option only — leave it off for the first test.
- The old llama.cpp Laguna server ran on ports 8000/8001; this server
  defaults to 8002 to avoid collisions.

## Getting the tree / updating it

The canonical dev tree is `/data/ds4_clone` on the workstation (branch
`laguna-crack`). To update the copy on this box after new commits, sync
from the workstation with the gitignore filter, which keeps the
workstation's x86-64 build outputs out of the transfer:

    rsync -a --filter=':- .gitignore' --exclude=graphify-rs-out/ \
      /data/ds4_clone/ \
      /run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/ds4-laguna-crack/

The workstation is x86-64 and this box is ARM64: a `./ds4` binary copied
across fails with `cannot execute binary file: Exec format error`. The
filter excludes `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`,
`ds4-eval`, `*.o` and the test binaries, so the tree here stays
source-only; always build on this box. (The first copy of this tree did
carry the workstation binaries; they were removed 2026-09-14.)

The branch is not pushed to the upstream remote (`antirez/ds4` is
read-only for us).

## What is verified and what is not

Verified before this copy was made:
- full sm_121 build (all binaries link) — done in a scratch copy on the
  workstation
- kernel numeric test 18/18 pass, bit-exact against CPU references
- `--inspect` on the real CRACK Q4_K_M file exits 0 with the expected
  tensor type histogram
- the engine passes every validation gate and begins loading on a
  12 GB GPU, stopping only on VRAM

NOT yet verified (this run is the test):
- end-to-end generation quality on real weights
- speed figures for this file on GB10
- the Q6_K export end to end (kernel coverage only)

If the generation smoke test (step 3) produces sane text, record the
numbers from the load line and the generated sample, and compare against
llama.cpp on the same prompt if you want a second opinion.
