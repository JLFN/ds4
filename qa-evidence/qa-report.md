ds4 QA report - H4b split-K Bonsai (qwen35) attention key range, CUDA path

Date: 2026-09-29
QA model: deepseek-v4.1-CC-flash. That is the slug this session runs (read from
  the session's summary.json) and the project's recorded QA model in the handoff
  frontmatter. Every check below was executed live by this run on this host;
  nothing is taken from the commit message.
Branch under test: feature/h4b-attn-splitk, HEAD a6c7bdd "perf(cuda): split the
  Bonsai attention key range across blocks", forked from dev 5a654a5.
Base ref: dev 5a654a5 "merge: Bonsai (qwen35) decode GEMV dispatch and
  fold-launch collapse into dev" (origin is the read-only upstream, main;
  origin/dev does not exist).
State tested: working tree clean (git status --porcelain -uall prints nothing;
  the only new files here are gitignored build outputs). Diff 5a654a5..HEAD is
  ds4.c +52 and tests/test_qwen35_cuda.cu +187/-13, two files, +226/-13.
  ds4_qwen4_cuda.cuh is unchanged by this commit: git diff 5a654a5..a6c7bdd --
  ds4_qwen4_cuda.cuh is empty, and its last change is fffeb19.
  sha256 ds4.c = 81440a9f5178882141d086e5a6618cb6ef1445d5fac8cff1a51d482635c1c462
  sha256 tests/test_qwen35_cuda.cu = 1c0583ded51a8bfeb4d95921b4776d8a92df9f1573c2067427737a2befebd103
This QA wrote no source, test, Makefile, script or git change in /data/ds4, did
  not commit and did not push. The only repo file it writes is this report. The
  control binary was built in a git worktree at /tmp/qa-h4b-base (detached at
  5a654a5); every run log is under /tmp.

Binaries as tested
  Under test (built by this QA):
    cd /data/ds4 && make cuda-generic CUDA_ARCH=native   -> EXIT=0, log
      /tmp/qa-build-h4b.log, zero warning/error lines
    cd /data/ds4 && make tests/test_qwen35_cuda CUDA_ARCH=native -> EXIT=0
    md5 ds4 74c4acdfb777288012f30c19e9bf79b2 (48636000 bytes, 17:52)
    md5 ds4-bench 66228bf4cd8390c977b8999b74acfbec (48377504 bytes, 17:52)
    md5 tests/test_qwen35_cuda 8f4bb15a7cfd25c016296d276d81c6dc (18:12)
    make -q ds4 ds4-bench ds4-server CUDA_ARCH=native exits 0 (targets current),
    so every run below exercised this tree.
  Control (dev 5a654a5):
    cd /tmp/qa-h4b-base && make cuda-generic CUDA_ARCH=native -> EXIT=0, log
      /tmp/qa-build-base.log, zero warnings
    md5 /tmp/qa-h4b-base/ds4-bench ef011f410d761aebf9f918ac1abf466d
  Static proof the change is in the emitted code: in the under-test ds4-bench
  the qwen35 attention call site (qwen35_graph_layer) pushes 0xb0(%rbx) as the
  dispatcher's partial argument, a load from the graph struct, while the base
  binary at the same site pushes $0x0. The qwen4 call sites push $0x0 (cT > 2)
  or the qwen4 graph's own partial in both binaries, unchanged.

Host
  RTX 4070 SUPER 12 GB (12282 MiB), sm_89, CUDA 13.3, driver 610.43.03,
  nsys 2026.1.3. The host is shared: nvidia-smi was read before every GPU run.
  One foreign compute process ran at 18:11-18:16 (another session's
  ./attn_repro_fixed micro-benchmark, 266 MiB); no timed run of this QA
  overlapped it (the interleaved pairs started 18:21 with the GPU empty and no
  foreign tenant appeared during them).

Surfaces covered
  ALL NEW SURFACES
  The gate's surface list is empty for this diff (it adds no new function
  definition in a non-test C/CU file: ds4.c only modifies existing functions
  and adds one struct field and one #define; the new test helper is in the
  test file, which the gate scans separately), so this blanket token is used
  per rule 19. The things actually verified are named explicitly here: the
  split-K dispatch path at runtime (nsys), the attn_partial allocation bound
  and every caller that can reach the dispatcher, the compare/session identity
  against the CPU reference, the attention-split test assertions, the
  interleaved base-vs-under-test performance pairs, and the qwen4 family's
  isolation from the change.

1. What was run (exact commands)

  Builds
    cd /data/ds4 && make cuda-generic CUDA_ARCH=native
    cd /data/ds4 && make tests/test_qwen35_cuda CUDA_ARCH=native
    cd /tmp/qa-h4b-base && make cuda-generic CUDA_ARCH=native
    cd /tmp/qa-h4b-base && make tests/test_qwen35_cuda CUDA_ARCH=native
    cd /data/ds4 && make pq2-0-test && make bonsai-fold-selftest

  Correctness
    cd /data/ds4 && ./run-bonsai.sh compare
    cd /data/ds4 && ./run-bonsai.sh session
    cd /data/ds4 && PROMPT="$(cat /tmp/bonsai-prompt-128.txt)" && \
        ./run-bonsai.sh compare "$PROMPT"          (logs /tmp/qa-compare-128.log)
    cd /data/ds4 && make test-qwen35-cuda CUDA_ARCH=native          (3 runs)
    /tmp/qa-h4b-base/tests/test_qwen35_cuda                         (3 runs)

  Runtime path evidence (nsys), each one model process at a time on an idle GPU
    nsys profile --trace=cuda --sample=none --cpuctxsw=none \
      -o /tmp/qa-nsys-h4b-32768 ./ds4-bench -m <model> --cuda \
      --prompt-file /tmp/bonsai-long-prompt.txt --ctx-start 32768 \
      --ctx-max 32768 --step-incr 32768 --gen-tokens 8
    nsys profile ... -o /tmp/qa-nsys-h4b-24 ... --ctx-start 24 --ctx-max 24 \
      --step-incr 24 --gen-tokens 8
    nsys profile ... -o /tmp/qa-nsys-base-2048  (base binary, ctx 2048 shape)
    nsys profile ... -o /tmp/qa-nsys-h4b-128p-b ./ds4 -m <model> --cuda \
      --first-token-test --raw -p "$(cat /tmp/bonsai-prompt-128.txt)"
    analysis:
      nsys stats --report cuda_gpu_kern_sum --force-export=true <rep>
      python3 /tmp/qa-nsys-analyze.py <rep>.sqlite attn_merge "attention<"
        (reads CUPTI_ACTIVITY_KIND_KERNEL: instance count and launch grid)

  Performance (interleaved pairs; script /tmp/qa-perf.sh, log
  /tmp/qa-perf-summary.log, raw CSV per run /tmp/qa-perf/<frontier>-<v>-<r>.out)
    for frontier F in 2048 8192 32768, three rounds, pair order base then h4b:
      /tmp/qa-h4b-base/ds4-bench -m <model> --cuda \
        --prompt-file /tmp/bonsai-long-prompt.txt --ctx-start F --ctx-max F \
        --step-incr F --gen-tokens 8
      /data/ds4/ds4-bench <same args>
    The script refuses to start a run when a foreign compute process is on the
    GPU and logs the wall time with the ds4-bench CSV row.
    GPU memory was sampled every 10 s during the whole run (/tmp/qa-vram.log).

2. What was observed

2.1 The split path is really taken at ctx 32768 (under test, nsys)

  cuda_gpu_kern_sum over /tmp/qa-nsys-h4b-32768.nsys-rep (frontier 32768,
  32768 prefill tokens + 8 decode tokens, log /tmp/qa-nsys-h4b-32768.out):
    attention<256>      32,896 instances, 84.53 s, 75.1 percent of kernel time
    attn_merge          32,864 instances,  1.41 s,  1.2 percent
    mul_mat_q (PQ2_0)   25,600 instances, 15.26 s, 13.6 percent
    gdn_scan             3,072 instances,  4.10 s
  attn_merge launches only when the dispatcher runs splits > 1 (it is called
  once after the attention kernel when splits > 1 and never with a NULL
  partial), so 32,864 attn_merge instances is direct runtime proof that the
  graph's non-NULL attn_partial reached the dispatcher and the split path ran.

  Launch grids from the trace sqlite:
    attention<256> grid=(6,16,64) x30752   prefill row batches, splits = 64
                   grid=(6,1,64)  x128     the 8 decode tokens x 16 layers
                   grid=(6,16,k)  x32 each, k = 1,2,3,... (early chunks)
    attn_merge     grid=(24,16,1) x32736, grid=(24,1,1) x128
  grid.z is the dispatcher's splits = min(64, (keys+31)/32): 64 at 32768 keys.
  The counts close exactly: 64 chunks x 16 attention layers x 32 row batches of
  16 = 32768 prefill calls, plus 8 x 16 = 128 decode calls = 32896 attention
  instances; the only one-split calls are the first chunk's first two row
  batches (keys 16 and 32, both <= 32), 32 of them, and 32896 - 32 = 32864
  attn_merge instances. Bonsai geometry confirmed from the GGUF metadata:
  64 blocks, full_attention_interval 4 (so 16 attention layers), head_count 24,
  head_count_kv 4, key/value length 256 - the same H/Hkv/D the buffer is sized
  from.

2.2 The short-context path is not the split path (under test, nsys)

  ctx 24 (24 prefill tokens + 8 decode, so every call has keys <= 32), log
  /tmp/qa-nsys-h4b-24.out:
    attention<256> 160 instances, grids (6,1,1) x128, (6,8,1) x16, (6,16,1)
      x16 - grid.z = 1 everywhere, one split.
    attn_merge: ZERO instances.
  So a short context runs the single-split kernel exactly as before and the
  buffer is allocated and passed but never written. The strict boundary is 32
  keys, not 64: at ctx 64 the last prefill batches (keys 48 and 64) already
  take 2 splits by the dispatcher's formula, which is why the split-free case
  was measured at ctx 24.

2.3 The control never takes the split path (base 5a654a5, nsys)

  ctx 2048, base binary, log /tmp/qa-nsys-base-2048.out:
    attention<256> 2,176 instances, all grid.z = 1 ((6,16,1) x2048 prefill,
      (6,1,1) x128 decode); attn_merge ZERO instances.
  splits is 1 whenever partial is NULL and nothing in that expression depends
  on the context, so the base ran the 32768-key serial walk as well.

2.4 The attn_partial bound cannot be overrun (code read, plus the dispatcher's
    own check)

  ds4.c qwen35_graph_open allocates
    attn_rows = min(cap_tokens, DS4_QWEN35_ATTN_ROWS = 16)
    g->attn_partial = attn_rows * DS4_N_HEAD(24) * DS4_QWEN35_ATTN_SPLITS(64)
                      * (DS4_N_HEAD_DIM(256) + 2) * 4
    = 25,362,432 bytes for cap_tokens >= 16 (the number the commit quotes).
  The only non-test caller that passes this buffer is qwen35_graph_attention
  (grep over the tree: ds4.c:69039 for qwen35; ds4.c:58911/58937 are the qwen4
  family; the rest are tests). It walks r0 in steps of 16 and passes
  rt = min(T - r0, 16) with T <= g->cap_tokens (qwen35_graph_forward rejects
  T > cap_tokens and is the only way in), so rt <= min(cap_tokens,16) =
  attn_rows always.
  The dispatcher (ds4_qwen4_cuda.cuh, untouched here) computes
  splits = min(64, (keys+31)/32) <= 64 and, before launching, does
  if (partial && !tensor(partial, T*H*splits*(D+2)*4)) return 0; the kernel's
  largest write is record ((T-1)*H + H-1)*splits + splits-1, floats 0..D+1,
  i.e. exactly T*H*splits*(D+2) floats. A caller that passed a bigger T than
  the buffer was sized for would get a clean failure, not an overrun; with this
  call site the check passes by construction. The qwen4 family's own partial
  (cT <= 2, its own graph-sized buffer) is unchanged by this commit.

2.5 Correctness against the CPU reference (all on the split path)

  ./run-bonsai.sh compare (default prompt "The capital of France is", 24 ids):
    IDENTICAL: all 24 generated token ids agree.
    /tmp/qa-ask-default-{cpu,cuda}.tokens, md5 a3e566aa910b83369ac007d208ab34cd
    both, 24 lines each.
  ./run-bonsai.sh session (DS4_QWEN35_SESSION=1, create/sync/eval, 24 ids):
    IDENTICAL. /tmp/bonsai-session.tokens md5 a3e566aa910b83369ac007d208ab34cd,
    the same oracle file as above.
  ./run-bonsai.sh compare "$(cat /tmp/bonsai-prompt-128.txt)" (139-token
  story prompt for this tokenizer, 24 greedy ids):
    IDENTICAL. /tmp/qa-ask-128-{cpu,cuda}.tokens md5
    38250280314c47e67304a7c4f57ffa24 both. Log /tmp/qa-compare-128.log
    (CUDA wall 5.66 s; CPU reference 587.02 s).
    That run's scenario was profiled separately (/tmp/qa-nsys-h4b-128p-b, the
    same prompt and entry point): attention<256> 2,464 calls with grid.z =
    1 x512, 2 x512, 3 x512, 4 x512, 5 x416 and attn_merge 1,952 = 2,464 - 512.
    The 512 single-split calls are the first 32 token steps (16 layers x 32),
    and the split counts follow keys 1..32, 33..64, 65..96, 97..128, 129..154
    exactly, i.e. the run the identity was proven on took the split path with
    up to 5 ranges per row (the commit's "splits <= 5").

2.6 test-qwen35-cuda attention-split assertions

  make test-qwen35-cuda CUDA_ARCH=native was run three times (logs
  /tmp/qa-test-qwen35-run1.log, -run2, -run3). Every run printed all eight
  "attention split:" assertion lines as PASS, including:
    T=1  ctx 2048 : split vs single max|d|=1.23e-07 (limit 1.14e-05) PASS;
                    split vs oracle 2.90e-08, single 9.97e-08
    T=1  ctx 32768: split vs single 1.54e-06 (limit 3.07e-06) PASS;
                    split vs oracle 1.00e-08, single 1.19e-06
    T=16 ctx 2048 : split vs single 3.13e-07 PASS; split 3.56e-08, single 1.84e-07
    T=16 ctx 32768: split vs single 2.16e-06 (limit 2.84e-06) PASS;
                    split vs oracle 1.19e-08, single 2.01e-06
  These reproduce the commit's claimed oracle distances (1.00e-08 / 1.19e-08
  split against 1.19e-06 / 2.01e-06 single) and its conclusion that the split
  reduction order is the more accurate one at 32K keys.
  Pre-existing failure, NOT this unit: every under-test run also prints
  "random1 MMQ (M=17408 K=5120 N=1): rel_l2=1 ... FAIL" and its MMVQ twin
  (all-zero output), which makes the binary exit 1 even though every
  attention-split assertion passes. The same failure reproduces on the BASE
  build: /tmp/qa-h4b-base/tests/test_qwen35_cuda run 1 PASS, runs 2 and 3 FAIL
  with the same rel_l2=1 line. It is intermittent and independent of this
  commit, matching the recorded open item. No other line fails in any of the
  six logs (checked with a FAIL grep).
  CPU-side: make pq2-0-test PASS (6 reference blocks, 34 bytes/block, 2.125
  bpw), make bonsai-fold-selftest PASS (/tmp/qa-cpu-checks.log).

2.7 Performance, interleaved pairs (base 5a654a5 vs under test)

  Each row is one process: one-shot prefill to the frontier with 8 greedy
  tokens, same prompt file both sides, base and under test alternating within
  each round. CSV columns are ds4-bench's
  ctx,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,....

  frontier 2048
    base prefill t/s  727.68 / 693.47 / 720.17   median 720.17
    h4b  prefill t/s  942.35 / 979.11 / 956.76   median 956.76   (1.33x)
    base decode  t/s   17.81 /  17.94 /  18.14   median  17.94
    h4b  decode  t/s   35.85 /  37.00 /  33.91   median  35.85   (2.00x)
    base first token ms 51.511 / 51.528 / 56.728 median 51.528
    h4b  first token ms 26.965 / 24.342 / 29.667 median 26.965   (1.91x)
    wall base 4.7/4.7/4.6 s, h4b 3.8/3.8/3.7 s
  frontier 8192
    base prefill 312.41 / 322.13 / 351.00   median 322.13
    h4b  prefill 683.41 / 677.22 / 712.90   median 683.41           (2.12x)
    base decode    6.46 / 6.79 / 7.06       median   6.79
    h4b  decode   33.86 / 35.04 / 36.66     median  35.04           (5.16x)
    base first ms 147.326 / 148.200 / 140.095 median 147.326
    h4b  first ms  28.071 /  29.324 /  26.639 median  28.071         (5.25x)
    wall base 28.8/28.0/25.8 s, h4b 13.6/13.7/13.1 s
  frontier 32768
    base prefill 69.03 / 68.53 / 66.08      median 68.53
    h4b  prefill 289.31 / 282.21 / 280.42   median 282.21           (4.12x)
    base decode    2.04 / 1.90 / 2.03       median   2.03
    h4b  decode   29.29 / 29.24 / 27.70     median  29.24           (14.4x)
    base first ms 490.721 / 527.497 / 494.578 median 494.578
    h4b  first ms  32.150 /  34.190 /  33.339 median  33.339         (14.8x)
    wall base 480.1/483.8/501.2 s, h4b 115.0/117.8/167.9 s (median 4.1x)

  The first-token claim (490 -> 36 ms at ctx 32768) reproduces: base median
  494.6 ms vs 33.3 ms here. The decode claim (2.03 -> 27.37 t/s) reproduces and
  is slightly better here: 2.03 -> 29.24 t/s. The prefill figure here is a
  one-shot prefill of the whole 32768-token frontier (prefill_tokens = 32768),
  not the last-16384-token window the commit quoted for its 44.29 -> 210.54, so
  those two numbers are not directly comparable; the direction and the
  magnitude of the win hold at every frontier. Every pair moved in the same
  direction, so no round was an outlier that changed a conclusion.

2.8 GPU memory

  Sampled every 10 s during the whole perf run (/tmp/qa-vram.log): the 8192
  runs peaked at 8375 MiB (base) and 8401 MiB (split-K); the 32768 runs stayed
  between 9921 and 9990 MiB for both variants (max sample 9990 MiB, during a
  split-K run). The card's 12282 MiB was never approached and no run failed to
  fit. The 25,362,432-byte buffer is smaller than the sampler's run-to-run
  spread, so its exact contribution could not be isolated this way (see 3).

2.9 The qwen4 family is untouched by this change

  Structural: ds4_qwen4_cuda.cuh (the dispatcher and both kernels) is
  byte-identical to the base; the only ds4.c hunks are inside qwen35_graph_*
  plus the struct field and #define. In the emitted binaries the qwen4 call
  sites of ds4_gpu_qwen4_attn_decode_tensor are the same instruction sequence
  in base and under test (push $0x0 / push the qwen4 graph's own attn_part).
  Live qwen4 model runs were not possible in this tree, see 3.

3. What could not be verified

  - The commit's absolute measurement numbers come from 2026-09-24 on
    different run conditions; the medians in 2.7 are this QA's own. Where they
    differ, the direction and size of the win reproduce. In particular the
    44.29 -> 210.54 t/s prefill figure is for the last-16384-token prefill
    window of the 32768 frontier, while this QA measured one-shot full-frontier
    prefill (68.53 -> 282.21 t/s), so that specific pair was not replicated.
  - The commit's VRAM claim (10160 -> 10184 MiB peak at ctx 32768) was not
    reproduced exactly: this QA's sampled peak was 9990 MiB, and the expected
    ~25 MB buffer delta is below the 10 s sampler's resolution. The substance
    (peak stays around 10 GB, far from 12282 MiB) holds.
  - The full-sweep wall-time claim (628 s -> 168 s) was not re-run as a full
    sweep; only the single-frontier pairs above were measured (their walls at
    32768 are 480-501 s vs 115-168 s).
  - The qwen4 regression test cannot be built in this tree on CUDA at either
    side: make tests/test_qwen4_cuda fails with an undefined reference to
    ds4_gpu_qwen4_attn_decode_rows_tensor, which is implemented only in
    ds4_metal.m. Reproduced identically in the base worktree, so it is
    pre-existing and not caused by this unit, but it means the cross-family
    check is structural (2.9), not a live qwen4 run.
  - The llama.cpp comparison quoted in the commit body is context from the
    handoff; it was not re-measured here.
  - One nsys capture artefact worth recording: the first 16-generation-step
    diagnostic profile showed about 9 percent fewer kernel instances than the
    run's 154 forwards imply; a repeat capture of the same command matched the
    expected counts exactly (2,464 attention calls = 154 x 16, and fold_rotate
    76,692 = 154 x 498), so the shortfall was an event-capture artefact of that
    first profile, not model behaviour. Only the repeat's numbers are quoted.

4. Verdict

  The unit passes on every check this QA could run live: the split-K dispatch
  path is proven at runtime and absent at short context and in the control, the
  partial buffer cannot be overrun by any reachable caller, compare and session
  are token-identical to the CPU reference on both the default and the 128/139
  token prompt, all attention-split assertions pass in three runs, the only
  failing assertions are the pre-existing intermittent random1 matmul case that
  fails on the base too, and the interleaved pairs show large, consistent wins
  (2.00x decode and 1.91x first-token at ctx 2048, 5.16x / 5.25x at 8192,
  14.4x / 14.8x at 32768) with no regression anywhere.

verdict: overall PASS
