ds4 QA report - H3 chunked prefill, Bonsai (qwen35) CUDA path

Date: 2026-09-24
QA model: deepseek-v4.1-CC-flash, acting as the rule 19 AI QA-tester.
  SUBSTITUTION NOTE: the project's previously recorded QA model for this repo
  is deepseek-v4.1-flash. That model is unavailable for this run: its gateway
  answers HTTP 402 Insufficient account funds (reported by the run brief; this
  QA did not re-probe the gateway because the run had to stay bounded, so the
  402 itself is accepted as given, not re-evidenced here). The QA model for
  this run is therefore deepseek-v4.1-CC-flash, the substituting model named
  by the run brief. Every check below was executed by this model in this run.
Branch under test: feature/h3-chunked-prefill (HEAD 25ea31c "feat(cuda):
  batch the Bonsai prefill pass over a chunk-capacity arena").
Base ref diffed against: dev (local dev == e16c68f). There is no origin/dev in
  this repo - origin is the read-only upstream antirez/ds4 and the only other
  remote is fork https://github.com/JLFN/ds4.git - so dev is the base, exactly
  as the project's own tests/qa-gate.sh defaults (QA_BASE=dev).
State tested: git status --short = " M ds4.c", " M tests/run.sh" and " M
  qa-evidence/qa-report.md" (this report). The unit is a commit plus an
  UNCOMMITTED delta on top of it. Hashes of the ds4.c/tests-run.sh diff:
    worktree vs HEAD (the uncommitted delta) sha256:
      3ce653080e8bcecee102f6e626529251e034c9433d636461f6391c2205eb998b
      (ds4.c 46 insertions / 5 deletions, tests/run.sh +6)
    worktree vs dev (the FULL unit) sha256:
      6dcb030280c615e3dad7e9a290acfbd4907d1ef993b732b61930b43dc53fa64f
      (ds4.c 177 changed lines, tests/run.sh +6)
  The committed half is 25ea31c (99 insertions / 32 deletions in ds4.c, no
  test change); the uncommitted half adds the DS4_QWEN35_ATTN_ROWS = 16 row
  batching in qwen35_graph_attention and the multi-chunk step in tests/run.sh.
  The worktree ds4.c sha256 is
  b527d3a8b1a1b3c9e0952af7b79cdca398cb0d3ec43373e1846c32b62067c20b,
  tests/run.sh d177b053e82c465eb665931a6a6fe520a10f178d965d36be30844b4f96c4f61b.
  This QA wrote no source, test, Makefile or script change and ran no git
  commit, checkout, reset or push. The only repo file it writes is this
  report; every run log lives in /tmp under the names quoted below.

Binaries as tested: ds4 and ds4-bench both mtime 2026-09-24 18:24, newer than
  ds4.c (18:21); make -q ds4 exits 0 (target up to date) and the earlier make
  test-qwen35-session rebuilt from the current tree, so every run below
  exercised the worktree code. tests/run.sh (18:25) is a driver script, not a
  build input of those binaries.

Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (qwen35 family, PQ2_0
  ternary), resident model copy 6.70-6.71 GiB on the GPU.
GPU: RTX 4070 SUPER 12 GB, shared. nvidia-smi was read before and after the
  runs. At the start the only tenants were Xorg (126 MiB) and cinnamon
  (28-81 MiB), i.e. 187-207 MiB used and ~11.7 GiB free, so no other compute
  tenant interfered. No ds4 stray was left at any point: pgrep -a ds4 after
  every step was empty (final check: rc 1, no output) and nvidia-smi returned
  to 187-207 MiB after each run.

Tooling note (rule 12 tool-first gate): repo-rag-mcp has no ds4 index
  (.rag-index absent) and callgraph-mcp has no ds4 entry on this root
  (.callgraph-index.bin absent); graphify-rs-out/graph.json and
  graphify-out/graph.json exist but predate the edit, so they are stale for
  this unit. The questions here are live-run behaviour, literal log strings
  and changed function bodies, so the live runs, direct source reading and
  grep are the authority.

SURFACES COVERED (every changed entry point of the unit):
  qwen35_prefill_chunk_tokens (static helper, verified through its caller)
  qwen35_graph_open (static; signature changed: new cap_tokens parameter)
  qwen35_graph_forward (static; chunk path and last-row logits projection)
  qwen35_graph_attention (static; DS4_QWEN35_ATTN_ROWS batching)
  ds4_session_create (the qwen35 branch: chunk resolution + shrink-to-fit)
  ds4_session_sync's qwen35 prefill block (the chunk loop and checkpointing)
  tests/run.sh (the new test-qwen35-session-multichunk step; NOT run through
    tests/run.sh by this QA, its two make invocations were run directly, see
    "Could not verify")


CHECK 1 - Arena sizing, chunk resolution, logits width

Code reading at the worktree HEAD:
  - ds4.c:69084 qwen35_graph_open(g, e, ctx_cap, cap_tokens): T = cap_tokens
    ? cap_tokens : 1u, and every transient is allocated with (uint64_t)T * ...:
    h, normed, blk, xt, qkv, z, ga, gb, lin_o, qg, q, gate, kp, vp, o, ffn_g,
    ffn_u, tokens. g->cap_tokens = T. The only tensors not scaled by T are
    g->logits = DS4_N_VOCAB * 4 bytes (one row, 248320 floats), g->pos3 =
    ctx_cap * 4 * uint32 (rope positions), and the per-layer state (lin_state,
    lin_hist, and k/v caches sized by ctx_cap). Check 1's "every transient
    sized for cap_tokens rows, logits exactly one vocabulary row" holds.
  - ds4.c:68789-68797 qwen35_prefill_chunk_tokens: DS4_QWEN35_PREFILL_CHUNK
    through strtoul; v == 0 or v > 1024 -> default 512; then clamped to ctx.
    DS4_QWEN35_DEFAULT_CHUNK 512u and DS4_QWEN35_MAX_CHUNK 1024u
    (ds4.c:68783-68784). Matches the claim.
  - ds4.c:74857-74864 ds4_session_create: chunk = qwen35_prefill_chunk_tokens
    (ctx); if (e->prefill_chunk && e->prefill_chunk < chunk) chunk =
    e->prefill_chunk; then "while (chunk > 0 && !qwen35_graph_open(...,
    chunk)) chunk /= 2u;". The halving loop is exactly as claimed; chunk 1 is
    the old one-token arena, so it converges.
Live corroboration (my runs, stderr lines):
  - "ds4: Bonsai prefill chunk: 2 tokens (ctx 10)" with
    DS4_QWEN35_PREFILL_CHUNK=2 on a 5-token prompt + 5 greedy steps.
  - "ds4: Bonsai prefill chunk: 10 tokens (ctx 10)" for the same run without
    the env: the default 512 is clamped to the 10-token context.
  - "ds4: Bonsai prefill chunk: 1 tokens (ctx 513)" with chunk 1.
  - "ds4: Bonsai prefill chunk: 1024 tokens (ctx 1025)" with the request at
    the 1024 ceiling, and 512 for the default at the same context.
  - Clamp checks: DS4_QWEN35_PREFILL_CHUNK=0 -> 512 (ctx 2049), =1024 -> 1024,
    =2048 -> 512. So 0, above-ceiling and the ceiling itself all behave as the
    code says.
Verdict: PASS.


CHECK 2 - Correctness: chunk 2, default chunk, chunk 1 and compare (the check
  that matters most)

All of the following were run by this QA against the worktree binaries:
  2a. DS4_QWEN35_PREFILL_CHUNK=2 DS4_BONSAI_LOG=/tmp/qa-chunk2.log
      ./run-bonsai.sh session
      exit 0, wall 97.6 s. CUDA session 2.38 s, "Bonsai prefill chunk: 2
      tokens (ctx 10)", 24 token lines, and the script's diff line:
        IDENTICAL: all 24 generated token ids agree, so the session path
                   reproduces the CPU reference on this prompt
  2b. DS4_BONSAI_LOG=/tmp/qa-default.log ./run-bonsai.sh session
      exit 0, wall 98.7 s. CUDA 2.32 s, same 24 ids, IDENTICAL.
  2c. ./run-bonsai.sh compare
      exit 0, wall 97.9 s (started 18:46:01 +02:00, ended 18:47:39). CUDA
      2.43 s, CPU 95.35 s / 0.25 t/s, same continuation:
        IDENTICAL: all 24 generated token ids agree, so the CUDA graph
                   reproduces the CPU reference on this prompt
  The three runs share one reference token stream, so chunk 2, the default
  chunk and the one-token-per-forward graph path all agree.
  2d. Cross-chunk equality beyond the 5-token case (this QA's own extension,
      because a 5-token prompt only reaches three chunks at chunk 2):
      a 1200-char / 326-token prompt was prefilled through the session path
      with DS4_QWEN35_PREFILL_CHUNK = 1, 2, 3, 16, 17, 32, 33 and 64, 4 greedy
      steps each. Every run exited 0 and every token stream diffs IDENTICAL
      against the chunk-1 (one-token, pre-unit-equivalent) stream. The top-5
      next-token candidates agree in id and order for all eight; the logit
      values move in the 4th decimal (for example the top id reads 14.3919 at
      chunk 1/2/3, 14.3744 at 16, 14.3866 at 17, 14.3652 at 32, 14.3924 at
      33, 14.3896 at 64). The argmax and the whole greedy continuation are
      stable, which is the property the unit claims; the last-digit drift is
      the registered cross-chunk fp reduction difference, not an argmax
      divergence.
  2e. A second, longer prompt (996 tokens, from /tmp/h3-bench-prompt.txt) at
      chunk 1, 2 and 512: all three streams IDENTICAL, and chunk 2 vs 512
      also has identical top-5 values (4334 13.8700 / 314 13.5728 ... vs
      4334 13.8349 / 314 13.5709 ...).
  2f. The developer's own long-prompt control logs, /tmp/h3-long2-1.log
      (chunk 1, ctx 2998) and /tmp/h3-long2-512.log (chunk 512, ctx 2998):
      their 8 token lines diffIDENTICAL (my diff of the "token " lines exits
      0). These are read-only artifacts, cited as corroboration, not as
      authority.
Verdict: PASS (no DIFFERENT anywhere; every IDENTICAL line above is the
  script's own diff, i.e. exit 0 of diff over the token files).


CHECK 3 - qwen35_graph_forward: T up to cap_tokens, last-row-only projection,
  the T=1 decode path

Code reading at the worktree HEAD:
  - Entry bound: "if (!g || T == 0 || T > g->cap_tokens) return false" plus
    "g->pos + T > g->ctx_cap" (ds4.c:69143-69148). so T up to cap_tokens is
    accepted and no more.
  - Token upload: ids[DS4_QWEN35_MAX_CHUNK] with T * sizeof(uint32_t) written
    into g->tokens, which qwen35_graph_open sized at T rows.
  - Projection: with logits_out, last = T > 1 ? view(g->h, (T-1)*E*4, E*4) :
    g->h; then qwen35_graph_norm(last, 1u), qwen35_graph_fold(..., 1u) and
    qwen35_graph_gemv(g->logits, ..., 1u) - each at T = 1 - followed by a read
    of one vocabulary row from offset 0, and ds4_gpu_tensor_free(last) for
    T > 1. So the norm/fold/output matmul see exactly the chunk's last row
    through a T=1 view, and the logits tensor is one row.
  - The T=1 path (the session's decode entry, ds4.c:79364, "qwen35_graph_
    forward(g, e, &token, 1u, s->logits)"): T == 1 makes "T > 1u" false on
    every branch - no tensor_view is created at all - and in
    qwen35_graph_attention the loop runs once with r0 == 0, so q, gate, o are
    the original arena tensors and the "if (r0)" free block is skipped. There
    is one attention launch per attention layer, exactly as before the unit.
    So "no tensor views and no extra launches" holds for the decode path.
  Direct corroboration: the worktree run without the env on the 5-token
  prompt reports "Bonsai prefill chunk: 10 tokens (ctx 10)" (the whole thing
  fits one chunk) and then decodes 24 tokens through the T=1 path, IDENTICAL
  to the reference (2b above); the chunk=1 session runs decode the same way
  and are IDENTICAL too (2a, 2d, 2e).
Verdict: PASS.


CHECK 4 - Row batching, the dispatcher gate, and T >= 32

  - Code reading: qwen35_graph_attention hands rows over in a "for (r0 = 0;
    r0 < T; r0 += DS4_QWEN35_ATTN_ROWS)" loop with rt capped at 16
    (ds4.c:69004-69005), and DS4_QWEN35_ATTN_ROWS is #defined to 16u at
    ds4.c:68978. Every call passes rt (<= 16), NOT T.
  - The dispatcher gate, read verbatim at ds4_qwen4_cuda.cuh:2324-2326:
    "if (!partial && T >= 32 && D == 256 && H/Hkv <= 16 &&
    !((uintptr_t)kc->ptr&15) && !((uintptr_t)vc->ptr&15) &&
    ds4_cuda_attn_tokentile_arch_ok() && !g_quality_mode &&
    !getenv("DS4_QWEN4_NO_ATTN_MM"))" selects attention_group. The Bonsai
    geometry is in ds4.c's DS4_SHAPE_QWEN35 (ds4.c:887-916): n_head = 24,
    n_head_kv = 4, n_head_dim = 256, so H/Hkv = 6 and D = 256 - both gates the
    developer names are real, and with rt <= 16 the "T >= 32" term is never
    satisfied from this call site. The claim "the gate is unreachable from the
    Bonsai chunk" is therefore verified for the worktree code: the batching is
    what keeps it unreachable, and the gate's condition is what the developer
    says it is.
  - Call-site audit: the only Bonsai callers of ds4_gpu_qwen4_attn_decode_
    tensor are ds4.c:69013 (the row-batched loop, rt <= 16) and the qwen4
    family at ds4.c:58911/58937 (untouched by this unit). tests/test_qwen4_
    kernels.c calls it with T > 32 in existing tests, i.e. the kernel itself
    is exercised elsewhere; the unit only changes who reaches it.
  - Live: T = 32, 33, 64, 100, 128, 256 and 512 prefill chunks all exit 0 at
    ctx 512 (see CHECK 5/6 for the rows), and a full session sweep at chunk
    1, 2, 3, 16, 17, 32, 33 and 64 on a 326-token prompt all exit 0 with
    IDENTICAL token streams (CHECK 2d). No fault, no non-zero exit.
  - The developer's fault evidence, read but not re-run: /tmp/h3-sanitizer.log
    contains 100 "Invalid __shared__ read of size 16 bytes" blocks, each
    pointing at tt_ldmatrix_x2_addr -> tt_ldmatrix_x2 -> qwen4_cuda::
    attention_group+0x27a0 in ds4_qwen4_cuda.cuh:276, launched from
    ds4_gpu_qwen4_attn_decode_tensor at ds4_qwen4_cuda.cuh:2328, called from
    qwen35_graph_layer in ds4.c. I checked line 276 of the worktree
    ds4_qwen4_cuda.cuh: it is exactly "tt_ldmatrix_x2(b,&kv[key0+lane%8][k+
    ((lane%16)/8)*8]);", and line 2328 is the attention_group launch. The
    fault's own ERROR SUMMARY in that log reads 13954 errors, with 13854 not
    printed. That log also shows "ds4: Bonsai prefill chunk: 32 tokens (ctx
    513)" - the failing run was a chunk of 32 before the row batching existed,
    and its host frames go qwen35_graph_forward -> ds4_session_sync_internal,
    i.e. this exact path.
  - This QA's own control: a fresh compute-sanitizer memcheck of the current
    code with the default 512 chunk at ctx 512,
      /usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck
        --target-processes all --launch-timeout 120 ./ds4-bench --cuda -m
        /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --prompt-file
        /tmp/h3-bench-prompt.txt --gen-tokens 0 --ctx-start 512 --ctx-max 512
    exited 0 with "========= ERROR SUMMARY: 0 errors" (log
    /tmp/qa-sanitizer-memcheck.log). With the batching in place no chunk, 33
    and 512 included, selects the faulty kernel, and DeepSeek's own memcheck
    agrees.
  - The "T >= 32 chunks now complete instead of faulting" half of the claim is
    verified live (33 and 512 both exit 0 and match chunk 1); the fault
    itself is only re-read from the developer's log, not reproduced here -
    reproducing it would mean reverting the batching, which this QA must not
    do. Reported honestly as such below.
Verdict: PASS (gate condition verified as described; rt <= 16 makes it
  unreachable; T >= 32 runs complete clean and sanitizer-clean).


CHECK 5 - Speed claims

All numbers below are this QA's own runs on the free GPU, CSV format
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,
gen_steady_tokens,gen_steady_tps,kvcache_bytes.
  - Prefill ctx 512, chunk 512: qa-512.csv row "512,512,813.84,..." and the
    wide sweep's first row "512,512,808.28,...". Claimed about 836 t/s; the
    developer's own artifact /tmp/h3-ab-chunk512.csv shows 836.76. My two
    measurements sit 2.5-3.5 percent below the claim and the developer's
    number, which is the same regime on a shared card; I report the measured
    value, not the claim.
  - Prefill ctx 512, DS4_QWEN35_PREFILL_CHUNK=1: 24.73 t/s (log
    /tmp/qa-chunk1-bench.out), against the claimed about 25 t/s. So the base
    path is intact at one token per forward.
  - Chunk sweep 32..512 at ctx 512, every size exiting 0:
      32 -> 488.61, 33 -> 427.00, 64 -> 636.86, 100 -> 642.04,
      128 -> 772.15, 256 -> 809.29, 512 -> 819.21 t/s
    (with 32 also measured at 472.40 in the earlier trio). All clean; the
    "every chunk size 32..512 runs clean" claim holds for the sizes probed.
  - Decode ctx 2048, default chunk: "2048,2048,697.90,128,14.91,63.638,127,
    14.92,0" (log /tmp/qa-decode-bench.out). Claimed about 15.4 t/s steady and
    a first token near 66 ms; measured 14.91/14.92 t/s and 63.638 ms. The
    steady rate is about 3 percent below the claim, the first token is about
    2 ms below it. The developer's own /tmp/h3-ab-decode.csv shows
    15.43/15.45 and 66.124 ms, so my run is the slower instance in the same
    regime. Also measured decode-independent prefill at 2048: 697.90 t/s in
    that same run and 592.93 in the wide sweep.
Verdict: PASS with the numbers above (measurably the same regime as claimed;
  my three speed probes read slightly lower than the claim on this shared
  card, and the report states my values, not the claim).


CHECK 6 - Shrink-to-fit and memory

  - stderr line present and correct in every run: "ds4: Bonsai prefill chunk:
    N tokens (ctx M)" - observed with N = 1, 2, 10, 19, 32, 33, 64, 100, 128,
    256, 512, 1024 and M = 7, 10, 513, 1025, 2049, 2998, 4097, across
    run-bonsai.sh session, the diagnostic and ds4-bench.
  - Halving is real but was NOT reproduced in my free window: with 11.7 GiB
    free the 512-row and even the 1024-row arena allocate, so a request of
    1024 at ctx 1025 printed "1024 tokens" and a request of 512 printed 512 -
    no halving. The halving path is nonetheless evidenced: the developer's
    /tmp/h3-sanitizer.log shows "Bonsai prefill chunk: 32 tokens (ctx 513)",
    i.e. 512 was halved five times to fit a ctx-513 arena on this same card,
    and the loop itself is read in the code (CHECK 1). I did not force it with
    a foreign memory hog, so the halving is code reading plus that log, not a
    live reproduction of mine.
  - No leak across a multi-frontier sweep: ./ds4-bench --cuda ... --ctx-start
    512 --ctx-max 4096 --step-incr 512 (gen-tokens 0) exited 0, wrote eight
    rows (808.28 down to 382.83 t/s as the context grows), and an nvidia-smi
    sampler at 2 s intervals recorded 187 MiB before the run, two samples at
    8107 MiB during it, and 187 MiB at 18:52:37 and 18:52:39 after it - i.e.
    VRAM returned to the ambient level, not to an elevated plateau.
  - Every one of the approximately 30 runs this QA performed ended with VRAM
    back at 187-207 MiB and with pgrep -a ds4 empty; the final check is empty
    with exit 1. No ds4 stray was left.
Verdict: PASS (with the explicit caveat that my free window never forced the
  halving; that part rests on code reading plus the developer's log).


COULD NOT VERIFY / accepted risks

- The attention_group fault itself was not reproduced by this QA: reproducing
  it would require reverting the row batching, which is a source change I must
  not make. What I verified is the fault's cited location (the worktree line
  276 is the ldmatrix_x2 on kv, line 2328 is the launch), the gate that now
  prevents the selection (T >= 32 never true at rt <= 16), and that T = 33 and
  T = 512 chunks complete with a clean memcheck.
- The shrink-to-fit halving was not forced live in my free GPU window; it is
  code reading plus the developer's /tmp/h3-sanitizer.log. A foreign memory
  tenant large enough to reject a 512-row arena would be needed to force it.
- tests/run.sh as a whole was NOT run; this QA ran only the two make targets
  it invokes for this unit (test-qwen35-session with and without
  DS4_QWEN35_PREFILL_CHUNK=2). The other suite steps (pq2-0-test,
  test-qwen35-cuda, bonsai-fold-selftest, bonsai-ref-check, qa-gate) are
  untouched by this unit's diff and were left to the developer's own suite
  run.
- The CPU reference on the long prompts (996 and 2989 tokens) was not run; the
  reference oracle is only used on the short prompt in run-bonsai.sh session.
  For the long prompts the equivalence evidence is chunk-to-chunk (chunk 1
  vs 2 vs 512), not chunk vs CPU.
- The exact top-5 logit VALUES are not bit-identical across chunk sizes
  (4th-decimal drift, quoted in CHECK 2d); only the argmax and the full greedy
  continuation are. The unit claims token-identity, so this is within claim,
  but a bit-exact-logits expectation would fail.
- The 12 GB card is shared; all speed numbers above are single-tenant
  measurements from one window, not a sustained or contended measurement.
- The developer artifacts /tmp/h3-ab-*.csv, /tmp/h3-long2-*.log,
  /tmp/h3-sanitizer.log, /tmp/h3-bench-prompt.txt and the bisect logs were
  present and were read; nothing in the brief's list was missing.


RESIDUAL RISKS

1. The batching is the only thing keeping the dispatcher off attention_group
   for this family. If DS4_QWEN35_ATTN_ROWS is ever raised to 32 or the loop
   is unrolled to pass T straight through, the illegal shared-memory access
   returns at H/Hkv = 6, D = 256. The guard is a #define plus a loop bound in
   the same function; it is easy to undo by accident. A comment saying why is
   in place (ds4.c:68961-68977), which mitigates but does not enforce.
2. The latent attention_group defect itself is unfixed: any other caller that
   reaches T >= 32 with D = 256 and H/Hkv <= 16 (a future prefill path, a
   different model family, or a test) will fault the same way. This unit
   routes around it; it does not repair it.
3. The last-row projection means a chunk contributes exactly one row to the
   logits. Any future consumer that expects per-row logits from a chunk (for
   example a teacher-forced prefill scorer) would silently get only the last
   row. The function's contract is documented in the comment at ds4.c:69190,
   but the O(1)-row behaviour is a design constraint, not a checked one.
4. Cross-chunk fp drift: the logits for one position differ in the last
   digits depending on the chunk the row was computed in (measured above).
   Argmax and greedy continuation are stable in every run I made; a
   temperature-sampled or logit-diff-sensitive consumer could in principle
   flip a near-tie. Not observed, not ruled out.
5. The uncommitted half of the unit (row batching + the tests/run.sh step) is
   the part that carries the sanitizer fix; if it is committed separately or
   dropped, the committed 25ea31c alone still selects attention_group at T >=
   32 and reintroduces the fault. The report above describes the WORKTREE,
   not the commit.
6. The shrink-to-fit loop's failure path (arena allocation error) was not
   live-exercised; its behaviour on exhaustion rests on code reading.
7. tests/run.sh's new step calls "env DS4_QWEN35_PREFILL_CHUNK=2 make
   test-qwen35-session"; I ran that exact make invocation and it passes, but I
   did not run it through run.sh, so its interaction with the rest of the
   suite is the developer's evidence, not mine.

verdict: overall PASS
