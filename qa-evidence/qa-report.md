ds4 QA report - CUDA resident model weights (unit: cuda_model_residency_fits)

Date: 2026-09-18
QA model: deepseek-v4.1-flash, acting as the rule 19 AI QA-tester
This is the RE-VERIFICATION after the guard fix. Round 1 tested the original
  guard "if (g_model_device_owned || ...) return 1;" and reported the risk
  that it returned success for ANY later map call once the primary model was
  device-owned, skipping the MTP/support map registration at ds4.c:73433.
  The guard has since been scoped to the primary image and the binaries were
  rebuilt (make cuda-generic, clean). Everything below that is marked as
  re-run was re-executed against the final code; the one item not re-run
  (the full tests/run.sh) is explicitly annotated with why it still stands.
Branch under test: feature/resident-weights
Base ref diffed against: dev (dev and HEAD are both
  b6cf33eac2cef55d7d07a1b50bcccd5d074ed4e3, so the unit is UNCOMMITTED and the
  working tree is the artifact).
State tested: git status --short: " M ds4_cuda.cu" and " M
  qa-evidence/qa-report.md" (this report) and nothing else. Code diff (git
  diff -- ds4_cuda.cu, 50 insertions) sha256:
  7ffcb4def5fca4511297c9432c9ef1d0250dd1165fd5f78b52a836dc001b04a5
  Round 1 code diff (original guard) sha256:
  b72ed417c57756da04db63c985f7d88a458ff5826319b97b58e832c74ee67324
  The QA wrote no source, Makefile or script change and ran no git commit,
  push or checkout. The only repo file it writes is this report; driver
  output lives in /tmp logs named where used.

Binaries as tested: ds4, ds4-bench and ds4-server all mtime 21:51, newer than
  ds4_cuda.cu (21:47) and ds4_cuda.o (21:48). make -q ds4, ds4-bench and
  ds4-server each exit 0. The resident run logs show the new build.
Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (arch qwen35, PQ2_0 ternary).
GPU: RTX 4070 SUPER 12 GB, shared. nvidia-smi was read before and after every
  run. During the middle of this QA the other session's llama-server held
  9158 MiB (pid 3893835, /data/llama-prisma-ml/.../llama-server, the same
  Bonsai GGUF); the QA did not touch that process, and the runs in that
  window either declined residency by design (see CHECK 6) or used the light
  mapped path. When the tenant exited, free windows (used 273 MiB) were used
  for the resident checks. No two QA GPU processes ran at once; no ds4 strays
  remained at any point (pgrep -a ds4 empty after every step).

Tooling note (rule 12 tool-first gate): repo-rag-mcp has no ds4 index and
  callgraph-mcp has no ds4 project (only ds4-on-spark, a different root);
  graphify-rs-out/graph.json exists but predates the edit, so it was stale
  for this unit. The questions here are live run behaviour and literal log
  strings plus the changed function bodies, so the live runs, direct source
  reading and grep are the authority.

SURFACES COVERED: ALL NEW SURFACES of this unit - the changed entry points
ds4_gpu_set_model_map_range and cuda_model_copy_chunked, and the new static
helper cuda_model_residency_fits (static, verified through its caller).


THE THREE CASES OF THE FINAL GUARD (code reading, ds4_cuda.cu:3925-3943)

Final guard, verbatim:
  if (g_model_device_owned) {
      if (model_map == g_model_host_base) return 1;
  } else if (cuda_model_residency_fits(model_size) &&
             cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
      return 1;
  }

Case 1 - nothing resident yet (g_model_device_owned == 0):
  The else-if runs. If cuda_model_residency_fits returns 0 (model_size == 0,
  g_n_gpus > 1, g_ssd_streaming_mode, any of the four env switches, free
  memory below the reserve/fit arithmetic, or cudaMemGetInfo failure), the
  else-if is false and control falls through to
  ds4_gpu_register_model_map_no_copy - the pre-unit mapped path.
  If it returns 1, cuda_model_copy_chunked runs; from a clean state (also
  g_model_registered == 0) it cudaMallocs, chunk-copies, sets
  g_model_host_base / g_model_registered_size / g_model_device_base and
  g_model_device_owned = 1, and the caller returns 1 (resident). On an
  allocation or copy error the helper frees its own buffer and returns 0, and
  the caller falls through to the mapped registration - graceful, no abort.
  Sub-case 1b (owned == 0 but g_model_registered == 1, i.e. a previous map
  was host-registered): cuda_model_copy_chunked hits its own
  "if (g_model_device_owned || g_model_registered) return 1;" (line 2545)
  and returns 1 WITHOUT copying and without setting g_model_host_base, so the
  caller reports the new map as mapped although it is neither resident nor
  registered. Pre-unit this call reached register_no_copy and registered it.
  Not reachable on CUDA today (needs a third map call; see residual risks).

Case 2 - the same map again (owned == 1 and model_map == g_model_host_base):
  returns 1 idempotently; no second copy, no re-registration. The check is
  pointer-only, so a same-pointer/different-size call would also be treated
  as already done; no current caller does that.

Case 3 - a different map while the primary is resident (owned == 1 and
model_map != g_model_host_base):
  The outer if is taken, the inner is false, so control falls out of the
  if/else chain to ds4_gpu_register_model_map_no_copy, exactly the path a
  second map took before this unit. cuda_register_model_map
  (ds4_cuda.cu:4021) then: releases the per-range/q8 caches; frees the primary
  resident copy (cudaFree(g_model_device_base), g_model_device_owned = 0,
  lines 4037-4039); unregisters the previous host registration; points
  g_model_host_base at the new map; cudaHostRegisters it and sets
  g_model_registered = 1. So the MTP/support map at ds4.c:73433 is registered
  again and can never be skipped by an early success return. The primary's
  resident copy does not survive the second registration (the pre-existing
  "last map wins" replacement semantics of cuda_register_model_map, which
  pre-unit had no device copy to free); see residual risks.

ORPHAN CHECK of cuda_model_copy_chunked (task question):
  The function never frees a prior g_model_device_base, so the question is
  whether its cudaMalloc (line 2550) can ever run while a previous malloc'd
  copy exists. It cannot, in the current tree: entry requires
  g_model_device_owned == 0 AND g_model_registered == 0 (line 2545 early
  return), g_model_device_owned is the only tracker for malloc'd copies and
  is set to 1 only immediately after a successful malloc+copy (lines 2617,
  3881), and every path that clears it frees first:
  cuda_register_model_map lines 4037-4039 (cudaFree then owned = 0),
  ds4_gpu_set_model_map lines 3850-3852, the shutdown/reset release lines
  3017-3027 (cudaFree then pointers cleared). The caller guard closes the
  remaining entrance: with owned == 1, case 2 returns and case 3 goes to
  register, so copy_chunked is unreachable while anything is resident.
  PLAINLY: no reachable call order orphans a previous resident copy today.
  The order that would resurrect the leak is a direct call to
  cuda_model_copy_chunked for map B while g_model_device_owned == 1 for map
  A (bypassing both the scoped caller guard and the helper's own early
  return): the malloc would overwrite g_model_device_base and A's copy would
  leak. That is dormant, with both guards in place.


CHECK 1 - Residency engages on this host (re-ran after the fix)

  ./ds4-bench --prompt-file /tmp/e2b-bench-prompt.txt -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
      --cuda --ctx-start 64 --ctx-max 64 --ctx-alloc 256 --gen-tokens 16
GPU before 273 MiB (free window). Log /tmp/qa2r-resident-bench.log:
  ds4: CUDA chunk-copying 6.71 GiB model image
  ds4: CUDA model chunk copy complete in 0.844s (6.70 GiB tensors)
  ds4: CUDA startup model preparation covered 6.70 GiB of tensor spans in 0.000s
  ds4: memory: KV 0.02 GiB ... + resident model 6.70 GiB = 6.72 GiB planned
Exit 0, wall 5 s, peak 7613 MiB sampled every 0.2 s, no "registered ...
  device access" line. Post-fix, the copy still happens and the startup
  preparation still reports the image as handled.
Verdict: PASS.


CHECK 2 - Correctness unchanged (re-ran after the fix)

  ./run-bonsai.sh compare "The capital of France is"
  ./run-bonsai.sh session "The capital of France is"
compare (exit 0, wall 110 s): cuda 2.30 s (resident; the 6.71 GiB chunk copy
  is in its log), cpu 108.13 s, both continuations "Paris.\nThe capital of
  Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is",
  and "IDENTICAL: all 24 generated token ids agree, so the CUDA graph
  reproduces the CPU reference on this prompt".
session (exit 0, wall 116 s): cuda (session) 2.61 s (resident), cpu 113.35 s,
  same continuation, "IDENTICAL: all 24 generated token ids agree, so the
  session path reproduces the CPU reference on this prompt".
Verdict: PASS.


CHECK 3 - Speedup real and reproducible (re-ran after the fix)

Three resident-bench processes, free GPU:
  64,64,27.22,16,27.54,38.480,15,27.73,0
  64,64,27.12,16,27.16,38.096,15,27.30,0
  64,64,27.20,16,27.49,37.258,15,27.61,0
(columns: ctx, prefill_tokens, prefill_tps, gen_tokens, gen_tps, gen_first_ms,
steady_tokens, steady_tps, kvcache_bytes.)
Prefill 27.12-27.22 t/s, decode 27.16-27.54 t/s, first token 37.3-38.5 ms.
Decode is inside the claimed 27-28 window; prefill sits at the top of the
claimed 26-27 band and the first token 1-3 ms above the claimed 35-37 band
(round 1 measured 35.1-38.5 across runs, so this is the same regime; the
shared tenant had been active shortly before, which is the likely cause and
is not proven). Still a 12x decode gain over the mapped path.
Verdict: PASS (exact numbers reported; no claim of better than measured).


CHECK 4 - Old behaviour restorable / env switches (re-ran after the fix)

  DS4_CUDA_NO_MODEL_COPY=1 ./ds4-bench ... (same command as CHECK 1)
Exit 0, wall 35 s, log /tmp/qa2r-nocopy-bench.log:
  ds4: CUDA (no-copy) registered 6.71 GiB model mapping for multi-tier selective cache
  CSV: 64,64,2.30,16,2.28,434.237,15,2.27,0 -> decode 2.27-2.30 t/s, first
  token 434 ms, no chunk-copy line. Matches the 2.2-2.3 t/s requirement.
The other three switches were re-run on the final code with a short bench
  (ctx 8, gen 4), each printing the same no-copy registration line and
  staying on the mapped path: DS4_CUDA_DIRECT_MODEL=1 2.17 t/s / 460 ms,
  DS4_CUDA_WEIGHT_CACHE=1 2.19 t/s / 457 ms, DS4_CUDA_WEIGHT_PRELOAD=1
  2.25 t/s / 446 ms (the shorter context explains the small drift from 2.28).
Verdict: PASS.


CHECK 5 - Suite and gate

The full suite was run in round 1 on the pre-fix guard and is not re-run
  here per the re-verification brief: exit 0, wall 147 s, 0 FAIL lines,
  "tests/run.sh: overall PASS", all six steps OK (pq2-0-test,
  test-qwen35-cuda, bonsai-fold-selftest, bonsai-ref-check,
  test-qwen35-session, qa-gate). It still stands as evidence for the final
  code because between round 1 and round 2 only the guard block inside
  ds4_gpu_set_model_map_range changed (round-1 vs round-2 code diff hashes
  above); every kernel, loader and test target is untouched by that delta,
  and the guard's paths were re-verified live in CHECKs 1-4 and the case
  walk-through above. Any suite statement that depends on the guard is
  therefore covered by the post-fix runs, not by the round-1 suite.
Gate, re-run now as requested against this report:
  QA_MODEL=deepseek-v4.1-flash bash tests/qa-gate.sh
  QA GATE: ALL PASS (no commits ahead of dev yet; nothing new to QA-tester)
  exit status: 0
  (Same expected message as round 1: dev and HEAD are the same commit while
  the unit is uncommitted. Once committed, the surface scan looks for added
  non-static function definitions and this unit's new function is static, so
  it would again report nothing to QA-tester; this report documents the
  three changed entry points regardless.)
Verdict: PASS (gate exit 0; suite evidence annotated).


CHECK 6 - Boundary honesty (updated: the does-not-fit branch is now live-verified)

While another session's llama-server held 9158 MiB, the CHECK 1 command was
  run and DECLINED residency by arithmetic, not by an env switch:
  GPU before 9387 MiB; log /tmp/qa2-resident-bench.log shows
  "ds4: CUDA (no-copy) registered 6.71 GiB model mapping for multi-tier
  selective cache", no chunk-copy line, and NO "allocation skipped" line, so
  the decline came from cuda_model_residency_fits before the malloc;
  CSV 64,64,2.28,16,2.30,431.427,15,2.30,0, wall 36 s, exit 0, peak 9935 MiB.
  The interrupted compare under the same pressure shows the same fallback on
  the graph path: cuda 13.18 s / 1.82 t/s where the resident path takes 2.30 s.
  The fit threshold for a 6.71 GiB image is free >= 1.5 GiB + 6.71*16/15 =
  about 8.66 GiB; observed free was about 2.7 GiB there (declined) versus
  about 11.9 GiB in the free window (engaged). The two live points bracket
  the decision; the exact threshold was not bisected.
Loader failures, re-confirmed as pre-existing (ds4.c is untouched by the
  diff; these fail at weight binding before any device mapping - their logs
  contain no chunk-copy/registration/CUDA-init line):
  Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf -> "ds4: tensor token_embd.weight has
    type 22, expected pq2_0", exit 1 in 1 s.
  Qwen3.8-9B-Q4_K_M.gguf -> "ds4: expected block_count=64 for Prism Bonsai 2
    27B, got 33", exit 1 in 0 s.
  Ternary-Bonsai-2-27B-PTQ1_0.gguf (not listed in the brief) -> 402
    "unsupported GGUF type 143" warnings then "tensor token_embd.weight has
    type 143, expected pq2_0", exit 1, no device mapping.
  --ssd-streaming on the Bonsai GGUF -> family refusal "Bonsai (qwen35) runs
    on the CPU reference ... or on single-GPU CUDA ...; tensor parallelism,
    SSD streaming, DSpark/MTP and steering are not supported", exit 1 in 1 s,
    before any CUDA work.
Not live-exercised, code reading only: n_gpus > 1 (single physical GPU);
  the g_ssd_streaming_mode decline inside the helper (the engine refuses the
  flag earlier); cases 1b/2/3 of the guard (need a second/third map call,
  i.e. an MTP-capable model - none installed, and the Bonsai family refuses
  MTP). --simulate-used-memory cannot help: it is a host-side mmap+mlock
  (ds4_ssd.c:142-201) that does not reduce CUDA free memory.
Verdict: PASS (the boundary story is stronger than round 1: the does-not-fit
  decline is now observed live under real memory pressure).


COULD NOT VERIFY (accepted risks)

- The MTP/support second-map call (ds4.c:73433) live: no MTP-capable model is
  installed and the Bonsai family refuses MTP/DSpark, so cases 2 and 3 are
  code reading, not live runs.
- The n_gpus > 1 decline: single physical GPU on this host.
- The g_ssd_streaming_mode decline inside the helper: the engine refuses
  --ssd-streaming for this family before the CUDA path.
- Non-Bonsai families on CUDA (DeepSeek V4, GLM): no model files installed.
- The 1.5 GiB reserve adequacy is design-asserted; the does-not-fit live point
  is a single bracket, not a bisected threshold.
- The round-1 comment figures (16.8 GB/s PCIe, 504 GB/s device) were not
  re-measured; this QA measures only the on/off delta, memory and the
  boundary behaviour.


RESIDUAL RISKS (updated after the fix)

1. RESOLVED - the flagged defect: the early return is scoped to
   model_map == g_model_host_base, and a different map falls through to
   ds4_gpu_register_model_map_no_copy, so the MTP/support map at ds4.c:73433
   is registered again exactly as before this unit.
2. Case 1b silent success (latent, currently unreachable): if a third
   set_model_map_range call arrives while owned == 0 and registered == 1,
   cuda_model_copy_chunked returns 1 without copying and the caller reports
   success although the new map was neither copied nor registered. On CUDA
   an engine has at most two map calls (primary at ds4.c:73399, MTP at
   73433; vision uses ds4_gpu_set_aux_model_map_range on Linux, the two
   __APPLE__ branches are not compiled here, and the CLI vision dump at
   ds4.c:74174 is a separate single-map process). Latent if a third map call
   is ever added on CUDA.
3. Case 3 frees the primary resident copy when a second map is registered
   (cuda_register_model_map frees before replacing). Consequence: for a
   fitting primary plus an MTP model, the primary reverts to the mapped path
   after the MTP registration. This is the pre-unit "last map wins" behaviour
   and is not a regression, but the residency gain is lost in that
   configuration; the same-map/one-map case that was measured is unaffected.
4. Orphan/double-alloc remains dormant: cuda_model_copy_chunked cannot
   allocate while anything is owned or registered, and all replacement paths
   free first. The order that would revive the leak is a direct
   cuda_model_copy_chunked call for a second map while a first copy is owned,
   bypassing both guards.
5. Timing drift under the shared GPU: the re-run first token (37.3-38.5 ms)
   is slightly above the 35-37 ms brief and round-1 values; the other tenant
   had been active shortly before, which is the likely cause and is not
   proven.
6. No leaks or strays: after every run nvidia-smi returned to the ambient
   level (273-276 MiB in free windows, the tenant's own usage otherwise) and
   pgrep -a ds4 found nothing; the teardown path frees the 6.70 GiB copy.

verdict: overall PASS
