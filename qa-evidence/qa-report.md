ds4 QA report - H4a decode GEMV dispatch + fold permute launch collapse, Bonsai (qwen35) CUDA path

Date: 2026-09-24
QA model: deepseek-v4.1-CC-flash, acting as the rule 19 AI QA-tester.
  SUBSTITUTION NOTE: the project's previously recorded QA model for this repo is
  deepseek-v4.1-flash. That model is unavailable for this run: its gateway
  answers HTTP 402 (reported by the run brief; this QA did not re-probe the
  gateway, so the 402 itself is accepted as given, not re-evidenced here). The
  QA model for this run is therefore deepseek-v4.1-CC-flash, the substituting
  model named by the run brief. Every check below was executed by this model in
  this run.
Branch under test: feature/h4a-decode-gemv-and-fold-launch, forked from dev
  54b9c37 ("merge: Bonsai (qwen35) chunked prefill into dev"). The unit is
  UNCOMMITTED: HEAD == dev == 54b9c37, git log dev..HEAD is empty, so the whole
  unit is the working-tree delta measured here.
Base ref: dev (local dev == 54b9c37). There is no origin/dev - origin is the
  read-only upstream antirez/ds4 - so dev is the base, as tests/qa-gate.sh
  defaults (QA_BASE=dev).
State tested: git status --short =
    M ds4_cuda.cu
    M ds4_qwen35_cuda.cuh
  No untracked files (git status --porcelain -uall shows the same two lines).
  git diff --stat: ds4_cuda.cu +13, ds4_qwen35_cuda.cuh 7 insertions/6
  deletions, total 20 insertions, 6 deletions, 2 files.
  sha256 of the whole git diff (worktree vs HEAD), unchanged at start and at
  the end of this QA:
    aa9ea72d49a10e5a6326a81db42c4686374fff7dc4145fd7dae6bc2037dd44e8
  Worktree file hashes:
    ds4_cuda.cu        2421dfb6a97427eaf88baeb360dd0dab6f9f1755f7056c9dab63440e39986b60
    ds4_qwen35_cuda.cuh ba35b1429fea6d60cb674b78e9fdc4b56e4705cc9d480787364db6a9d47774ef
  This QA wrote no source, test, Makefile, script or git change in /data/ds4.
  The only repo file it writes is this report; every run log lives in /tmp under
  the names quoted below. The one clone it made for the A/B control is in
  /tmp/qa-base and never touches /data/ds4/.git.

Binaries as tested: ds4 2026-09-24 19:16:07, ds4-bench 19:16:13 - both newer
  than the changed sources (ds4_cuda.cu 19:12:32, ds4_qwen35_cuda.cuh 19:12:27)
  and newer than ds4_cuda.o (19:13:58). make -q ds4 ds4-bench CUDA_ARCH=native
  exits 0 (targets up to date), so every run below exercised the worktree code.
  Disassembly confirms the dispatch is compiled in: in /data/ds4/ds4 and
  /data/ds4/ds4-bench the dense wrapper has exactly one call to
  ds4_mmq_pq2_0_dense_vec (0x315330 in ds4) and one to ds4_mmq_pq2_0_dense
  (0x314580); the control binary built from the base ref (see the tile-versus-
  vec A/B below) has zero calls to the vec entry and one to the tile entry.

Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (qwen35 family, PQ2_0
  ternary), resident model copy 6.70-6.71 GiB on the GPU.
GPU: RTX 4070 SUPER 12 GB, shared. nvidia-smi was read before and after every
  run. No compute tenant appeared at any point: nvidia-smi
  --query-compute-apps was empty at every check, and the only occupants were
  Xorg (166 MiB), cinnamon (59 MiB) and two browser clients (85+61 MiB), i.e.
  a 250-425 MiB baseline. Machine load average during the runs was 1.92-4.00 on
  28 cores, so this is a loaded host; the perf section says which numbers are
  interleaved A/Bs for that reason. No ds4 stray was ever left: pgrep for
  /data/ds4/ds4 and qa-base/ds4 was empty after every step, and free VRAM
  returned to baseline (257 MiB used / 11638 MiB free at the start of the
  memory check; 267 MiB used after a full decode and a full prefill; 425 MiB at
  the end, all of it the display clients named above).

Tooling note (rule 12 tool-first gate): repo-rag-mcp has no ds4 project
  (.rag-index absent, confirmed with repo-rag-mcp list_projects); callgraph-mcp
  has no ds4 entry (callgraph_list_projects shows none and
  .callgraph-index.bin is absent); graphify-rs-out/graph.json exists but is
  dated 2026-09-18 21:52, i.e. older than HEAD 54b9c37 (2026-09-24 19:07), so it
  is stale for this unit and was not used. The questions here are live-run
  behaviour, launch counts and changed function bodies, so direct source
  reading, git grep and the live runs are the authority.

SURFACES COVERED (blanket token: ALL NEW SURFACES; the unit adds no new
function definitions - it changes two existing ones, so the covered surfaces
are named individually as well):
  1. ds4_cuda.cu, cuda_matmul_mmq_dense_quant, case 142u - the dense PQ2_0
     dispatch now sends n_tok <= 8 to ds4_mmq_pq2_0_dense_vec.
  2. ds4_qwen35_cuda.cuh, qwen35_cuda::fold_gdn_permute - row indexing moved to
     blockIdx.y.
  3. ds4_qwen35_cuda.cuh, ds4_qwen35_fold_launch - one launch for the whole
     chunk instead of one per row.

================================================================================
CHECK 1 - claim 1 is present and the vec entry was genuinely dead before it
================================================================================
Command: git grep -n 'ds4_mmq_pq2_0_dense_vec' HEAD -- .
Observed at HEAD 54b9c37, the vec entry appears only as
  cuda/mmq/ds4_mmq.cu:4993 (its definition),
  cuda/mmq/ds4_mmq.h:183 (its declaration),
  tests/test_qwen35_cuda.cu:361 (the kernel unit test, guarded by n_tok == 1).
No production caller exists at HEAD. The change's comment "the vec entry was
previously never called from anywhere" is therefore true for production code;
the precise form is "called only by the kernel unit test, at N = 1".
Command: git diff (case 142u hunk)
Observed: the wrapper's 142 case is now
  if (n_tok <= 8u) { ds4_mmq_pq2_0_dense_vec(...); break; }
  ds4_mmq_pq2_0_dense(...);
with the same arguments as the tile call, so the only behavioural difference is
which kernel runs.

Reachability of the vec entry with N > 8 or K % 256 != 0 (adversarial check):
Command: read cuda/mmq/ds4_mmq.cu:3109-3140 (ds4_mmq_dense_vec_impl)
Observed: the shared vec impl rejects K % 256 != 0 (lines 3125-3128) and
N > MMVQ_MAX_BATCH_SIZE (lines 3131-3134) with return -1. MMVQ_MAX_BATCH_SIZE
is 8 (cuda/mmq/mmvq.cuh:4). A violation would therefore fail loudly through the
wrapper's rc != 0 path ("ds4: CUDA dense PQ2_0 MMQ failed"), never silently.
Because the wrapper gates on n_tok <= 8u, the cap is a safety net that this
caller cannot trip. GGML_TYPE_PQ2_0 = 142 (cuda/mmq/ds4_ggml_stubs.h:173) and
DS4_TENSOR_PQ2_0 = 142 (ds4.c:2449); the token-embedding row lookups use the
separate rows entry (ds4.c:5982, ds4.c:67724), not this wrapper.

K multiples checked against the real file, not only the claim:
Command: python3 /tmp/qa-gguf-dims.py
  /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf   (a QA scratch script under
  /tmp; it reads the GGUF header only)
Observed: file has 851 tensors, 402 of them type 142 (PQ2_0). Shapes (rows x
cols, as stored): (1024,5120) x32, (5120,6144) x64, (5120,17408) x64,
(6144,5120) x48, (10240,5120) x48, (12288,5120) x16, (17408,5120) x128,
(248320,5120) x2. The reduction width (K, i.e. dim[0] in ds4's mapping) is
therefore 5120, 6144 or 17408 in every dense PQ2_0 matmul of this model, and
the script's check "PQ2_0 tensors whose K is not a multiple of 256" returned an
empty list. The claim's K list said "5120/6144/10240/12288/17408"; in the file
10240 and 12288 are output widths (the second dim), not K, but every PQ2_0
dimension in the file is a multiple of 256 either way, so the precondition
holds.

Blast radius: read ds4.c:882-902 (the shape table comment) and the shape table
itself; the PQ2_0-bearing family is DS4_SHAPE_QWEN35 only (48 gated delta-net
layers, 16 attention layers, dense FFN). Every other family's dense weights are
Q8_0/Q4_0/F16/F32/BF16/MXFP4/IQ2_XXS/Q2_K/Q4_K, which take their own cases in
cuda_matmul_mmq_dense_quant, so the new branch cannot be reached by another
family's model. Confirmed by the model file too: its only tensor types are
0 (F32), 30 (BF16) and 142 (PQ2_0).

================================================================================
CHECK 2 - claim 2: the permute change is element-identical and collapses the
launch count
================================================================================
Command: git diff (ds4_qwen35_cuda.cuh hunk) and read of the file
Observed: old code looped t = 0..n_tok-1 launching
  fold_gdn_permute<<<(n+255)/256, 256>>>(tmp + t*n, x + t*n, n, hd, nk, rep)
per row; new code launches once as dim3((n+255)/256, n_tok) and the kernel adds
  const uint64_t off = (uint64_t) blockIdx.y * n;
  dst[off + i] = src[off + gdn_src(i, hd, nk, rep)];
For row y and element i the store is dst[y*n + i] = src[y*n + gdn_src(i)] -
byte-for-byte the same element map, the same kernel body and the same
gdn_src. The sibling kernel fold_rotate in the same file already indexed rows
with blockIdx.y, so the pair is now consistent rather than divergent. With
n_tok = 1 the new grid is (ceil(n/256), 1), i.e. the old single launch.

Direct launch-count measurement (nsys, identical command, base binary vs
worktree binary):
Command: nsys profile --force-overwrite=true -o <out> --stats=false env
  DS4_QWEN35_PREFILL_CHUNK=512 <bin> --cuda -m <model> --prompt-file
  /tmp/h3-bench-prompt.txt --gen-tokens 0 --ctx-start 512 --ctx-max 512
  then: nsys stats --force-export=true --report cuda_gpu_kern_sum <rep>
Observed, base (54b9c37) binary:
  qwen35_cuda::fold_gdn_permute   24,576 instances, 24.05 ms total, avg 978.6 ns
  qwen35_cuda::fold_rotate           498 instances, 53.64 ms total, avg 107.7 us
Observed, worktree binary, same command:
  qwen35_cuda::fold_gdn_permute       48 instances,  1.37 ms total, avg 28.6 us
  qwen35_cuda::fold_rotate           498 instances, 51.57 ms total, avg 103.6 us
This is the exact claim: 24,576 -> 48 permute launches for one 512-token chunk
(48 gated delta-net layers x 512 rows before, one per layer now), while the
rotation work is untouched (498 instances in both runs, same size). The brief's
arithmetic "48 layers x 512 rows = 24,576" is confirmed by measurement, and the
per-instance time (~1 us) matches the profile note that drew the unit's
attention.

================================================================================
CHECK 3 - correctness: the CPU-reference gate and the tile-versus-vec A/B
================================================================================
The three required runs, exact commands and observed results:
1) DS4_QWEN35_PREFILL_CHUNK=2 ./run-bonsai.sh session
   Observed: CUDA session 2.07 s wall; CPU reference 95.87 s;
   "IDENTICAL: all 24 generated token ids agree".
2) ./run-bonsai.sh session
   Observed: CUDA session 2.05 s; CPU reference 96.07 s;
   "IDENTICAL: all 24 generated token ids agree".
3) ./run-bonsai.sh compare
   Observed: CUDA graph 2.08 s (11.54 tokens/s including model load); CPU
   reference 96.52 s; "IDENTICAL: all 24 generated token ids agree".
Cross-check of the token streams themselves (not only the script's verdict):
  sha256sum /tmp/bonsai-cpu.tokens /tmp/bonsai-cuda.tokens /tmp/bonsai-session.tokens
  -> 812c3b894021b7d8434bf91fe0bb43074471eb585b2b0c640c685f69f3d542a8 for all
  three files; diff cpu-vs-cuda and cpu-vs-session report no difference.
  Continuation text in all three: "Paris. / The capital of Germany is Berlin. /
  The capital of Italy is Rome. / The capital of Spain is".

TILE-VERSUS-VEC A/B (the check the brief feared could not be constructed):
there is indeed no env knob that forces the tile path (searched
ds4_cuda.cu/mmq for a PQ2_0 dispatch switch - none exists), so instead of
editing source I built the control from the base ref in a throwaway clone:
Command: git clone -q --shared /data/ds4 /tmp/qa-base &&
  cd /tmp/qa-base && git checkout -q 54b9c37 &&
  make ds4 ds4-bench CUDA_ARCH=native -j28
Observed: clean tree at 54b9c37, build exit 0. Proof that this binary really is
the pre-change dispatch (not just an older timestamp):
  nm /tmp/qa-base/ds4 -> ds4_mmq_pq2_0_dense 0x314580, _vec 0x315330
  objdump -d /tmp/qa-base/ds4     | grep -c 'call.*315330' -> 0
  objdump -d /tmp/qa-base/ds4     | grep -c 'call.*314580' -> 1
  objdump -d /data/ds4/ds4        | grep -c 'call.*315330' -> 1
  objdump -d /data/ds4/ds4        | grep -c 'call.*314580' -> 1
  (same result for ds4-bench: base 0/1, worktree 1/1)
Greedy streams from both binaries on two prompts, at the default chunk (tile
path for the prompt) and at chunk 6 (vec path for the prompt) - all four cell
comparisons:
  "The capital of France is" (24 greedy ids, DS4_QWEN35_STEPS=24):
    chunk 512: base == new, base == CPU oracle, new == CPU oracle
    chunk 6:   base == new, base == CPU oracle, new == CPU oracle
  "The capital of Germany is Berlin. The capital of France is Paris. The
   capital of Italy is Rome. The capital of" (41 prompt tokens; 16 greedy ids at
  the default step count): chunk 512 and chunk 6 both base == new, and both
  equal the CPU oracle.
This is a genuine tile-versus-vec A/B on the model's own output: with the 41-
token prompt at the default chunk, the prompt is one tile-path chunk and every
greedy step is a vec-path matmul, and the base binary (tile everywhere) and the
worktree binary (tile prompt + vec decode) produce the identical token stream,
which also equals the CPU oracle.

Small-N boundary and both-paths-used evidence:
Command: DS4_QWEN35_SESSION=1 DS4_QWEN35_STEPS=24 ./ds4 ... with
  DS4_QWEN35_PREFILL_CHUNK in 1..8, diffed against the CPU oracle stream
Observed: chunks 1,2,3,4,5,6,7,8 all print the chunk the kernel actually got
("Bonsai prefill chunk: N tokens") and all are IDENTICAL to the CPU oracle on
24 ids. For the 5-token prompt chunks 1-5 exercise n_tok 1-5; a 41-token prompt
with chunks 6,7,8 (16 ids each) exercises N=6,7,8 identically. So the vec route
is proven at every N in 2..8, not only at 1.
Command: nsys on the worktree binary, chunk 512 prefill and an 8-token decode
Observed: the chunk-512 prefill capture contains 400 instances of
mul_mat_q<(ggml_type)142,(int)128> (the tile path for chunk rows) plus 1
instance of mul_mat_vec_q<(ggml_type)142,(int)1> (the last-row logits
projection); the decode capture contains 3,209 instances of
mul_mat_vec_q<142,1> alongside 400 tile instances from the prompt. Both paths
are live in one process, and the n_tok <= 8 gate is the only selector.

The project's own kernel test, rebuilt against the changed sources:
Command: cp /data/ds4/ds4_cuda.cu /data/ds4/ds4_qwen35_cuda.cuh /tmp/qa-base/ &&
  cd /tmp/qa-base && make tests/test_qwen35_cuda CUDA_ARCH=native -j8 &&
  ./tests/test_qwen35_cuda
Observed: build exit 0; every case PASS, including all fold cases with n_tok
2,4,5 and the gdn-reorder case: "fold gdn reorder + forward: max_abs=0 (at 0)
failures=0/24576: PASS" (n=6144, n_tok=4; the test's gdn branch also runs for
n=5120 rows=5 and n=17408 rows=2, both PASS), "fold forward/inverse round trip
n=5120 rows=5", "n=6144 rows=4", "n=17408 rows=2" all PASS, and "PQ2_0 CUDA
parity: PASS" (MMQ rel_l2 0.0037-0.0041 and MMVQ rel_l2 0.0029-0.0041 against
the model's own CPU reference, tolerance 0.05; exact-activation MMQ max_abs
5.7e-06 to 2.5e-05 with 0 failures). The gdn case runs only where
hd*nk*rep == n, i.e. for the three folded widths, and it does run at n_tok > 1,
which is exactly the geometry this unit changed. The in-tree tests/test_qwen35_cuda binary is dated 2026-09-18 and predates
this unit, which is why the test was rebuilt in the throwaway clone instead of
trusting that binary.

================================================================================
CHECK 4 - measured perf, base binary versus worktree binary, interleaved
================================================================================
All numbers below are from the two binaries described in CHECK 3, run on the
same card with the same model and prompt, interleaved base/new/base/new so
drift in machine load affects both sides. Exact command shape:
  <bin> --cuda -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
        --prompt-file /tmp/h3-bench-prompt.txt --gen-tokens N --ctx-start C
        --ctx-max C
(the frontier sweep adds --gen-tokens 0 --step-incr 512; the chunk sweep adds
DS4_QWEN35_PREFILL_CHUNK=<T>).

Decode ctx 64, 128 greedy tokens, steady-state gen_steady_tps (nine runs each,
from /tmp/qa-ab-perf.log, /tmp/qa-repeat.log and /tmp/qa-dec4.log):
  base: 25.80, 25.87, 25.89, 25.92, 25.94, 25.94, 26.02, 27.37, 27.48
  new:  36.31, 36.44, 36.45, 36.70, 36.74, 37.95, 38.20, 39.05, 39.05
  first token over the same runs: base 35.2-40.3 ms, new 24.4-26.3 ms.
  The brief's 27.3 -> 37.4 t/s is the same effect; my base side reads a little
  lower (25.9-27.5) and the new side sits at 36.3-39.1.
Decode ctx 2048, 128 greedy tokens:
  steady: base 14.40, 15.66, 15.22  ->  new 17.77, 18.83, 17.99 t/s
  first token: base 73.1, 62.3, 62.3 ms  ->  new 56.6, 53.3, 51.6 ms
  The brief's 15.43 -> 17.99 t/s and 66.1 -> 58.2 ms are reproduced within noise.
Prefill frontiers, chunk 512, prefill_tps, 3 interleaved repeats each:
  512:  base 817.12 / 839.14 / 835.51   new 885.08 / 880.07 / 885.25
  1024: base 851.58 / 848.04 / 851.99   new 897.03 / 891.48 / 889.20
  1536: base 710.52 / 688.46 / 708.47   new 741.18 / 739.76 / 741.10
  2048: base 606.21 / 605.77 / 610.24   new 632.31 / 631.69 / 632.51
Chunk sweep at ctx 512, prefill_tps, 3 interleaved repeats each:
  T=32:  base 474.70 / 486.76 / 486.57  new 489.36 / 490.52 / 491.56
  T=128: base 768.41 / 776.71 / 771.89  new 798.90 / 813.19 / 809.61
  T=512: base 835.91 / 818.17 / 813.96  new 865.70 / 872.82 / 851.37
Interpretation: the decode win is large, unambiguous and reproducible; the
prefill win is real in every interleaved pair but modest (about 1-8 percent
here) and must be read against the loaded host (load average 1.92-4.00) and the
shared card. The brief's absolute frontier and chunk-sweep figures are close to
my base side but its new-side figures are higher than mine at some points (for
example its T=128 878.9 and T=512 928.9 versus my 798.9-813.2 and 851.4-872.8);
single non-interleaved runs I took earlier did reach 843.9 and 917.7, so the
difference is run-to-run variance on this loaded machine, not a different code
path.

The unit's own code comment, audited against measurement (nsys on the BASE
binary, 8 greedy tokens at ctx 64):
  observed: mul_mat_q<(ggml_type)142,(int)8> 3,209 instances, 62.5 percent of
  captured GPU kernel time, avg 68.2 us (median 58.3); plus
  mul_mat_q<(ggml_type)142,(int)64> 400 instances (12.8 percent).
  3209 / 8 = 401.1 instances per decode token -> the comment's "401 calls per
  token" is exact, and its "66 us per call" matches my 68.2 us average.
  Its "42.5 percent of decode GPU time" is NOT what I measure in this capture
  (62.5 percent, with the capture also containing the 64-token prefill and the
  model load); the exact share depends on the capture window, and the
  actionable part of the comment (401 calls/token at ~66 us) is confirmed.
  The same capture on the worktree binary: 3,209 instances of
  mul_mat_vec_q<142,1> at 53.2 percent (avg 44.8 us) replacing them, which is
  where the decode speedup comes from.

================================================================================
CHECK 5 - compute-sanitizer
================================================================================
Command: compute-sanitizer --tool memcheck --print-limit 20 ./ds4-bench --cuda
  -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --prompt-file
  /tmp/h3-bench-prompt.txt --gen-tokens 8 --ctx-start 64 --ctx-max 64
Observed: "========= ERROR SUMMARY: 0 errors". This run exercises the vec path:
the prompt is prefill and all 8 greedy steps route n_tok=1 through
ds4_mmq_pq2_0_dense_vec. Exit status 0.
Command: DS4_QWEN35_PREFILL_CHUNK=512 compute-sanitizer --tool memcheck
  --print-limit 20 ./ds4-bench --cuda -m <model> --prompt-file
  /tmp/h3-bench-prompt.txt --gen-tokens 0 --ctx-start 512 --ctx-max 512
Observed: "========= ERROR SUMMARY: 0 errors", log line "Bonsai prefill chunk:
512 tokens (ctx 513)". This run exercises the tile path at N=512 plus the
collapsed fold launches. Exit status 0.

================================================================================
CHECK 6 - memory, strays, and repo hygiene
================================================================================
Commands: nvidia-smi --query-gpu=memory.used,memory.free before and after each
run, nvidia-smi --query-compute-apps after each run, pgrep -a -f
'/data/ds4/ds4|qa-base/ds4' after each run, git status --short, git diff --stat.
Observed: 257 MiB used / 11638 MiB free before the decode run; 267 MiB after
the 128-token decode at ctx 64; 267 MiB after the 2048-token prefill; the
resident 6.70 GiB model copy is released when the process exits, so there is no
leak across runs. No compute app and no ds4 process remained at any check. The
final reading of 425 MiB used is Xorg (166) + cinnamon (59) + two browser
clients (85+61), all type G display clients, not compute. git status --short is
still exactly " M ds4_cuda.cu" and " M ds4_qwen35_cuda.cuh", no untracked
files, and the diff sha256 is unchanged
(aa9ea72d49a10e5a6326a81db42c4686374fff7dc4145fd7dae6bc2037dd44e8), so this
QA changed no tracked state.
Command: QA_MODEL=deepseek-v4.1-CC-flash bash tests/qa-gate.sh
Observed: "QA GATE: ALL PASS (no commits ahead of dev yet; nothing new to
QA-tester)", exit 0. Expected: the unit is uncommitted, so HEAD == dev and the
gate has no committed diff to enumerate. This report will be the gate's
evidence once the unit is committed.

================================================================================
COULD NOT VERIFY / ACCEPTED RISKS
================================================================================
1. The brief's absolute perf figures were not reproduced exactly. My base side
   reads slower on decode ctx 64 (25.9-27.5 versus the claimed 27.3) and both
   sides differ on the prefill frontiers and chunk sweep by up to about 6
   percent in either direction. The host was loaded (load average 1.92-4.00)
   and the card is shared; no other compute tenant was present during my runs.
   The direction and rough size of every claimed delta is confirmed by the
   interleaved A/B in CHECK 4; the exact numbers in the brief are not, and the
   numbers in this report are mine.
2. The code comment's "42.5 percent of decode GPU time" is not reproduced by my
   nsys capture (I measure 62.5 percent for the same kernel in a capture that
   also contains a 64-token prefill). Declared as a discrepancy rather than a
   failure: the same measurement confirms the comment's 401 calls per token and
   its ~66 us per call.
3. The prefill speedup is small (about 1-8 percent in my interleaved pairs) and
   is partly below the noise floor of a loaded shared host. I would not quote
   my own single-run prefill numbers as a regression or improvement signal; the
   decode side is the robust part of the unit.
4. The kernel unit test rebuilt for this QA linked
   ds4_cuda_test_hooks.o from the worktree build tree (object mtime 18:54),
   which predates the last ds4.c touch (19:07). ds4.c is not part of this
   unit's diff and the hooks only expose the reference Hadamard/dequant
   functions this unit does not change, so the oracle is unaffected; but the
   object was not rebuilt from scratch in my run.
5. tests/run.sh as a whole was not run: the model-backed CPU side takes about
   96 seconds per oracle run and the suite drives several. I ran the pieces
   that cover the changed files (the rebuilt kernel test, the session paths,
   compare, the chunk sweep) plus the gate script itself; the aggregate suite
   result is the developer's evidence, not mine.
6. The fold_rotate instance count (498 per 512-token chunk) is measured as
   unchanged between the two binaries but I did not decompose it against the
   source's fold call sites; only fold_gdn_permute's 24,576 -> 48 change is
   load-bearing for this unit and that one is exact.

================================================================================
RESIDUAL RISKS
================================================================================
1. The vec route is now on the decode hot path for every PQ2_0 dense matmul,
   including any future model whose K is not a multiple of 256. The wrapper's
   gate is only n_tok <= 8; the K % 256 precondition is enforced inside the vec
   impl (which returns -1, a loud failure), so a violating model would fail at
   load time, not produce wrong numbers - but the correctness of the vec kernel
   at other K values rests on that guard, not on a live run here. All K values
   in the shipped Bonsai file are 5120, 6144 and 17408 and were parsed from the
   file.
2. The tile-versus-vec A/B proves agreement on two prompts and two chunk
   regimes, not on every tensor of the trunk; the strongest general evidence is
   still the 24-id CPU-oracle agreement plus the rebuilt kernel parity test.
   The two kernels accumulate in different orders, so last-bit differences
   exist by construction (the unit test records MMVQ exact-activation max_abs
   up to 1.1e-3 against a reference that is itself fp32); greedy argmax was
   stable in every comparison I ran, but a near-tie sampled by temperature
   could in principle flip. Not observed, not ruled out.
3. The fold collapse changes nothing per element, but it does change the launch
   geometry from N one-block grids to one N-row grid. At n_tok = 1 the grid is
   identical, so decode behaviour is untouched; the session runs at chunk 1
   through 8 that I made all pass, which covers that boundary.
4. This unit is uncommitted on a branch with no commits ahead of dev, so the
   evidence above describes the working tree and nothing else. If the delta is
   edited or split before commit, this report's diff hash
   (aa9ea72d49a10e5a6326a81db42c4686374fff7dc4145fd7dae6bc2037dd44e8) no
   longer applies.

verdict: overall PASS
