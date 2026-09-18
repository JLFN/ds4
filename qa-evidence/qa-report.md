AI QA-TESTER REPORT - ds4 / Prism Bonsai (qwen35, PQ2_0) on CUDA
QA model: deepseek-v4.1-flash
Date: 2026-09-18 (first pass), 2026-09-18 follow-up to verify the fix for
finding 5; both passes used the same QA model.
Repo: /data/ds4, branch feature/qwen35-pq2-0, HEAD fffeb19
Verified tree: HEAD fffeb19. The author then applied a 12-line fix to ds4.c
(the token-id guard described under FIX RE-VERIFICATION below); git diff after
that change shows ds4.c carries only that one hunk and no other file changed,
and the follow-up pass rebuilt ds4 and re-ran the checks listed at the end.
Unit under test: git log --oneline dev..HEAD (6 commits: 11bdbd0, 80b1a5f, d24eeaa, d7face8, 98a3b78, fffeb19)
Model used: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (7206168928 bytes, GGUF arch qwen35)
GPU: NVIDIA GeForce RTX 4070 SUPER, 12282 MiB, sm_89. Checked with
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv before
every CUDA run; no other compute app was ever present, never two model processes.
Constraint honored: no llama.cpp command was run; every acceptance below is
inside ds4, and the oracle is ds4's own CPU reference in ds4.c.

BUILD EVIDENCE

make cuda-generic (CUDA_ARCH=native) completed with exit 0 in 246 s and linked
all five binaries: ds4 (48573880 bytes), ds4-server (49852568), ds4-bench
(48319472), ds4-eval (48661704), ds4-agent (49800448), all timestamped 16:38.
The CUDA object list includes ds4_cuda.o and the vendored mmq objects
(cuda/mmq/ds4_mmq.o, mmvq.o, vecdotq via mmq.cuh), so the pq2_0 additions are
compiled into the deliverable.

FIX RE-VERIFICATION (FOLLOW-UP PASS, 2026-09-18)

What was fixed: finding 5, the missing token-id bound check on the Bonsai CPU
reference diagnostic path. The fix is at the input boundary of
qwen35_first_token_test in ds4.c (lines 69123 to 69136 in the fixed working
tree), immediately after the prompt token list is built and before any
forward runs. In plain words, it loops over every element of seq and, when an
element is below 0 or at least V, prints
"ds4: Bonsai token id %d is outside the vocabulary (0..%u)" with the offending
id and V - 1, frees seq and returns 1. The C condition is
seq[t] below 0 or seq[t] at least (int)V.

Placement check (code read): the loop runs on seq[] after the single if/else
that fills it from DS4_QWEN35_TOKENS (ds4_parse_token_list) or from prompt->v,
so it covers BOTH sources; it also runs before the on_gpu branch, so the CPU
reference and the CUDA graph both pass through it. The CUDA graph keeps its own
check as a second layer (ds4.c:69049), now redundant for this caller.

Rebuild: make ds4 CUDA_ARCH=native reported "make: 'ds4' is up to date" because
ds4.o and the binary had already been rebuilt after the edit (mtimes: ds4.c
16:52:26, ds4.o 16:53:07, ds4 16:53:08). strings -a ds4 contains exactly one
"ds4: Bonsai token id %d is outside the vocabulary (0..%u)", confirming the
running binary carries the fix. The parity-test objects were rebuilt from the
fixed ds4.c by make test-qwen35-cuda (ds4_cuda_test_hooks.o 16:56:35,
tests/test_qwen35_cuda 16:56:36).

Edge case 5 re-run, CPU path, all three ids now refuse with a nonzero exit:

    DS4_QWEN35_TOKENS=248320 DS4_QWEN35_STEPS=1 ./ds4 -m ...gguf --cpu --first-token-test --raw -p "x"
    ds4: Bonsai token id 248320 is outside the vocabulary (0..248319)
    exit=1                      (before the fix: exit 0 and a wrong embedding row)

    DS4_QWEN35_TOKENS=100000000 ... --cpu ...
    ds4: Bonsai token id 100000000 is outside the vocabulary (0..248319)
    exit=1                      (before the fix: SIGSEGV, exit 139)

    DS4_QWEN35_TOKENS=-5 ... --cpu ...
    ds4: Bonsai token id -5 is outside the vocabulary (0..248319)
    exit=1                      (the parser uses atoi, so a negative id is representable and is now caught)

CUDA path, same three ids, same message, exit=1 for each (the boundary guard
now fires before the graph's own check).

No regression on valid input, proved by byte-identical dumps rather than by
inspection: the Run 1 logits command (DS4_QWEN35_TOKENS=760,6511,314,9338,369,11751,
STEPS=6) rerun after the fix produced /tmp/qa_cpu_fixed.bin and
/tmp/qa_cuda_fixed.bin, and cmp reports both files byte-identical to the
pre-fix runs (/tmp/qa_cpu.bin, /tmp/qa_cuda.bin), with the same token streams
(13, 198, 760, 6511, 314, 9564) and the same CPU-vs-CUDA diff (6/6 argmax,
max_abs/ref_max 0.00393, per-row rel_l2 0.00214 to 0.00283). The fix changes
only the acceptance of out-of-range ids, not any computation.

Parity test after the fix: make test-qwen35-cuda rebuilds the hook object from
the fixed ds4.c and passes in full again (51 PASS lines, 0 FAIL, final line
"PQ2_0 CUDA parity: PASS", exit 0). make pq2-0-test and make bonsai-fold-selftest
were also re-run on the fixed tree and still pass.

SURFACES COVERED

FUNCTION ds4_gpu_qwen35_attn_prep_tensor
Covered by: my scratch guard probe (built in /tmp from the tree's objects,
/tmp/qa_edge.cu, /tmp/qa_edge.log) shows the entry runs when pos0+T equals the
cache cap and is refused when pos0+T exceeds it; plus the model-level CUDA run,
where the 16 attention layers (blk.3,7,11,...,63) call it for every one of 37
positions and the resulting logits agree with the fp32 CPU reference 37/37 at
argmax with 0.25% rms error. No isolated bit-exact unit test exists for this
entry (stated under not covered).

FUNCTION ds4_gpu_qwen35_fold_forward_tensor
Covered by: make test-qwen35-cuda, fold forward cases at n=5120 (1 and 5 rows),
6144 x 4, 17408 x 2, 1024 x 1, all max_abs=0 vs the ds4_test_hadamard_fold
oracle (bit-exact), plus the gdn reorder + forward case max_abs=0, plus my
scratch probe (malformed shapes refused, legal shape accepted, refused call is
a no-op on the buffer), plus every folded matmul input in the model-level run.

FUNCTION ds4_gpu_qwen35_fold_inverse_tensor
Covered by: make test-qwen35-cuda, fold inverse cases max_abs=0 at the five
widths, the forward/inverse round trips (max_abs 4.3e-07 to 5.4e-07), and the
explicit normalized Sylvester matrix check at n=1024 (max_abs=5.36e-07). My
probe adds the constant-block identity: forward of an all-ones block gives
[sqrt(1024), 0, ...] with deviation 0 and the inverse returns it to 1 with
deviation 0. This entry runs in every model forward because the GGUF declares
prism.hadamard.inverse_weight_names = token_embd.weight.

FUNCTION ds4_gpu_qwen35_gdn_out_tensor
Covered by: make test-qwen35-cuda test_gdn_out_gates, "gdn out norm, silu
(Bonsai) gate: max_abs=9.53674e-07 failures=0/18432 PASS"; my probe confirms the
head-dim guard (D=16, D=160, D=256 refused, D=64 runs). Runs in all 48 linear
layers of the model-level CUDA run.

FUNCTION ds4_mmq_pq2_0_dense
Covered by: make test-qwen35-cuda, MMQ at all four model shapes in random and
exact-activation modes: random1 rel_l2 0.0029 to 0.0041 (tol 0.05), prefill 8,
64 and 256 rel_l2 0.0037; exact1 MMQ max_abs 5.7e-06 to 2.5e-05, exact64 MMQ
max_abs=9.54e-06 over 1114112 values, zero failures everywhere. This is the
kernel the model graph dispatches for every PQ2_0 matmul at every batch size
(ds4_cuda.cu case 142u), so the logits comparison below is its end-to-end
evidence.

FUNCTION ds4_mmq_pq2_0_dense_vec
Covered by: make test-qwen35-cuda, the MMVQ twin at the four shapes, n_tok=1:
random1 rel_l2 0.0029 to 0.0041, exact1 MMVQ max_abs 3.97e-04 to 1.12e-03 with
failures=0. Not reached by the current graph (all batch sizes go to _dense);
see not covered.

FUNCTION ds4_mmq_pq2_0_rows_f32
Covered by: make test-qwen35-cuda row lookup, bit-exact (got[i] != ref[i]
comparison, zero mismatches) for sampled rows of a 64-row table and for an
8-token multi-row lookup driven by a device token array, plus the host-wiring
checks (single-token embed bit-exact for tokens 0,3,6; 4-token embed_tokens
bit-exact). Also refuses in_dim not a multiple of 128 (shape guard PASS).

FUNCTION ds4_mmq_pq2_0_rows_kernel
Covered by: the same calls above, which launch this kernel inside
ds4_mmq_pq2_0_rows_f32; the bit-exact comparison is against
ds4_test_pq2_0_ref_row, so a wrong code unpack, wrong byte index or wrong
(level - 1) offset in the kernel would show. Its negative-index path (src < 0
writes 0) is not driven by any caller in the test or the graph.

FUNCTION ds4_test_hadamard_fold
Covered by: make test-qwen35-cuda, where this hook is the oracle for the fold
rotate (op 0), forward (op 1), inverse (op 2) and gdn reorder + forward (op 3)
cases, all matching bit-exactly (max_abs=0); it is the reference for every
device fold check above. Code read at ds4.c:71953 confirms it drives
ds4_hadamard_rotate/forward/inverse/gdn_permute themselves.

FUNCTION ds4_test_pq2_0_ref_row
Covered by: make test-qwen35-cuda row lookup and host-wiring checks; a ds4.c
code read (ds4.c:71924) confirms it calls pq2_0_row_f32 directly, so the test
compares CUDA against the loader's own dequantizer, not a transcription.

FUNCTION ds4_test_pq2_0_ref_matvec
Covered by: make test-qwen35-cuda; it is the oracle for every MMQ and MMVQ
comparison at all four shapes, in random and exact-activation modes. Code read
(ds4.c:71932) confirms it is pq2_0_row_f32 plus ref_row_dot, the double
precision row dot.

FUNCTION fold_gdn_permute
Covered by: make test-qwen35-cuda "fold gdn reorder + forward" case, max_abs=0
at n=6144 rows=4 (hd=128, nk=16, rep=3, the model's grouped geometry); the
model-level run applies it in every ssm_out projection because
prism.hadamard.gdn_v_grouped = 1. My probe confirms the entry refuses a gdn
geometry that does not tile the row.

FUNCTION fold_rotate
Covered by: make test-qwen35-cuda fold rotate/forward/inverse/round-trip cases
at five widths (max_abs=0 against the CPU hook, round trips 4.3e-07 to
5.4e-07), the explicit Sylvester matrix check, and my probe's constant-block
property (deviation 0 both ways). It is the butterfly under both fold entries.

FUNCTION gdn_out
Covered by: make test-qwen35-cuda test_gdn_out_gates, which checks BOTH gates
of the shared kernel against a double-precision reference: "gdn out norm,
sigmoid (qwen4exp) gate: max_abs=3.57628e-07 failures=0/18432 PASS" and the
silu (Bonsai) gate "max_abs=9.53674e-07 failures=0/18432 PASS". Code read
confirms the qwen4 call site now passes gate_silu=0u (ds4_qwen4_cuda.cuh:1831)
and the new qwen35 entry passes 1u (ds4_qwen35_cuda.cuh:165). I could not run
make test-qwen4-cuda (pre-existing dead target, see not covered), so the dual
gate check above is the qwen4-behavior evidence for this changed kernel.

END-TO-END EVIDENCE (real measured numbers)

Run 1, prompt DS4_QWEN35_TOKENS=760,6511,314,9338,369,11751,
DS4_QWEN35_STEPS=6, both backends, DS4_QWEN35_LOGITS to /tmp/qa_cpu.bin and
/tmp/qa_cuda.bin:
- Identical greedy streams, both printed: token 6: 13, 7: 198, 8: 760 The,
  9: 6511 capital, 10: 314 of, 11: 9564 Germany.
- Top-5 of the last prompt position: CPU 13(15.4021) 11(13.3783) 198(11.1009)
  271(11.0023) 318(10.7048); CUDA 13(15.4131) 11(13.3801) 198(11.1099)
  271(10.9948) 318(10.6942).
- Logits diff (python over the raw [6][248320] f32 files): argmax agreement
  6/6; global max abs difference 0.0604718 on a reference maximum of 15.4021,
  i.e. 0.39% of the reference maximum (this is the branch's about-0.4% figure,
  and the comparison confirms it); per-row relative L2 error 0.00214 to
  0.00283 (0.21% to 0.28%); rms abs difference 0.0107225 against rms reference
  4.23238 (0.25%); Pearson r = 0.999982. Individual elements near zero carry a
  large per-element relative error (unavoidable with Q8_1 activation
  quantization and irrelevant to the argmax); the worst absolute deviation
  sits at reference value -3.397, not at a decision boundary.
- Wall time: CPU 1m07s, CUDA 0m08s.

Run 2, two raw-text prompts, DS4_QWEN35_STEPS=10 each, both backends
("The capital of France is" and "Once upon a time there was a"):
- diff of the prompt and token lines is empty for both prompts (byte-identical
  streams). Prompt 1: 11751 Paris, 13 ., 198, 760 The, 6511 capital, 314 of,
  9564 Germany, 369 is, 19241 Berlin, 13 .. Prompt 2: 2545 little, 3616 girl,
  852 who, 11815 lived, 303 in, 264 a, 2526 small, 13721 village, 13 .,
  6056 Her.

Run 3, longer context (37-token natural prompt: "The old clockmaker opened the
wooden box ..."), DS4_QWEN35_STEPS=4, logits dumped from both backends:
- Identical greedy streams: 271, 760 The, 6321 letter, 1018 said.
- argmax agreement 37/37. Per-row relative L2 error 0.00217 to 0.00365 over
  the 37 positions (no growth with position beyond noise; correlation between
  relative L2 and row index is 0.47 at this length, but the level stays at the
  same 0.2% to 0.4% band). Worst absolute difference 0.102114 at row 21
  (CUDA 9.0349 vs CPU 8.93278, row reference maximum 15.24).
- Wall time: CPU 4m58s (47m30s user), CUDA 0m34s including the second graph
  the dump path opens.

Follow-up: Run 1 was repeated on the fixed tree and both logits dumps are
byte-identical to the files measured above (cmp clean), with the same streams
and the same CPU-vs-CUDA diff, so every number in this section holds for the
tree with the token-id guard as well.

EDGE CASES TRIED (beyond what the branch already checks)

1. Fold width/block refusals (scratch probe /tmp/qa_edge, all PASS): width 1000
   with block 1024 refused; block 2048 refused (over 1024); block 768 refused
   (not a power of two); block 0 refused; width 0 refused; zero rows refused;
   a gdn geometry 128x15x3 that does not tile the row refused; a signs tensor
   shorter than the width refused; a null tensor refused by the inverse entry;
   and after all refusals the activation buffer is byte-unchanged. Positive
   control: n=2048, 3 rows, block 1024 runs. The loader has a second gate for
   this (ds4.c:7527 ds4_die when the block size does not divide a folded
   weight's input width), so a bad model file cannot reach the kernel either.

2. Constant-block transform (scratch probe): forward with all-ones signs of a
   constant 1 block gives exactly [sqrt(1024), 0, ...] (max deviation 0) and
   the inverse maps that back to 1 (max deviation 0). This is an independent
   normalization cross-check with no ds4 code in the expectation.

3. gdn_out head dim (scratch probe): D=16, D=160 and D=256 all refused with
   return 0 (D=160 and 256 are the tempting ones: multiples of 32 but outside
   32..128); D=64 accepted and runs.

4. Context cap (scratch probe): attn_prep runs when pos0+T equals the cap and
   returns 0 when pos0+T is one past the cap. The graph's own guard
   (qwen35_graph_forward: g->pos + T > g->ctx_cap, ds4.c:69043) prints
   "Bonsai CUDA graph: context of N tokens is full" and fails the forward
   instead of wrapping; it cannot be driven past from outside because the test
   driver sizes the cap to n_seq+steps+1, so the probe above is the live
   evidence for the guard.

5. Out-of-vocabulary token id. FOUND in the first pass, FIXED and RE-VERIFIED
   in the follow-up (full detail under FIX RE-VERIFICATION above).
   - Original defect (first pass, HEAD fffeb19): the CPU reference path had no
     token bound check. DS4_QWEN35_TOKENS=248320 exited 0 and read the row past
     the end of the embedding table (a silent wrong tensor: the table ends at
     file offset 686592352 inside a 7206168928-byte mapping), and
     DS4_QWEN35_TOKENS=100000000 died with SIGSEGV, exit 139.
   - Fixed tree: all three ids (248320, 100000000, -5) refuse on both the CPU
     and the CUDA path with "ds4: Bonsai token id <id> is outside the vocabulary
     (0..248319)" and exit 1; valid ids (760,6511,...) still run and their
     logits are byte-identical to the pre-fix files (cmp clean).
   - Status: verified fixed. The guard sits at the input boundary of
     qwen35_first_token_test and covers both the env-supplied list and the
     prompt tokens.

6. fp16 k/v cache versus the fp32 CPU reference (the risk called out in the
   assignment). Measured at 37 tokens (run 3): 37/37 argmax, per-row relative
   L2 flat in the 0.2% to 0.4% band, worst absolute deviation 0.102 (0.6% of
   the row maximum), no runaway drift. At this length the fp16 cache does not
   dominate the error budget (the Q8_1 activation quantization does). No
   evidence was gathered for thousands of tokens: the CPU oracle needs about
   5 s per token at 28 cores because its attention is quadratic, so a
   1000-token comparison is out of budget for this pass. The 37-token result
   is a floor, not a long-context proof.

ROUTE AUDIT FOR OUT-OF-RANGE TOKEN IDS (follow-up pass)

Every route that can carry a token id into the qwen35 CPU reference or the CUDA
graph was enumerated by reading the call graph; those I could drive live were
driven.

Checked, now guarded:
- DS4_QWEN35_TOKENS env list: the new boundary guard rejects ids outside
  0..248319 (live-verified on both backends, see FIX RE-VERIFICATION).
- Prompt tokens (the path taken when DS4_QWEN35_TOKENS is absent): the same
  loop runs on seq[] after the else branch fills it from prompt->v, so it is
  covered by the same check (code read). The tokenizer produces ids as indices
  of the vocab table built from tokenizer.ggml.tokens (ds4.c:43807), i.e. in
  range by construction; and if a malformed file ever had more tokens than the
  compiled shape, the new guard turns that into a clean refusal instead of an
  out-of-bounds read. This path cannot be made to emit an out-of-range id with
  a valid model, so there is no live negative test for it.
- Greedy argmax inside the test: best is chosen from logits indices 0..V-1 in
  the reporting loop and in the decode loop, so it is always in range; NaN
  logits cannot win a comparison and leave best at 0. The decode loop then
  feeds best straight back into the graph/reference (ds4.c:69167, 69199, 69251),
  which is the only other way seq values reach the forward, and it is in range
  by construction (code read).
- CUDA graph entry: qwen35_graph_forward keeps its own check (tokens[t] < 0 or
  >= DS4_N_VOCAB, ds4.c:69049 to 69052) for any future caller, now redundant
  for the diagnostic. ds4_gpu_embed_token_quant_tensor checks token >= n_vocab
  (ds4_cuda.cu:27712), and the multi-token entry only accepts a device token
  array whose only qwen35 caller is the validating graph.
- Server: ds4_server.c has zero references to qwen35 or Bonsai; the only caller
  of qwen35_first_token_test is the --first-token-test diagnostic
  (ds4.c:69584), so no server request can inject a raw id into this path.
- Other qwen35 diagnostic knobs: DS4_QWEN35_STEPS (a count), DS4_QWEN35_LOGITS
  (a file path) and DS4_QWEN35_FOLD_SELFTEST (returns before the token list is
  built) carry no token ids.

Checked, NOT guarded (out of this unit's scope, stated plainly):
- The qwen4 diagnostics parse DS4_QWEN4_FT_TOKENS (ds4.c:69368) and
  DS4_QWEN4_FT_LIST (ds4.c:69316) through the same unguarded
  ds4_parse_token_list and feed their own CPU reference; that is a different
  model family's pre-existing diagnostic, not a route into the qwen35
  reference, and I did not execute it (a qwen4 model is present at
  /data/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf, but running it is outside this
  pass).
- The device-side row kernel ds4_mmq_pq2_0_rows_kernel bounds only src < 0,
  not src >= n_rows (cuda/mmq/ds4_mmq.cu). It is reached only through entries
  whose qwen35 callers now validate; I did not test an out-of-range row index
  at the kernel level because doing so reads arbitrary device memory by design.

ACCEPTED RISKS AND NOT COVERED

- No isolated bit-exact test for ds4_gpu_qwen35_attn_prep_tensor; its evidence
  is the guard probe plus the model-level agreement over 16 attention layers,
  which is strong but indirect.
- ds4_mmq_pq2_0_dense_vec is exercised only by the isolated kernel test; the
  model graph dispatches all batch sizes, including 1, to
  ds4_mmq_pq2_0_dense. If the vec entry is intended for a decode fast path
  later, it has no end-to-end coverage yet.
- make test-qwen4-cuda does not link in this tree: undefined references to
  ds4_gpu_qwen4_moe_mid_grouped_tensor and the qwen4 rows batch entries, which
  have no CUDA definition (the ds4.c call site is guarded by
  DS4_HAS_QWEN4_METAL, and git grep finds no CUDA definition at dev either, so
  this predates the unit). The qwen4 behavior of the changed gdn_out kernel is
  therefore verified through the dual-gate test in test_qwen35_cuda.cu instead.
- fp16 k/v cache at long context: see edge case 6.
- The negative-token path of ds4_mmq_pq2_0_rows_kernel (src < 0 writes 0) is
  not driven by any caller in the test or the graph.
- The out-of-vocabulary CPU diagnostic gap (edge case 5) was found in the
  first pass, fixed by the author at the input boundary of
  qwen35_first_token_test, and re-verified in the follow-up pass on both
  backends; it is no longer an accepted risk. The residual notes are the
  unguarded pre-existing qwen4 diagnostics and the kernel-level row index
  (both outside this unit, see the route audit above).

COMMANDS RE-RUN FOR THIS REPORT
First pass (all exits 0 unless noted):
- make cuda-generic
- make test-qwen35-cuda (final line "PQ2_0 CUDA parity: PASS")
- make pq2-0-test (final line "pq2_0: all checks passed (6 reference blocks,
  34 bytes/block, 2.125 bpw)")
- make bonsai-fold-selftest (final line "fold selftest: blocks 2, 4 and 1024
  match the explicit Hadamard matrix, blocks stay independent, forward/inverse
  round-trips, and the gdn permutation follows the tiled-to-grouped index map")
- the CPU and CUDA logits runs and the two greedy prompts above
- /tmp/qa_edge (scratch CUDA guard probe, 21 checks, PASS)
- make test-qwen4-cuda (FAIL to link; pre-existing, see not covered)
Follow-up pass on the fixed tree (all exits 0 unless noted):
- git diff (only the 12-line ds4.c guard), code read of the guard placement
- strings -a ds4 (the new message is present in the rebuilt binary)
- the three out-of-vocabulary ids on --cpu and --cuda (exit 1 with the range
  message; pre-fix these were exit 0, SIGSEGV and untested respectively)
- valid-input reruns: CPU 1m06s and CUDA 0m08s, cmp byte-identical to the
  pre-fix logits files, same streams and same CPU-vs-CUDA diff
- make test-qwen35-cuda again (rebuilt hooks, 51 PASS, 0 FAIL, parity PASS)
- make pq2-0-test and make bonsai-fold-selftest again (pass)
- route audit: callers of qwen35_first_token_test, ds4_qwen35_ref_forward_token,
  qwen35_graph_forward and ds4_parse_token_list, plus a qwen35 search in
  ds4_server.c (code reads)

verdict: overall PASS
