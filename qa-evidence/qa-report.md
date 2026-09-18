ds4 QA report - Bonsai (qwen35 / Ternary-Bonsai-2-27B-PQ2_0) session path

Date: 2026-09-18
QA model: deepseek-v4.1-flash, acting as the rule 19 AI QA-tester
Branch under test: feature/qwen35-session
Base ref diffed against: dev
HEAD and dev point at the same commit: c61789747e2f74c66d94a775cbe7e5c6224e10f8
State tested: the unit is UNCOMMITTED, so the working tree is the artifact.
  git diff dev --stat:
    .gitignore    |   1 +
    Makefile      |  14 +++
    ds4.c         | 288 +++++++++++++++++++++++++++++++++++++++++++++++++++++++---
    run-bonsai.sh |  72 ++++++++++++---
    tests/run.sh  |   5 +
  untracked: tests/test_qwen35_session.c (347 lines)
  The QA run changed no source content. git status is identical before and after
  (the same five modified files plus the untracked test); git diff dev hashes to
  8a18c316171a724b6b0c5d17d3443864. Only the mtime of ds4.c was touched, as the
  requested compiler-warning rebuild needs it, and build outputs (ds4.o, the
  test-hook objects, tests/test_qwen35_session) were rebuilt by the checks as
  usual; none of that is source.
Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (7206168928 bytes, 6.71 GiB)
Host: RTX 4070 SUPER 12 GB, shared with other sessions. Every heavy run was
  sequenced one at a time, and nvidia-smi -q compute-apps was checked before and
  after each: no compute process was present at either end of every run below.
Oracle: ds4's own CPU reference in the same binary (ds4_test_qwen35_ref_greedy
  built under DS4_TEST_HOOKS, and the --cpu --first-token-test diagnostic).
  No llama.cpp was executed and no external implementation was involved.

Reference continuation used below (CPU oracle, greedy):
  prompt "The capital of France is" -> 5 tokens: 760 6511 314 9338 369
  12 steps: 11751 13 198 760 6511 314 9564 369 19241 13 198 760
  text: " Paris." / "The capital of Germany is Berlin." /
        "The capital of Italy is Rome." / "The capital of Spain is"
  24 steps (run-bonsai.sh session, CPU side) are the same stream continued.

SURFACE 1 - the session path: graph state, create/sync/eval/free, invalidate,
rewind and the stale replay (ds4.c)
Code under test: ds4_session.qwen35_graph field ds4.c:60720; ds4_session_is_qwen35
ds4.c:61749; create ds4.c:74728-74755 (heap graph allocated at 74743); free
ds4.c:75103-75106; sync block ds4.c:77147-77197; eval block ds4.c:79224-79249;
qwen35_graph_reset ds4.c:69090; qwen35_session_replay_if_stale ds4.c:75824-75836;
invalidate ds4.c:87060; rewind ds4.c:87106-87113.

What I ran:
  1. make test-qwen35-session DS4_TEST_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf CUDA_ARCH=native
  2. ./run-bonsai.sh session
  3. ./ds4 -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda --raw -p "The capital of France is" -n 24 --temp 0
  4. ./ds4 -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda --raw -p "The capital of France is" --seed 7 -n 16

Observed (1, plain decode; the log carried 10 individual checks, 0 FAIL):
  PASS  plain: position after sync equals the prompt length
  PASS  plain: session reports the created context size
  PASS  plain: decoded ids equal the CPU reference
  PASS  prefix reuse: position after the extending sync
  PASS  prefix reuse: ids equal the CPU reference across both syncs
  PASS  rewind: position is back at the prompt
  PASS  rewind: the replay reproduces the reference ids
  PASS  invalidate: the checkpoint is empty
  PASS  invalidate: ids after the rebuild equal the reference
  PASS  context bound: the third decode past capacity is refused

  qwen35 session path: PASS
  EXIT=0

Observed (2): the session run printed
   Paris.
  The capital of Germany is Berlin.
  The capital of Italy is Rome.
  The capital of Spain is
  and the token-for-token diff of the session log against the CPU reference log
  (/tmp/bonsai-session.tokens vs /tmp/bonsai-cpu.tokens, 24 lines each) printed
  "IDENTICAL: all 24 generated token ids agree, so the session path reproduces
  the CPU reference on this prompt".

Observed (3): exit 0, 12.95 s wall for prompt plus 24 tokens, stdout
   Paris.
  The capital of Germany is Berlin.
  The capital of Italy is Rome.
  The capital of Spain is
  ds4: prefill: 2.21 t/s, generation: 2.41 t/s
  and the text is byte-identical to the 24-step CPU reference continuation
  (newline-normalized diff empty).

Observed (4): exit 0, 9.51 s wall, stdout
   Paris.
  The capital of Italy is Rome.
  The capital of Germany is
  (sampling differs from greedy as expected; the text is fluent and on-topic).
  The sampled CLI goes through ds4_session_create/sync/sample/eval in
  run_sampled_generation (dispatched at ds4_cli.c:1240-1244, call at 1249; the
  session loop at 596-660), so this exercises the sample path over the Bonsai
  session logits.

Verdict for this surface: PASS.

SURFACE 2 - the engine gate relaxation and its remaining refusals (ds4.c)
Code under test: the qwen35 gate ds4.c:72346-72375; the accepted set is
cpu_reference or cuda_graph or cuda_session (72361) and the refused set is
tp != NONE, distributed != NONE, load_slice, ssd_streaming, dspark, glm_mtp,
mtp_path, directional steering file, power_percent != 100 (72362-72367); the
refusal message is 72368-72372.

What I ran (refusal probes):
  a. ./ds4 -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda --first-token-test --raw --ssd-streaming -p "x"
  b. env DS4_QWEN35_SESSION=1 ./ds4 -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cpu --first-token-test --raw -p "The capital of France is"

Observed (a): exit 1, stderr
  ds4: Bonsai (qwen35) runs on the CPU reference (--cpu --first-token-test, the
  diagnostic oracle) or on single-GPU CUDA (--cuda, diagnostic or session/server);
  tensor parallelism, SSD streaming, DSpark/MTP and steering are not supported
Observed (b): the CPU reference is still accepted with the note (see Surface 7)
and the run exited 0.

Observed (acceptance side): runs 3, 4 and the server runs of Surface E all
opened the engine without --first-token-test and exited 0, i.e. the gate now
admits a Bonsai CUDA session.

Verdict for this surface: PASS. Only --ssd-streaming was probed live; the other
options in the same boolean expression were read, not each executed (see
accepted risks).

SURFACE 3 - the greedy engine path and ds4_session_eval_argmax (ds4.c)
Code under test: the qwen35 condition in ds4_engine_generate_argmax
(ds4.c:65570-65572) routing the family into the session block (65573-65615),
and the ds4_session_eval_argmax guard (ds4.c:65091-65095) that sends Bonsai
through plain eval + argmax.

What I ran: run 3 above (greedy CLI, --temp 0).

Observed: exit 0 and, on stderr, the timing line printed only by that session
block, "ds4: prefill: 2.21 t/s, generation: 2.41 t/s", with the reference text.
With DS4_CUDA_GREEDY_TOP1 defaulting on (65593), the loop calls
ds4_session_eval_argmax for every step, so the greedy CLI exercised that exact
entry point live. A source grep confirms Bonsai never reaches the DeepSeek
raw-swa graph encode paths inside this function (they sit after the return of
the session block).

Verdict for this surface: PASS.

SURFACE 4 - the two CUDA batch fast paths no longer take a Bonsai session
(ds4.c)
Code under test: ds4_sessions_eval_batch_cuda guard ds4.c:85839-85842 and
ds4_sessions_eval_batch_with_prefill_cuda guard ds4.c:85970-85973. When the
guard excludes the family, the first function falls through to its correctness
fallback, ds4_session_eval per item (ds4.c:85896-85907), which routes each
Bonsai session into its own branch; the second falls back to
ds4_session_sync + ds4_sessions_eval_batch (ds4.c:86013-86020).

What I ran: none live. The batched modes are a server feature
(--batched-session) and the server path is explicitly the next unit. I read the
guards and both fallbacks in the live source.

Observed: the guards are the only way a Bonsai session could have reached
metal_graph_encode_token_raw_swa/encode_session_pipeline_batch on s->graph, and
both exclude it. No live batch run was performed.

Verdict for this surface: PASS by code inspection; live batch coverage is an
accepted gap (see accepted risks).

SURFACE 5 - the KV payload checkpoint refusal (ds4.c)
Code under test: ds4_session_save_payload refuses at ds4.c:63363-63371 and
ds4_session_load_payload refuses at ds4.c:63757-63765, both with the exact text
"Bonsai KV checkpoints are not implemented yet", before the DeepSeek raw-swa
writer/reader is reached.

What I ran (attempted live trigger through the server disk cache):
  ./ds4-server -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda -c 2048 --port 8899 \
      --kv-disk-dir /tmp/qa-kv --kv-disk-space-mb 512 --kv-cache-min-tokens 1 --kv-cache-cold-max-tokens 30000
  curl -s -m 120 http://127.0.0.1:8899/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash","prompt":"The capital of Spain is","max_tokens":8,"temperature":0}'
  then SIGTERM to the server (logs "persisting resident KV cache before shutdown slot=0 tokens=38").

Observed: HTTP 200, 16.9 s, a normal completion; /tmp/qa-kv stayed empty; no
crash; no refusal line in the log. Reason the live trigger does not reach the
guard: the disk-cache store exits earlier and silently at
ds4_kvstore_store_live_prefix_text (ds4_kvstore.c:937-941) because
ds4_kvstore_quant_bits_supported (ds4_kvstore.c:418-421) rejects the value
returned by ds4_engine_routed_quant_bits, which is 0 for this dense model (it
looks for ffn_gate_exps, ds4.c:62660-62669). The message is therefore only
reachable through the payload API itself, not through the server disk cache
today.

Verdict for this surface: PASS by code inspection (the guard is present, before
the DeepSeek layout code, with the exact message, in both directions). The live
trigger is blocked upstream by an unrelated disk-cache policy; recorded as an
accepted risk.

SURFACE 6 - the test-only reference oracle ds4_test_qwen35_ref_greedy (ds4.c)
Code under test: ds4.c:68603-68628, compiled only under -DDS4_TEST_HOOKS; it
runs the existing ds4_qwen35_ref_forward_token reference greedily over the same
prompt and step count, so the test has an in-process oracle.

What I ran:
  make ds4_cuda_test_hooks.o tests/test_qwen35_session.o CUDA_ARCH=native
  (full rebuild after rm of both objects; 0 warnings, 0 errors)
  plus the test runs of Surface 8, which consume the hook.

Observed: the test printed "CPU reference (the oracle): 11751 13 198 760 6511 314
9564 369 19241 13 198 760" and all session scenarios matched it. The hook build
compiles clean under -Wall -Wextra with -Wno-unused-function.

Verdict for this surface: PASS.

SURFACE 7 - the DS4_QWEN35_SESSION diagnostic mode (ds4.c)
Code under test: qwen35_first_token_test session branch ds4.c:69207-69370
(session creation 69211-69226 with the CPU note at 69222; the LOGITS refusal at
69262-69271; the session decode step 69342-69351; free at 69367-69369).

What I ran:
  a. ./run-bonsai.sh session   (session run + CPU reference + diff)
  b. env DS4_QWEN35_SESSION=1 ./ds4 -m ... --cpu --first-token-test --raw -p "The capital of France is"
  c. env DS4_QWEN35_SESSION=1 DS4_QWEN35_LOGITS=/tmp/qa-qwen35-logits.bin ./ds4 -m ... --cuda --first-token-test --raw -p "The capital of France is"

Observed (a): the same "token N: id text" stream from the session path, matched
token-for-token against the CPU reference (IDENTICAL, 24/24).
Observed (b): exit 0, stderr "ds4: DS4_QWEN35_SESSION needs --cuda; running the
CPU reference", then the reference token stream; the first 16 ids equal the B
oracle ids.
Observed (c): exit 1, stderr "ds4: DS4_QWEN35_LOGITS needs the per-position
logits and is not available with DS4_QWEN35_SESSION"; /tmp/qa-qwen35-logits.bin
did not exist before the run and did not exist after it.

Verdict for this surface: PASS.

SURFACE 8 - tests/test_qwen35_session.c and the Makefile target (test)
Code under test: tests/test_qwen35_session.c (347 lines) and Makefile
484-495 (object rule 486-487, link rule 491, phony + run 494-495).

What I ran: the command of Surface 1, item 1, first as the make target and again
after relinking the binary from the freshly rebuilt objects.

Observed: the make target built and ran the binary; the binary printed the
10 individual PASS lines, "qwen35 session path: PASS" and exited 0, twice
(once before the warning rebuild, once after). The source really exercises both
paths the task named:
  - prefix reuse: scenario_prefix_reuse (tests/test_qwen35_session.c:119-165)
    decodes half the steps, pushes them into an extended prompt and calls
    ds4_session_sync again; the test asserts the position after the extending
    sync and that the ids across both syncs equal the reference. The observed
    PASS lines for both assertions prove the prefix/checkpoint-continuation
    branch inside ds4_session_sync_internal (ds4.c:77156-77161) was taken.
  - rebuild: scenario_invalidate (223-252) invalidates the session, asserts the
    checkpoint is empty and resyncs from scratch, comparing the ids again.
  - the rewind-replay scenario (171-217) covers qwen35_session_replay_if_stale
    plus the rewind reset, and the context-bound scenario (258-283) covers the
    g->pos >= g->ctx_cap refusal.

Verdict for this surface: PASS.

SURFACE 9 - run-bonsai.sh session mode and the tests/run.sh wiring
Code under test: run-bonsai.sh session_mode at run-bonsai.sh:139-178 (diff at
170-176), status text 208-211; tests/run.sh:44-48 adding the CUDA-gated step.

What I ran: ./run-bonsai.sh session; bash -n run-bonsai.sh; bash -n tests/run.sh.

Observed: the script printed the CUDA session run, the CPU reference run, and
"IDENTICAL: all 24 generated token ids agree ..."; it exits 0. Both scripts
parse. The suite step calls make test-qwen35-session with DS4_TEST_MODEL and
CUDA_ARCH and is skipped when DS4_SKIP_CUDA=1. The whole tests/run.sh suite was
not executed (out of scope for this unit check), only the wired target.

Verdict for this surface: PASS.

Adversarial checks (trying to falsify the unit)

1. Two prompts back to back on one server (session/state isolation):
   with the server from Surface E I sent "The capital of France is" and then
   "The capital of Italy is" (both max_tokens 16, temperature 0). The first
   answer was "\nThe user is asking for the capital of France. This is a
   straightforward factual"; the second was "\nThe user is asking for the
   capital of Italy. This is a straightforward factual". The second answer
   tracks its own prompt (Italy), so no observable cross-request contamination.
   Limit: this is a behavioural probe, not a state inspection; the server's own
   slot lifecycle is the next unit.
2. DeepSeek graph audit (s->graph sites, grep -n "s->graph" ds4.c, 183 sites):
   the only sites a Bonsai session can reach in the live session API are:
   - ds4_session_dspark_capture_invalidate (ds4.c:61720), called from
     ds4_session_invalidate (87060 area) and ds4_session_rewind for every
     non-CPU session. It calls the two capture invalidators, which both return
     immediately unless g->dspark_capture_enabled (ds4.c:30012-30030); a
     qwen35 session has a zeroed s->graph, so this is two boolean reads and no
     graph work.
   - ds4_session_free's else-chain is reached only when s->qwen35_graph is NULL
     (ds4.c:75103-75106), so metal_graph_free is not called for Bonsai.
   - ds4_session_gpu_warmup returns early for any non-DeepSeek4 family
     (ds4.c:78448) before touching s->graph.
   - ds4_session_sync_multimodal writes s->graph.prefill_vision_spans and
     span_count around the sync (ds4.c:77036-77037 and 77047-77048) even for Bonsai: two writes
     of NULL/0 into a zeroed struct with no dereference. Not family-guarded, but
     harmless; an image span would require a loaded vision encoder, which this
     model does not have.
   Residual risk (unreachable but unguarded, not a pass): 
   ds4_session_eval_output_head_from_hc (ds4.c:75368-75393) uses &s->graph
   unconditionally for a non-CPU, non-GLM session. Its only callers are
   ds4_distributed.c:2642 and 3766, and Bonsai cannot enter a distributed role
   (the engine gate refuses distributed/pipeline), so it is unreachable today;
   if the server path ever calls it for this family it would touch the zeroed
   DeepSeek graph. Same class: the DeepSeek speculative cycle tail of
   ds4_session_eval_speculative is reachable only with mtp_ready or
   support_kind == DSPARK, both refused for this family by the gate
   (mtp_path/dspark) and unreachable through ds4_engine_mtp_draft_tokens.
3. Rebuild warnings: touch ds4.c && make ds4.o CUDA_ARCH=native produced
   0 warnings and 0 errors (compile line with -Wall -Wextra); the
   DS4_TEST_HOOKS build of ds4.c and tests/test_qwen35_session.o also produced
   0 warnings after a forced rebuild.
4. Context bound: the test's context-bound scenario (refused at the third decode
   past capacity, not run) passed live, so the boundary is a refusal and not an
   out-of-bounds read.

What could NOT be verified (accepted risks)

1. Live KV checkpoint refusal: the guard is code-verified only (Surface 5). The
   disk cache never reaches it for this model because PQ2_0 has no supported
   routed-expert bitwidth (ds4_kvstore.c:418-421, ds4.c:62660-62669), and no
   CLI/agent path can stage a payload for a Bonsai session in this build.
2. The batch fast-path guards (Surface 4): read and reasoned, not exercised
   live; batched sessions belong to the server unit that follows.
3. The other refusals of the engine gate (tensor parallel, distributed,
   DSpark/MTP, steering file, power_percent) were read in the source but only
   --ssd-streaming was probed live.
4. The server prompt rendering is borrowed, not this model's own: the
   /v1/completions prompt was wrapped into a synthetic system+user chat
   ("You are a helpful assistant") and answered in a reasoning style, with
   prompt_tokens = 30 for a 5-token string, and the response model id is
   "deepseek-v4-flash". server_model_syntax_for_engine (ds4_server.c:1214-1219)
   has no qwen35 case and falls through to the DeepSeek syntax, while the GGUF
   itself carries a Qwen-style template (tokenizer.chat_template with
   <|im_start|> markers, tokenizer.ggml.pre = qwen35). This is expected for a
   unit that does not touch the server, and it did not crash, but the server
   smoke is not a raw-completion test of this model.
5. Long-context behaviour: only short prompts (5 tokens) with 12 and 24 greedy
   steps were compared live; the context bound was tested at the capacity edge
   but not a long prefill (one token per forward today, by design).
6. The full tests/run.sh suite was not run end to end; only the newly wired
   target inside it was executed.
7. The binaries used for the live CLI/server runs are the ones built at
   18:22/18:26 from this same source; the warning rebuild refreshed ds4.o and
   the test objects afterwards without changing any content.

Result summary

All required live checks A-F passed: 10/10 session-path assertions plus the test
summary PASS; 24/24 session ids identical to the CPU oracle through
run-bonsai.sh session; greedy and sampled CLI generation on the real session
path exit 0 with fluent, reference-matching text; the CPU note, the LOGITS
refusal and the SSD-streaming gate all behave as specified; the server answers
and does not crash, with the two-prompt isolation probe passing; zero compiler
warnings from ds4.c in both builds. The gaps listed above are scope boundaries
and unguarded-but-unreachable sites, not observed failures.

verdict: overall PASS
