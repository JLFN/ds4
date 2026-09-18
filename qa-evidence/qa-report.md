ds4 QA report - Bonsai (qwen35 / Ternary-Bonsai-2-27B-PQ2_0) server path close (unit E2b)

Date: 2026-09-18
QA model: deepseek-v4.1-flash, acting as the rule 19 AI QA-tester
Branch under test: feature/qwen35-server
Base ref diffed against: dev (local integration branch, dev = 648bb3f)
State tested: the unit is UNCOMMITTED, so the working tree is the artifact.
  git status --short before the QA run: M ds4_server.c, M run-bonsai.sh
  git diff dev --stat: ds4_server.c 9 insertions; run-bonsai.sh 69 changed lines
  (66 insertions, 3 deletions). Nothing else is touched: ds4.c, ds4.h, ds4_kvstore.c
  and the Makefile are byte-identical to dev, which is why every kernel, tokenizer and
  session-path regression below MUST be unchanged.
  git diff dev sha256: c25fcf85ba89b08833ed2409335d8a895cc6c65c10eb1b6f1049233ab88c2be8
  The QA wrote no source, Makefile or script change. The only file it writes in the
  repo is this report; the scripts that drive the checks live in /tmp and are named
  where used so a reviewer can re-run every one of them.

Binaries as tested: ds4-server mtime 2026-09-18 19:55:29, newer than ds4_server.c
  mtime 19:54:12, and the new notice is present in the binary
  (strings ds4-server | grep -c "session checkpoints are not implemented for this model"
  returns 1). No rebuild was needed and none was done.
Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (arch qwen35, PQ2_0 ternary).
GPU: RTX 4070 SUPER 12 GB, shared with other sessions. nvidia-smi
  --query-compute-apps=pid,used_memory was read before and after every run and
  returned empty every time; every server started for a check was stopped, after which
  pgrep -a ds4-server found nothing; no other compute process appeared during the run.
  Every heavy run was strictly sequential.

Tooling note (rule 12 tool-first gate): repo-rag-mcp has no ds4 index registered
  (list_projects shows 21 projects, none is ds4); callgraph-mcp has no ds4 project
  (only ds4-on-spark, a different root, and no /data/ds4/.callgraph-index.bin);
  a fresh graphify-rs-out/graph.json exists (mtime 2026-09-18 19:53). The checks of
  this unit are live behaviour and literal log strings, so the live runs, direct
  source reading and grep are the authority; the indexes had nothing to add.


SURFACE 1 - ds4_server.c: --kv-disk-dir for this family logs an explicit notice

What I ran (script /tmp/qa-e2b-c-check.sh, one server instance, one request):
  cd /data/ds4
  ./ds4-server -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda -c 2048 \
      --port 8899 --kv-disk-dir /tmp/qa-e2b-kv
  curl -s -m 600 "http://127.0.0.1:8899/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"prism-bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of Italy?"}],"max_tokens":8,"temperature":0}'

Literal key output (grep -nE "KV disk cache|session checkpoints" on /tmp/qa-e2b-c.log):
  11:0918 20:09:18 ds4-server: KV disk cache /tmp/qa-e2b-kv (budget=4096 MiB, cross-quant=accept, min=512, cold_max=30000, continued=10000, trim=32, align=2048, hit_half_life=21600s)
  12:0918 20:09:18 ds4-server: session checkpoints are not implemented for this model; --kv-disk-dir stays unused
The notice sits on the line immediately after the existing "KV disk cache ..." line,
exactly the placement the brief asked to confirm.
The small request still completes: client wall 30.06s, finish_reason "length" at the
8-token cap, reasoning "The user asks: \"What is the capital of the", content empty,
usage prompt_tokens 59 / completion_tokens 8 / total_tokens 67. (The 8-token cap is far
below this model's think length; the same budget effect is measured in SURFACE 3 and 4.)
Directory state: ls -la /tmp/qa-e2b-kv right after startup and again after the request
shows the empty directory only (drwx------, 40 bytes); find /tmp/qa-e2b-kv -type f | wc -l
returns 0. No checkpoint was written, no crash.
Shutdown: log tail "shutdown requested, draining requests"; pgrep -a ds4-server finds
nothing; nvidia-smi compute apps empty.
Surface verdict: PASS.


SURFACE 2 - run-bonsai.sh: the new server mode, end to end

What I ran:
  cd /data/ds4 && DS4_BONSAI_CTX=2048 ./run-bonsai.sh server "What is the capital of Portugal?"

Literal output (exit code 0, 44.6s wall clock end to end):
  model:   /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
  prompt:  What is the capital of Portugal?
  ctx:     2048 on port 8899

  server:  up (pid 3846679), listening on http://127.0.0.1:8899

  wall:    43.26s for the request
  answer:   The capital of Portugal is Lisbon.
  reasoning: User asks: "What is the capital of Portugal?" Simple factual question. Need answer: Lisbon. Keep concise.
  tokens:   {'prompt_tokens': 65, 'completion_tokens': 33, 'total_tokens': 98, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 65}}

  server:  stopped (log at /tmp/bonsai-run.log.server)

Server log key lines (from the check A live capture; the runbook reuses the fixed path
/tmp/bonsai-run.log.server, and the later F4 attempt in this QA overwrote it with its
own failed-listen log, so that file on disk now holds the F4 content rather than this):
  chat ctx=0..65:65 prompt done 28.773s
  chat ctx=65..98:33 gen=33 decoding chunk=2.28 t/s avg=2.28 t/s 14.477s
  chat ctx=0..65:65 gen=33 finish=stop 43.250s
  shutdown requested, draining requests
After the run: pgrep -a ds4-server exit code 1 (no process); nvidia-smi compute apps
empty.
Surface verdict: PASS (listening line, wall time, exact answer, reasoning line, usage,
clean stop, no stray process, GPU free).


SURFACE 3 - concurrency and isolation with --batched-session

3a. Brief check B, exactly as specified (three concurrent requests, max_tokens 40,
temperature 0). Script /tmp/qa-e2b-b-check.sh.

What I ran:
  ./ds4-server -m /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf --cuda -c 4096 \
      --port 8899 --batched-session 3
  then three background curls in one command, each:
  curl -s -m 900 "http://127.0.0.1:8899/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"prism-bonsai-2-27b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":40,"temperature":0}'
  (same body for Japan and Portugal)

Literal key output:
  spawn line: 0918 20:06:15 ds4-server: batched mode enabled resident_sessions=3 prefill_quantum=2048 mixed_prefill_quantum=128 decode_coalesce_us=2000
  curl exits 0/0/0; aggregate wall 131s; client walls: france 130.72s, japan 129.84s, portugal 128.97s
  france:   finish_reason 'length'; content 'The capital of France';     reasoning head 'The user asks: "What is the capital of France?" This is a simple factual question. The answer is Paris. I should respond'
  japan:    finish_reason 'length'; content 'The capital of Japan is Tokyo'; reasoning head 'The user asks: "What is the capital of Japan?" This is a simple factual question. The answer is Tokyo. I should provide '
  portugal: finish_reason 'length'; content 'The capital of Portugal is'; reasoning head 'The user asks: "What is the capital of Portugal?" This is a simple factual question. The answer is Lisbon. I need to pro'
  each usage: prompt_tokens 59, completion_tokens 40, total 99
Server log per-request lines (grep -E "prompt done|finish=" /tmp/qa-e2b-b.log):
  chat ctx=0..59:59 prompt done 26.461s
  chat ctx=0..59:59 prompt done 25.848s
  chat ctx=0..59:59 prompt done 53.017s
  chat ctx=59..99:40 gen=40 decoding chunk=0.39 t/s avg=0.39 t/s 102.518s   (plus the matching finish=length 128.979s)
  chat ctx=59..99:40 gen=40 decoding chunk=0.52 t/s avg=0.52 t/s 77.540s    (finish=length 103.388s)
  chat ctx=59..99:40 gen=40 decoding chunk=0.78 t/s avg=0.78 t/s 51.242s    (finish=length 104.259s)
Literal brief criterion "each must finish with finish_reason stop": NOT MET at
max_tokens 40. All three responses report finish_reason "length" and the France and
Portugal contents stop before the city; only Japan's content happens to include Tokyo.
Everything else the brief requires of B holds: each response's reasoning names its own
correct city for its own question, there is no crash, no cross-contamination, all three
exit cleanly, the server shuts down ("shutdown requested, draining requests"), no
process is left and the GPU is free.
Root cause, measured not guessed: the same three prompts at temperature 0 need 41-43
completion tokens (see 3b: 42, 41, 43, all finish stop). The brief's 40-token budget is
1-3 tokens short of what the model emits, so the server correctly announces "length".
This is OpenAI max_tokens semantics; it is not a defect of this unit, which touches no
request handling (ds4_server.c gains only the notice, ds4.c is untouched).

3b. Same scenario, swapped order, max_tokens 64 (script /tmp/qa-e2b-b2-check.sh).
  ./ds4-server ... -c 4096 --port 8899 --batched-session 3
  three background curls fired in the order Portugal, Japan, France (swapped relative to B).
Literal key output:
  portugal: finish 'stop'; content 'The capital of Portugal is Lisbon.';   completion 42; wall 130.24s
  japan:    finish 'stop'; content 'The capital of Japan is Tokyo.';        completion 41; wall 130.24s
  france:   finish 'stop'; content 'The capital of France is Paris.';       completion 43; wall 131.99s
  each reasoning head names its own country (quoted in the capture); aggregate wall 132s
  server log: gen=42 finish=stop 130.248s; gen=41 finish=stop 130.248s; gen=43 finish=stop 131.998s
  clean shutdown; pgrep empty; GPU empty.
Verdict: PASS (order swap changes nothing; the answers never mix, and with a budget
above the model's completion length every property of the brief's B holds).

3c. Brief check F1: --batched-session 4 with only two requests (script
/tmp/qa-e2b-d2f1-check.sh, second half).
  ./ds4-server ... -c 4096 --port 8899 --batched-session 4
  two concurrent curls, Italy and Spain, max_tokens 64.
Literal key output:
  0918 20:16:07 ds4-server: batched mode enabled resident_sessions=4 prefill_quantum=2048 mixed_prefill_quantum=128 decode_coalesce_us=2000
  italy: finish 'stop'; content 'The capital of Italy is Rome.';   completion 43; wall 90.52s
  spain: finish 'stop'; content 'The capital of Spain is Madrid.'; completion 43; wall 90.07s
  server log contains exactly two sessions: prompt done 26.109s / 26.106s; gen=43 finish=stop 90.074s / 64.412s
  clean shutdown; no strays; GPU empty.
Verdict: PASS (two idle slots cause nothing: both sessions answer correctly, the two
unused slots are never reported active, and shutdown is clean).

3d. Aggregate behaviour, stated plainly.
  While three requests were in flight, each session decoded at 0.39, 0.52 and 0.78 t/s
  (the three quote lines in 3a), against 2.28 t/s for a solo decode on this host (SURFACE 2
  log line) - every request sees roughly one third to one sixth of the solo rate.
  Prefill is serialized one session at a time: the three "prompt done" stamps are 26.5s,
  25.8s and 53.0s, the third including about 27.6s of queue wait (its first chunk line
  reads "avg=0.04 t/s 27.598s" because the average starts at admission).
  Aggregate decode throughput in the window is about 1.15 t/s (120 tokens in 104s; B2 at
  64 tokens: about 1.19 t/s over 106s), roughly half the solo 2.28 t/s, and a 64-token
  answer that takes 44.85s solo (SURFACE 4 D2) takes about 130s with two others in
  flight. The serial-eval fallback therefore costs per-request latency (about 3x here)
  and some aggregate throughput; there is no parallelism gain, which is what "native
  batching does not cover this family" means in practice for this build.
Surface verdict: PASS for concurrency and isolation (the properties the Phase H gate
exists for), with the literal finish_reason wording at max_tokens 40 reported in the
deviation note below rather than hidden.


SURFACE 4 - streaming under the new build

Literal brief check D, max_tokens 16 (script /tmp/qa-e2b-d-check.sh). Three prompts
were tried, each one streaming request against a fresh server
(./ds4-server ... --cuda -c 2048 --port 8899):
  d1 "Reply with exactly: hello", d2 "Say hi", d3 "What is the capital of France?"
  body: {"model":"prism-bonsai-2-27b","messages":[{"role":"user","content":"..."}],
         "max_tokens":16,"temperature":0,"stream":true}
Parsed SSE for all three is the same shape:
  role deltas: 1; reasoning deltas: 16; content deltas: 0; finish_reason: length; [DONE]: True
  assembled content: ''   (assembled reasoning for d1 starts 'The user wants me to reply with exactly "hello". This is a simple request')
  first SSE line: ": prefill";   last SSE line: "data: [DONE]"
So role, reasoning_content deltas, finish_reason and [DONE] are all present, but the
literal D criterion "must produce ... content deltas" is NOT met at max_tokens 16:
the model spends the whole 16-token budget inside its think block, so no content token
can exist yet.
Root cause, measured: the think block alone is 33 tokens for this prompt (D2 below),
so content can only start at token 34; a 16-token budget cannot reach it at
temperature 0.
D2 diagnostic (script /tmp/qa-e2b-d2f1-check.sh, first half), same streaming request
with max_tokens 64:
  role deltas: 1; reasoning deltas: 33; content deltas: 8; finish_reason: stop; [DONE]: True
  assembled content: '\n\nThe capital of Portugal is Lisbon.'
  client wall 44.85s; server log "chat ctx=0..59:59 gen=42 finish=stop 44.848s"
  first line ": prefill", last line "data: [DONE]"
Surface verdict: PASS for the streaming machinery (role, reasoning deltas, content
deltas, finish_reason stop and [DONE] were all produced and the content assembles),
with the literal 16-token content requirement reported in the deviation note below
rather than hidden.


SURFACE 5 - foreign model id and /v1/models (brief F3)

What I ran (script /tmp/qa-e2b-f3b-check.sh, one server instance):
  GET  http://127.0.0.1:8899/v1/models
  POST http://127.0.0.1:8899/v1/chat/completions with
       {"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Reply with exactly: hello"}],"max_tokens":64,"temperature":0}
Literal key output:
  /v1/models: id prism-bonsai-2-27b | name Prism Bonsai 2 27B
              id prism-bonsai-2-27b-chat | name Prism Bonsai 2 27B
              id prism-bonsai-2-27b-reasoner | name Prism Bonsai 2 27B
  foreign id: echoed model 'deepseek-v4-flash'; finish_reason stop; content 'hello';
              reasoning head 'The user wants me to reply with exactly "hello". This is a simple request. I should output exactly the word "hello" with';
              usage prompt_tokens 57, completion_tokens 35; client wall 39.82s
  An earlier probe with the same foreign id at max_tokens 32 answered in the same
  ChatML form but was cut inside the think block (finish length, content empty) - the
  same budget effect as SURFACE 4, not an id problem.
Server stopped cleanly; no strays; GPU empty.
Surface verdict: PASS (the server still serves a foreign id with this family's ChatML
rendering, and the three family ids are unchanged).


SURFACE 6 - port already in use (brief F4)

What I ran (script /tmp/qa-e2b-f4-check.sh):
  python3 -m http.server 8899 --bind 127.0.0.1 &     (occupies the port; ss -tln confirms LISTEN 127.0.0.1:8899)
  DS4_BONSAI_CTX=2048 ./run-bonsai.sh server "What is the capital of Austria?"
Literal output:
  runbook exit code: 1
  server exited before listening (exit status above); last lines:
  ...
  0918 20:17:46 ds4-server: failed to listen on 127.0.0.1:8899: Address already in use
  "ds4-server strays after the failed runbook run": none
  holder stopped; GPU after: empty
Surface verdict: PASS (fails fast with a clear message, no request is attempted, no
process is left behind).


SURFACE 7 - the E3 KV checkpoint refusal still guards both payload functions

What I ran:
  grep -n "Bonsai KV checkpoints are not implemented yet" ds4.c
Literal output:
  63389:        payload_set_err(err, errlen, "Bonsai KV checkpoints are not implemented yet");
  63783:        payload_set_err(err, errlen, "Bonsai KV checkpoints are not implemented yet");
Surrounding lines, save path (ds4_session_save_payload, ds4.c:63384-63391):
  if (ds4_session_is_qwen35(s)) {
      /* Its state is the gated delta-net recurrent state, the conv history
       * and the fp16 k/v caches, not the DeepSeek raw-swa layout the generic
       * writer below assumes.  Refuse until that serializer exists. */
      payload_set_err(err, errlen, "Bonsai KV checkpoints are not implemented yet");
      return 1;
  }
Surrounding lines, load path (ds4_session_load_payload, ds4.c:63778-63784):
  if (ds4_session_is_qwen35(s)) {
      /* Its state is the gated delta-net recurrent state, the conv history
       * and the fp16 k/v caches, not the DeepSeek raw-swa layout the generic
       * reader below assumes.  Refuse until that serializer exists. */
      payload_set_err(err, errlen, "Bonsai KV checkpoints are not implemented yet");
      return 1;
  }
In both functions the qwen35 guard sits ahead of the generic DeepSeek payload
writer/reader (the qwen35 branch is the third family branch, before the glm branch and
the generic writer that follows it). ds4.c is entered by zero lines of this unit's diff
(git diff dev --stat lists only ds4_server.c and run-bonsai.sh).
Surface verdict: PASS.


SURFACE 8 - regressions (brief E), all sequential after the live checks

Literal commands and key output (full logs in /tmp/qa-e-*.log via script /tmp/qa-e2b-e-check.sh):
1. make pq2-0-test
   EXIT=0; "pq2_0: all checks passed (6 reference blocks, 34 bytes/block, 2.125 bpw)"
2. make test-qwen35-cuda
   EXIT=0; "exact1 MMQ ... failures=0/6144: PASS", "exact1 MMVQ ... failures=0/6144: PASS",
   "exact64 MMQ ... failures=0/1114112: PASS", "PQ2_0 CUDA parity: PASS"
3. make bonsai-fold-selftest
   EXIT=0; "fold selftest: blocks 2, 4 and 1024 match the explicit Hadamard matrix,
   blocks stay independent, forward/inverse round-trips, and the gdn permutation
   follows the tiled-to-grouped index map"
4. make bonsai-ref-check
   EXIT=0; tokens 5..16 are 11751 13 198 760 6511 314 9564 369 19241 13 198 760,
   the same reference continuation the previous QA recorded (token 16 text " The").
5. make test-qwen35-session DS4_TEST_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
   EXIT=0; 11 lines carrying PASS, 0 carrying FAIL: the 10 checks are plain
   position/ids, prefix reuse position/ids, rewind position/replay, invalidate
   empty/rebuild, context bound, plus the summary line "qwen35 session path: PASS".
6. ./run-bonsai.sh compare
   EXIT=0; cuda 13.31s / cpu 108.50s for 24 greedy tokens;
   "IDENTICAL: all 24 generated token ids agree, so the CUDA graph"
7. ./run-bonsai.sh session
   EXIT=0; cuda (session) 13.37s / cpu 107.96s;
   "IDENTICAL: all 24 generated token ids agree, so the session path"
After the suite: nvidia-smi compute apps empty; no ds4 process left.
Note recorded as requested: ds4.c is untouched by this unit (git diff dev --stat
shows only ds4_server.c 9 insertions and run-bonsai.sh 69 changed lines), so these
must be - and are - the same results the earlier QA recorded.
Surface verdict: PASS.


TWO LITERAL BRIEF CRITERIA THAT DID NOT HOLD AT THE BRIEF'S BUDGETS (not hidden)

1. Brief B, "each must finish with finish_reason stop" at max_tokens 40.
   Observed: all three concurrent B responses returned finish_reason "length"; France
   and Portugal contents stop before the city. Measured cause: the same prompts need
   41-43 completion tokens at temperature 0 (B2: 42, 41, 43, all stop at 64). The
   brief's budget is 1-3 tokens short; the server reports that honestly as "length".
   Fix for a re-run of the gate as written: set max_tokens 48 or higher (the runbook's
   own default is 64, which passes).
2. Brief D, "must produce ... content deltas" at max_tokens 16.
   Observed: reasoning deltas 16, content deltas 0, finish length for three prompts.
   Measured cause: the think block alone is 33 tokens for this model/prompt, so content
   starts at token 34; a 16-token budget cannot reach it. The same request at 64 tokens
   produces role, 33 reasoning deltas, 8 content deltas, finish stop and [DONE] (D2).
Both deviations are properties of the requested token budgets against a reasoning model
at temperature 0, not of the unit under test: the unit changes no request handling,
ds4.c is untouched, and every other property of those checks (own correct city, no
crash, no cross-contamination, role/reasoning/finish/[DONE], clean shutdown) holds at
the brief's own settings.

Small print found while running (reported, none of it a failure):
- Per-request walls in B were 129.0-130.7s, slightly above the brief's 90-125s estimate;
  the shape of the estimate (three serialized sessions) is right.
- run-bonsai.sh line 29 still carries the header comment "the server path is a later
  unit and is not wired into this script yet"; the status text and the usage block were
  updated by the unit, that one sentence was not. Documentation only (the usage block
  prints lines 2-13 and is correct).
- A response cut inside the think block makes the server log "thinking not closed,
  ignoring incomplete Qwen tool calls in reasoning" (seen in SURFACE 1 and SURFACE 4);
  a parser note, no failure.
- The server-side "finish=Xs" timer starts at session admission, not at curl dispatch,
  so in B the server prints 128.98s / 103.39s / 104.26s while all three client walls
  read about 129-131s. Explained, not an error.
- The third queued prefill line reads "avg=0.04 t/s 27.598s" at chunk 1/59 because its
  average includes the queue wait; its completed prefill is 26s of work behind it.

COULD NOT VERIFY (accepted risks)

- The mixed prefill/decode overlap was never triggered. In every server log captured for
  this QA (checks A, B, B2, C, D, D2, F1, F3b and the F4 attempt) the only occurrence of
  "mixed" is the startup configuration value mixed_prefill_quantum=128 in the
  batched-mode line where one exists; no request produced a mixed-phase line. The brief
  records that the previous run saw no occurrence either. What this does not prove: that
  the overlap path is correct; only that it did not engage under these requests on this
  build.
- The non-Bonsai families with the new code: the notice is inside the
  "if (cfg.kv_disk_dir)" branch and additionally gated on ds4_engine_is_qwen35(engine)
  (ds4_server.c:15794-15806 read directly), so for any other engine nothing executes.
  No DeepSeek or GLM model is installed, so this is code-level reasoning, not a live run.
- Long generations and long chats for this family: not exercised (the brief forbids
  queueing long generations on this slow shared GPU); the longest answer generated in
  this QA was 43 tokens.
- Four or more concurrent requests: only 2 and 3 concurrent requests were exercised
  (plus the 4-slot server with two requests). A 4-request burst was not run.
- The report file is the only repo change the QA makes; no source, Makefile or script
  was modified, and the working tree after the run is still exactly the unit's two
  modified files plus this rewritten report.

verdict: overall PASS
