ds4 QA report - Bonsai (qwen35 / Ternary-Bonsai-2-27B-PQ2_0) tokenizer dispatch and
ChatML-with-reasoning chat syntax

Date: 2026-09-18
QA model: deepseek-v4.1-flash, acting as the rule 19 AI QA-tester
Branch under test: feature/qwen35-chat-syntax
Base ref diffed against: dev (the local integration branch)
HEAD and dev point at the same commit: a69f4c02709b0513382218a3ddf58515e205cf93
State tested: the unit is UNCOMMITTED, so the working tree is the artifact.
  git status --short: M ds4.c, M ds4.h, M ds4_agent.c, M ds4_server.c
  git diff dev --stat: ds4.c 87, ds4.h 1, ds4_agent.c 4, ds4_server.c 17 changed
                       lines (75 insertions, 34 deletions over 4 files)
  git diff dev sha256: aabc80dcd771ca3c24f4c8137a6f3e21a4e73244dd738a37f7e807575bc36e7a
  The QA run changed no source file, Makefile or script. git status is identical
  before and after. Only ds4.c's mtime was touched (the requested warning-free
  rebuild needs it) and build outputs were rebuilt by their own targets (ds4.o,
  tests/test_pq2_0, tests/test_qwen35_session and the objects it links); none of
  that is source. Everything else the QA wrote lives outside the repo (/tmp).
Model: /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf (arch qwen35, PQ2_0 ternary,
  851 tensors, vocab 248320, 6.71 GiB).
GPU: RTX 4070 SUPER 12 GB, shared with other sessions. Every heavy run was
  sequenced one at a time; nvidia-smi --query-compute-apps=pid,used_memory was
  read before and after each run and returned empty every time. Both servers
  started for the checks were stopped and the GPU verified empty afterwards.
Oracle: ds4's own CPU reference in the same binary and the model's own metadata
  (the GGUF tokenizer.chat_template and tokenizer.ggml.pre keys). No llama.cpp
  binary was built or executed. /data/llama.cpp-prism was read only, as the
  reference source for the pre-tokenizer rule.
Surface coverage: this report covers ALL NEW SURFACES the unit adds (the ChatML
  renderer, the qwen35 tokenizer dispatch, the engine predicate, and the server
  and agent family wiring). Every check below states the literal command and the
  literal key output. For the tokenizer A/B I built the pre-unit tree read-only
  from git (git archive dev -> /tmp/ds4-base, make ds4 CUDA_ARCH=native); the
  repo itself was never rebuilt from the base sources.

Reference continuation used by the regression checks, prompt
"The capital of France is" -> 5 tokens 760 6511 314 9338 369, then 24 greedy
tokens: 11751 13 198 760 6511 314 9564 369 19241 13 198 760 (text: " Paris." /
"The capital of Germany is Berlin." / "The capital of Italy is Rome." /
"The capital of Spain is").


SURFACE 1 - qwen35 pre-tokenizer dispatch: bpe_tokenize_text now selects
bpe_tokenize_text_qwen35 for the Bonsai family
(ds4.c:43701-43706; before the unit only ds4_model_is_qwen4() selected it)

What I ran (all CPU, no GPU):
  1. python3 reader of the GGUF KV header (the bundled gguf python reader
     rejects PQ2_0 type 142) -> printed general.architecture, tokenizer.ggml.pre
  2. sed -n '375,390p' /data/llama.cpp-prism/src/llama-vocab.cpp
  3. ./ds4 -m MODEL --cpu --raw --dump-tokens -p "<each test text>"
     and the same command with /tmp/ds4-base/ds4 (the pre-unit binary built
     from dev) for the A/B.
  4. The recorded pre-change capture /tmp/e2-tokenizer-before.txt (6 lines).

Observed:
  - GGUF metadata: general.architecture = 'qwen35'; tokenizer.ggml.pre =
    'qwen35'; tokenizer.ggml.model = 'gpt2'; bos 248044; eos 248046;
    add_bos_token False. The pre-type string is what the fork maps to its own
    pre-tokenizer.
  - Reference: /data/llama.cpp-prism/src/llama-vocab.cpp:2191-2192 maps
    tokenizer_pre == "qwen35" to LLAMA_VOCAB_PRE_TYPE_QWEN35, and
    llama-vocab.cpp:382-388 gives that pre-type its own ordered alternation
    (the only one with \p{M} in the letter class and no CJK isolation):
      case LLAMA_VOCAB_PRE_TYPE_QWEN35:
          regex_exprs = {
              // original regex from tokenizer.json
              // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
              "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
          };
          break;
  - ASCII battery, current binary vs the same 6 inputs on the pre-unit binary:
    all six id lists are identical between the two binaries, and they match the
    recorded pre-change capture /tmp/e2-tokenizer-before.txt. The capture's
    first line shows one extra tail (21 22 23 984 = "678" + " %"), i.e. its ids
    belong to the string "The capital of France is 12,345.678 %", not to the
    label it records; both binaries reproduce that id list exactly for that
    string, and both give [760, 6511, 314, 9338, 369, 220, 16, 17, 11, 18, 19,
    20, 13] for the labeled string. Literal current output:
      The capital of France is 12,345. -> [760, 6511, 314, 9338, 369, 220, 16, 17, 11, 18, 19, 20, 13]
      don't stop                       -> [14572, 914, 2842]
      a  b                             -> [64, 220, 292]
      Hello  世界!                   -> [9419, 220, 220, 96748, 0]
      1+1=2                            -> [16, 10, 16, 28, 17]
      The capital of France is         -> [760, 6511, 314, 9338, 369]
  - Combining mark: ./ds4 -m MODEL --cpu --raw --dump-tokens -p "$(printf 'cafe\xcc\x81')"
      current (qwen35 rule): [895, 1795, 52033]   (ca, fe, U+0301)
      pre-unit binary (DeepSeek rule): [895, 1795, 52033]
    The two rules produce the SAME ids for this input. Stated plainly: I could
    not measure any id difference, so I do not claim one. The reason is visible
    in both implementations in the tested tree: the DeepSeek/JoyAI path treats
    every non-ASCII byte as letter-like (ds4.c:43259-43273, joyai_letter_like_at
    returns true for all bytes >= 128, consumed by joyai_consume_letters at
    43275), so the combining mark stays inside the same piece as "cafe", exactly
    as the qwen35 rule's [\p{L}\p{M}]+ run does (ds4.c:43596-43608 and
    43624-43642). Extra discriminating inputs ("你a", "a你b", "caféx", "1+mark",
    "1234", "123456789", "0420") were also run through both binaries: identical
    ids in every case, because the BPE merges rejoin what the two rules split
    differently. The dispatch change is real and correct by the model's own
    metadata and the reference regex, but on the inputs I could construct it is
    not id-visible; a difference, if any exists, is in untested input shapes.
Surface verdict: PASS (dispatch matches the model's declared pre-type and the
  reference fork's rule; ASCII and combining-mark behaviour measured and
  unchanged)


SURFACE 2 - the generalized ChatML-with-reasoning renderer: chatml_chat_open,
chatml_chat_close, chatml_assistant_prefix, chatml_system, the effort strings
DS4_CHATML_REASONING_XHIGH / DS4_CHATML_REASONING_LOW, and their selection in
encode_chat_prompt (ds4.c:43985-44045; chat_push_think_prefix ds4.c:43960-43973)

What I ran:
  1. Extract tokenizer.chat_template from the GGUF (python reader) -> 8952 bytes;
     read the xhigh sentence at line 52, the low sentence at line 54, the system
     turn at lines 79-83, the user turn at line 109, and the generation prompt at
     lines 163-169.
  2. Reconstruct the template's rendering for system="You are a helpful
     assistant", user="What is the capital of France?", thinking on, and compare
     the two token streams:
       ./ds4 -m MODEL --cpu --dump-tokens -p "What is the capital of France?" > struct.out
       RAW=$(cat expected.txt); ./ds4 -m MODEL --cpu --raw --dump-tokens -p "${RAW}"$'\n' > raw.out
       diff struct.out raw.out   (also diffed the id lines only)
  3. Same A-check for thinking off: --nothink structured vs the template's
     enable_thinking=false text (system turn with no instruction, assistant
     prefix "<think>\n\n</think>\n\n"), the raw prompt passed with $'\n\n' so the
     final blank line is a real part of the text.
  4. Server token-level check of the effort variants: one request with
     "chat_template_kwargs":{"reasoning_effort":"low"} against
     ./ds4-server -m MODEL --cuda -c 4096 --port 8899 --trace /tmp/qa_low_trace.log,
     then read the rendered prompt from the trace.

Observed (2):
  expected text (354 bytes, the template's own concatenation):
    <|im_start|>system\n
    Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n
    You are a helpful assistant<|im_end|>\n
    <|im_start|>user\nWhat is the capital of France?<|im_end|>\n
    <|im_start|>assistant\n<think>\n
  structured id stream (65 tokens):
    [248045, 8678, 198, 24342, 286, 4879, 369, 716, 310, 830, 11553, 13, 5044, 1683, 15060, 1472, 279, 3274, 11, 9307, 1328, 30800, 11, 2814, 47675, 25605, 11, 321, 60445, 55404, 11, 27224, 11, 321, 30246, 303, 279, 1534, 4087, 13, 271, 2523, 513, 264, 10631, 17313, 248046, 198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 198]
  raw id stream: byte-identical to the structured one; diff of the full outputs
  (ids and the token table) is empty, i.e. the renderer equals the template
  token for token, including the final newline after <think>.
Observed (3): the --nothink structured stream equals the raw tokenization of the
  template's enable_thinking=false text, id for id:
    [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 248046, 198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
  (29 tokens; diff of the two dumps is empty)
Observed (4): prompt_tokens 53 and the trace shows the template's low sentence
  verbatim in the system turn:
    <|im_start|>system
    Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.

    You are a helpful assistant<|im_end|>
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
    <think>
  The xhigh sentence emitted by the default path is exactly the template's
  line-52 sentence (compare the structured stream above against the template
  text); the medium level renders no instruction and the thinking-off prefix is
  the template's empty-think form (checked above).
Surface verdict: PASS (renderer == tokenizer.chat_template, token for token,
  for the default xhigh path, the low path, and thinking off)


SURFACE 3 - multi-turn history appenders and the CLI chat path:
ds4_chat_append_message, ds4_chat_append_assistant_prefix,
ds4_chat_append_multimodal_message (ds4.c:44232-44248, 44296-44300, 73941-73965)

What I ran:
  1. ./ds4 -m MODEL --cuda -p "What is the capital of France? Answer in one short sentence." -n 80 --temp 0
  2. Server multi-turn: system + user Japan + assistant(content "The capital of
     Japan is Tokyo.", reasoning_content replayed) + user "And France?" through
     POST /v1/chat/completions (see SURFACE 5 for the server mechanics).

Observed (1): exit 0 in 49 s; stdout:
    The user asks for the capital of France and requests a one short sentence. The answer is Paris. I need to provide a concise sentence.


    The capital of France is Paris.
  stderr: "processing 71 input tokens: 1/71 (1.4%)" ... "71/71 (100.0%)" and
    "ds4: prefill: 2.26 t/s, generation: 2.29 t/s". 71 = the 65-token
  system+user rendering plus the 6 tokens of " Answer in one short sentence.",
  i.e. the appenders produce the same ChatML stream as the single-turn dump.
Observed (2): the replayed assistant turn reproduced the live token stream
  exactly (100 live tokens, 115 prompt tokens, cached_tokens 100), whose only
  way to happen is that the assistant-history appender re-renders
  "<|im_start|>assistant\n<think>\n" + reasoning_content + "\n</think>\n\n" +
  content as the template prescribes.
Surface verdict: PASS for the text and multi-turn paths. The multimodal appender
  is code-only verified (see could-not-verify).


SURFACE 4 - new public predicate ds4_engine_is_qwen35 (ds4.c:73643-73646,
declaration ds4.h:311)

What I ran: the predicate is the switch every surface above goes through, so it
is verified by the live paths themselves (CLI, server, agent all took the qwen35
branch only if it returns true for this model; the DeepSeek fallback would have
rendered none of the ChatML text above). Symbol presence checked in the diff.
Observed: every check in this report that depends on the family mapping
  (ChatML render, server model list, agent system turn) behaved as the qwen35
  branch prescribes; the tokenizer A/B additionally shows the dispatch reached
  bpe_tokenize_text_qwen35. No separate runtime probe exists for a one-line
  metadata predicate.
Surface verdict: PASS (verified through all three live consumers)


SURFACE 5 - server wiring: server_model_syntax_for_engine
(ds4_server.c:1214-1219), server_model_id_from_engine (1223-1229),
server_model_alias_known (1233-1256), send_models (14949-14966)

What I ran: ./ds4-server -m MODEL --cuda -c 4096 --port 8899 (twice, once with
--trace /tmp/qa_C_trace.log) and:
  1. curl -s http://127.0.0.1:8899/v1/models
  2. curl -s http://127.0.0.1:8899/v1/models/prism-bonsai-2-27b
  3. POST /v1/chat/completions, system + user "What is the capital of France?",
     max_tokens 80, temperature 0 (non-streaming)
  4. the same body with "stream":true, raw SSE captured to a file and assembled
  5. France, then Japan, back to back on one server instance
  6. a continuation that replays the Japan reply and asks about France
  7. adversarial requests (SURFACE 6/E block below)
  8. kill of the server, then nvidia-smi

Observed (1): three models, each name "Prism Bonsai 2 27B":
    prism-bonsai-2-27b | Prism Bonsai 2 27B
    prism-bonsai-2-27b-chat | Prism Bonsai 2 27B
    prism-bonsai-2-27b-reasoner | Prism Bonsai 2 27B
Observed (2): 200, id prism-bonsai-2-27b, name "Prism Bonsai 2 27B",
  context_length 4096, supported_parameters [tools, tool_choice, max_tokens,
  temperature, top_p, top_k, min_p, ignore_eos, stop, seed, stream,
  reasoning_effort].
Observed (3): 200 in 47.8 s:
    content: "The capital of France is Paris."
    reasoning_content: "The user asks: \"What is the capital of France?\" This is a simple factual question. The answer is Paris. I should respond directly and concisely."
    usage: prompt_tokens 65, completion_tokens 43, total_tokens 108,
           prompt_tokens_details {cached_tokens 0, cache_write_tokens 65}
  The 65 equals the CLI structured rendering's token count for the same two
  messages (SURFACE 2), measured independently: CLI 65, server 65.
Observed (4): SSE stream, 100 lines; first chunks literal:
    data: {"id":"chatcmpl-59a8bfd9...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
    data: {...,"delta":{"reasoning_content":"T"},...}
    data: {...,"delta":{"content":"\n\n"},...}
    ...
    data: {...,"delta":{},"finish_reason":"stop"}
    data: [DONE]
  Assembled content: "\n\nThe capital of France is Paris."; assembled
  reasoning_content: "The user asks: \"What is the capital of France?\" This is a
  simple factual question. The answer is Paris. I should respond directly and
  concisely.\n". One role chunk, finish_reason "stop", [DONE] present.
Observed (5): France answered "The capital of France is Paris.", Japan answered
  "The capital of Japan is Tokyo." independently on the same instance. The log
  lines show a memory miss for both, with the shared prefix measured and the
  prefill starting at 0:
    0918 19:19:56 ds4-server: live kv cache miss live=108 prompt=65 common=65 vision=match reason=token-mismatch
    0918 19:19:56 ds4-server: chat ctx=0..65:65 prefill chunk 1/65 (1.5%) chunk=0.00 t/s avg=2.16 t/s 0.463s
    0918 19:20:25 ds4-server: chat ctx=0..65:65 prompt done 28.855s
    0918 19:20:44 ds4-server: live kv cache miss live=108 prompt=65 common=56 vision=match reason=token-mismatch
    0918 19:20:44 ds4-server: chat ctx=0..65:65 prefill chunk 1/65 (1.5%) chunk=0.00 t/s avg=2.23 t/s 0.448s
  So for two DIFFERENT prompts back to back the server reports the shared prefix
  (common=56 for Japan: the system turn plus "What is the capital of ") but does
  NOT reuse it; it prefills the full 65 tokens again. This is the designed
  condition, not a chat-syntax defect: reuse requires common == old_pos and
  prompt_len >= old_pos (ds4_server.c:11758-11763, trace_cache_memory_reusable),
  and the rewind path that could truncate a longer live checkpoint is gated to
  the GLM DSA family (ds4_server.c:11866-11870 live_prefix_rewind_target, called
  at 13388-13389 with ds4_engine_is_glm_dsa(s->engine)). The live checkpoint
  after an answer is 108 tokens (65 prompt + 43 generated), so a fresh 65-token
  prompt can never satisfy that condition. Prefix reuse for this family works
  where the protocol allows it, and I measured it (6):
Observed (6): continuation request (system + Japan + the Japan reply + "And
  France?"), 115 prompt tokens:
    live_tokens_before: 100
    live_prompt_common: 100
    memory_token_reusable: 1
    memory_miss_reason: live-prefix-match
    cache_source: memory-token
    cached_tokens: 100
    usage: prompt_tokens 115, completion_tokens 35, cached 100
    content: "The capital of France is Paris."   (22.3 s vs 47.8 s full prefill)
Observed (8): both server processes were killed by PID; pgrep -x ds4-server is
  empty and nvidia-smi --query-compute-apps returns nothing.
Surface verdict: PASS (model ids, aliases, syntax selection and the model list
  are correct; request rendering, usage accounting and streaming are correct).
  The back-to-back-independent-prompts reuse expectation in the brief does not
  hold and cannot hold under the current server policy; see ANOMALIES 2.


SURFACE 6 - agent wiring: agent_tool_syntax_for_engine (ds4_agent.c:412-413) and
agent_worker_build_system_tokens (ds4_agent.c:5123-5134)

What I ran (live, one non-interactive turn, trace on):
  timeout 900 ./ds4-agent -m MODEL --cuda --non-interactive -n 12 --temp 0 \
      -p "Reply with exactly: hello" --trace /tmp/qa_agent_trace.log
  then extracted the prompt tokens from the trace.

Observed: the run finished (exit 0, 640 s, GPU empty after). The trace's first
  tokens are the effort system turn the unit wires in, followed by the agent's
  own system prompt, all as ChatML:
    <|im_start|> system \n Reason|ing| effort is set to x|high|. Please ... final
    answer . <|im_end|> \n <|im_start|> system \n You are a coding agent running
    in a local workspace . Use tools for local ...
  (token indices 0..41 are the effort turn; the effort sentence occupies 3..39;
  prompt = 1406 tokens; the model then started a reply that -n 12 cut short, and
  the agent reported no error.)
Surface verdict: PASS (the agent builds the qwen35 ChatML system turn live; the
  tool syntax constant is AGENT_TOOL_SYNTAX_QWEN for this engine, which is what
  the ChatML <tool_call> rendering used by the same trace's tools system turn
  requires). One anomaly on this path is recorded in ANOMALIES 3.


REGRESSIONS (all run after the unit was already in the tree, sequential)

  1. make pq2-0-test
     -> "pq2_0: all checks passed (6 reference blocks, 34 bytes/block, 2.125 bpw)"
  2. make test-qwen35-cuda
     -> exit 0; last line "PQ2_0 CUDA parity: PASS"; all kernel cases PASS
        (fold round-trip max_abs=4.32e-07, both gates of the shared norm kernel,
        MMQ/MMVQ rel_l2 ~3-4e-3 vs tol 0.05, exact-row cases 0 failures).
  3. make bonsai-fold-selftest
     -> "fold selftest: blocks 2, 4 and 1024 match the explicit Hadamard matrix,
        blocks stay independent, forward/inverse round-trips, and the gdn
        permutation follows the tiled-to-grouped index map"
  4. make bonsai-ref-check
     -> tokens 5..16: 11751 13 198 760 6511 314 9564 369 19241 13 198 760
        (identical to the reference continuation above)
  5. make test-qwen35-session DS4_TEST_MODEL=/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf
     -> exit 0. The log carries 10 individual PASS lines plus the summary line
        "qwen35 session path: PASS" (11 PASS-bearing lines, 0 FAIL):
        plain position/size/ids, prefix reuse position/ids, rewind position/replay,
        invalidate empty/rebuild, context bound. Note: the brief said 12 PASS
        lines; what the suite prints is 10 checks plus its one-line summary.
  6. ./run-bonsai.sh compare
     -> CUDA 13.32 s / CPU 107.86 s; both continuations identical; final
        "IDENTICAL: all 24 generated token ids agree".
  7. ./run-bonsai.sh session
     -> CUDA session 12.96 s / CPU reference 107.46 s; final
        "IDENTICAL: all 24 generated token ids agree, so the session path
        reproduces the CPU reference on this prompt".
  8. touch ds4.c && make ds4.o CUDA_ARCH=native 2>&1 | grep -c warning
     -> 0


ADVERSARIAL CHECKS (trying to falsify the unit)

E1 foreign model ids. POST with "model":"deepseek-v4-flash" and with
  "model":"qwen3.8-flash-next": both returned HTTP 200 in 47.8 s, the response
  echoes the requested id in "model", the content and reasoning_content are the
  same correct answer as with the native id, usage prompt_tokens 65. The server
  does not validate the model id on this route; nothing was served from another
  family (the engine is the loaded Bonsai model and the ChatML syntax applied,
  as the identical 65-token rendering shows).
E2 chat_template_kwargs enable_thinking false. HTTP 200; usage prompt_tokens 29
  (the template's thinking-off rendering, 10+12+7 tokens); reasoning_content
  null; content "The capital of France is **Paris**."; the trace's rendered
  prompt is the thinking-off form (system turn with no effort sentence,
  assistant prefix "<think>" ... "</think>"). Token count and text both match
  the CLI --nothink tokenization of SURFACE 2.
E3 tools supplied. POST with one OpenAI-style function schema: HTTP 200,
  finish_reason "tool_calls", tool_calls
  [{"id":"call_e4b9a5fa...","type":"function","function":{"name":"get_capital",
  "arguments":"{\"country\":\"France\"}"}}], reasoning_content present, content
  empty, usage prompt_tokens 323. The trace shows the ChatML tools block
  rendered inside the system turn exactly as the template specifies:
    <|im_start|>system
    Reasoning effort is set to xhigh. ... final answer.

    # Tools

    You have access to the following functions:

    <tools>
    {"type": "function", "function": {"name": "get_capital", ...}}
    </tools>

    If you choose to call a function ONLY reply in the following format with NO suffix:
    ... <IMPORTANT> ... </IMPORTANT>

    You are a helpful assistant<|im_end|>
E4 reasoning split with thinking on. Non-streaming France: content has no
  <think> or </think>; the trace's parsed message shows finish stop, generated
  tokens 43, and the reasoning separated from content. CLI run B likewise printed
  reasoning then the answer with no tags. Streaming carries reasoning only in
  reasoning_content deltas. No raw think marker leaked into content in any
  response tested.
E5 template variants. reasoning_effort "low" -> HTTP 200, prompt_tokens 53,
  trace shows the template's low sentence (see SURFACE 2). The default and high
  paths emit the xhigh sentence. Medium adds no instruction by design (code
  path ds4_qwen4_reasoning_effort_text returns NULL; not exercised live because
  no client flag reaches DS4_THINK_MEDIUM for this family).


ANOMALIES (reported, none of them introduced by a line this unit changes)

1. Streaming and non-streaming disagree on trimming at the think boundary. The
   same request returned content "The capital of France is Paris." without
   leading whitespace when non-streaming, but the streaming path's first content
   delta is "\n\n" and its reasoning_content keeps a trailing "\n". The
   splitter/trim code is not touched by this unit (the diff touches only the
   family selection), and both paths still separate reasoning from content
   correctly. Small, cosmetic, recorded so it is not lost.
2. Two different prompts back to back do not reuse the live KV; the second is
   prefilled in full (quoted in SURFACE 5). The brief expected the second request
   to prefill only its new tail over the shared system turn. The observed
   behaviour is what the server code prescribes for non-GLM families (reuse only
   when the incoming prompt extends the live checkpoint; CUDA has no rewind).
   Reuse itself works and was measured with a continuation (cached_tokens 100).
   This is a property of the server's cache policy for this family, not of the
   tokenizer or ChatML rendering the unit adds, and it makes long chats on this
   family pay a full re-prefill whenever the client does not replay the
   previous answer. Worth a follow-up unit; it is not a reason to fail this one.
3. The agent could not persist its system-prompt KV:
   "ds4-agent: failed to save system prompt KV: unsupported routed quantization
   for KV save" (ds4_agent.c:5392; the error text is produced at ds4_agent.c:5019).
   Observed on the live agent run above; the unit's two agent lines are the
   syntax selection and the system-turn construction and neither touches the KV
   store. Preexistence was NOT verified against the pre-unit tree (an agent
   rebuild plus a second 10-minute run was out of proportion for this check).
4. The pre-change capture /tmp/e2-tokenizer-before.txt mislabels its first line:
   the ids it records belong to "The capital of France is 12,345.678 %", not to
   "The capital of France is 12,345." It is a record-keeping slip in the capture,
   not a tokenizer difference; both binaries agree on both strings.


COULD NOT VERIFY (accepted risks)

  - The DeepSeek (DSML/joyai-llm) and GLM families' rendering after the shared
    renderer refactor: no model of those families is present, and the brief
    forbids running llama.cpp, so their live rendering could not be exercised.
    Mitigation in the diff itself: their encode_chat_prompt and
    chat_push_think_prefix branches are unchanged (the conditions only gained an
    OR), the renamed chatml_* helpers are called only from the qwen4/qwen35
    branch, and the effort text function's HIGH/MAX and LOW cases are the same
    strings under new names. Residual risk: a DeepSeek/GLM regression is
    possible but not plausible from this diff; it must be covered by the next
    session that runs those models.
  - The qwen4 (Qwen3.8) family itself: the condition now reads
    ds4_model_is_qwen4() || ds4_model_is_qwen35(), so qwen4 takes exactly the
    same helper calls it took before the rename, but no qwen4 model is installed
    to run live. Same mitigation as above.
  - The multimodal appender (ds4_chat_append_multimodal_message) for this
    family: it needs a vision encoder and image inputs for a Bonsai model; the
    chat path was code-reviewed only (ds4.c:73938-73963 now selects the chatml_*
    helpers for qwen35), and the text/multimodal split was verified in the
    non-mm path.
  - Server disk KV checkpoints: the default min_tokens is 512 and no
    --kv-disk-dir was used, so disk checkpointing was not exercised; only the
    in-memory reuse path was (measured above).
  - The agent's tool-call round trip for this family (AGENT_TOOL_SYNTAX_QWEN)
    was verified at the prompt level (the system turn and the tool syntax in the
    trace) but no real tool was executed inside the agent; the same ChatML
    <tool_call> block was exercised live through the server (E3).
  - Fixing any of the anomalies is explicitly out of scope for this QA run.


verdict: overall PASS
