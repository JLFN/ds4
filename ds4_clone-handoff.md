---
project: ds4-laguna-crack
plan_start_commit: 448d5695d1c86401a4e9447c440feb983b73e6de
last_updated_commit: 0285c6328b76ca8d1627a6ac9116bc9c13443014
branch: laguna-crack
remote: https://github.com/antirez/ds4.git
handoff_written_at_context_usage: 20%
handoff_written_at: 2026-09-14T18:30:00+02:00
session_id: 01a0a01d-70db-7a83-a72c-05cc17e69943
---

Handoff + executable plan — ds4-laguna-crack (2026-09-14)

0. Rules for writing THIS document (read before editing)

- Only write or replace this file at a clean, verified, committed boundary.
- Every claim in "Current state" must be re-verified in the writing session.
- Hard caps: Current state 15 lines, Guardrails/open items 15 lines.
- History is absent by design: git log is the record.

1. Session protocol

This project is a sequence of units; each unit ends with a verified, pushed
commit set.

1. Read context usage from the session's signals.json under the open-grok
   home (sessions/<percent-encoded-cwd>/<session-id>/), newest session dir
   for this cwd.
2. If usage < 40%: continue immediately — write the next unit's plan into
   section 3, execute it, verify, commit, re-check, repeat.
3. At ~40%: do not pause or ask. Finish this handoff, then hand off.
   mid-unit, record exact state in section 2 and the open items.

2. Current state (2026-09-14 — COMPLETE, verified)

- Work tree: /data/ds4_clone (branch laguna-crack, clean, 6 commits on
  top of the branch point 448d569).
- The CRACK mixed-quant CUDA support is implemented, built, unit-verified
  and copied to the GX10 box as ~/ds4-laguna-crack (HEAD 0285c63).
- FIRST REAL RUN SUCCEEDED on the box (2026-09-14): build via
  make cuda-spark, then a 49-token prompt generated a coherent haiku at
  prefill 41.17 t/s / generation 22.05 t/s on the GB10 (sm_121), with
  KV 1.57 GiB and scratch 5862 MiB at ctx 32768. So the port works on
  real weights.
- Unit 2 (commit 0285c63) fixed the long-prefill bug the first server
  run exposed: a 51415-token prompt died after 1/48 layers with "CUDA
  Laguna routed MoE intermediate quantize launch failed: invalid
  argument". Cause: the routed MoE quantizes n_tokens * n_expert rows
  and the Q6_K gate/up launches one block per pair, both on blockIdx.y,
  which CUDA caps at 65535 (16384-token chunk x 10 experts = 163840).
  Fix: the three q8_K quantizers and the Q6_K gate/up kernel fold
  blockIdx.z into the row/pair index, with q8_K_row_grid()/launch
  helpers. Prompts longer than ~6553 tokens were affected; short ones
  were not, which is why the smoke test passed.
- Verification: make test-laguna-crack-kernels now 27/27 PASS including
  three large-batch cases at pair_count=80000 (past the cap, bit-exact
  vs CPU refs); make q4k-dot-test 4/4; all five binaries build.
- Model facts (GGUF headers, inventories in /data/ds4_crack_refs/):
  Q4_K_M 72,748,281,728 B (the file on the box), Q6_K 97,346,154,368 B,
  Q2_K 45,099,593,600 B.
- KV sizing (ds4.c:47946): 36/48 layers keep a 512-token window, so KV
  = 1.57 GiB at 32768 ctx and 48.07 GiB at 1048576. Measured on the box:
  32768 ctx plans 69.32 GiB total (67.75 model + 1.57 KV) plus 5.86 GiB
  scratch. One caveat: the engine allocates the graph and the server may
  also hold its own copy, so 1M needs about 121.6 GiB and does not fit
  the box's ~115-118 GiB usable; the practical ceiling is roughly 800k
  context. Recommended default 262144.
- /data/ds4 (original) untouched: main @ a04f46f.

3. Next unit plan — long-context confirmation and speed (2026-09-14)

Goal: confirm the grid fix on the box with the exact prompt that failed,
then decide the context default and the speed path.

Phase A — rebuild and unit-test on the box
  cd ~/ds4-laguna-crack
  make cuda-spark
  make test-laguna-crack-kernels        # expect 27/27 PASS
  Acceptance: zero failures; the three "large batch ... pair_count=80000"
  lines all print ok.

Phase B — the regression that mattered
  Stop any running server, then rerun the request that failed: a long
  prompt (>6553 tokens; the failure was at 51415). Watch for "routed MoE
  intermediate quantize launch failed" — it must NOT appear, and prefill
  must complete past layer 1.
  Acceptance: the server answers; the log shows no "failed in routed
  experts".

Phase C — decide the context default
  The CRACK export declares context_length 1048576 and the engine adopts
  it, so -c up to 1048576 is accepted. Measured ceiling on the box: the
  model is 67.75 GiB resident, scratch is fixed at ~5.9 GiB (prefill cap
  16384), and KV costs 49,152 bytes per token (12 layers full + 36 at a
  512 window). 1M needs ~121.6 GiB and does not fit ~115-118 GiB usable;
  about 800k is the practical ceiling. Test upward (e.g. 262144 -> 524288
  -> 786432) and keep the highest that starts cleanly; default 262144.
  Note: prefill is chunked at 16384, so use the server for 1M-class
  prompts rather than one huge CLI prompt.

Phase D — speed (measure before changing code)
  Known levers, in order:
  1. DFlash speculative decoding: fetch poolside's draft with
     ./download_model.sh laguna-dflash (laguna-s-2.1-DFlash-Q8_0.gguf,
     1.11 GiB, verified present in the antirez Laguna repo), then pass
     --dflash <file> (CUDA default --dflash-draft 15). This is the main
     decode lever and is untested here.
  2. Prefill: the review found the mixed-down layers (23 of 47 in the
     Q4_K_M file) pay the full tiled expert setup in prefill and then
     re-run the down projection per token; and laguna_routed_moe_tc_prefill
     only accepts Q2_K/Q3_K, so the CRACK files never use the tensor-core
     path. Both are code changes in ds4_cuda.cu.
  Measure with ds4-bench (--ctx-start/--ctx-max/--gen-tokens, --csv) and
  record numbers in the closing commit.

Notes/decisions:
- Do NOT pass --prefill-chunk or --power below 100 for Laguna: the engine
  rejects them ("standard local graph path only"). SSD streaming is also
  rejected for Laguna. The only flag-level speed lever is DFlash.
- The Q6_K CRACK export (90.66 GiB) fits only at small context now that
  the real KV math is known; Q4_K_M is the right file for this box.

4. Project facts a fresh session cannot re-derive

- Clone path (workstation): /data/ds4_clone, branch laguna-crack; the
  original /data/ds4 is untouched on main.
- Box copy: ~/ds4-laguna-crack on gx10-e1d2 (192.168.1.91), reachable
  from the workstation at /run/user/1000/gvfs/sftp:host=192.168.1.91,
  user=leandro/home/leandro/ (gvfs mount, read-write; SSH key auth was
  refused, this mount is the only channel).
- Reference material: /data/ds4_crack_refs/ — crack_tensors.txt and
  q6_tensors.txt (full tensor inventories), pr594_* (upstream open CUDA
  PR used as the implementation reference), pr633.diff (upstream mixed-
  quant validator + metadata rope), ds4-before (the pre-change binary),
  parse_gguf.py (header parser).
- Box model files: /home/leandro/models/ has Laguna-S-2.1-CRACK-Q4_K_M.gguf
  (used here), DeepSeek Flash files, and the DSpark support file. No
  DFlash draft, no Q6_K CRACK file.
- Model weights reference on the workstation: none local (the CRACK file
  lives only on the box; local tests use synthetic weights).
- GPU on the workstation: RTX 4070 SUPER, 12 GB — cannot run the real
  model, only the synthetic kernel tests.

5. Guardrails and open items

- Global rules apply (plain text/no emojis, conventional commits with ISO
  bodies, source-of-truth verification, this handoff protocol).
- The clone is derived from a fork/fork-like tree: do NOT push to
  antirez/ds4 (it is not our remote to write). origin is set to the
  upstream URL only so `git fetch` works; keep the branch local.
- Rule 12: a graphify graph exists at /data/ds4_clone/graphify-rs-out
  (AST-only, built 2026-09-14); rebuild at commit time when code changes.
- OPEN: independent code review of the diff (subagent 01a0a0b6, started
  2026-09-14 ~18:18, still running at handoff time) — read its findings
  and fix anything critical/major before trusting the port on the box.
- OPEN: the DGX verification run (section 3) — the only step that turns
  "loads and passes unit tests" into "generates correct text".
- OPEN: llama.cpp cross-check on identical prompts (the user's working
  reference for this file).

6. Progress and staleness check (run this first, every session)

Progress meter:
    cd /data/ds4_clone
    git log --oneline 448d569..HEAD
    git diff --stat 448d569..HEAD
    git log --grep "Unit: .* complete" 448d569..HEAD
Trust git over the plan headings: continue with the first unit whose work
is NOT in the diff.

Staleness check:
    cd /data/ds4_clone && git log --oneline f45d5f4..HEAD
Any output means this handoff predates newer commits — read them first
and update last_updated_commit.

Box-side staleness: the copy at ~/ds4-laguna-crack is a point-in-time
copy; re-run the rsync from /data/ds4_clone (or fetch the branch) after
any new commit here:
    rsync -a --exclude=graphify-rs-out/ /data/ds4_clone/ \
      /run/user/1000/gvfs/sftp:host=192.168.1.91,user=leandro/home/leandro/ds4-laguna-crack/
