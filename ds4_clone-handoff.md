---
project: ds4-laguna-crack
plan_start_commit: 448d5695d1c86401a4e9447c440feb983b73e6de
last_updated_commit: 2b66121a84e8c1833f0fe5b05c96ddd54b87ff3c
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

- Work tree: /data/ds4_clone (branch laguna-crack, clean, 4 commits on
  top of the branch point 448d569).
- The CRACK mixed-quant CUDA support is implemented, built, unit-verified
  and copied to the GX10 box as ~/ds4-laguna-crack (git history included,
  HEAD f45d5f4; verified over the sftp mount 2026-09-14T18:13).
- Unit 1 (commits 6670ce7, df2b7ff, b2bcc7d, f45d5f4):
  - ds4.c: metadata-driven rope adoption (context_length == orig_ctx *
    factor, bounded factor/attn checks) and the new mixed-layout
    validation (Q6_K embedding marker; attn/shexp family probing; Q4_K or
    Q6_K dense gate/up; Q4_K/Q3_K/Q2_K/Q6_K routed gate/up with coherent
    gate/up and Q4_K-or-Q6_K down).
  - ds4_cuda.cu: cuda_block_q6_K + assert; dev_q4_K_value/dev_q6_K_value/
    dev_dot_q6_K_q8_K_block; Q4_K/Q6_K token embedding kernels + dispatch;
    laguna_moe_gate_up_q6K_kernel + laguna_moe_down_q6K_sum_kernel;
    q34_batch takes down_type; laguna_matmul_q4_K_f32/_q6_K_f32 kernels
    with dequant+GEMM path (cuda_laguna_matmul_k_tensor), wired into
    ds4_gpu_matmul_q6_K_tensor and matmul_quant.
  - tests/test_laguna_crack_kernels.c + make target: 18/18 PASS on the
    RTX 4070 (sm_89), worst=0.000 of tolerance (bit-exact vs CPU refs).
- Pre-verified for the DGX: full sm_121 build (make cuda CUDA_ARCH=sm_121)
  links all five binaries cleanly in a scratch copy.
- Model facts (from GGUF headers, both saved in /data/ds4_crack_refs/):
  Q4_K_M 72,748,281,728 B (on the box, loads past every gate; local run
  stops only on the 4070's 12 GB), Q6_K 97,346,154,368 B, Q2_K
  45,099,593,600 B.
- KV sizing (per ds4.c:47946): 36/48 layers keep a 512-token sliding
  window, so KV = 12.07 GiB at 262144 ctx, 48.07 GiB at 1048576.
- /data/ds4 (original) untouched: main @ a04f46f.

3. Next unit plan — DGX verification run (2026-09-14)

Goal: prove the CRACK Q4_K_M file generates correct text on the GB10 and
leave a working server. Status: not yet started; the tree is already on
the box at ~/ds4-laguna-crack (all phases below run ON the box).

Ground truth first (reference-implementation rule): the user's llama.cpp
on the box runs this exact file (engine per ~/models/README.md is
mainline llama.cpp; /data/DGX_lagunaS21/LAGUNA_GX10_SETUP.md says the
DFlash draft needs poolside's fork — verify which build is current
before relying on either as the reference).

Phase A — build on the box
  cd ~/ds4-laguna-crack
  make cuda-spark            # GB10; forces a clean rebuild, no -arch
  Expected: ds4, ds4-server, ds4-agent, ds4-bench, ds4-eval built; the
  Makefile's CUDA_HOME default (/usr/local/cuda) matches the box.
  Acceptance: all five binaries exist; `./ds4 --help` prints.

Phase B — kernel test + load check
  make test-laguna-crack-kernels     # expect 18/18 PASS, 0 failures
  ./ds4 --inspect -m /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf
  Expected: exit 0, arch laguna, 814 tensors, types f32/q8_0/q4_k/q6_k,
  file size 67.75 GiB. (Same output already verified from the workstation
  over the mount; this re-verifies the aarch64 build's own binary.)

Phase C — generation smoke
  ./ds4 --cuda -m /home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf \
    -c 32768 -p "Write a haiku about mountains." -n 32
  Acceptance: coherent output (not loops/garbage — the Q6_K-misalignment
  failure mode is degenerate repetition, so read the text, do not just
  check exit status). Compare against llama.cpp on the same prompt if in
  doubt. Watch stderr for the "Laguna GPU graph: ctx=... KV ..." line.
  If it OOMs at 262144, retry at 32768 first (12 GB is enough for the
  graph at that ctx; the weights still need 67.75 GiB resident).

Phase D — serve
  ./start-laguna-crack-ds4.sh start   # ds4-server --cuda on :8002
  curl -s http://127.0.0.1:8002/v1/models    # alias laguna-s-2.1
  Compare with the workstation's open-grok model entry
  laguna-s-2-1-crack (base_url http://192.168.1.91:8002/v1, currently
  served by llama.cpp; switching it to this server is the end goal).
  Note: the old start-ds4-laguna.sh in ~ uses llama.cpp flags; it does
  not apply to ds4-server. Do not use it.

Phase E — record and close
  Record evidence (numbers, sample output) in the commit that closes the
  unit ("Unit: 2 complete" trailer), update this handoff's section 2, and
  if the box is reachable from open-grok, note the swap in memory.

Notes/decisions:
- speed numbers for the CRACK file on GB10 are unknown; capture prefill
  and decode t/s from the server log or ds4-bench as part of phase C/D.
- DFlash: the draft file laguna-s-2.1-DFlash-Q4_K.gguf is NOT on the box
  (checked 2026-09-14). If wanted, fetch poolside's Q8_0 draft via the
  branch's ./download_model.sh laguna-dflash (antirez repo) — the ds4
  DFlash needs the antirez Q8_0 file, not the Myric Q4_K one.
- Q6_K is untested end-to-end (only its kernels are covered); it fits
  the box (90.66 GiB + 12.07 KV at 262144) if the user wants it later.

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
