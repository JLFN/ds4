#!/usr/bin/env bash
# tests/run.sh — the ds4 suite entry point.  Runs the unit tests that need no
# model first, then the model-backed reference checks when the Bonsai GGUF is
# present, then the rule 19 QA gate (tests/qa-gate.sh), which is what fails the
# run until fresh QA evidence exists for everything the branch added.
#
# The CUDA targets need the CUDA toolchain and a GPU: pass CUDA_ARCH=native,
# or set DS4_SKIP_CUDA=1 to run only the CPU-side checks on a machine without
# one.  Nothing here calls llama.cpp: the CPU reference in ds4 is the oracle.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL=${DS4_BONSAI_MODEL:-/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}
CUDA_ARCH=${CUDA_ARCH:-native}
rc=0

step() {
  echo
  echo "== $* =="
}

run() {
  local what=$1; shift
  if "$@"; then
    echo "OK   $what"
  else
    echo "FAIL $what"
    rc=1
  fi
}

step "PQ2_0 block format (CPU)"
run "pq2-0-test" make pq2-0-test

if [[ ${DS4_SKIP_CUDA:-0} != 1 ]]; then
  step "PQ2_0 + fold kernels against the CPU reference (CUDA)"
  run "test-qwen35-cuda" make test-qwen35-cuda CUDA_ARCH="$CUDA_ARCH"
fi

if [[ -f "$MODEL" ]]; then
  step "Bonsai reference checks (model: $MODEL)"
  run "bonsai-fold-selftest" make bonsai-fold-selftest
  run "bonsai-ref-check" make bonsai-ref-check
  if [[ ${DS4_SKIP_CUDA:-0} != 1 ]]; then
    step "Bonsai session path against the CPU reference (CUDA)"
    run "test-qwen35-session" make test-qwen35-session \
      DS4_TEST_MODEL="$MODEL" CUDA_ARCH="$CUDA_ARCH"
  fi
else
  echo
  echo "SKIP model-backed reference checks ($MODEL not present)"
fi

step "Rule 19 QA gate (QA_MODEL=${QA_MODEL:-unset})"
run "qa-gate" bash tests/qa-gate.sh

echo
if (( rc == 0 )); then
  echo "tests/run.sh: overall PASS"
else
  echo "tests/run.sh: FAILURES above"
fi
exit $rc
