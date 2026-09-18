#!/usr/bin/env bash
# tests/qa-gate.sh — rule 19 guardrail mount for ds4 (canonical generic source:
# ~/.opengrok/templates/qa-gate.sh). Enforces: before a unit is committed for
# delivery or pushed, an AI QA-tester — the model is chosen per project with
# QA_MODEL and is never hardcoded — must verify every surface the unit added
# and write a FRESH qa-evidence/qa-report.md whose LAST non-empty line is
# "verdict: overall PASS".  This mount fails the local gate (exit 1) until
# that holds and SKIPS in real CI, where the QA subagent cannot run.
#
# ds4's surfaces are CUDA host entries and device kernels, not DB functions,
# API endpoints or client methods, so the surface list is the set of function
# DEFINITIONS the diff added under the family's files; a report may also
# satisfy the gate with the blanket token "ALL NEW SURFACES" that rule 19
# allows.
#
# Env knobs:
#   QA_BASE     ref the unit is diffed against (default: local dev, which is
#               the integration point for this repo; origin is upstream and
#               read-only, so it is never the base here)
#   QA_MODEL    QA-tester model name, shown in messages and required to be set
#   QA_REPORT   report path (default qa-evidence/qa-report.md)
#   QA_SURFACE_PATHS  override the "path:type" list (type: fn)
cd "$(dirname "$0")/.." || exit 1

if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  echo "SKIP tests/qa-gate.sh (AI QA-tester is a local-session step, not provisioned in CI)"
  exit 0
fi

BASE=${QA_BASE:-dev}
MODEL=${QA_MODEL:-""}
REPORT=${QA_REPORT:-qa-evidence/qa-report.md}

if [[ -z "$MODEL" ]]; then
  echo "QA GATE: FAIL (QA_MODEL is not set; rule 19 requires the per-project QA model to be named)"
  exit 1
fi

BASE_SHA=$(git rev-parse --verify --quiet "${BASE}" || echo "")
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
if [[ -z "$BASE_SHA" ]]; then
  echo "QA GATE: FAIL (base '${BASE}' does not resolve; set QA_BASE to the integration ref)"
  exit 1
fi
if [[ "$BASE_SHA" == "$HEAD_SHA" ]]; then
  echo "QA GATE: ALL PASS (no commits ahead of ${BASE} yet; nothing new to QA-tester)"
  exit 0
fi

# Files that carry the engine's own surfaces: test files are the verification
# instrument, not a surface, and are scanned by their own targets.  Static
# helpers are internal and are verified through their callers.
FILES=$(git diff --name-only "${BASE_SHA}" "${HEAD_SHA}" | while read -r f; do
  [[ "$f" == tests/* ]] && continue
  git cat-file -e "${HEAD_SHA}:${f}" 2>/dev/null && [[ "$f" =~ \.(c|cu|cuh|h)$ ]] && echo "$f"
done)

# Surfaces the unit added: non-static C entry points (the exported ds4_gpu_*
# and ds4_test_* ABI) and the CUDA device kernels those entries launch.
added=$(git diff "${BASE_SHA}" "${HEAD_SHA}" -- $FILES 2>/dev/null | grep -E '^\+' | sed -E 's/^\+//')
surfaces=$(
  {
    grep -E '^(extern "C" )?(int|void|bool)[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*\(' <<< "$added" \
      | sed -E 's/^(extern "C" )?(int|void|bool)[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*)\(.*/\3/'
    grep -E '^__global__ (static )?void [a-zA-Z_][a-zA-Z0-9_]*\(' <<< "$added" \
      | sed -E 's/^__global__ (static )?void ([a-zA-Z_][a-zA-Z0-9_]*)\(.*/\2/'
  } | grep -vE '^(if|for|while|switch|return|sizeof)$' | sort -u)

if [[ -z "$surfaces" ]]; then
  echo "QA GATE: ALL PASS (no new function definitions relative to ${BASE}; nothing to QA-tester)"
  exit 0
fi

echo "New surfaces to be QA'd (${MODEL}):"
echo "$surfaces" | sed 's/^/  /'

failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures+1)); fi; }

check "QA report exists ($(basename "$REPORT"))" "[[ -f '$REPORT' ]]"
base_sec=$(git log -1 --format=%ct "$BASE_SHA" 2>/dev/null || echo 0)
check "QA report is fresh (>= ${BASE} commit)" \
  "[[ -f '$REPORT' ]] && [[ '$(stat -c %Y "$REPORT" 2>/dev/null || echo 0)' -ge '$base_sec' ]]"

# The operative verdict is the report's LAST non-empty line and nothing else:
# a report whose last line reads "verdict: overall FAIL" stays red, and a fenced
# block appended after the verdict cannot become the operative one.
last_verdict=$(grep -v '^[[:space:]]*$' "$REPORT" 2>/dev/null | tail -1 | tr 'A-Z' 'a-z' | tr -s ' ' | sed 's/[[:space:]]*$//')
check "operative verdict is PASS (last non-empty line: ${last_verdict:-none})" \
  "[[ '$last_verdict' == 'verdict: overall pass' ]]"

blanket=0
grep -qF 'ALL NEW SURFACES' "$REPORT" 2>/dev/null && blanket=1
if (( blanket == 1 )); then
  echo "PASS  report uses the blanket token ALL NEW SURFACES (covers the list above)"
else
  while IFS= read -r surf; do
    [[ -z "$surf" ]] && continue
    check "report covers surface: $surf" "grep -qF '$surf' '$REPORT'"
  done <<< "$surfaces"
fi

if (( failures == 0 )); then
  echo "QA GATE: overall PASS (${MODEL} evidence fresh and covering)"
  exit 0
else
  echo "QA GATE: FAIL (${failures} check(s) red). Run the AI QA-tester on ${MODEL},"
  echo "have it verify the surfaces above live, and update $REPORT with"
  echo "'verdict: overall PASS' before committing or pushing."
  exit 1
fi
