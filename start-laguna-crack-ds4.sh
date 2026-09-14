#!/bin/bash
# Start Laguna S 2.1 (community CRACK Q4_K_M export) with the ds4-server
# CUDA path on the GX10 / DGX Spark.
#
# This is the ds4 engine, not llama-server. The CRACK mixed layout (Q4_K
# embedding and dense/routed gate/up, Q8_0 attention and shared experts,
# mixed Q4_K/Q6_K routed down, Q6_K output, 1M-context YaRN rope) is
# accepted on CUDA by the laguna-crack branch built into this tree.
#
# Context sizing (measured on this box, 2026-09-14):
#   - the model needs 67.75 GiB resident;
#   - the graph scratch is fixed at about 5.9 GiB (prefill cap 16384);
#   - KV costs 49,152 bytes per token: 36 of the 48 layers keep a
#     512-token sliding window, only 12 hold the full context, so
#     32768 tokens cost 1.57 GiB and 1048576 cost 48.07 GiB.
#   Total at 1M is about 121.6 GiB, more than this box can give, so 1M is
#   NOT reachable; roughly 800k is the ceiling. 262144 is the default and
#   the shipped configuration.
#
# The script refuses a context that cannot fit and says so, instead of
# letting the load fail with an allocation error. Override with
# LAGUNA_FORCE=1 if you want to try anyway.
#
# Usage: ./start-laguna-crack-ds4.sh [start|stop|status|plan]
set -euo pipefail

MODEL="${LAGUNA_CRACK_MODEL:-$HOME/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf}"
SERVER="${DS4_SERVER:-$(cd -- "$(dirname -- "$0")" && pwd)/ds4-server}"
CTX="${LAGUNA_CTX:-262144}"
PORT="${LAGUNA_PORT:-8002}"
LOG="${LAGUNA_LOG:-/tmp/laguna-crack-ds4.log}"
PIDFILE="${LAGUNA_PIDFILE:-/tmp/laguna-crack-ds4.pid}"
DFLASH="${LAGUNA_DFLASH:-}"
BUDGET_GIB="${LAGUNA_BUDGET_GIB:-115}"   # usable unified memory to plan against
FORCE="${LAGUNA_FORCE:-0}"

GIB=$((1024 * 1024 * 1024))

# KV bytes = 4096 * (12 * ctx + 36 * 512); see ds4.c:47946.
kv_gib() {
    python3 -c "print(f'{(4096 * (12 * int('$1') + 36 * 512)) / $GIB:.2f}')"
}

plan() {
    local model_gib scratch_gib kv total
    model_gib=$(python3 -c "import os;print(f'{os.path.getsize(\"$MODEL\") / $GIB:.2f}')")
    scratch_gib=5.9
    kv=$(kv_gib "$CTX")
    total=$(python3 -c "print(f'{float(\"$model_gib\") + float(\"$kv\") + $scratch_gib:.2f}')")
    printf 'model %s GiB + KV %s GiB (ctx %s) + scratch %s GiB = %s GiB (budget %s GiB)\n' \
        "$model_gib" "$kv" "$CTX" "$scratch_gib" "$total" "$BUDGET_GIB"
    python3 -c "import sys; sys.exit(0 if float('$total') <= float('$BUDGET_GIB') else 1)"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "running pid=$(cat "$PIDFILE") port=$PORT ctx=$CTX"
        curl -sS --max-time 3 "http://127.0.0.1:$PORT/v1/models" | head -c 300
        echo
        return 0
    fi
    echo "not running"
    return 1
}

stop() {
    if [ -f "$PIDFILE" ]; then
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
        echo "stopped"
    else
        echo "not running"
    fi
}

case "${1:-start}" in
start)
    if status >/dev/null 2>&1; then
        echo "already running (pid=$(cat "$PIDFILE"))"
        exit 0
    fi
    [ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
    [ -x "$SERVER" ] || { echo "server not found: $SERVER" >&2; exit 1; }
    echo "plan: $(plan || true)"
    if ! plan >/dev/null 2>&1; then
        echo "refusing: ctx $CTX does not fit ($BUDGET_GIB GiB budget)." >&2
        echo "1M is not reachable on this box; try up to ~800k, or raise" >&2
        echo "LAGUNA_BUDGET_GIB, or set LAGUNA_FORCE=1 to attempt it anyway." >&2
        [ "$FORCE" = "1" ] || exit 2
        echo "LAGUNA_FORCE=1: continuing anyway" >&2
    fi
    EXTRA=()
    if [ -n "$DFLASH" ] && [ -f "$DFLASH" ]; then
        EXTRA+=(--dflash "$DFLASH" --dflash-draft 15)
        echo "DFlash draft: $DFLASH"
    fi
    nohup "$SERVER" --cuda -m "$MODEL" -c "$CTX" \
        --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}" \
        >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "started pid=$(cat "$PIDFILE"); log $LOG"
    echo "watch: tail -f $LOG"
    ;;
stop) stop ;;
status) status ;;
plan) plan ;;
*) echo "usage: $0 [start|stop|status|plan]" >&2; exit 2 ;;
esac
