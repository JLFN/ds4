#!/bin/bash
# Start Laguna S 2.1 (community CRACK Q4_K_M export) with the ds4-server
# CUDA path on the GX10 / DGX Spark.
#
# This is the ds4 engine, not llama-server. The CRACK mixed layout (Q4_K
# embedding and dense/routed gate/up, Q8_0 attention and shared experts,
# mixed Q4_K/Q6_K routed down, Q6_K output, 1M-context YaRN rope) is
# accepted on CUDA by the laguna-crack branch built into this tree.
#
# Context: the deployed GGUF carries 1048576 as its context_length and the
# engine adopts the export's rope config, so -c may go up to 1M; KV is cheap
# because 36 of 48 layers keep a 512-token sliding window (about 12 GiB at
# 262144 tokens, about 48 GiB at 1M). Start at 262144 and raise only if the
# free memory allows: the model itself needs 67.75 GiB resident.
#
# Usage: ./start-laguna-crack-ds4.sh [start|stop|status]
set -euo pipefail

MODEL="${LAGUNA_CRACK_MODEL:-/home/leandro/models/Laguna-S-2.1-CRACK-Q4_K_M.gguf}"
SERVER="${DS4_SERVER:-/home/leandro/ds4-laguna-crack/ds4-server}"
CTX="${LAGUNA_CTX:-262144}"
PORT="${LAGUNA_PORT:-8002}"
LOG="${LAGUNA_LOG:-/tmp/laguna-crack-ds4.log}"
PIDFILE="${LAGUNA_PIDFILE:-/tmp/laguna-crack-ds4.pid}"
DFLASH="${LAGUNA_DFLASH:-}"

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "running pid=$(cat "$PIDFILE") port=$PORT"
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
    EXTRA=()
    if [ -n "$DFLASH" ] && [ -f "$DFLASH" ]; then
        EXTRA+=(--dflash "$DFLASH" --dflash-draft 15)
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
*) echo "usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
