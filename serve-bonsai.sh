#!/bin/bash
# serve-bonsai.sh — run ds4-server for Prism Bonsai 2 27B (qwen35 + PQ2_0) so an
# open-grok session can talk to it as a normal OpenAI-compatible model.
#
# The engine is the ds4 CUDA backend; the weights are copied into VRAM at
# startup when they fit (6.70 GiB on this host), which is what makes it fast.
# If another process is holding the card, the same binary keeps working over a
# mapped host image and is simply slower, so a start is never a failure on VRAM.
#
# Usage
#   ./serve-bonsai.sh start [--ctx N] [--port N] [--foreground]
#   ./serve-bonsai.sh stop
#   ./serve-bonsai.sh restart
#   ./serve-bonsai.sh status
#   ./serve-bonsai.sh logs [N]          last N log lines (default 20) and follow
#   ./serve-bonsai.sh models            the model ids the server advertises
#   ./serve-bonsai.sh url               base_url, model id, ready-to-run open-grok command
#   ./serve-bonsai.sh smoke [prompt]    one greedy chat request through the API
#   ./serve-bonsai.sh install           add the model block to ~/.opengrok/config.toml
#   ./serve-bonsai.sh help
#
# Env overrides
#   DS4_BONSAI_BIN    server binary   (default /data/ds4/ds4-server)
#   DS4_BONSAI_MODEL  GGUF path       (default /data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf)
#   DS4_BONSAI_PORT   port            (default 8899)
#   DS4_BONSAI_CTX    context tokens  (default 65536; see the sizing note below)
#   DS4_BONSAI_HOST   bind address    (default 127.0.0.1)
#   DS4_BONSAI_RUN    runtime dir     (default /tmp/ds4-bonsai; holds the pid and the log)
#   DS4_BONSAI_WAIT   startup timeout seconds (default 300)
#   DS4_BONSAI_TAG    config section / model key for `install` (default bonsai-local)
#
# Sizing the context (measured on the 12 GB RTX 4070 SUPER)
#   Two walls bracket it.  Below, open-grok's own system prompt is about 38k
#   tokens, so anything under ~49152 cannot fit one open-grok turn at all.  Above,
#   this family's graph allocates its fp16 k/v caches up front at roughly 64 KiB
#   per token of context (16 attention layers x 4 kv heads x 256 dims x 2 fp16),
#   so 65536 needs about 4 GiB of KV and a start measures 11637 of 12282 MiB;
#   131072 fails with "CUDA tensor alloc failed: out of memory".  65536 is
#   therefore the largest workable value on a free card and the default here.
#   If a start fails with that message, lower --ctx (49152 leaves about 1.3 GiB
#   of headroom) or free VRAM from the other session sharing this card.

set -u

BIN="${DS4_BONSAI_BIN:-/data/ds4/ds4-server}"
MODEL="${DS4_BONSAI_MODEL:-/data/models/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
PORT="${DS4_BONSAI_PORT:-8899}"
CTX="${DS4_BONSAI_CTX:-65536}"
HOST="${DS4_BONSAI_HOST:-127.0.0.1}"
RUN_DIR="${DS4_BONSAI_RUN:-/tmp/ds4-bonsai}"
WAIT="${DS4_BONSAI_WAIT:-300}"
TAG="${DS4_BONSAI_TAG:-bonsai-local}"

# The id ds4-server advertises for this family (server maps the qwen35 family to
# its own ChatML syntax and this name; see ds4_server.c and the handoff, E2a).
SERVED_ID="prism-bonsai-2-27b"
# A chat answer needs ~35 output tokens on this model: its reasoning block alone
# runs about 33 before any content appears.
SMOKE_TOKENS=48

PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
CONFIG="${OPENGROK_CONFIG:-$HOME/.opengrok/config.toml}"

say() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

pid_of() {
  # The recorded pid, but only while it is really a live ds4-server process.
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

listening() {
  curl -s -m 3 -o /dev/null "http://$HOST:$PORT/v1/models"
}

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -q ":$PORT "
  else
    curl -s -m 2 -o /dev/null "http://$HOST:$PORT/v1/models"
  fi
}

check_env() {
  [ -x "$BIN" ] || die "server binary not found or not executable: $BIN
       build it: cd /data/ds4 && make cuda-generic"
  [ -f "$MODEL" ] || die "model not found: $MODEL"
}

gpu_note() {
  command -v nvidia-smi >/dev/null 2>&1 || { say "gpu:     nvidia-smi not found"; return; }
  say "gpu:     $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
  # Our own server holds ~9 GiB by design; it is not a competing tenant.
  local self=""
  [ -f "$PID_FILE" ] && self=$(cat "$PID_FILE" 2>/dev/null)
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
         | { if [ -n "$self" ]; then grep -v "^$self,"; else cat; fi; } | head -3)
  if [ -n "$busy" ]; then
    say "         another process holds VRAM (the 12 GB card is shared):"
    say "$busy" | sed 's/^/           /'
    say "         a start with that tenant resident keeps the weights host-mapped:"
    say "         correct, but about 12x slower to decode."
  fi
}

cmd_start() {
  local foreground=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --foreground) foreground=1 ;;
      --ctx) shift; [ $# -gt 0 ] || die "--ctx needs a value"; CTX="$1" ;;
      --port) shift; [ $# -gt 0 ] || die "--port needs a value"; PORT="$1" ;;
      *) die "unknown start option: $1" ;;
    esac
    shift
  done

  check_env
  mkdir -p "$RUN_DIR"

  local running
  if running=$(pid_of); then
    if listening; then
      say "already running (pid $running) on http://$HOST:$PORT — nothing to do."
      say "use ./serve-bonsai.sh restart to reload, or stop to end it."
      return 0
    fi
    say "a recorded pid ($running) exists but the port is not answering; restarting."
    "$0" stop >/dev/null 2>&1 || true
  fi

  if port_busy; then
    die "port $PORT is already in use by something else (try --port N, or DS4_BONSAI_PORT=N)"
  fi

  say "model:   $MODEL"
  say "context: $CTX tokens"
  say "port:    http://$HOST:$PORT"
  gpu_note
  say ""

  : > "$LOG_FILE"
  if [ "$foreground" -eq 1 ]; then
    say "running in the foreground; logs also at $LOG_FILE"
    exec "$BIN" -m "$MODEL" --cuda --host "$HOST" --port "$PORT" -c "$CTX" 2>&1 | tee -a "$LOG_FILE"
    return
  fi

  nohup "$BIN" -m "$MODEL" --cuda --host "$HOST" --port "$PORT" -c "$CTX" >>"$LOG_FILE" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  local waited=0
  while [ "$waited" -lt "$WAIT" ]; do
    if grep -q "listening on" "$LOG_FILE" 2>/dev/null; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      say "the server exited before listening; last log lines:"
      tail -15 "$LOG_FILE" | sed 's/^/  /'
      rm -f "$PID_FILE"
      return 1
    fi
    if [ $((waited % 15)) -eq 0 ] && [ "$waited" -gt 0 ]; then
      say "  still loading... ${waited}s"
    fi
    sleep 1
    waited=$((waited + 1))
  done

  if ! grep -q "listening on" "$LOG_FILE" 2>/dev/null; then
    say "no listener after ${waited}s; last log lines:"
    tail -15 "$LOG_FILE" | sed 's/^/  /'
    return 1
  fi

  local resident
  resident=$(grep -c "chunk-copying" "$LOG_FILE" 2>/dev/null || printf 0)
  say "started (pid $pid) in ${waited}s"
  if [ "$resident" -gt 0 ]; then
    say "weights: in VRAM (fast path)"
    grep -oE "CUDA model chunk copy complete in [0-9.]+s" "$LOG_FILE" | sed 's/^/         /'
  else
    say "weights: host-mapped (slower path; see the VRAM note above)"
  fi
  say ""
  cmd_url
}

cmd_stop() {
  local pid
  if ! pid=$(pid_of); then
    say "not running (no live pid in $PID_FILE)"
    rm -f "$PID_FILE"
    return 0
  fi
  say "stopping pid $pid..."
  kill "$pid" 2>/dev/null
  local waited=0
  while [ "$waited" -lt 30 ] && kill -0 "$pid" 2>/dev/null; do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    say "still alive after ${waited}s; sending SIGKILL"
    kill -9 "$pid" 2>/dev/null
    sleep 1
  fi
  rm -f "$PID_FILE"
  say "stopped."
}

cmd_status() {
  say "binary:  $BIN"
  say "model:   $MODEL"
  local pid
  local running=0
  if pid=$(pid_of); then
    running=1
    say "server:  pid $pid"
    if listening; then
      say "http:    answering on http://$HOST:$PORT"
    else
      say "http:    not answering yet (still loading, or wedged)"
    fi
  else
    say "server:  not running"
  fi
  if [ -f "$LOG_FILE" ] && [ "$running" -eq 1 ]; then
    if grep -q "chunk-copying" "$LOG_FILE" 2>/dev/null; then
      say "weights: in VRAM (fast path)"
    elif grep -q "no-copy) registered" "$LOG_FILE" 2>/dev/null; then
      say "weights: host-mapped (slower path)"
    fi
  fi
  gpu_note
  say "log:     $LOG_FILE"
}

cmd_logs() {
  local n="${1:-20}"
  [ -f "$LOG_FILE" ] || die "no log yet at $LOG_FILE"
  say "--- last $n lines of $LOG_FILE (Ctrl-C to stop following) ---"
  tail -n "$n" -f "$LOG_FILE"
}

cmd_models() {
  listening || die "the server is not answering on http://$HOST:$PORT (start it first)"
  curl -s -m 20 "http://$HOST:$PORT/v1/models" | tr ',' '\n' | grep '"id"' | sed 's/.*"id"[: ]*//;s/"//g' | sed 's/^/  /'
}

cmd_url() {
  say "base_url:  http://$HOST:$PORT/v1"
  say "model id:  $SERVED_ID"
  local registered=0
  [ -f "$CONFIG" ] && grep -qE "^\[model\.(\"$TAG\"|$TAG)\]" "$CONFIG" && registered=1
  if [ "$registered" -eq 1 ]; then
    say "open-grok: open-grok -m $TAG      (registered in $CONFIG)"
  else
    say "open-grok: open-grok -m $TAG      (run ./serve-bonsai.sh install once first)"
  fi
  say "           or inside a session:  /model $TAG"
}

cmd_smoke() {
  local prompt="${1:-Reply with the single word OK.}"
  listening || die "the server is not answering on http://$HOST:$PORT (start it first)"
  say "prompt: $prompt"
  local t0 t1 wall body
  t0=$(date +%s.%N)
  body=$(curl -s -m 600 "http://$HOST:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$SMOKE_TOKENS,\"temperature\":0}")
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  printf '%s' "$body" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("raw response:", sys.stdin.read()[:400]); raise SystemExit(1)
if "error" in d:
    print("error:", json.dumps(d["error"])[:400]); raise SystemExit(1)
c = d["choices"][0]["message"]
print("content:  ", (c.get("content") or "").strip() or "(empty: raise max_tokens, the reasoning block eats about 33)")
rc = (c.get("reasoning_content") or "").strip()
if rc: print("reasoning:", rc[:160])
print("usage:    ", d.get("usage"))'
  say "wall:      ${wall}s (includes prefill; the prompt is short)"
}

cmd_install() {
  [ -f "$CONFIG" ] || die "no config at $CONFIG (set OPENGROK_CONFIG if it lives elsewhere)"
  if grep -qE "^\[model\.(\"$TAG\"|$TAG)\]" "$CONFIG"; then
    say "[model.$TAG] is already in $CONFIG — nothing to add."
    return 0
  fi
  local backup="$CONFIG.bak-$(date +%Y%m%d-%H%M%S)"
  cp -p "$CONFIG" "$backup" || die "could not back up $CONFIG"
  say "backed up: $backup"
  cat >> "$CONFIG" <<EOF

[model.$TAG]
model = "$SERVED_ID"
name = "Prism Bonsai 2 27B (ds4, CUDA)"
base_url = "http://$HOST:$PORT/v1"
api_backend = "chat_completions"
api_key = "dummy"
context_window = $CTX
max_completion_tokens = 4096
supports_images = false
EOF
  say "added [model.$TAG] to $CONFIG (context_window = $CTX, matching this script's --ctx)."
  say "if you change --ctx later, keep context_window in that block in step with it."
  say ""
  cmd_url
}

usage() {
  sed -n '2,41p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-start}" in
  start)   shift; cmd_start "$@" ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; say ""; cmd_start ;;
  status)  cmd_status ;;
  logs)    shift; cmd_logs "${1:-20}" ;;
  models)  cmd_models ;;
  url)     cmd_url ;;
  smoke)   shift; cmd_smoke "${*:-Reply with the single word OK.}" ;;
  install) cmd_install ;;
  help|-h|--help) usage ;;
  *) die "unknown command: $1 (try: $0 help)" ;;
esac
