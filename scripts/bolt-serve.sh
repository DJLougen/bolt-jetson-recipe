#!/usr/bin/env bash
# Nemotron-3.5-Bolt-E4B-4GB (Q4_K_M) + DFlash drafter on 8 GB unified-memory Jetson (sm_87).
# usage: bolt-serve.sh [dflash|plain] [extra llama-server args...]
#
# Env overrides (defaults in parens):
#   BOLT_MODEL_DIR   where the GGUFs live        (~/models/bolt)
#   LLAMA_BIN        llama-server directory      (~/llama.cpp-bolt/build-cuda/bin)
#   BOLT_CTX         context size                (16384)
#   BOLT_DRAFT       drafter GGUF filename       (Nemotron-3.5-Bolt-E4B-DFlash-Q8_0.gguf)
#   BOLT_NMAX        max draft tokens            (7)
#   BOLT_HOST        bind address(es)            (127.0.0.1 -- do NOT use 0.0.0.0
#                                                 unless you add an API key)
#   BOLT_PORT        port                        (8000)
set -euo pipefail
MODE="${1:-dflash}"; shift || true
M="${BOLT_MODEL_DIR:-$HOME/models/bolt}"
BIN="${LLAMA_BIN:-$HOME/llama.cpp-bolt/build-cuda/bin}"
export GGML_CUDA_NO_PINNED=1
ARGS=(-m "$M/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf" --alias nemotron-bolt
      -c "${BOLT_CTX:-16384}" -np 1 -ngl 99 -fa on --jinja
      -lm "${BOLT_LOADMODE:-mmap}" -lzm "${BOLT_LAZY:-on}"
      --host "${BOLT_HOST:-127.0.0.1}" --port "${BOLT_PORT:-8000}")
if [ "$MODE" = dflash ]; then
  ARGS+=(-md "$M/${BOLT_DRAFT:-Nemotron-3.5-Bolt-E4B-DFlash-Q8_0.gguf}" --spec-type draft-dflash
         --spec-draft-n-max "${BOLT_NMAX:-7}" --spec-draft-ngl 99)
fi
exec "$BIN/llama-server" "${ARGS[@]}" "$@"
