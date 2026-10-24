#!/usr/bin/env bash
# Nemotron-3.5-Bolt-E4B-4GB (Q4_K_M) + DFlash drafter on Jetson Orin Nano Super (sm_87, 7.4 GiB unified).
# llama.cpp b11840 (36e36e63) at ~/llama.cpp-bolt/build-cuda.
# usage: bolt-serve.sh [dflash|plain] [extra llama-server args...]
set -euo pipefail
MODE="${1:-dflash}"; shift || true
M=/home/djl/models/bolt
export GGML_CUDA_NO_PINNED=1
ARGS=(-m "$M/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf" --alias nemotron-bolt
      -c "${BOLT_CTX:-32768}" -np 1 -ngl 99 -fa on --jinja
      -lm "${BOLT_LOADMODE:-mmap}" -lzm "${BOLT_LAZY:-on}"
      --host "${BOLT_HOST:-0.0.0.0}" --port "${BOLT_PORT:-8000}")
if [ "$MODE" = dflash ]; then
  ARGS+=(-md "$M/${BOLT_DRAFT:-Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf}" --spec-type draft-dflash
         --spec-draft-n-max "${BOLT_NMAX:-7}" --spec-draft-ngl 99)
fi
exec /home/djl/llama.cpp-bolt/build-cuda/bin/llama-server "${ARGS[@]}" "$@"
