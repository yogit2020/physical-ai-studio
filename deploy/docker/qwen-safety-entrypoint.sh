#!/bin/bash
# Start vLLM with the safety monitor's settings; extra arguments are passed through to vLLM.
set -eo pipefail
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" --dtype bfloat16 \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 1 --max-num-batched-tokens 1024 \
    --limit-mm-per-prompt '{"image": 2}' --mm-processor-kwargs '{"max_pixels": 200704}' \
    "$@"
