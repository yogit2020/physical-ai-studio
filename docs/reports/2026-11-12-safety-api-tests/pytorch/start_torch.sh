#!/bin/bash
# Pi0.5 210base 65ft, PyTorch export (exports/torch/pi05.pt) on B70 GPU 1 (card2/renderD129), XPU.
# Reachable only on 127.0.0.1:18767. Stop with: docker rm -f yogini-pi05-torch
docker rm -f yogini-pi05-torch >/dev/null 2>&1
docker run -d --name yogini-pi05-torch \
  --device /dev/dri/card2 --device /dev/dri/renderD129 --group-add 990 \
  -p 127.0.0.1:18767:18767 \
  -v /home/devcloud/yogini/pi05-210base-65ft/exports:/models/exports:ro \
  -v /home/devcloud/yogini/remote-inference/v2/remote_predictor.py:/app/application/backend/src/workers/remote_predictor.py:ro \
  -v /home/devcloud/yogini/remote-inference/v2/remote_policy_server.py:/srv/remote_policy_server.py:ro \
  -v /home/devcloud/yogini/bench/pg_cache:/hf-cache:ro -e HF_HUB_CACHE=/hf-cache \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint /app/application/backend/.venv/bin/python \
  ghcr.io/open-edge-platform/physical-ai-studio-xpu:main \
  /srv/remote_policy_server.py --export-dir /models/exports/torch \
    --backend torch --device xpu --host 0.0.0.0 --port 18767
