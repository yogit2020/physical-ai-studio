#!/usr/bin/env bash
# Direct curl tests of the Pi0.5 policy server and the Qwen safety model, through the SSH tunnel.
# Read-only for the robot: these calls never reach the arm.
set -u
IMG=${1:-/tmp/apitest}
PI=http://172.18.0.1:18765
QW=http://172.18.0.1:18766/v1
T='\n  -> HTTP %{http_code}  connect %{time_connect}s  first byte %{time_starttransfer}s  total %{time_total}s  sent %{size_upload} B\n'
b64() { base64 -w0 "$1"; }
PROMPT=$(sed -n "/^SCENE_PROMPT='/,/'$/p" "$(dirname "$0")/../../../application/backend/scripts/safety.sh" | sed "1s/^SCENE_PROMPT='//; \$s/'$//")
ANSWER='

Answer with exactly one word: SAFE, UNSAFE_HAND or UNSAFE_OBJECT.'

echo "### 1. Pi0.5  GET /info"
echo "\$ curl $PI/info"
curl -sS --noproxy '*' -w "$T" $PI/info

pi_predict() {  # $1 label, $2 topview jpg, $3 arm jpg
  printf '{"state": [[-4.76, -56.04, 44.27, 43.44, -50.23, 50.06]], "images": {"images.topview": "%s", "images.arm-camera": "%s"}, "task": "Sort this item into the correct bin"}' \
    "$(b64 "$2")" "$(b64 "$3")" > /tmp/apitest/pi_body.json
  echo; echo "### Pi0.5  POST /predict  ($1)"
  echo "\$ curl -X POST $PI/predict -H 'Content-Type: application/json' -d @pi_body.json"
  curl -sS --noproxy '*' -X POST $PI/predict -H 'Content-Type: application/json' -d @/tmp/apitest/pi_body.json -w "$T" -o /tmp/apitest/pi_out.json
  python3 $(dirname "$0")/summarize_pi.py /tmp/apitest/pi_out.json
}
for i in 1 2 3 4 5; do pi_predict "run $i" $IMG/clean-topview.jpg $IMG/clean-arm-camera.jpg; done

qwen() {  # $1 label, $2 topview, $3 arm
  python3 - "$2" "$3" "$PROMPT$ANSWER" > /tmp/apitest/qw_body.json <<'PY'
import base64, json, sys
img = lambda p: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
print(json.dumps({"model": "qwen-safety", "temperature": 0, "max_tokens": 6, "logprobs": True, "top_logprobs": 5,
  "chat_template_kwargs": {"enable_thinking": False},
  "structured_outputs": {"choice": ["SAFE", "UNSAFE_HAND", "UNSAFE_OBJECT"]},
  "messages": [{"role": "user", "content": [img(sys.argv[2]), img(sys.argv[1]), {"type": "text", "text": sys.argv[3]}]}]}))
PY
  echo; echo "### Qwen  POST /v1/chat/completions  ($1)"
  echo "\$ curl -X POST $QW/chat/completions -H 'Content-Type: application/json' -d @qw_body.json"
  curl -sS --noproxy '*' -X POST $QW/chat/completions -H 'Content-Type: application/json' -d @/tmp/apitest/qw_body.json -w "$T" -o /tmp/apitest/qw_out.json
  python3 $(dirname "$0")/summarize_qwen.py /tmp/apitest/qw_out.json || head -c 400 /tmp/apitest/qw_out.json
}
for i in 1 2 3; do qwen "hand in view, run $i" $IMG/hand-topview-448.jpg $IMG/hand-arm-camera-448.jpg; done
for i in 1 2 3; do qwen "old July scene, black mat, run $i" $IMG/clean-topview-448.jpg $IMG/clean-arm-camera-448.jpg; done
