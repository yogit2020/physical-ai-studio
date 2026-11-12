#!/bin/bash
# Drive a Studio runtime session's safety latch and Qwen monitor from the shell.
#
#   ./safety.sh status                 latch state, monitor health, last verdict, home pose
#   ./safety.sh capture-home           take the arm's current position as its home pose
#   ./safety.sh go-home                move the arm home now (only while nothing drives it)
#   ./safety.sh stop ["reason"]        hazard stop: end Pi0.5, glide home, latch
#   ./safety.sh hold ["reason"]        hazard stop that freezes the arm where it is
#   ./safety.sh ack                    clear the latch (nothing restarts by itself)
#   ./safety.sh monitor-on             Qwen on the B70 watches both cameras (scene prompt below)
#   ./safety.sh monitor-off
#   ./safety.sh events                 recent safety events, newest first
#   ./safety.sh frame <event_id> <camera> [out.jpg]   a frame Qwen judged for a stop
#
# The session is the one open in the Studio GUI (start one by opening Run model / Record).
set -euo pipefail

STUDIO=${STUDIO:-http://localhost:7860}
QWEN_ENDPOINT=${QWEN_ENDPOINT:-http://172.18.0.1:18766/v1}
PY=${PY:-"$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"}

SCENE_PROMPT='You are the safety monitor for a robot arm work cell. Image 1 is the wrist camera on the arm, image 2 is the overhead camera.
THE WORK AREA is the white mat. Judge ONLY what is on or above the white mat. Ignore everything beyond its edges: the wooden desk, the wall, a camera tripod, cables, a power strip, and anything else lying on the desk, such as a phone.
ALLOWED on the mat: the white robot arm and gripper including its exposed circuit boards, servos and wires; small wooden cubes (red, blue, black, green), also while held in the gripper; a green bin with a white TRASH logo; a blue bin with a white RECYCLE logo; the bins'"'"' small stands.
The wrist camera moves with the gripper. When the gripper holds a cube, the cube fills the middle of the wrist image: that is ALLOWED. When the arm lifts or reaches over a bin, the wrist camera looks past the mat at the desk, wall, tripod, cables and power strip: those are off the mat, ignore them. Bins, logos, cubes and the gripper fingers are often very close or cut off at the edges, and dark areas are bin walls or shadows: ALLOWED.
For objects, trust the overhead camera (image 2): it shows the whole mat. A hand counts in either image.
UNSAFE: any human hand, finger, arm, sleeve or person on or above the mat, or any object on the mat that is not in the allowed list (for example a cup, phone, tool, pen, bottle, box).'

session_id() {
    # The sessions socket sends the live sessions on connect.
    NO_PROXY='*' no_proxy='*' "$PY" - "$STUDIO" <<'EOF'
import asyncio, json, sys
import websockets

async def main() -> None:
    url = sys.argv[1].replace("http", "ws", 1) + "/api/runtime/sessions/ws"
    async with websockets.connect(url, proxy=None) as ws:
        sessions = json.loads(await asyncio.wait_for(ws.recv(), 10))["sessions"]
    if not sessions:
        sys.exit("No runtime session is open: open Run model in the Studio GUI first.")
    if len(sessions) > 1:
        print(f"{len(sessions)} sessions open; using {sessions[0]['environment_name']!r}", file=sys.stderr)
    print(sessions[0]["session_id"])

asyncio.run(main())
EOF
}

api() {  # method path [json]
    local method=$1 path=$2 body=${3:-}
    curl -sS --noproxy '*' -X "$method" "$STUDIO/api/runtime/sessions/$SID$path" \
        ${body:+-H 'Content-Type: application/json' -d "$body"}
}

pretty() { "$PY" -m json.tool 2>/dev/null || cat; }

summary() {
    "$PY" -c "$(cat <<'EOF'
import json, sys
s = json.load(sys.stdin)
if "detail" in s or "message" in s:
    print(json.dumps(s, indent=2)); sys.exit()
cause, mon, cfg = s.get("latched_by"), s["monitor"], s["config"]
print(f'latch:      {s["state"]}' + (f'  (by {cause["source"]}: {cause["reason"]})' if cause else ''))
print(f'home pose:  {cfg["home_pose"] or "not captured"}  (on stop: {cfg["on_stop"]})')
v = mon.get("last_verdict")
print(f'monitor:    {"on" if cfg["monitor"]["enabled"] else "off"}, health {mon["health"]}, guarding {mon["guarding"]}'
      + (f', error: {mon["error"]}' if mon.get("error") else ''))
if v:
    print(f'last check: {v["answer"]} ({v["confidence"]:.2f}) in {v["latency_ms"]:.0f} ms at {v["at"]}')
print(f'counters:   {s["counters"]}')
EOF
)"
}

cmd=${1:-status}
SID=$(session_id)

case "$cmd" in
    status) api GET /safety | summary ;;
    capture-home) api POST /safety/home/capture | summary ;;
    go-home) api POST /safety/home/go | summary ;;
    stop | hold)
        reason=${2:-manual test stop}
        hold=$([ "$cmd" = hold ] && echo true || echo false)
        api POST /safety/stop "{\"reason\": \"$reason\", \"source\": \"safety.sh\", \"hazard\": \"other\", \"hold\": $hold}" | summary
        ;;
    ack) api POST "/safety/acknowledge?source=operator" | summary ;;
    monitor-on)
        body=$("$PY" -c 'import json, sys; print(json.dumps({"monitor": {
            "enabled": True, "endpoint": sys.argv[1], "model": "qwen-safety",
            "cameras": ["arm-camera", "topview"], "check_hz": 2.0, "min_confidence": 0.6,
            "unsafe_consecutive": 1, "fail_safe_after_s": 2.0, "prompt": sys.argv[2]}}))' "$QWEN_ENDPOINT" "$SCENE_PROMPT")
        api PUT /safety "$body" | summary
        ;;
    monitor-off) api PUT /safety '{"monitor": {"enabled": false}}' | summary ;;
    events) api GET "/safety/events?limit=${2:-20}" | pretty ;;
    frame)
        out=${4:-$2-$3.jpg}
        curl -sS --noproxy '*' -o "$out" "$STUDIO/api/runtime/sessions/$SID/safety/events/$2/frames/$3" && echo "saved $out"
        ;;
    *) sed -n '2,16p' "$0"; exit 1 ;;
esac
