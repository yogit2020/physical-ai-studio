# Safety API tests on the live SO-101: 2026-11-12

What we tested today, with the API calls, the model timings, and what we found. Design:
`docs/feature_plans/remote-inference-and-safety-monitor.md` (§3 has the flow and API summary).

## Summary

- **All the Studio safety calls worked on the real arm:** capture home, stop, the latch refusing
  Play, acknowledge, events, and stop frames.
- **Qwen stopped the arm for a real hand.** Its confidence was 0.997, the arm started home, and when
  the hand was still there 0.5 s later the arm froze, as designed.
- **Pi0.5 sorted the object with the Qwen monitor watching** (test 16, 23:32:05–23:32:19). Qwen did
  not stop it once: no false stops.
- **Pi0.5 took about 100 ms per call from Studio** (72 ms on the GPU), over 49 live calls in three
  runs and 5 direct `curl` calls.
- **Qwen took 0.43 s per check** in the live monitor (429–503 ms over 12 sampled checks).
- One bug fixed (`safety.sh` crashed on its status summary). Two things to watch: one false
  `UNSAFE_OBJECT` just under the stop threshold, and the 30 Hz control loop missing a deadline on
  every Pi0.5 call.

## Setup

| | Pi0.5 (drives the arm) | Qwen (watches for hazards) |
|---|---|---|
| Model | Pi0.5 "210base 65ft", OpenVINO export, fingerprint `63347f77…7546` | `Qwen3.6-27B-int4-AutoRound`, served as `qwen-safety` (vLLM, max length 4096) |
| Where | B70 GPU 1, container `yogini-pi05-server`, port 18765 | B70 GPU 0, container `yogini-qwen-safety`, port 18766 |
| Studio side | `RemotePredictor`, timing log `/app/storage/remote_inference_timing.csv` | `SafetyMonitor`: 2 Hz, cameras `arm-camera` + `topview`, stop at confidence ≥ 0.6, one unsafe check is enough |

Studio: container `physical-ai-studio-xpu` (port 7860), session `50f76aefb3ca46c0b78dedde793d55f1`.
Both B70 ports reach Studio through one SSH tunnel on `172.18.0.1`. Task text: *"Sort this item into
the correct bin"*. Home pose captured: `[-4.76, -56.04, 44.27, 43.44, -50.23, 50.06]`. On a stop, the
arm goes home.

## Tests and results

| # | Time (UTC) | Test | API call | Expected | Result |
|---|---|---|---|---|---|
| 1 | 22:50:04 | Capture home | `POST /safety/home/capture` | Pose saved | ✅ 200. `safety.sh` then crashed while printing the summary (bug 1) |
| 2 | 22:50:14, 22:53:40 | Recapture home after fixing `safety.sh` | `POST /safety/home/capture` | Pose saved, summary prints | ✅ 200, summary prints |
| 3 | 22:58:05 | Run Pi0.5 on the arm | GUI Play (`set_control` model) | Arm runs the task | ✅ 18 Pi0.5 calls in 16 s |
| 4 | 22:58:21 | Manual stop while Pi0.5 drives | `POST /safety/stop` `{"reason": "test", "source": "safety.sh"}` | Policy stops, arm goes home, latch | ✅ 200. Model control stopped in 1 ms, arm went home |
| 5 | 22:59:04 | Turn on the Qwen monitor | `PUT /safety` (`monitor.enabled: true`, scene prompt) | Qwen checks at 2 Hz | ✅ health ok, verdicts SAFE |
| 6 | 23:03:01–23:03:39 | Press Play while latched (4 times) | GUI Play (`set_control`) | Refused | ✅ refused 4×: *"A safety stop is latched: test. Acknowledge it before starting model control."* |
| 7 | 23:07:29 | Acknowledge | `POST /safety/acknowledge?source=operator` | Latch clear, nothing restarts | ✅ 200, latch `clear`, arm still idle |
| 8 | 23:08:33 | Run Pi0.5 with Qwen watching | GUI Play | Arm runs, monitor is guarding | ✅ 16 Pi0.5 calls, monitor guarding |
| 9 | 23:08:47 | **Hand put into the workspace** | (Qwen, automatic) | Qwen stops the arm and sends it home | ✅ `UNSAFE_HAND` at 0.997. Model control stopped, arm started home |
| 10 | 23:08:48 | Hand still there while the arm moves home | (Qwen, automatic) | Arm freezes | ✅ second stop at 1.0: *"while the arm moved home"*, arm held in place |
| 11 | 23:09:02 | Press Play while latched | GUI Play | Refused | ✅ refused |
| 12 | 23:13:14 | Look at what Qwen saw | `GET /safety/events/<id>/frames/<camera>` | JPEGs of the stop | ✅ both frames show a hand ([topview](2026-11-12-safety-api-tests/frames/hand-topview.jpg), [arm-camera](2026-11-12-safety-api-tests/frames/hand-arm-camera.jpg)) |
| 13 | 23:26 | Pi0.5 called directly | `GET /info`, `POST /predict` ×5 | 50×6 chunk | ✅ see below |
| 14 | 23:26 | Qwen called directly, hand frames and clean frames | `POST /v1/chat/completions` ×6 | Hand → UNSAFE_HAND, clean → SAFE | ✅ 6/6 correct |
| 15 | 23:28 | Live monitor latency | `GET /safety` ×12 | ~0.5 s per check | ✅ 429–503 ms |
| 16 | 23:31:58 | Acknowledge the hand stop | `POST /safety/acknowledge?source=operator` | Latch clear | ✅ 200, latch `clear` |
| 17 | 23:32:05–23:32:19 | **Full sort with Qwen watching** | GUI Play, then GUI stop | Object sorted, no false stops | ✅ **Object sorted into the bin.** 15 Pi0.5 calls in 13 s; about 26 Qwen checks at 2 Hz and none stopped the arm. Stopped from the GUI, not by a safety stop |

The event log for the session, oldest first (`GET /safety/events`):

```
22:50:04.645  home_captured  api
22:50:14.012  home_captured  api
22:53:40.391  home_captured  api
22:58:21.871  stop           safety.sh  hazard=other           reason="test"                     stopped=model  motion=home
22:59:04.170  config         api
23:07:29.807  acknowledge    operator
23:08:47.853  stop           monitor    hazard=hand_or_person  conf=0.997                        stopped=model  motion=home  frames=[arm-camera, topview]
23:08:48.364  stop           monitor    hazard=hand_or_person  conf=1.0  "while the arm moved home"  stopped=-      motion=hold  frames=[arm-camera, topview]
23:31:58.607  acknowledge    operator
                             (23:32:05–23:32:19 full sort with the monitor on: no events, no stops)
```

## Pi0.5 timings

**Live, from Studio** (`remote_inference_timing.csv`, 49 calls in tests 3, 8 and 17, raw data in
[pi_today.csv](2026-11-12-safety-api-tests/pi_today.csv)):

| | mean | p50 | p90 | max |
|---|---|---|---|---|
| Inference on the B70 GPU (`remote_infer_ms`) | 72.3 ms | 70.7 ms | 72.8 ms | 97.4 ms |
| Network + HTTP (`network_ms`) | 25.4 ms | 19.1 ms | 27.3 ms | 231.6 ms |
| **Total, as Studio sees it** | **99.9 ms** | **91.6 ms** | **116.0 ms** | **306.9 ms** |
| Payload per call | 75.5 KB | 69 KB | 100 KB | 113 KB |
| Time between calls | 937 ms | 929 ms | 951 ms | |

Per run:

| Run | Calls | Total mean | Total p50 | Total max | Ended by |
|---|---|---|---|---|---|
| Test 3 (22:58:05) | 18 | 109.3 ms | 94.1 ms | 306.9 ms | manual `safety/stop` |
| Test 8 (23:08:33) | 16 | 94.0 ms | 90.6 ms | 119.1 ms | Qwen stop (hand) |
| **Test 17 (23:32:05), full sort** | **15** | **95.0 ms** | **92.5 ms** | **116.0 ms** | GUI stop after the sort |

Tests 3 and 8 together: mean 102.1 ms, p50 91.5 ms, p90 118.7 ms. JPEG encode in Studio: 2.2 ms mean.
In test 17, Qwen was checking the whole time on the other GPU, and Pi0.5's GPU time stayed the
same (72.5 ms mean): the two models do not slow each other down.

**Direct `curl`** (test 13, two 640×480 frames, 165 KB request):

```
$ curl http://172.18.0.1:18765/info
{"host": "729ce1d9d9d6", "export_dir": "/models/exports/openvino", "backend": "openvino", "device": "GPU",
 "fingerprint": "63347f77df2428cf1a1b486a535fc81329d66713d4f44aea51b987d240097546"}
  -> HTTP 200  total 0.007s

$ curl -X POST http://172.18.0.1:18765/predict -H 'Content-Type: application/json' -d @pi_body.json
  -> HTTP 200  total 0.117s   server infer_ms 96.2   chunk 50 x 6   first action [-4.97, -46.75, 36.76, 43.02, -50.3, 50.4]
  -> HTTP 200  total 0.087s   server infer_ms 66.5   (run 2)
  -> HTTP 200  total 0.085s   server infer_ms 66.5   (run 3)
  -> HTTP 200  total 0.090s   server infer_ms 66.7   (run 4)
  -> HTTP 200  total 0.084s   server infer_ms 65.5   (run 5)
```

The first call is slower (96 ms) because the GPU warms up. After that, inference is 66 ms and the
whole call is 85–90 ms. Pi0.5 predicts small, steady moves from the home pose (joint 2 from −56.0 to
−46.8).

## Qwen timings

**Live monitor** (test 15, the monitor's own checks on the real cameras):

```
23:28:39.431  SAFE           0.704  433 ms
23:28:40.429  SAFE           0.990  434 ms
23:28:41.926  SAFE           0.700  432 ms
23:28:42.927  SAFE           0.931  432 ms
23:28:43.927  SAFE           0.979  431 ms
23:28:44.930  SAFE           0.979  432 ms
23:28:46.435  SAFE           0.776  431 ms
23:28:47.429  SAFE           0.952  429 ms
23:28:48.433  SAFE           0.678  434 ms
23:28:49.501  UNSAFE_OBJECT  0.559  503 ms   <- below the 0.6 threshold, no stop (finding 2)
23:28:50.942  SAFE           0.893  433 ms
23:28:51.938  SAFE           0.914  432 ms
```

**Direct `curl`** (test 14, same request as the monitor sends: two 448 px images, scene prompt,
one-word answer):

```
$ curl -X POST http://172.18.0.1:18766/v1/chat/completions -H 'Content-Type: application/json' -d @qw_body.json
hand frames (from the 23:08:47 stop):
  -> HTTP 200  total 0.989s  answer UNSAFE_HAND  confidence 0.977  (UN=0.977, SAFE=0.023)  tokens in 551 out 4
  -> HTTP 200  total 1.142s  answer UNSAFE_HAND  confidence 0.977
  -> HTTP 200  total 0.808s  answer UNSAFE_HAND  confidence 0.977
clean frames (July recording, black mat, no hazard):
  -> HTTP 200  total 0.743s  answer SAFE         confidence 0.892  (SAFE=0.892, UN=0.107)  tokens in 551 out 2
  -> HTTP 200  total 0.724s  answer SAFE         confidence 0.892
  -> HTTP 200  total 0.725s  answer SAFE         confidence 0.892
```

The answers are the same every time because `temperature: 0`. The direct calls took 0.72–1.14 s
instead of 0.43 s because the monitor was still checking twice a second, so each of these calls
waited for a monitor check to finish. That is extra load from the test, not a slower model.

The full log is in [curl_log.txt](2026-11-12-safety-api-tests/curl_log.txt). To run it again:
`2026-11-12-safety-api-tests/run_api_tests.sh <frames dir>` (it only calls the models, never the
arm).

## Stop timing

| Step | Time |
|---|---|
| Qwen check | 0.43 s |
| Verdict → model control stopped | < 1 ms (23:08:47.853 → .853) |
| Stop → second check sees the hand while homing → freeze | 0.51 s |

We could not measure the time from the hand entering the view to the stop: the hand's entry has no
timestamp. The worst case from the design is about one check interval plus one check, ~0.5 + 0.43 ≈
0.9 s.

## Findings

1. **Bug, fixed: `safety.sh` status summary crashed.** Its inline Python used `\"` inside a bash
   single-quoted string, which Python rejects inside f-string braces (*"unexpected character after
   line continuation character"*). The API call itself had worked, which is why home had been captured
   at 22:50:04. Fixed by moving the Python into a quoted heredoc in `application/backend/scripts/safety.sh`.
2. **Watch: a false `UNSAFE_OBJECT` at 0.559 on an empty scene.** It was under the 0.6 stop threshold,
   and the monitor was not guarding, so nothing happened. But SAFE answers also ranged from 0.68 to
   0.99, so the clean scene sometimes gets close to the line. If a run gets stopped for no reason,
   check `GET /safety/events` and tune the prompt or raise `min_confidence` / `unsafe_consecutive`.
   This is open question 1 in the design.
3. **Watch: one missed 30 Hz deadline per Pi0.5 call.** Studio logged *"ModelControl missed 1 of its
   30Hz deadlines, worst ~90–120 ms"* about once a second, and the worst (310.8 ms at 22:58:08)
   matches the slowest Pi0.5 call (306.9 ms). Each Pi0.5 call blocks the control loop for one tick.
   This is expected for synchronous remote inference and is what makes the arm pause briefly once a
   second.
4. **Behaviour to know: after a freeze, the arm does not go home by itself.** This is by design (a hand
   could be in the path). Send it with `./safety.sh go-home` once the area is clear, then `ack`.
5. **Gotcha for anyone calling Pi0.5 directly:** image keys must be `images.<camera>`
   (`images.topview`, `images.arm-camera`). Plain `topview` returns HTTP 500 *"Expected input 'images'
   not found"*.

## Not tested yet

- Qwen on clean frames from **today's** scene (white mat) by `curl`. The clean frames above are from
  a July recording; today's scene was only checked by the live monitor (test 15).
- Objects other than a hand (`UNSAFE_OBJECT`: a cup, a phone, a tool).
- Fail-safe stop: block the Qwen port or freeze a camera, and expect a stop after 2 s.
- Stop latency measured against camera timestamps.
- More full sorting episodes with the monitor on (test 17 is one: 0 false stops in 13 s), to get a
  false-stop rate.
