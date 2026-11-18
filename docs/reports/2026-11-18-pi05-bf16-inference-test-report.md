# Pi0.5 bf16 inference (muniker's code): test report

**Date:** 2026-11-18 · **Robot:** SO-101, sorting cubes into two bins ·
**Branch:** [`yogit2020/physical-ai-studio` · `feature/remote-inference-safety`](https://github.com/yogit2020/physical-ai-studio/tree/feature/remote-inference-safety)
· Follows [2026-11-17-remote-inference-test-report.md](2026-11-17-remote-inference-test-report.md)

## 1. Why

The policy model, **Pi0.5 "210base 65ft"** (`muniker/pi05-so101-sort-210base-65ft`), was trained with
muniker's Pi0.5 code (`munikera/physical-ai-studio`, branch `pi05-training-on-arc-pro`). Its weights
are **all bfloat16**: 823 of 823 tensors, vision tower and layernorms included (`dtype: bfloat16`).

The stock code runs it differently from how it was trained. At load it moves the vision tower,
multi-modal projector and layernorms to float32, without any warning. Every inference path we had
used the stock code: local Studio and both B70 servers run the `main` image. The OpenVINO export is
a fixed graph and is not affected; the PyTorch export (`pi05.pt`) is rebuilt from the installed code,
so it is.

## 2. Change

Inference only, in `library/src/physicalai/policies/pi05/`. Muniker's training changes are left out.

| File | Change |
|---|---|
| `model.py` | With `precision == "bfloat16"`, nothing is kept in float32 (muniker) |
| `model.py` | Noisy actions and time embeddings are cast to the projection weight dtype; `suffix_out` stays in the `action_out_proj` dtype instead of float32 (muniker) |
| `model.py` | Image and language prefix embeddings are concatenated in the language embedding's dtype (muniker's code alone fails here: `mat1 and mat2 must have the same dtype`) |
| `pi_gemma.py` | The AdaRMS conditioning is cast to the `dense` weight dtype (same error otherwise) |
| `preprocessor.py` | The state is cast to float32 before it is discretised for the prompt (muniker) |

The last two fixes come from `fix/pi05-inference-bf16-dtype` (2026-08-05). The flow-matching steps
still integrate in float32, which the RTC code expects. Tests: `test_pi05.py`, `test_pi05_noise_input.py`,
`test_pi05_rtc.py` pass (215).

## 3. Same start noise, stock vs bf16 code

40 dataset frames (8 episodes × 5) from `muniker/so101-sort-colored-blocks-65-dataset`, PyTorch on
the B70 CPU, torch reseeded before every call so both versions start from the same noise.

| | MAE first 20 steps (task) | MAE 50 steps (task) | MAE first 20 (no task) |
|---|---|---|---|
| Stock code | 1.181 | 1.623 | 1.386 |
| bf16 code | 1.183 | 1.622 | 1.385 |
| Hold still | 3.77 | | 3.77 |

The actions differ by **0.04** joint units on average (max 0.5). The inference-only diff in this
branch gives bit-identical actions to the full muniker patch tested on the arm (max difference 0.0).

**The precision mismatch hardly changes the actions, so it does not explain why our measurements
differed from muniker's.** The start noise matters far more: the same stock code on the same frames
scored 1.39 on 11-05 and 1.16 on 11-18 without a fixed seed. Comparisons need a fixed seed.

## 4. B70 PyTorch server and the real arm

Server `yogini-pi05-torch` (port 18767, `torch.compile`), weights fingerprint unchanged.

| | Stock (11-17) | bf16 code (11-18) |
|---|---|---|
| Inference on the B70 GPU | 93 ms | **92 ms** median (p90 96) |
| Round trip from Studio | 115 ms | **112.5 ms** median (p90 116) |
| GPU memory, torch model | 9.0 GB | **8.0 GB** |

Real arm with the Qwen3.8 safety monitor on: 64 inference calls, about 67 s under model control,
no errors, GPU memory flat. **2 hand stops, both correct** (confidence 1.00): the policy stopped and
the arm went home; the hand was still there 0.6 s later, so the arm held where it was. **0 false stops.**
The gripper held a red cube at the stop.

## 5. Deploy

- B70: `~/yogini/remote-inference/start_torch_muniker.sh` mounts this branch's `model.py`,
  `pi_gemma.py` and `preprocessor.py` over the image copy (`start_torch.sh` is the stock version).
- Local Studio still runs the stock library. It only matters for local PyTorch inference.
