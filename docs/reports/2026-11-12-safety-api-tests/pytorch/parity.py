"""Same frames, state, task and start noise into the OpenVINO and the PyTorch export of one Pi0.5; compare actions."""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from physicalai.inference import InferenceModel

W = Path("/work")
CASES = {  # dataset frame -> recorded joint state at that frame
    "hold-02051": [-9.744, 1.490, -26.789, 43.394, -58.926, 55.922],
    "hold-30482": [-13.407, 10.379, -23.321, 30.354, -57.802, 56.068],
    "uni-20722": [-14.530, 15.409, -41.392, 42.027, -64.103, 55.946],
}
TASK = "Sort this item into the correct bin"

def obs(case, noise):
    o = {"state": np.asarray([CASES[case]], np.float32), "task": [TASK], "noise": noise}
    for cam in ("arm-camera", "topview"):
        o[f"images.{cam}"] = np.asarray(Image.open(W / f"{case}-{cam}.jpg").convert("RGB"))[np.newaxis]
    return o

models = {}
for name, path, backend, device in [("openvino", W / "ov_noise", "openvino", "GPU"), ("torch", Path("/models/exports/torch"), "torch", "xpu")]:
    t = time.perf_counter(); models[name] = InferenceModel(path, backend=backend, device=device)
    print(f"loaded {name} in {time.perf_counter() - t:.0f}s", flush=True)

import torch
FORCED = {}
policy = models["torch"].adapter._policy
samplers = [m for m in policy.modules() if hasattr(m, "sample_noise")]
assert samplers, "no sample_noise in the torch policy"
for m in samplers:
    def forced(shape, device, _m=m):
        n = torch.from_numpy(FORCED["noise"]).to(device)
        assert tuple(n.shape) == tuple(shape), (n.shape, shape)
        return n
    m.sample_noise = forced
print(f"patched sample_noise on {len(samplers)} module(s)", flush=True)
_predict = models["torch"].predict_action_chunk
def torch_predict(o):
    FORCED["noise"] = o["noise"].copy()
    return _predict(o)
models["torch"].predict_action_chunk = torch_predict

rows, timing = [], {"openvino": [], "torch": []}
for case in CASES:
    for seed in (0, 1, 2):
        noise = np.random.default_rng(seed).standard_normal((1, 50, 32), dtype=np.float32)
        out = {}
        for name, m in models.items():
            t = time.perf_counter(); out[name] = np.asarray(m.predict_action_chunk(obs(case, noise.copy()))); timing[name].append((time.perf_counter() - t) * 1000)
        d = np.abs(out["openvino"] - out["torch"])
        rows.append(dict(case=case, seed=seed, max_abs=float(d.max()), mean_abs=float(d.mean()), first_ov=out["openvino"][0].round(2).tolist(), first_pt=out["torch"][0].round(2).tolist()))
        print(f"{case} seed {seed}: max |ov-pt| {d.max():6.3f}  mean {d.mean():6.3f}  first ov {out['openvino'][0].round(2)}  pt {out['torch'][0].round(2)}", flush=True)

# Same noise twice: each backend must repeat itself. Different noise: how much does noise alone move the actions?
n0 = np.random.default_rng(0).standard_normal((1, 50, 32), dtype=np.float32)
n1 = np.random.default_rng(1).standard_normal((1, 50, 32), dtype=np.float32)
for name, m in models.items():
    a = np.asarray(m.predict_action_chunk(obs("hold-02051", n0.copy()))); b = np.asarray(m.predict_action_chunk(obs("hold-02051", n0.copy()))); c = np.asarray(m.predict_action_chunk(obs("hold-02051", n1.copy())))
    print(f"{name}: same noise twice max diff {np.abs(a - b).max():.4f} | noise 0 vs noise 1 max diff {np.abs(a - c).max():.3f} mean {np.abs(a - c).mean():.3f}")
for name, v in timing.items():
    print(f"{name}: predict ms first {v[0]:.0f}, rest mean {np.mean(v[1:]):.1f} min {min(v[1:]):.1f} max {max(v[1:]):.1f}")
json.dump(rows, open(W / "parity_results.json", "w"), indent=1)
