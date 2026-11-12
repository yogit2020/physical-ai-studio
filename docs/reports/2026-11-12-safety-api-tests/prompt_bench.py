"""Compare safety prompts on frame pairs: <kind>-<id>-arm-camera.jpg + <kind>-<id>-topview.jpg."""
import base64, glob, json, math, os, sys, time, urllib.request
from collections import defaultdict

URL = os.environ.get("QWEN_URL", "http://172.18.0.1:18766/v1/chat/completions")
ANSWER = " Answer with exactly one word: SAFE, UNSAFE_HAND or UNSAFE_OBJECT."
os.environ["no_proxy"] = "*"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def ask(prompt, wrist, top):
    img = lambda p: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
    body = {"model": "qwen-safety", "temperature": 0, "max_tokens": 6, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}, "structured_outputs": {"choice": ["SAFE", "UNSAFE_HAND", "UNSAFE_OBJECT"]},
            "messages": [{"role": "user", "content": [img(wrist), img(top), {"type": "text", "text": prompt + ANSWER}]}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter(); r = json.load(opener.open(req, timeout=30)); ms = (time.perf_counter() - t) * 1000
    c = r["choices"][0]
    return c["message"]["content"], math.exp(c["logprobs"]["content"][0]["logprob"]), ms

prompts = {name: open(f"{os.path.dirname(__file__)}/prompt_{name}.txt").read().strip() for name in sys.argv[2:]}
pairs = sorted({p[: -len("-arm-camera.jpg")] for d in sys.argv[1].split(",") for p in glob.glob(f"{d}/*-arm-camera.jpg")})
rows = []
for base in pairs:
    kind = os.path.basename(base).split("-")[0]
    for name, prompt in prompts.items():
        answer, conf, ms = ask(prompt, base + "-arm-camera.jpg", base + "-topview.jpg")
        stop = answer != "SAFE" and conf >= 0.6
        rows.append(dict(pair=os.path.basename(base), kind=kind, prompt=name, answer=answer, conf=round(conf, 3), ms=round(ms), stop=stop))
        if kind != "uni" and kind != "hold" or stop:
            print(f"{os.path.basename(base):45} {name:4} {answer:13} {conf:.3f} {ms:4.0f} ms {'STOP' if stop else ''}", flush=True)
json.dump(rows, open(os.environ.get("OUT", "prompt_bench_results.json"), "w"), indent=1)
agg = defaultdict(lambda: [0, 0, 0, []])
for r in rows:
    a = agg[(r["kind"], r["prompt"])]; a[0] += 1; a[1] += r["stop"]; a[2] += r["answer"] != "SAFE"; a[3].append(r["ms"])
print("\nkind       prompt  pairs  stops(>=0.6)  any-unsafe  mean ms  p90 ms")
for (k, p), (n, s, u, ms) in sorted(agg.items()):
    ms.sort(); print(f"{k:10} {p:6} {n:6} {s:8} {u:12} {sum(ms)/n:9.0f} {ms[int(n*.9)]:7}")
