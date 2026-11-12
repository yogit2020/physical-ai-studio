import json, math, sys
r = json.load(open(sys.argv[1]))
c = r["choices"][0]
t = c["logprobs"]["content"][0]
alts = ", ".join(f'{x["token"]}={math.exp(x["logprob"]):.3f}' for x in t["top_logprobs"][:3])
print(f'  -> answer {c["message"]["content"]}  confidence {math.exp(t["logprob"]):.3f}  (top first tokens: {alts})  tokens in {r["usage"]["prompt_tokens"]} out {r["usage"]["completion_tokens"]}')
