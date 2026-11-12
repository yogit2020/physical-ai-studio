import json, sys
r = json.load(open(sys.argv[1]))
if "action" not in r:
    print("  ->", r); sys.exit()
a = r["action"]
print(f'  -> server infer_ms {r["infer_ms"]:.1f}; chunk {len(a)} x {len(a[0])}; first action {[round(x, 2) for x in a[0]]}')
