import sys, json, os
from ga_tabu_route import RouteGaTabu
data, cap, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = RouteGaTabu(data, seed=seed, batch_capacity=cap)
for kv in sys.argv[4:]:
    k, v = kv.split('='); setattr(s, k, json.loads(v))
s.run()
os.makedirs('out', exist_ok=True)
f = f"out/{os.path.splitext(os.path.basename(data))[0]}_cap{cap}_seed{seed}.xlsx"
s.save_results(f)
r = s.analyze_results(); r.update(data=data, cap=cap, seed=seed, file=f)
print(json.dumps(r, ensure_ascii=False), flush=True)
