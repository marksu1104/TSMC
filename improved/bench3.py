"""Scale benchmark for the route-based GA-TS. Usage: python bench3.py LOG data.xlsx:cap:seeds ..."""
import sys, json, os, subprocess
from ga_tabu_route import RouteGaTabu
log = sys.argv[1]
os.makedirs('out', exist_ok=True)
for spec in sys.argv[2:]:
    data, cap, seeds = spec.split(':'); cap = int(cap)
    for seed in range(int(seeds)):
        s = RouteGaTabu(data, seed=seed, batch_capacity=cap)
        s.time_limit = 300
        s.run()
        f = f"out/{os.path.splitext(os.path.basename(data))[0]}_cap{cap}_seed{seed}.xlsx"
        s.save_results(f)
        chk = subprocess.run([sys.executable, 'check_schedule.py', data, f, str(cap)], capture_output=True, text=True)
        r = s.analyze_results(); r.update(data=data, cap=cap, seed=seed, check_ok=chk.returncode == 0,
                                          first_hit_sec=next(t for t, v in s.history if v == min(v for _, v in s.history)))
        with open(log, 'a') as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
