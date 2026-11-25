"""Best schedule reachable by the GA/GA-TS notebooks' encoding.

Those notebooks decide each cart's pickup/delivery order with a greedy rule: from the current location,
go to the nearest pending pickup or delivery (ties: pickups in list order first, then deliveries in pickup order).
The chromosome only controls which WIPs a cart gets and their list order (tie-breaking).
Step 1: for every group of 1..CAP WIPs and every list order, simulate that rule; keep the cheapest.
Step 2: set partitioning (<= 20 carts) with OR-Tools (SCIP).
"""
import itertools, sys, time
import numpy as np, pandas as pd
from ortools.linear_solver import pywraplp

x = pd.ExcelFile(sys.argv[1]); CAP = int(sys.argv[2])
cart = pd.read_excel(x, "CART"); wip = pd.read_excel(x, "WIP"); xf = pd.read_excel(x, "XFER_TIME")
locs = sorted(set(xf.FROM) | set(xf.TO)); L = {l: i for i, l in enumerate(locs)}
D = np.zeros((len(locs), len(locs)), dtype=np.int64)
for r in xf.itertuples():
    D[L[r.FROM], L[r.TO]] = r.XFER_TIME
assert xf.FROM.iloc[0] == cart.INIT_LOC.iloc[0] and cart.INIT_LOC.nunique() == 1
start = L[cart.INIT_LOC.iloc[0]]
src = np.array([L[v] for v in wip.FROM]); dst = np.array([L[v] for v in wip.TO]); q = wip["Remaining Q-Time"].to_numpy()
n = len(wip)
INF = 10**9

def greedy_cost(S, perm):
    N, k = S.shape
    pos = np.empty(k, dtype=np.int64); pos[list(perm)] = np.arange(k)   # list position of each slot
    cur = np.full(N, start); t = np.zeros(N, dtype=np.int64); scrap = np.zeros(N, dtype=np.int64)
    picked = np.zeros((N, k), bool); done = np.zeros((N, k), bool); pstep = np.zeros((N, k), dtype=np.int64)
    for step in range(2 * k):
        key = np.full((N, k), INF, dtype=np.int64); isdel = np.zeros((N, k), bool)
        for i in range(k):
            w = S[:, i]
            dp = D[cur, src[w]] * 1000 + pos[i]
            dd = D[cur, dst[w]] * 1000 + k + pstep[:, i]
            key[:, i] = np.where(~picked[:, i], dp, np.where(~done[:, i], dd, INF))
            isdel[:, i] = picked[:, i] & ~done[:, i]
        j = key.argmin(axis=1); r = np.arange(N)
        d = isdel[r, j]; w = S[r, j]
        nxt = np.where(d, dst[w], src[w])
        t = t + D[cur, nxt]; cur = nxt
        scrap += d & (t > q[w])
        done[r[d], j[d]] = True
        newp = ~d
        picked[r[newp], j[newp]] = True; pstep[r[newp], j[newp]] = step
    return scrap * 10000 + t

t0 = time.time(); groups, costs = [], []
for k in range(1, CAP + 1):
    S = np.array(list(itertools.combinations(range(n), k)))
    best = np.full(len(S), np.iinfo(np.int64).max)
    for perm in itertools.permutations(range(k)):
        best = np.minimum(best, greedy_cost(S, perm))
    groups.extend(map(tuple, S)); costs.append(best)
costs = np.concatenate(costs)
solver = pywraplp.Solver.CreateSolver("SCIP")
y = [solver.BoolVar(f"y{j}") for j in range(len(groups))]
cover = [[] for _ in range(n)]
for j, g in enumerate(groups):
    for w in g:
        cover[w].append(y[j])
for w in range(n):
    solver.Add(solver.Sum(cover[w]) == 1)
solver.Add(solver.Sum(y) <= len(cart))
solver.Minimize(solver.Sum([int(c) * v for c, v in zip(costs, y)]))
status = solver.Solve()
sel = [j for j in range(len(groups)) if y[j].solution_value() > 0.5]; tot = int(costs[sel].sum())
print(f"CAP={CAP} 只用最近鄰排序時的最佳解（OR-Tools SCIP）：{'OPTIMAL' if status == 0 else status}，"
      f"目標值 {round(solver.Objective().Value())}，下界 {solver.Objective().BestBound():.0f}，"
      f"報廢 {tot // 10000}、距離 {tot % 10000}，使用推車 {len(sel)} 台（{time.time()-t0:.0f} 秒）")
if len(sys.argv) > 3:
    import json
    json.dump([[str(wip.WIP_ID.iloc[w]) for w in groups[j]] for j in sel], open(sys.argv[3], 'w'))
