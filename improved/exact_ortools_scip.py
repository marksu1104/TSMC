"""每台推車 0 到 CAP 批的最佳解（OR-Tools 整數規劃介面，SCIP 求解器）。

步驟 1：列舉 1 到 CAP 批的所有 WIP 組合；每個組合試過所有合法取送順序，留下成本最低的
        （成本 = 報廢數 × 10000 + 距離，和 GA 的適應值相同）。
步驟 2：整數規劃。每個組合一個 0-1 變數，每批 WIP 恰好被一個組合涵蓋、組合數不超過推車數，總成本最小。
        CAP = 2 時就是 WIP 兩兩配對的模型。
用法：python exact_ortools_scip.py data.xlsx 4 [時間上限秒數]
"""
import itertools, sys, time
import numpy as np, pandas as pd

x = pd.ExcelFile(sys.argv[1]); CAP = int(sys.argv[2]); LIMIT = float(sys.argv[3]) if len(sys.argv) > 3 else 600
cart = pd.read_excel(x, "CART"); wip = pd.read_excel(x, "WIP"); xf = pd.read_excel(x, "XFER_TIME")
assert cart.INIT_LOC.nunique() == 1, "這個模型假設所有推車從同一個位置出發"
locs = sorted(set(xf.FROM) | set(xf.TO)); L = {l: i for i, l in enumerate(locs)}
D = np.zeros((len(locs), len(locs)), dtype=np.int64)
for r in xf.itertuples():
    D[L[r.FROM], L[r.TO]] = r.XFER_TIME
start = L[cart.INIT_LOC.iloc[0]]
src = np.array([L[v] for v in wip.FROM]); dst = np.array([L[v] for v in wip.TO]); q = wip["Remaining Q-Time"].to_numpy()
n = len(wip)


def legal_orders(k):
    out = []
    def rec(seq, picked, done):
        if len(seq) == 2 * k:
            out.append(list(seq)); return
        for i in range(k):
            if i not in picked:
                rec(seq + [(0, i)], picked | {i}, done)
            elif i not in done:
                rec(seq + [(1, i)], picked, done | {i})
    rec([], frozenset(), frozenset())
    return out


t0 = time.time()
groups, costs = [], []
for k in range(1, CAP + 1):
    S = np.array(list(itertools.combinations(range(n), k)))
    best = np.full(len(S), np.iinfo(np.int64).max)
    for order in legal_orders(k):
        loc_prev = np.full(len(S), start); t = np.zeros(len(S), dtype=np.int64); scrap = np.zeros(len(S), dtype=np.int64)
        for is_d, slot in order:
            w = S[:, slot]
            nxt = dst[w] if is_d else src[w]
            t = t + D[loc_prev, nxt]; loc_prev = nxt
            if is_d:
                scrap += t > q[w]
        best = np.minimum(best, scrap * 10000 + t)
    groups.extend(map(tuple, S)); costs.extend(best.tolist())
t_enum = time.time() - t0

from ortools.linear_solver import pywraplp
solver = pywraplp.Solver.CreateSolver("SCIP")
solver.SetTimeLimit(int(LIMIT * 1000))
y = [solver.BoolVar(f"y{j}") for j in range(len(groups))]
cover = [[] for _ in range(n)]
for j, g in enumerate(groups):
    for w in g:
        cover[w].append(y[j])
for w in range(n):
    solver.Add(solver.Sum(cover[w]) == 1)
solver.Add(solver.Sum(y) <= len(cart))
solver.Minimize(solver.Sum([c * v for c, v in zip(costs, y)]))
t1 = time.time()
status = solver.Solve()
name = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(status, str(status))
chosen = [groups[j] for j in range(len(groups)) if y[j].solution_value() > 0.5]
obj = round(solver.Objective().Value())
print(f"CAP={CAP}：{len(groups):,} 個變數（列舉 {t_enum:.1f} 秒），SCIP {name}，"
      f"目標值 {obj}（報廢 {obj // 10000}、距離 {obj % 10000}），下界 {solver.Objective().BestBound():.0f}，"
      f"求解 {time.time() - t1:.1f} 秒，使用推車 {len(chosen)} 台")
