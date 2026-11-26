"""方法 B 到 E 的實作。每個函式都回傳結果字典，並把排程寫成和原版相同格式的 Excel，
之後一律交給 improved/check_schedule.py 重新計算與檢查。

B. 路徑式 GA–TS：直接呼叫 improved/ga_tabu_route.py 的 RouteGaTabu，只設定時間上限。

C. 精確法（列舉 + 集合分割整數規劃，SCIP）：
   列舉 1 到 cap 批的所有 WIP 組合，每個組合試過所有合法取送順序，留下成本最低的
   （成本 = 報廢數 × 10000 + 距離）。整數規劃：每個組合一個 0-1 變數，每批 WIP 恰好被一個組合涵蓋，
   組合數不超過推車數。和 improved/exact_ortools_scip.py 相同，多了輸出排程、
   以及列舉前先估計組合數，超過上限就不執行並回傳原因。
   成立條件：所有推車從同一地點出發（本實驗都是 LOC1）。
   注意：limit 只限制 SCIP 的求解，不含列舉與載入模型；SCIP 在預處理階段也可能超過上限
   （80 批、容量 4 設 1800 秒，實際約 3000 秒才停）。

D. 不列舉的路徑整數規劃（SCIP）：
   節點：起點 0、第 w 批的取貨點 P_w、送貨點 D_w、終點 S（開放路徑，到 S 的距離為 0）。
   變數：x[i,j] 是否從節點 i 直接走到 j；T_i 抵達時間；u_i 該節點是路線上的第幾個事件；
         v_i 路線標籤（等於該路線第一個取貨點的編號 + 1，0 不使用）；z_w 第 w 批是否報廢。
   限制：
   - 每個取貨、送貨點恰好進一次、出一次；從起點出發的路線數 = 進入終點的路線數 ≤ 推車數
   - 時間：x[i,j] = 1 時 T_j ≥ T_i + 距離(i,j)。不另外加「T_{D_w} ≥ T_{P_w} + 距離(FROM_w, TO_w)」：
     距離表有 1148 處不符合三角不等式（繞路最多可以少 1），這條會擋掉合法的路線
   - 順序：x[i,j] = 1 時 u_j ≥ u_i + 1，u_{D_w} ≥ u_{P_w} + 1，1 ≤ u ≤ 2 × cap
     （u 消除不經過起點的迴圈；一條路線最多 2 × cap 個事件，也就是最多 cap 批，容量因此成立）
   - 同車：x[0,P_w] = 1 時 v_{P_w} = w + 1；x[i,j] = 1 時 v_i = v_j；v_{P_w} = v_{D_w}
   - 報廢：T_{D_w} ≤ Q_w + M × z_w
   目標：距離總和 + 10000 × Σ z_w。大 M 用一條路線可能的最長時間 2 × cap × 最大距離。
   變數約 (2n)² 個 x，不需要列舉組合，但線性鬆弛很弱（大 M 限制）。

E. OR-Tools 路徑求解器（pickup and delivery）：
   每批 WIP 一組取貨、送貨節點，同一台車、取貨在前；開放路徑（終點是距離 0 的虛擬節點）。
   容量：計數維度，取貨點 +1、送貨點 +0，上限 cap（代表分到的 WIP 數，不是同時在車上的數量）。
   Q-time：時間維度（時間 = 距離、不能等待），送貨點設軟上限 Q_w，係數 10000。
   注意 OR-Tools 的軟上限是「每超過 1 單位時間罰 10000」，不是「每違反一次罰 10000」；
   超過 1 單位以上時懲罰比題目大。所有結果最後都用題目的成本重新計算。
   第一個解用 PARALLEL_CHEAPEST_INSERTION，之後用 GUIDED_LOCAL_SEARCH 改善到時間上限。
"""
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "improved"))
PENALTY = 10000


class Instance:
    def __init__(self, path):
        x = pd.ExcelFile(path)
        self.cart = pd.read_excel(x, "CART")
        self.wip = pd.read_excel(x, "WIP")
        xf = pd.read_excel(x, "XFER_TIME")
        locs = sorted(set(xf.FROM) | set(xf.TO))
        L = {l: i for i, l in enumerate(locs)}
        self.D = np.zeros((len(locs), len(locs)), dtype=np.int64)
        for r in xf.itertuples():
            self.D[L[r.FROM], L[r.TO]] = r.XFER_TIME
        assert self.cart.INIT_LOC.nunique() == 1, "假設所有推車從同一地點出發"
        self.start = L[self.cart.INIT_LOC.iloc[0]]
        self.src = np.array([L[v] for v in self.wip.FROM])
        self.dst = np.array([L[v] for v in self.wip.TO])
        self.q = self.wip["Remaining Q-Time"].to_numpy().astype(np.int64)
        self.n, self.m = len(self.wip), len(self.cart)

    def write(self, routes, out_xlsx):
        """routes：每台推車一個事件串列，事件 (w, 0) 取貨、(w, 1) 送貨。回傳 (適應值, 報廢, 距離, 使用台數)。"""
        rows, rej, dist = [], 0, 0
        for c, r in enumerate(routes):
            loc, t = self.start, 0
            for k, (w, d) in enumerate(r, 1):
                nxt = self.dst[w] if d else self.src[w]
                t += int(self.D[loc, nxt])
                loc = nxt
                if d and t > self.q[w]:
                    rej += 1
                rows.append([self.cart.CART_ID.iloc[c], k, self.wip.WIP_ID.iloc[w], "DELIVERY" if d else "PICKUP", t])
            dist += t
        pd.DataFrame(rows, columns=["CART_ID", "ORDER", "WIP_ID", "ACTION", "COMPLETE_TIME"]).to_excel(out_xlsx, index=False)
        return rej * PENALTY + dist, rej, dist, sum(1 for r in routes if r)


# ---------------------------------------------------------------- B
def run_b(data, cap, seed, limit, out_xlsx):
    from ga_tabu_route import RouteGaTabu
    s = RouteGaTabu(data, seed=seed, batch_capacity=cap)
    s.time_limit = limit
    s.run()
    s.save_results(out_xlsx)
    r = s.analyze_results()
    best = min(v for _, v in s.history) if s.history else s.best_fit
    r["first_hit_sec"] = next((t for t, v in s.history if v == best), None)
    r["hit_limit"] = r["seconds"] >= limit
    r["history"] = s.history
    return r


# ---------------------------------------------------------------- C
def legal_orders(k):
    out = []

    def rec(seq, picked, done):
        if len(seq) == 2 * k:
            out.append(list(seq))
            return
        for i in range(k):
            if i not in picked:
                rec(seq + [(0, i)], picked | {i}, done)
            elif i not in done:
                rec(seq + [(1, i)], picked, done | {i})

    rec([], frozenset(), frozenset())
    return out


def exact_size(n, cap):
    """變數數（組合數）與要算的「組合 × 順序」次數。"""
    ks = range(1, min(cap, n) + 1)
    groups = sum(math.comb(n, k) for k in ks)
    evals = sum(math.comb(n, k) * math.factorial(2 * k) // 2 ** k for k in ks)
    return groups, evals


def run_c(data, cap, limit, out_xlsx, max_groups=3_000_000):
    I = Instance(data)
    n = I.n
    groups_est, evals_est = exact_size(n, cap)
    info = {"variables": groups_est, "order_evals": evals_est}
    if groups_est > max_groups:
        info.update(status="SKIPPED", note=f"組合數 {groups_est:,} 超過上限 {max_groups:,}")
        return info
    t0 = time.time()
    blocks = []                                           # 每個 k 一組：(組合陣列 S, 最佳成本, 最佳順序編號)
    for k in range(1, min(cap, n) + 1):
        S = np.array(list(itertools.combinations(range(n), k)), dtype=np.int32).reshape(-1, k)
        best = np.full(len(S), np.iinfo(np.int64).max)
        arg = np.zeros(len(S), dtype=np.int32)
        for oi, order in enumerate(legal_orders(k)):
            prev = np.full(len(S), I.start)
            t = np.zeros(len(S), dtype=np.int64)
            scrap = np.zeros(len(S), dtype=np.int64)
            for is_d, slot in order:
                w = S[:, slot]
                nxt = I.dst[w] if is_d else I.src[w]
                t = t + I.D[prev, nxt]
                prev = nxt
                if is_d:
                    scrap += t > I.q[w]
            c = scrap * PENALTY + t
            better = c < best
            best = np.where(better, c, best)
            arg = np.where(better, oi, arg)
        blocks.append((S, best, arg))
    t_enum = time.time() - t0

    # 整數規劃：用稀疏矩陣一次建好（逐一建立變數物件在 80 批、容量 4 時會用掉超過 12 GB 記憶體）
    import scipy.sparse as sp
    from ortools.linear_solver.python import model_builder as mb
    G = sum(len(b[0]) for b in blocks)
    rows, cols, off = [], [], 0
    for S, _, _ in blocks:
        k = S.shape[1]
        rows.append(S.ravel())
        cols.append(np.repeat(np.arange(off, off + len(S), dtype=np.int64), k))
        off += len(S)
    rows.append(np.full(G, n, dtype=np.int64))            # 最後一列：組合數 ≤ 推車數
    cols.append(np.arange(G, dtype=np.int64))
    A = sp.csr_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n + 1, G))
    cost = np.concatenate([b[1] for b in blocks]).astype(float)
    model = mb.Model()
    model.helper.fill_model_from_sparse_data(np.zeros(G), np.ones(G), cost,
                                             np.concatenate([np.ones(n), [0]]), np.concatenate([np.ones(n), [I.m]]), A)
    for j in range(G):
        model.helper.set_var_integrality(j, True)
    solver = mb.Solver("scip")
    solver.set_time_limit_in_seconds(limit)
    t1 = time.time()
    status = solver.solve(model)
    t_solve = time.time() - t1
    name = {mb.SolveStatus.OPTIMAL: "OPTIMAL", mb.SolveStatus.FEASIBLE: "FEASIBLE"}.get(status, str(status))
    info.update(status=name, enum_sec=round(t_enum, 2), solve_sec=round(t_solve, 2), seconds=round(t_enum + t_solve, 2))
    if name not in ("OPTIMAL", "FEASIBLE"):
        return info
    info["bound"] = round(solver.best_objective_bound, 1)
    val = np.array([solver.value(model.var_from_index(j)) for j in range(G)]) if G < 200_000 else None
    if val is None:
        val = solver.values(model.get_variables()).to_numpy()
    chosen = np.flatnonzero(val > 0.5)
    routes, starts, tpls = [], np.cumsum([0] + [len(b[0]) for b in blocks]), {}
    for j in chosen:
        k = int(np.searchsorted(starts, j, side="right"))   # 第 k 組（1 起算）
        S, _, arg = blocks[k - 1]
        i = j - starts[k - 1]
        tpls.setdefault(k, legal_orders(k))
        routes.append([(int(S[i][slot]), is_d) for is_d, slot in tpls[k][arg[i]]])
    routes += [[] for _ in range(I.m - len(routes))]
    f, rej, dist, used = I.write(routes, out_xlsx)
    info.update(fitness=f, rejections=rej, distance=dist, carts_used=used)
    return info


# ---------------------------------------------------------------- D
def run_d(data, cap, limit, out_xlsx):
    from ortools.linear_solver import pywraplp
    I = Instance(data)
    n, m, D = I.n, I.m, I.D
    P = lambda w: 1 + w
    Dl = lambda w: 1 + n + w
    S = 2 * n + 1
    loc = np.concatenate(([I.start], I.src, I.dst))     # 節點 0..2n 的地點
    t0 = time.time()
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetTimeLimit(int(limit * 1000))
    maxd = int(D.max())
    Tmax = 2 * cap * maxd
    arcs = {}
    for w in range(n):
        arcs[(0, P(w))] = int(D[I.start, I.src[w]])
        arcs[(Dl(w), S)] = 0
    for i in range(1, 2 * n + 1):
        for j in range(1, 2 * n + 1):
            if i == j:
                continue
            wi, di = (i - 1) % n, i > n
            wj, dj = (j - 1) % n, j > n
            if di and not dj and wi == wj:                # D_w → P_w 不可能
                continue
            arcs[(i, j)] = int(D[loc[i], loc[j]])
    x = {a: solver.BoolVar("") for a in arcs}
    T = [solver.NumVar(0, Tmax, "") for _ in range(2 * n + 1)]
    u = [solver.NumVar(1, 2 * cap, "") for _ in range(2 * n + 1)]
    v = [solver.NumVar(0, n, "") for _ in range(2 * n + 1)]
    z = [solver.BoolVar("") for _ in range(n)]
    out = [[] for _ in range(2 * n + 2)]
    inn = [[] for _ in range(2 * n + 2)]
    for (i, j), var in x.items():
        out[i].append(var)
        inn[j].append(var)
    for k in range(1, 2 * n + 1):
        solver.Add(solver.Sum(out[k]) == 1)
        solver.Add(solver.Sum(inn[k]) == 1)
    solver.Add(solver.Sum(out[0]) == solver.Sum(inn[S]))
    solver.Add(solver.Sum(out[0]) <= m)
    U = 2 * cap
    for (i, j), var in x.items():
        if j == S:
            continue
        d = arcs[(i, j)]
        if i == 0:
            solver.Add(T[j] >= d * var)
            w = j - 1
            solver.Add(v[j] >= (w + 1) * var)               # 路線第一個取貨點：標籤 = w + 1
            solver.Add(v[j] <= (w + 1) * var + n * (1 - var))
            continue
        solver.Add(T[j] >= T[i] + d - (Tmax + d) * (1 - var))
        solver.Add(u[j] >= u[i] + 1 - U * (1 - var))
        solver.Add(v[j] - v[i] <= n * (1 - var))
        solver.Add(v[i] - v[j] <= n * (1 - var))
    for w in range(n):
        solver.Add(u[Dl(w)] >= u[P(w)] + 1)
        solver.Add(v[P(w)] == v[Dl(w)])
        solver.Add(T[Dl(w)] <= int(I.q[w]) + Tmax * z[w])
    solver.Minimize(solver.Sum([c * x[a] for a, c in arcs.items() if c]) + PENALTY * solver.Sum(z))
    t_build = time.time() - t0
    status = solver.Solve()
    sec = time.time() - t0
    name = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(status, f"STATUS{status}")
    info = {"status": name, "variables": solver.NumVariables(), "constraints": solver.NumConstraints(),
            "build_sec": round(t_build, 2), "seconds": round(sec, 2)}
    if name not in ("OPTIMAL", "FEASIBLE"):
        info["note"] = "時間內沒有找到可行解"
        return info
    info["bound"] = round(solver.Objective().BestBound(), 1)
    succ = {i: j for (i, j), var in x.items() if var.solution_value() > 0.5}
    routes = []
    for (i, j), var in x.items():
        if i == 0 and var.solution_value() > 0.5:
            r, k = [], j
            while k != S:
                r.append(((k - 1) % n, int(k > n)))
                k = succ[k]
            routes.append(r)
    routes += [[] for _ in range(m - len(routes))]
    f, rej, dist, used = I.write(routes, out_xlsx)
    info.update(fitness=f, rejections=rej, distance=dist, carts_used=used, mip_obj=round(solver.Objective().Value()))
    return info


# ---------------------------------------------------------------- E
def run_e(data, cap, limit, out_xlsx):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    I = Instance(data)
    n, m = I.n, I.m
    N = 2 * n + 2                                         # 0 起點、1..n 取貨、n+1..2n 送貨、2n+1 虛擬終點
    END = 2 * n + 1
    loc = np.concatenate(([I.start], I.src, I.dst, [I.start]))
    M = I.D[np.ix_(loc, loc)].astype(int)
    M[:, END] = 0
    t0 = time.time()
    mgr = pywrapcp.RoutingIndexManager(N, m, [0] * m, [END] * m)
    rt = pywrapcp.RoutingModel(mgr)
    mat = M.tolist()
    cb = rt.RegisterTransitCallback(lambda a, b: mat[mgr.IndexToNode(a)][mgr.IndexToNode(b)])
    rt.SetArcCostEvaluatorOfAllVehicles(cb)
    rt.AddDimension(cb, 0, int(M.max()) * 2 * cap + 1, True, "Time")
    tdim = rt.GetDimensionOrDie("Time")
    cnt = rt.RegisterUnaryTransitCallback(lambda a: 1 if 1 <= mgr.IndexToNode(a) <= n else 0)
    rt.AddDimension(cnt, 0, cap, True, "Count")
    for w in range(n):
        p, d = mgr.NodeToIndex(1 + w), mgr.NodeToIndex(1 + n + w)
        rt.AddPickupAndDelivery(p, d)
        rt.solver().Add(rt.VehicleVar(p) == rt.VehicleVar(d))
        rt.solver().Add(tdim.CumulVar(p) <= tdim.CumulVar(d))
        tdim.SetCumulVarSoftUpperBound(d, int(I.q[w]), PENALTY)
    prm = pywrapcp.DefaultRoutingSearchParameters()
    prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    prm.time_limit.FromMilliseconds(int(limit * 1000))
    first, hist = {}, []

    def on_solution():
        first.setdefault("sec", round(time.time() - t0, 2))
        hist.append((round(time.time() - t0, 2), rt.CostVar().Max()))
    rt.AddAtSolutionCallback(on_solution)
    sol = rt.SolveWithParameters(prm)
    sec = time.time() - t0
    info = {"status": "FEASIBLE" if sol else "NO_SOLUTION", "seconds": round(sec, 2), "first_sol_sec": first.get("sec"), "history": hist}
    if not sol:
        return info
    routes = []
    for k in range(m):
        idx, r = rt.Start(k), []
        idx = sol.Value(rt.NextVar(idx))
        while not rt.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            r.append(((node - 1) % n, int(node > n)))
            idx = sol.Value(rt.NextVar(idx))
        routes.append(r)
    f, rej, dist, used = I.write(routes, out_xlsx)
    info.update(fitness=f, rejections=rej, distance=dist, carts_used=used, solver_obj=sol.ObjectiveValue())
    return info
