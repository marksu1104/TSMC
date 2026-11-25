"""推車排程：遺傳演算法 + 禁忌搜尋（路徑式，適用於大規模與高容量）

和 GAwithTabu 的關鍵差異：每台推車直接存「取送順序」（路徑），而不是只存分到哪些 WIP。
所有操作都建立在同一個動作上：最便宜插入。

最便宜插入：把一批 WIP 的取貨點和送貨點插進某台推車的路徑，試過所有「取貨在送貨之前」的位置組合，
選增加成本最少的一組。路徑有 L 個事件時，位置組合約 L²/2 個，全部用 numpy 一次算完。
不需要列舉整條路徑的所有順序，所以每台推車載多少批都能用。

成本：報廢數 × 10000 + 行駛距離（和原版相同）。時間等於距離，送達時間超過 Q-time 就算報廢。
容量：每台推車分配的 WIP 數上限（和原版相同）。

演算法結構沿用原版：
- 初始族群：隨機順序 + 最便宜插入（每批從成本最低的前幾台推車隨機選一台，保持多樣性）
- 交配：一半推車沿用父代 1 的路徑，其餘推車沿用父代 2 的路徑並刪掉重複的 WIP，缺的 WIP 用最便宜插入補回
- 突變：拿掉幾批 WIP 或整台推車的 WIP，再用最便宜插入放回（破壞與重建）
- 禁忌搜尋：搬移（從一台推車拿出，插到另一台）、交換、同車重新插入；禁忌清單記錄「WIP 不能回到原推車」
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd

PENALTY = 10000
INF = 10**12


class RouteGaTabu:
    def __init__(self, data_file="data.xlsx", seed=42, batch_capacity=2, sheet_wip="WIP"):
        self.rng = random.Random(seed)
        cart = pd.read_excel(data_file, sheet_name="CART")
        wip = pd.read_excel(data_file, sheet_name=sheet_wip)
        xf = pd.read_excel(data_file, sheet_name="XFER_TIME")
        locs = sorted(set(xf["FROM"]) | set(xf["TO"]))
        lid = {v: i for i, v in enumerate(locs)}
        self.D = np.zeros((len(locs), len(locs)), dtype=np.int64)
        for f, t, x in xf[["FROM", "TO", "XFER_TIME"]].itertuples(index=False):
            self.D[lid[f], lid[t]] = x
        self.wip_ids, self.cart_ids = list(wip["WIP_ID"]), list(cart["CART_ID"])
        self.n, self.m = len(self.wip_ids), len(self.cart_ids)
        self.src = np.array([lid[v] for v in wip["FROM"]])
        self.dst = np.array([lid[v] for v in wip["TO"]])
        self.q = wip["Remaining Q-Time"].to_numpy().astype(np.int64)
        self.start = np.array([lid[v] for v in cart["INIT_LOC"]])
        # 事件編號：2w 是第 w 批取貨，2w+1 是送貨
        self.ev_loc = np.empty(2 * self.n, dtype=np.int64)
        self.ev_loc[0::2], self.ev_loc[1::2] = self.src, self.dst
        self.ev_due = np.full(2 * self.n, INF, dtype=np.int64)
        self.ev_due[1::2] = self.q
        self.cap = batch_capacity

        self.pop_size = 30
        self.generations = 200
        self.elite_size = 3
        self.tournament_size = 3
        self.crossover_rate = 0.8
        self.mutation_rate = 0.5
        self.ruin_max = 0.1             # 破壞與重建：最多拿掉 10% 的 WIP
        self.dissolve_rate = 0.3        # 突變時改成拿掉整台推車的機率
        self.init_top = 3               # 初始解從成本最低的前幾台推車中隨機選
        self.earlier_ga_stop = 30
        self.tabu_frequency = 5
        self.tabu_elite = 2
        self.tabu_iterations = 300
        self.earlier_ts_stop = 100
        self.tabu_tenure = (5, 20)
        self.exact_k = 4                # 一台車不超過這個批數時，直接列舉所有合法順序（2 批 6 種、4 批 2,520 種）
        small = batch_capacity <= self.exact_k
        self.ts_wips = None if small else 20    # 每次迭代檢查幾批 WIP（None：全部）
        self.ts_carts = None if small else 10   # 每批 WIP 檢查幾台候選推車（None：全部）
        self._tpl = {k: self._orders(k) for k in range(1, self.exact_k + 1)}
        self._exact_cache = {}
        self.time_limit = None          # 秒數上限（None 表示不限）
        self.best_routes, self.best_fit = None, None
        self.history = []
        self.execution_time = 0.0

    # ------------------------------------------------------------------ 單條路徑
    @staticmethod
    def _orders(k):
        """k 批的所有合法順序。編號 0..k-1 是取貨、k..2k-1 是送貨。"""
        out = []

        def rec(seq, picked, done):
            if len(seq) == 2 * k:
                out.append(seq)
                return
            for i in range(k):
                if i not in picked:
                    rec(seq + [i], picked | {i}, done)
                elif i not in done:
                    rec(seq + [k + i], picked, done | {i})

        rec([], frozenset(), frozenset())
        return np.array(out)

    def exact_route(self, c, wips):
        """批數少時：列舉所有合法順序取最佳，同一組 WIP 只算一次。回傳 (成本, 路徑)。"""
        if not wips:
            return 0, []
        key = (int(self.start[c]), frozenset(wips))
        hit = self._exact_cache.get(key)
        if hit is not None:
            return hit
        w = np.array(sorted(wips))
        k = len(w)
        tpl = self._tpl[k]
        locs = np.concatenate([self.src[w], self.dst[w]])[tpl]
        prev = np.concatenate([np.full((len(tpl), 1), self.start[c]), locs[:, :-1]], axis=1)
        t = np.cumsum(self.D[prev, locs], axis=1)
        due = np.concatenate([np.full(k, INF), self.q[w]])[tpl]
        cost = (t > due).sum(axis=1) * PENALTY + t[:, -1]
        b = int(np.argmin(cost))
        route = [int(2 * w[e] if e < k else 2 * w[e - k] + 1) for e in tpl[b]]
        res = (int(cost[b]), route)
        self._exact_cache[key] = res
        return res

    @staticmethod
    def wips_of(r):
        return [e >> 1 for e in r if not e & 1]

    def after_remove(self, c, r, w):
        """拿掉第 w 批之後的路徑與成本（批數少時重新取最佳順序）。"""
        rr = self.remove(r, w)
        if len(rr) // 2 <= self.exact_k:
            return self.exact_route(c, self.wips_of(rr))
        return self.route_cost(c, rr), rr

    def route_cost(self, c, r):
        if not r:
            return 0
        ev = np.asarray(r)
        locs = self.ev_loc[ev]
        prev = np.concatenate(([self.start[c]], locs[:-1]))
        t = np.cumsum(self.D[prev, locs])
        return int((t > self.ev_due[ev]).sum()) * PENALTY + int(t[-1])

    def best_insert(self, c, r, w):
        """把第 w 批的取貨與送貨插進推車 c 的路徑 r。回傳 (新成本, 新路徑)。"""
        if len(r) // 2 + 1 <= self.exact_k:
            return self.exact_route(c, self.wips_of(r) + [w])
        D, P, Q, qw = self.D, self.src[w], self.dst[w], self.q[w]
        s = self.start[c]
        L = len(r)
        if L == 0:
            t = D[s, P] + D[P, Q]
            return int(t > qw) * PENALTY + int(t), [2 * w, 2 * w + 1]
        ev = np.asarray(r)
        loc = self.ev_loc[ev]
        prevloc = np.concatenate(([s], loc))                 # 第 g 個空隙前一個地點（g = 0..L）
        t = np.cumsum(self.D[prevloc[:-1], loc])
        prevt = np.concatenate(([0], t))
        slack = self.ev_due[ev] - t                          # 取貨事件的 slack 為無限大
        dist0 = t[-1]
        E = np.arange(L)
        # 情況 A：取貨後立刻送貨，插在第 g 個空隙
        g = np.arange(L + 1)
        nxt = np.concatenate((loc, [loc[-1]]))
        tail = np.where(g < L, D[Q, nxt] - D[prevloc, nxt], 0)
        addA = D[prevloc, P] + D[P, Q] + tail
        delivA = prevt + D[prevloc, P] + D[P, Q]
        shiftA = np.where(E[None, :] >= g[:, None], addA[:, None], 0)
        lateA = (shiftA > slack[None, :]).sum(1) + (delivA > qw)
        costA = lateA * PENALTY + dist0 + addA
        # 情況 B：取貨插在空隙 i，送貨插在空隙 j（i < j），中間隔著原本的事件
        I, J = np.triu_indices(L + 1, k=1)
        d1 = D[prevloc[I], P] + D[P, loc[I]] - D[prevloc[I], loc[I]]
        lj = loc[J - 1]
        tailB = np.where(J < L, D[Q, nxt[J]] - D[lj, nxt[J]], 0)
        d2 = D[lj, Q] + tailB
        delivB = t[J - 1] + d1 + D[lj, Q]
        shiftB = d1[:, None] * (E[None, :] >= I[:, None]) + d2[:, None] * (E[None, :] >= J[:, None])
        lateB = (shiftB > slack[None, :]).sum(1) + (delivB > qw)
        costB = lateB * PENALTY + dist0 + d1 + d2
        a, b = int(np.argmin(costA)), int(np.argmin(costB))
        if costA[a] <= costB[b]:
            return int(costA[a]), r[:a] + [2 * w, 2 * w + 1] + r[a:]
        i, j = int(I[b]), int(J[b])
        return int(costB[b]), r[:i] + [2 * w] + r[i:j] + [2 * w + 1] + r[j:]

    @staticmethod
    def remove(r, w):
        return [e for e in r if e >> 1 != w]

    def improve_route(self, c, r, cost=None):
        """同車重新插入：每批 WIP 拿出來再插回最好的位置，直到沒有改善。"""
        if len(r) // 2 <= self.exact_k:
            return self.exact_route(c, self.wips_of(r))[::-1]
        cost = self.route_cost(c, r) if cost is None else cost
        improved = True
        while improved:
            improved = False
            for w in [e >> 1 for e in r if not e & 1]:
                rr = self.remove(r, w)
                nc, nr = self.best_insert(c, rr, w)
                if nc < cost:
                    r, cost, improved = nr, nc, True
        return r, cost

    # ------------------------------------------------------------------ 整體解
    def evaluate(self, routes):
        return sum(self.route_cost(c, r) for c, r in enumerate(routes))

    def size(self, r):
        return len(r) // 2

    def cheapest_cart(self, routes, costs, w, exclude=None, top=1):
        """所有有空位的推車中，插入第 w 批增加成本最少的（起點相同的空車只算一台）。"""
        cands, seen_empty = [], set()
        for c in range(self.m):
            if c == exclude or self.size(routes[c]) >= self.cap:
                continue
            if not routes[c]:
                if self.start[c] in seen_empty:
                    continue
                seen_empty.add(self.start[c])
            nc, nr = self.best_insert(c, routes[c], w)
            cands.append((nc - costs[c], c, nc, nr))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[self.rng.randrange(min(top, len(cands)))] if top > 1 else cands[0]

    def insert_all(self, routes, costs, wips, exclude=None, top=1):
        for w in wips:
            pick = self.cheapest_cart(routes, costs, w, exclude, top)
            if pick is None:
                pick = self.cheapest_cart(routes, costs, w, None, top)
            _, c, nc, nr = pick
            routes[c], costs[c] = nr, nc

    def new_solution(self):
        routes, costs = [[] for _ in range(self.m)], [0] * self.m
        wips = list(range(self.n))
        self.rng.shuffle(wips)
        self.insert_all(routes, costs, wips, top=self.init_top)
        return routes, costs

    def crossover(self, p1, p2):
        carts = list(range(self.m))
        self.rng.shuffle(carts)
        routes, used = [[] for _ in range(self.m)], set()
        for c in carts[: self.m // 2]:
            routes[c] = list(p1[0][c])
            used.update(e >> 1 for e in routes[c])
        for c in carts[self.m // 2:]:
            routes[c] = [e for e in p2[0][c] if e >> 1 not in used]
            used.update(e >> 1 for e in routes[c])
        costs = [self.route_cost(c, r) for c, r in enumerate(routes)]
        missing = [w for w in range(self.n) if w not in used]
        self.rng.shuffle(missing)
        self.insert_all(routes, costs, missing)
        return routes, costs

    def mutate(self, sol):
        routes, costs = [list(r) for r in sol[0]], list(sol[1])
        if self.rng.random() < self.dissolve_rate:                 # 拿掉整台推車的 WIP
            c = self.rng.choice([c for c in range(self.m) if routes[c]])
            out = [e >> 1 for e in routes[c] if not e & 1]
            routes[c], costs[c] = [], 0
            self.rng.shuffle(out)
            self.insert_all(routes, costs, out, exclude=c)
            return routes, costs
        k = self.rng.randint(1, max(2, int(self.n * self.ruin_max)))  # 拿掉幾批 WIP
        out = self.rng.sample(range(self.n), k)
        outs = set(out)
        for c in range(self.m):
            if any(e >> 1 in outs for e in routes[c]):
                routes[c] = [e for e in routes[c] if e >> 1 not in outs]
                costs[c] = self.route_cost(c, routes[c])
        self.insert_all(routes, costs, out)
        return routes, costs

    # ------------------------------------------------------------------ 禁忌搜尋
    def tabu_search(self, sol, deadline=None):
        routes, costs = [list(r) for r in sol[0]], list(sol[1])
        where = {e >> 1: c for c in range(self.m) for e in routes[c]}
        total = sum(costs)
        best = ([list(r) for r in routes], list(costs))
        best_total, stall, tabu = total, 0, {}
        for it in range(1, self.tabu_iterations + 1):
            if deadline and time.time() > deadline:
                break
            pick, pick_val = None, None
            W = range(self.n) if self.ts_wips is None else self.rng.sample(range(self.n), min(self.ts_wips, self.n))
            for w in W:
                a = where[w]
                ca, ra = self.after_remove(a, routes[a], w)
                if self.size(routes[a]) > self.exact_k:          # 同車重新插入（批數少時順序已是最佳）
                    nc, nr = self.best_insert(a, ra, w)
                    val = total - costs[a] + nc
                    if nc < costs[a] and (pick_val is None or val < pick_val):
                        pick, pick_val = (("same", w, a, nr, nc),), val
                # 搬移或交換到其他推車
                others = [c for c in range(self.m) if c != a]
                empty_seen = set()
                cand = []
                for c in self.rng.sample(others, len(others)):
                    if not routes[c]:
                        if self.start[c] in empty_seen:
                            continue
                        empty_seen.add(self.start[c])
                    cand.append(c)
                    if self.ts_carts is not None and len(cand) >= self.ts_carts:
                        break
                for b in cand:
                    if self.size(routes[b]) < self.cap:              # 搬移
                        nb_cost, nb = self.best_insert(b, routes[b], w)
                        val = total - costs[a] - costs[b] + ca + nb_cost
                        if tabu.get((w, b), 0) > it and val >= best_total:
                            pass
                        elif pick_val is None or val < pick_val:
                            pick, pick_val = (("move", w, a, b, ra, ca, nb, nb_cost),), val
                    if not routes[b]:
                        continue
                    vs = self.wips_of(routes[b])                       # 交換：和 b 的一批互換
                    if self.ts_carts is not None:
                        vs = [self.rng.choice(vs)]
                    for v in vs:
                        _, rb = self.after_remove(b, routes[b], v)
                        nb_cost, nb = self.best_insert(b, rb, w)
                        na_cost, na = self.best_insert(a, ra, v)
                        val = total - costs[a] - costs[b] + na_cost + nb_cost
                        if (tabu.get((w, b), 0) > it or tabu.get((v, a), 0) > it) and val >= best_total:
                            continue
                        if pick_val is None or val < pick_val:
                            pick, pick_val = (("swap", w, v, a, b, na, na_cost, nb, nb_cost),), val
            if pick is None:
                continue
            mv = pick[0]
            tenure = self.rng.randint(*self.tabu_tenure)
            if mv[0] == "same":
                _, w, a, nr, nc = mv
                routes[a], costs[a] = nr, nc
                changed = (a,)
            elif mv[0] == "move":
                _, w, a, b, ra, ca, nb, nb_cost = mv
                routes[a], costs[a], routes[b], costs[b] = ra, ca, nb, nb_cost
                where[w] = b
                tabu[(w, a)] = it + tenure
                changed = (a, b)
            else:
                _, w, v, a, b, na, na_cost, nb, nb_cost = mv
                routes[a], costs[a], routes[b], costs[b] = na, na_cost, nb, nb_cost
                where[w], where[v] = b, a
                tabu[(w, a)] = tabu[(v, b)] = it + tenure
                changed = (a, b)
            for c in changed:                                   # 被改到的推車重新整理順序
                routes[c], costs[c] = self.improve_route(c, routes[c], costs[c])
            total = sum(costs)
            if total < best_total:
                best, best_total, stall = ([list(r) for r in routes], list(costs)), total, 0
            else:
                stall += 1
                if stall >= self.earlier_ts_stop:
                    break
        # 最後把每台推車的順序再整理一次
        routes, costs = best
        for c in range(self.m):
            if routes[c]:
                routes[c], costs[c] = self.improve_route(c, routes[c], costs[c])
        return (routes, costs), sum(costs)

    # ------------------------------------------------------------------ 主流程
    def tournament(self, pop, fits):
        idx = self.rng.sample(range(len(pop)), self.tournament_size)
        return pop[min(idx, key=lambda i: fits[i])]

    def run(self, verbose=False):
        t0 = time.time()
        deadline = t0 + self.time_limit if self.time_limit else None
        pop = [self.new_solution() for _ in range(self.pop_size)]
        fits = [sum(s[1]) for s in pop]
        stall = 0
        self.best_fit, self.history = None, []
        for gen in range(self.generations):
            if deadline and time.time() > deadline:
                break
            order = sorted(range(len(pop)), key=lambda i: fits[i])
            if gen % self.tabu_frequency == 0:
                for i in order[: self.tabu_elite]:
                    s, f = self.tabu_search(pop[i], deadline)
                    if f < fits[i]:
                        pop[i], fits[i] = s, f
                order = sorted(range(len(pop)), key=lambda i: fits[i])
            gb = fits[order[0]]
            self.history.append((round(time.time() - t0, 2), gb))
            if self.best_fit is None or gb < self.best_fit:
                self.best_fit, self.best_routes, stall = gb, [list(r) for r in pop[order[0]][0]], 0
                if verbose:
                    print(f"第 {gen} 代：{gb}（{time.time() - t0:.1f} 秒）", flush=True)
            else:
                stall += 1
                if stall >= self.earlier_ga_stop:
                    break
            new = [([list(r) for r in pop[i][0]], list(pop[i][1])) for i in order[: self.elite_size]]
            while len(new) < self.pop_size:
                p1, p2 = self.tournament(pop, fits), self.tournament(pop, fits)
                child = self.crossover(p1, p2) if self.rng.random() < self.crossover_rate else p1
                if self.rng.random() < self.mutation_rate or child is p1:
                    child = self.mutate(child)
                new.append(child)
            pop, fits = new, [sum(s[1]) for s in new]
        best = (self.best_routes, [self.route_cost(c, r) for c, r in enumerate(self.best_routes)])
        s, f = self.tabu_search(best, None if not deadline else max(deadline, time.time() + 5))
        if f < self.best_fit:
            self.best_fit, self.best_routes = f, s[0]
        self.execution_time = time.time() - t0
        return self.best_routes, self.best_fit

    # ------------------------------------------------------------------ 輸出（格式和原版相同）
    def generate_schedule(self):
        rows = []
        for c, r in enumerate(self.best_routes):
            loc, t = self.start[c], 0
            for k, e in enumerate(r, 1):
                t += self.D[loc, self.ev_loc[e]]
                loc = self.ev_loc[e]
                rows.append([self.cart_ids[c], k, self.wip_ids[e >> 1], "DELIVERY" if e & 1 else "PICKUP", int(t)])
        return pd.DataFrame(rows, columns=["CART_ID", "ORDER", "WIP_ID", "ACTION", "COMPLETE_TIME"])

    def save_results(self, filename):
        self.generate_schedule().to_excel(filename, index=False)

    def analyze_results(self):
        used = sum(1 for r in self.best_routes if r)
        return {"fitness": self.best_fit, "rejections": self.best_fit // PENALTY, "distance": self.best_fit % PENALTY,
                "carts_used": used, "seconds": round(self.execution_time, 2)}
