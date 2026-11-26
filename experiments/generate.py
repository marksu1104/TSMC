"""生成規模實驗用的資料（固定種子、可重現）。

做法：
1. 讀 data.xlsx，沿用它的 50 個地點與 XFER_TIME（距離表原封不動複製）。
2. 統計原始 40 批 WIP（數字由本程式印出，也寫在 experiments/README.md）：
   - FROM：40 批用到 34 個不同地點，沒有特別集中的地點（最多重複 2 次）→ 從 50 個地點均勻抽樣。
   - TO：40 批中有 1 批 FROM 等於 TO（W24，LOC18→LOC18）→ 以 1/40 的機率令 TO = FROM，
     其餘從另外 49 個地點均勻抽樣。
   - Remaining Q-Time：36 到 115，和距離幾乎無關（和「LOC1→FROM→TO」直送距離的相關係數 0.13）；
     但每批都滿足 Q-time ≥ 直送距離 + 1（最小餘裕 1），也就是單獨派一台車一定來得及。
     → 從原始 40 個 Q-time 值重複抽樣（bootstrap），抽到「Q-time ≤ 直送距離」就重抽。
3. WIP 數 n ∈ {40, 80, 160}，推車數 m = n / 2，全部從 LOC1 出發；每種 n 生成 3 組（種子 1、2、3）。
輸出：experiments/data/gen_n{n}_s{seed}.xlsx，格式和 data.xlsx 相同（CART、WIP、XFER_TIME 三張表）。
用法：python generate.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data.xlsx")
OUT = os.path.join(HERE, "data")

xf = pd.read_excel(SRC, sheet_name="XFER_TIME")
wip0 = pd.read_excel(SRC, sheet_name="WIP")
locs = sorted(set(xf.FROM) | set(xf.TO), key=lambda s: int(s[3:]))
D = {(a, b): t for a, b, t in xf[["FROM", "TO", "XFER_TIME"]].itertuples(index=False)}
Q0 = wip0["Remaining Q-Time"].to_numpy()
direct0 = np.array([D[("LOC1", f)] + D[(f, t)] for f, t in zip(wip0.FROM, wip0.TO)])
p_same = (wip0.FROM == wip0.TO).mean()


def summary():
    print(f"原始資料：{len(wip0)} 批，FROM 有 {wip0.FROM.nunique()} 個不同地點、TO 有 {wip0.TO.nunique()} 個，"
          f"FROM = TO 的有 {(wip0.FROM == wip0.TO).sum()} 批")
    print(f"Q-time：最小 {Q0.min()}、中位數 {np.median(Q0):.0f}、最大 {Q0.max()}；"
          f"Q-time − 直送距離 最小 {np.min(Q0 - direct0)}；和直送距離的相關係數 {np.corrcoef(Q0, direct0)[0, 1]:.2f}")


def make(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        f = locs[rng.integers(len(locs))]
        if rng.random() < p_same:
            t = f
        else:
            others = [l for l in locs if l != f]
            t = others[rng.integers(len(others))]
        direct = D[("LOC1", f)] + D[(f, t)]
        while True:
            q = int(Q0[rng.integers(len(Q0))])
            if q > direct:
                break
        rows.append((f"W{i + 1:03d}", q, f, t))
    wip = pd.DataFrame(rows, columns=["WIP_ID", "Remaining Q-Time", "FROM", "TO"])
    cart = pd.DataFrame({"CART_ID": [f"C{i + 1:02d}" for i in range(n // 2)], "INIT_LOC": "LOC1"})
    path = os.path.join(OUT, f"gen_n{n}_s{seed}.xlsx")
    with pd.ExcelWriter(path) as w:
        cart.to_excel(w, sheet_name="CART", index=False)
        wip.to_excel(w, sheet_name="WIP", index=False)
        xf.to_excel(w, sheet_name="XFER_TIME", index=False)
    print(f"{path}: {n} 批、{n // 2} 台推車，FROM = TO 有 {(wip.FROM == wip.TO).sum()} 批，"
          f"Q-time 中位數 {wip['Remaining Q-Time'].median():.0f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    summary()
    for n in (40, 80, 160):
        for seed in (1, 2, 3):
            make(n, seed)
