"""獨立檢查排程結果：只讀原始資料和輸出的排程 Excel，不使用排程器的任何程式碼。

檢查：每批 WIP 恰好由同一台推車先取貨、後送貨各一次；每台推車的 WIP 數不超過容量；
完成時間等於從推車起點依序累加的移動距離；最後重新計算報廢數與總距離。
用法：python check_schedule.py data.xlsx schedule.xlsx 容量
"""
import sys

import pandas as pd

data, sched_file, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
cart = pd.read_excel(data, sheet_name="CART").set_index("CART_ID")
wip = pd.read_excel(data, sheet_name="WIP").set_index("WIP_ID")
xfer = pd.read_excel(data, sheet_name="XFER_TIME")
dist = {(f, t): x for f, t, x in xfer[["FROM", "TO", "XFER_TIME"]].itertuples(index=False)}
sched = pd.read_excel(sched_file)

problems, total_dist, total_rej = [], 0, 0
seen = {}
for c, g in sched.groupby("CART_ID"):
    g = g.sort_values("ORDER")
    if c not in cart.index:
        problems.append(f"{c}: 不在 CART 表中")
        continue
    loc, t, onboard, done = cart.loc[c, "INIT_LOC"], 0, set(), set()
    for r in g.itertuples(index=False):
        w = r.WIP_ID
        nxt = wip.loc[w, "FROM"] if r.ACTION == "PICKUP" else wip.loc[w, "TO"]
        t += dist[(loc, nxt)]
        loc = nxt
        if t != r.COMPLETE_TIME:
            problems.append(f"{c} {w} {r.ACTION}: 完成時間 {r.COMPLETE_TIME}，重算為 {t}")
        if r.ACTION == "PICKUP":
            if w in seen:
                problems.append(f"{w}: 被取貨兩次")
            seen[w] = c
            onboard.add(w)
        else:
            if w not in onboard:
                problems.append(f"{c} {w}: 送貨前沒有取貨")
            onboard.discard(w)
            done.add(w)
            if t > wip.loc[w, "Remaining Q-Time"]:
                total_rej += 1
    if onboard:
        problems.append(f"{c}: {sorted(onboard)} 取貨後沒有送貨")
    if len(done) > cap:
        problems.append(f"{c}: 分到 {len(done)} 批，超過容量 {cap}")
    total_dist += t
missing = set(wip.index) - set(seen)
if missing:
    problems.append(f"沒有被排入的 WIP：{sorted(missing)}")
print(f"{sched_file}: 報廢 {total_rej}、距離 {total_dist}、適應值 {total_rej * 10000 + total_dist}、"
      f"使用推車 {sched.CART_ID.nunique()} 台、問題 {len(problems)} 個")
for p in problems[:20]:
    print("  ", p)
sys.exit(1 if problems else 0)
