"""確認方法 D（不列舉的路徑整數規劃）的模型正確：在小資料上和方法 C（列舉 + SCIP，已知是精確解）比較。

小資料：取 data.xlsx 前 k 批 WIP（k = 6, 8, 10, 12），推車 k / 2 台，全部從 LOC1 出發；
容量 2 與 4。D 不設時間上限太久的話以 600 秒為上限，記錄是否證明最佳。
用法：python validate_d.py
"""
import json
import os

import pandas as pd

import methods

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "data", "small"), exist_ok=True)
os.makedirs(os.path.join(HERE, "out", "small"), exist_ok=True)
x = pd.ExcelFile(os.path.join(HERE, "..", "data.xlsx"))
cart, wip, xf = (pd.read_excel(x, s) for s in ("CART", "WIP", "XFER_TIME"))
rows = []
for k in (6, 8, 10, 12):
    path = os.path.join(HERE, "data", "small", f"data_first{k}.xlsx")
    with pd.ExcelWriter(path) as w:
        cart.head(k // 2).to_excel(w, sheet_name="CART", index=False)
        wip.head(k).to_excel(w, sheet_name="WIP", index=False)
        xf.to_excel(w, sheet_name="XFER_TIME", index=False)
    for cap in (2, 4):
        c = methods.run_c(path, cap, 600, os.path.join(HERE, "out", "small", f"first{k}_cap{cap}_C.xlsx"))
        d = methods.run_d(path, cap, 600, os.path.join(HERE, "out", "small", f"first{k}_cap{cap}_D.xlsx"))
        row = {"k": k, "cap": cap, "C": c.get("fitness"), "C_status": c["status"], "C_sec": c.get("seconds"),
               "D": d.get("fitness"), "D_status": d["status"], "D_bound": d.get("bound"), "D_sec": d["seconds"],
               "D_vars": d["variables"]}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
with open(os.path.join(HERE, "logs", "validate_d.jsonl"), "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
