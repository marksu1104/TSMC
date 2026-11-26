"""排定全部實驗，用 4 個行程平行執行（每個行程跑一個設定，彼此獨立）。

時間上限 T(n)：n = 40 為 60 秒、80 為 120 秒、160 為 300 秒。
- A（原版）、B（路徑式 GA–TS）：每個設定 5 個種子（0 到 4），上限 T(n)
  （B 有自己的提前停止規則，多數設定在上限前就結束）
- C（列舉 + SCIP）：組合數不超過 300 萬才執行，SCIP 上限 1800 秒；其餘只記錄規模估計
- D（路徑整數規劃，SCIP）、E（OR-Tools 路徑求解器）：上限 T(n)。兩者沒有隨機種子
  （同樣的輸入每次走同樣的搜尋），所以各跑一次
已經在結果檔裡的設定會跳過，可以中斷後重跑。
用法：python run_all.py [方法字母，例如 BE]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "results.jsonl")
LIMIT = {40: 60, 80: 120, 160: 300}
CAPS = (2, 4, 10, 20)
WORKERS = 4


def datasets():
    yield 40, os.path.join(HERE, "..", "data.xlsx")
    for n in (40, 80, 160):
        for s in (1, 2, 3):
            yield n, os.path.join(HERE, "data", f"gen_n{n}_s{s}.xlsx")


def jobs(methods):
    for n, path in datasets():
        for cap in CAPS:
            for m in methods:
                if m in "AB":
                    for seed in range(5):
                        yield m, path, cap, seed, LIMIT[n]
                elif m == "C":
                    yield m, path, cap, 0, 1800
                else:
                    yield m, path, cap, 0, LIMIT[n]


def done():
    out = set()
    if os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ERROR":          # 出錯的工作下次重跑
                continue
            out.add((r["method"], r["data"], r["cap"], r["seed"]))
    return out


def run(job):
    m, path, cap, seed, lim = job
    subprocess.run([sys.executable, os.path.join(HERE, "run_one.py"), m, path, str(cap), str(seed), str(lim), LOG],
                   cwd=HERE, stdout=subprocess.DEVNULL, stderr=open(os.path.join(HERE, "logs", "errors.log"), "a"))
    print("完成", m, os.path.basename(path), cap, seed, flush=True)


if __name__ == "__main__":
    methods = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
    finished = done()
    todo = [j for j in jobs(methods)
            if (j[0], os.path.splitext(os.path.basename(j[1]))[0], j[2], j[3]) not in finished]
    # 長的工作先跑（n 大的先），讓 4 個行程盡量同時結束
    todo.sort(key=lambda j: -j[4])
    print(f"共 {len(todo)} 個工作", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, todo))
