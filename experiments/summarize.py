"""整理結果：logs/results.jsonl → results.csv、results.md（Markdown 表）、figures/time_quality_n{n}.png。

- 適應值一律用 check_schedule.py 重新算出的值（check_fitness）；檢查沒通過的解不計入。
- 參考值：C 證明最佳時用 C 的最佳解（標「最佳」）；否則用所有方法、所有種子中最好的解（標「已知最佳」）。
- 差距 = (適應值 − 參考值) / 參考值。
- A、B 每個設定 5 個種子，回報最好、中位數、最差；C、D、E 沒有隨機性，只跑一次。
用法：python summarize.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "logs", "results.jsonl"), encoding="utf-8")]
df = pd.DataFrame(rows)
# 同一個設定若有多筆紀錄（例如出錯後重跑），只留最後一筆
df = df.drop_duplicates(["method", "data", "cap", "seed"], keep="last").reset_index(drop=True)
df["n"] = df["data"].map(lambda s: 40 if s == "data" else int(s.split("_n")[1].split("_")[0]))
df["ok"] = df.get("check_ok", False).fillna(False).astype(bool)
df["fit"] = np.where(df["ok"], df.get("check_fitness"), np.nan)
df["sec"] = df["seconds"].fillna(df["wall_sec"])

# 參考值
ref = {}
for (d, cap), g in df.groupby(["data", "cap"]):
    c = g[(g.method == "C") & (g.status == "OPTIMAL") & g.ok]
    if len(c):
        ref[(d, cap)] = (float(c.fit.iloc[0]), "最佳")
    elif g.fit.notna().any():
        ref[(d, cap)] = (float(g.fit.min()), "已知最佳")
df["ref"] = [ref.get((d, c), (np.nan, ""))[0] for d, c in zip(df.data, df.cap)]
df["ref_kind"] = [ref.get((d, c), (np.nan, ""))[1] for d, c in zip(df.data, df.cap)]
df["gap_pct"] = (df.fit - df.ref) / df.ref * 100

cols = ["n", "data", "cap", "method", "seed", "limit", "status", "fit", "rejections", "distance", "carts_used",
        "sec", "hit_limit", "first_hit_sec", "generations", "bound", "variables", "constraints", "enum_sec",
        "solve_sec", "ref", "ref_kind", "gap_pct", "ok", "note"]
df[[c for c in cols if c in df]].sort_values(["n", "data", "cap", "method", "seed"]).to_csv(
    os.path.join(HERE, "results.csv"), index=False, encoding="utf-8-sig")

NAME = {"A": "A 原版 GA–TS", "B": "B 路徑式 GA–TS", "C": "C 列舉 + SCIP", "D": "D 路徑 IP（SCIP）", "E": "E OR-Tools routing"}


def cell(g):
    """一個 (資料, 容量, 方法) 的文字：A/B 為 最好 / 中位數 / 最差。"""
    f = g.fit.dropna()
    if g.method.iloc[0] == "C" and g.status.iloc[0] == "SKIPPED":
        return f"未執行（{int(g.variables.iloc[0]):.2e} 個變數）"
    if not len(f):
        st = g.status.iloc[0] if "status" in g and isinstance(g.status.iloc[0], str) else "無解"
        return f"無可行解（{st}）"
    if len(f) == 1:
        s = f"{int(f.iloc[0])}"
        b = g.bound.iloc[0] if "bound" in g else np.nan
        if g.method.iloc[0] in "CD" and pd.notna(b):
            s += f"（下界 {b:.0f}）" if g.status.iloc[0] != "OPTIMAL" else "（最佳）"
        return s
    return f"{int(f.min())} / {int(f.median())} / {int(f.max())}"


lines = ["# 實驗結果", "",
         "A、B：5 個種子的「最好 / 中位數 / 最差」；C、D、E：只跑一次（沒有隨機性）。"
         "數字是適應值（報廢數 × 10000 + 總距離），全部由 improved/check_schedule.py 重新計算。", ""]
for n in (40, 80, 160):
    sub = df[df.n == n]
    if sub.empty:
        continue
    lines += [f"## n = {n}（{n // 2} 台推車，時間上限 {int(sub[sub.method == 'B'].limit.max()) if (sub.method == 'B').any() else '-'} 秒）", ""]
    ms = [m for m in "ABCDE" if (sub.method == m).any()]
    lines.append("| 資料 | 容量 | 參考值 | " + " | ".join(NAME[m] for m in ms) + " |")
    lines.append("|" + " --- |" * (3 + len(ms)))
    for (d, cap), g in sub.groupby(["data", "cap"]):
        r, k = ref.get((d, cap), (np.nan, ""))
        cells = [cell(g[g.method == m]) if (g.method == m).any() else "—" for m in ms]
        lines.append(f"| {d} | {cap} | {r:.0f}（{k}） | " + " | ".join(cells) + " |")
    lines.append("")
    # 依容量彙總：差距中位數與秒數中位數（跨資料、跨種子）
    lines += [f"### n = {n}：和參考值的差距（%）與秒數，跨資料與種子取中位數", ""]
    lines.append("| 容量 | " + " | ".join(NAME[m] for m in ms) + " |")
    lines.append("|" + " --- |" * (1 + len(ms)))
    for cap, g in sub.groupby("cap"):
        cs = []
        for m in ms:
            h = g[g.method == m]
            if h.fit.notna().any():
                cs.append(f"{h.gap_pct.median():.1f}%，{h.sec.median():.0f} 秒（{h.fit.notna().sum()}/{len(h)} 有解）")
            else:
                cs.append("無解" if len(h) and not (h.status == "SKIPPED").all() else "未執行")
        lines.append(f"| {cap} | " + " | ".join(cs) + " |")
    lines.append("")
# ------------------------------------------------------------ 以 B 的實際執行時間為準
# 每次 B 執行花了 t 秒，就取 E 在 t 秒時的目前最好解（E 的歷程）。A、D 沒有時間歷程，但它們跑滿上限，
# 用上限時的結果比較只會對 A、D 較有利。E 的歷程值是求解器目標（軟上限罰分），沒有報廢時等於題目成本。
def e_at(hist, t):
    # 歷程記的是每個新找到的解（GLS 會接受較差的解），所以取到 t 秒為止的最小值
    v = [o for s, o in hist if s <= t]
    return min(v) if v else np.nan


tab = []
n_pen = n_none = 0
for (n, cap), g in df.groupby(["n", "cap"]):
    win = tie = lose = 0
    bs, es, ts = [], [], []
    for d, h in g.groupby("data"):
        e = h[(h.method == "E") & h.fit.notna()]
        if e.empty or "history" not in e:
            continue
        hist = e.history.iloc[0]
        for _, b in h[(h.method == "B") & h.fit.notna()].iterrows():
            # 已過 E 最後一次改進的時間：E 的解就是最後交出的解，用檢查程式算的題目成本
            ev = float(e.fit.iloc[0]) if b.sec >= hist[-1][0] else e_at(hist, b.sec)
            if np.isnan(ev):                     # B 結束時 E 還沒有任何解：算 B 較好
                n_none += 1
                ev = np.inf
            elif b.sec < hist[-1][0] and ev >= 10000:
                n_pen += 1                       # 用的是含遲到罰分的求解器目標，比題目成本高
            bs.append(b.fit); es.append(ev); ts.append(b.sec)
            win += b.fit < ev; tie += b.fit == ev; lose += b.fit > ev
    if bs:
        es = [x for x in es if np.isfinite(x)] or [np.nan]
        tab.append(f"| {n} | {cap} | {np.median(ts):.0f} | {np.median(bs):.0f} | {np.nanmedian(es):.0f} | {win} / {tie} / {lose} |")
lines += ["## 以 B 的執行時間為準：E 在同一秒數的解", "",
          "每次 B 執行花了 t 秒，就取 E 在第 t 秒時的目前最好解。t 已超過 E 最後一次改進的時間時，直接用 E 最後交出的解（題目成本）。"
          "E 的歷程記的是求解器目標（遲到 1 單位時間罰 10000），有遲到時會比題目成本高。"
          + (f"本表有 {n_pen} 次比較用到含罰分的歷程值，對 E 不利。" if n_pen else "本表沒有任何一次比較用到含罰分的歷程值，所以 E 的值都等於題目成本。")
          + (f"另有 {n_none} 次 B 結束時 E 還沒有解，算 B 較好。" if n_none else "")
          + "A、D 沒有時間歷程，比較時用它們跑滿上限的結果，只會對它們較有利。", "",
          "| n | 容量 | B 秒數中位數 | B 中位數 | E 在 B 秒數時（中位數） | B 較好 / 平手 / E 較好（次數） |",
          "| --- | --- | --- | --- | --- | --- |"] + tab
lines.append("")
open(os.path.join(HERE, "results.md"), "w", encoding="utf-8").write("\n".join(lines))

# ------------------------------------------------------------ 圖：每種規模一張（取種子 1 的資料），四個容量各一格
# 曲線 = 執行過程中「目前最好解」和參考值的差距：B 五個種子各一條，E 一條（E 的歷程是求解器目標，有遲到時偏高）。
# 點 = 只有最終結果的方法：A（5 個種子）、C、D，畫在它們結束的秒數。差距超過 YMAX 的畫在上緣，實際值寫在圖下方。
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
for fam in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(fam in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = fam
        plt.rcParams["axes.unicode_minus"] = False
        break
from matplotlib.ticker import FuncFormatter
STYLE = {"B": ("#257de0", "o"), "A": ("#f47230", "s"), "E": ("#00b894", "^"), "D": ("#6a4ce2", "D"), "C": ("#191919", "*")}
YMAX = 60
for n in (40, 80, 160):
    d = f"gen_n{n}_s1"
    sub = df[(df.data == d) & df.fit.notna()]
    if sub.empty:
        continue
    lim = LIM = float(sub[sub.method == "B"].limit.max())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, cap in zip(axes, (2, 4, 10, 20)):
        g = sub[sub.cap == cap]
        if g.empty or (d, cap) not in ref:
            ax.set_title(f"容量 {cap}（沒有結果）", fontsize=10.5)
            continue
        r = ref[(d, cap)][0]
        gap = lambda v: (v - r) / r * 100
        out = []
        for m in "BE":
            for k, (_, x) in enumerate(g[g.method == m].iterrows()):
                h = x.history if isinstance(x.history, list) else []
                if not h:
                    continue
                t = [a for a, _ in h] + [x.sec]
                v = list(np.minimum.accumulate([gap(b) for _, b in h])) + [x.gap_pct]
                ax.step(t, np.clip(v, None, YMAX), where="post", color=STYLE[m][0], lw=1.3 if m == "E" else 1.0,
                        alpha=1 if m == "E" else 0.75, label=NAME[m] if k == 0 else None)
            last = g[g.method == m]
            if len(last) and (last.gap_pct > YMAX).any():
                out.append(f"{m} 最終 {last.gap_pct.max():,.0f}%")
        for m in "ACD":
            h = g[g.method == m]
            if h.empty:
                continue
            col, mk = STYLE[m]
            ax.scatter(h.sec, h.gap_pct.clip(upper=YMAX), s=70 if m == "C" else 34, c=col, marker=mk,
                       label=NAME[m], edgecolors="white", linewidths=1.0, zorder=3, clip_on=False)
            if (h.gap_pct > YMAX).any():
                out.append(f"{m} {h.gap_pct.min():,.0f}%" + (f"～{h.gap_pct.max():,.0f}%" if len(h) > 1 else ""))
        ax.set_xscale("log")
        ax.set_xlim(min(0.1, g.sec.min() * 0.7), max(LIM, g.sec.max()) * 1.3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(-1, YMAX)
        ax.set_title(f"容量 {cap}（參考值 {r:.0f}，{ref[(d, cap)][1]}）", fontsize=10.5)
        ax.set_xlabel("秒數（對數刻度）" + ("\n上緣的點實際差距：" + "；".join(out) if out else ""), fontsize=9)
        ax.axvline(LIM, color="#bbbbbb", lw=0.8, ls="--")
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("和參考值的差距（%）")
    hs, ls = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in ls:
                hs.append(hh); ls.append(ll)
    fig.legend(hs, ls, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"{d}（{n} 批、{n // 2} 台推車）：執行過程中的目前最好解，虛線為時間上限 {LIM:.0f} 秒", y=1.1, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figures", f"time_quality_n{n}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
print("已寫出 results.csv、results.md、figures/")
