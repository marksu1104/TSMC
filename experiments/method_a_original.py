"""方法 A：原版 GAwithTabu。

做法：直接從 ../GAwithTabu.ipynb 讀出第一個程式格（GaTabuScheduler 類別）的原始碼並執行，
參數照 notebook 的設定（族群 1000、最多 1000 代、精英 50、突變率 0.4、錦標賽 10、
連續 50 代沒改善就停、禁忌搜尋 50 次迭代、期限 10、每 5 代一次、rep 5、連續 40 次沒改善就停）。

唯一的改動是加上時間上限（原版沒有）：
- 每次計算適應值前檢查時間，超過上限就丟出 TimeUp，結束演化，回傳目前最好的解；
- 第一代族群一定算完（否則沒有任何解可以回傳），所以第一代本身超時時，實際秒數會超過上限；
- 時間到之後不做最後一次禁忌搜尋。
結果欄位 hit_limit 記錄是否因時間上限而停止。

原版其他行為不變，包括：取送順序用最近鄰、起點用距離表的第一個地點
（本實驗的距離表第一個地點是 LOC1，和所有推車的 INIT_LOC 相同，所以這個問題在這裡不影響結果）。
"""
import contextlib
import io
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "GAwithTabu.ipynb")


def load_class():
    nb = json.load(open(NB, encoding="utf-8"))
    src = "".join(next(c for c in nb["cells"] if c["cell_type"] == "code")["source"])
    ns = {}
    exec(compile(src, NB, "exec"), ns)
    return ns["GaTabuScheduler"]


class TimeUp(Exception):
    pass


def run(data, cap, seed, limit, out_xlsx):
    Base = load_class()

    class Limited(Base):
        deadline = None
        armed = False

        def fitness(self, chromosome):
            if self.armed and time.time() > self.deadline:
                raise TimeUp
            return super().fitness(chromosome)

    s = Limited(data_file=data, seed=seed)
    # notebook 的參數（三個容量的程式格參數相同，只有 batch_capacity 不同）
    s.batch_capacity = cap
    s.pop_size, s.generations, s.elite_size = 1000, 1000, 50
    s.mutation_rate, s.tournament_size, s.earlier_ga_stop = 0.4, 10, 50
    s.tabu_iterations, s.tabu_tenure, s.tabu_frequency = 50, 10, 5
    s.tabu_elite_ratio, s.rep, s.earlier_ts_stop = 0.001, 5, 40

    t0 = time.time()
    s.deadline = t0 + limit
    hit = False
    # 第一代算完才開始檢查時間：用 evolution_history 判斷
    orig_fitness = s.fitness

    def fitness(ch):
        if not s.armed and s.evolution_history["best"]:
            s.armed = True
        return orig_fitness(ch)

    s.fitness = fitness
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            s.run()
        except TimeUp:
            hit = True
            s.armed = False
            fit, det, met = Base.fitness(s, s.best_solution)
            s.best_fit, s.best_details, s.best_metrics = fit, det, met
        s.save_results(out_xlsx)
    sec = time.time() - t0
    rej = sum(m["rejections"] for m in s.best_metrics.values())
    dist = sum(m["distance"] for m in s.best_metrics.values())
    used = sum(1 for v in s.best_solution.values() if v)
    return {"fitness": rej * 10000 + dist, "rejections": rej, "distance": dist, "carts_used": used,
            "seconds": round(sec, 2), "hit_limit": hit, "generations": len(s.evolution_history["best"])}
