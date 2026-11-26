"""跑一個設定：一個方法 × 一組資料 × 一個容量 × 一個種子，結果附加到 JSON Lines 檔。

每個解都另外呼叫 improved/check_schedule.py 重新計算報廢數與距離；
check_ok 記錄檢查是否通過，check_fitness 是檢查程式算出的適應值（表格一律用這個值）。
用法：python run_one.py 方法(A/B/C/D/E) 資料 容量 種子 時間上限秒數 結果檔.jsonl
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "..", "improved", "check_schedule.py")


def main():
    method, data, cap, seed, limit, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    name = os.path.splitext(os.path.basename(data))[0]
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", f"{name}_cap{cap}_{method}_s{seed}.xlsx")
    rec = {"method": method, "data": name, "cap": cap, "seed": seed, "limit": limit}
    t0 = time.time()
    try:
        if method == "A":
            import method_a_original
            r = method_a_original.run(data, cap, seed, limit, out)
        else:
            import methods
            if method == "B":
                r = methods.run_b(data, cap, seed, limit, out)
            elif method == "C":
                r = methods.run_c(data, cap, limit, out)
            elif method == "D":
                r = methods.run_d(data, cap, limit, out)
            elif method == "E":
                r = methods.run_e(data, cap, limit, out)
        rec.update(r)
    except Exception as e:
        rec.update(status="ERROR", note=f"{type(e).__name__}: {e}")
        traceback.print_exc()
    rec["wall_sec"] = round(time.time() - t0, 2)
    if "fitness" in rec and os.path.exists(out):
        chk = subprocess.run([sys.executable, CHECK, data, out, str(cap)], capture_output=True, text=True)
        rec["check_ok"] = chk.returncode == 0
        mm = re.search(r"適應值 (\d+)", chk.stdout)
        rec["check_fitness"] = int(mm.group(1)) if mm else None
        lines = chk.stdout.strip().splitlines()
        rec["check_msg"] = lines[0] if lines else chk.stderr[-300:]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
