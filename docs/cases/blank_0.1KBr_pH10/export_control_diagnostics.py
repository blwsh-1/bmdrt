"""把 6 个 control 的严格门控诊断导出为随包发布的 CSV。

来源：网格 run.log 的 R-hat / bulk ESS / tail ESS，以及 real_*.json 的
kk.status / sample_grade / divergences / tree_cap。
导出后 replay_identifiability.py 优先读 control_diagnostics.csv，
因此在没有原始 run.log 的独立环境中也能复现门控判定。

运行：
  D:/autoeis_env/Scripts/python.exe export_control_diagnostics.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(sys.argv[1]) if len(sys.argv) > 1 else None)
NAMES = {"none": ("G_K1_none", "G_K2_none"),
         "Ws": ("G_K1_Ws", "G_K2_Ws"),
         "Wo": ("G_K1_Wo", "G_K2_Wo")}


def main():
    if REPO is None or not REPO.exists():
        raise SystemExit("找不到仓库根目录；请把 bmdrt 仓库路径作为第 1 个参数传入")
    rows = []
    for fam, pair in NAMES.items():
        for K, run_id in zip((1, 2), pair):
            js_p = glob.glob(str(ROOT / "results" / run_id / "real_*.json"))[0]
            js = json.loads(Path(js_p).read_text(encoding="utf-8"))
            logs = sorted(glob.glob(str(REPO / "out/_grid" / run_id / "**" / "run.log"),
                                    recursive=True))
            rhat = bulk = tail = None
            if logs:
                txt = Path(logs[0]).read_text(encoding="utf-8", errors="replace")
                m = re.search(r"R̂max=([\d.]+)\s+bulk=([\d.]+)\s+tail=([\d.]+)", txt)
                if m:
                    rhat, bulk, tail = (float(m.group(1)), float(m.group(2)),
                                        float(m.group(3)))
            rows.append(dict(
                run_id=run_id, family=fam, K=K,
                kk_status=(js.get("kk") or {}).get("status"),
                sample_grade=js.get("sample_grade"),
                divergences=js.get("divergences"),
                tree_cap=js.get("tree_cap"),
                rhat_max=rhat, ess_bulk=bulk, ess_tail=tail,
                source="grid run.log + real_*.json"))
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "control_diagnostics.csv", index=False, float_format="%.6g")
    print(d.to_string(index=False))
    print("DONE ->", ROOT / "control_diagnostics.csv")


if __name__ == "__main__":
    main()
