"""3E 数据能不能用来做残差重复性：先过「拟合是否收敛」这道门槛。

**只读 + 三次诊断性拟合，不判验收、不改判定、不写进 §13/§14。**

为什么必须先过这道门：M-b / E1 用的残差来自后验中位曲线。若采样未收敛
（`R̂>1.05` 或有发散），残差本身不可信，拿它去比"重复性"没有意义。

取 3E 里数据层最紧的一组（同条件三次实验的谱差最小）——`3E-2` 的
`full_electrolysis_cell`，同电流 exp1/2/3（数据对数据 |ΔZ|/|Z| ≈ 2.8%），
用 §14 冻结档（`standard` + 逐点 2%σ + `parallel/L=2` + seed 0）各拟一次。

判据（只作门槛，不构成新验收）：
- `R̂max ≤1.05` 且 `div=0` → 收敛，可继续做「跨实验重复」分析；
- 否则 → 该批谱在 nested 下不可用，只能停在数据层面。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e3_conv_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))

import bm_real as R          # noqa: E402
import m4_holdout2 as H      # noqa: E402

SPEC_DIR = os.path.join("EISmeasurements", "csv_3e")
GROUP = ("3E-2_full_electrolysis_cell_exp1_0100mA.csv",
         "3E-2_full_electrolysis_cell_exp2_0100mA.csv",
         "3E-2_full_electrolysis_cell_exp3_0100mA.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("out", "e3_conv_probe.json"))
    a = ap.parse_args(argv)

    rows = []
    for fn in GROUP:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, fn))
        tag = os.path.splitext(fn)[0]
        res = H._fit_set(f, Z, [0], save_dir=os.path.join("out", "e3_probe"),
                         tag=tag)            # §14 冻结档，唯一拟合入口
        r = res["rows"][0]
        rows.append(dict(file=fn, n=int(f.size), rms_rel=float(r["rms_rel"]),
                         params=r["params"], ci90=r["ci90"],
                         acf=float(r["acf"]), acf_re=float(r["acf_re"]),
                         acf_im=float(r["acf_im"]),
                         rhat_max=float(r["rhat_max"]), div=int(r["div"]),
                         ess_min=float(r["ess_min"]),
                         converged=bool(r["rhat_max"] <= H.RHAT_MAX and r["div"] == 0)))
        print(f"{fn:<48} rms {r['rms_rel']:6.2f}%  acf {r['acf']:.3f}  "
              f"R̂ {r['rhat_max']:.3f}  div {r['div']:>2}  ESS {r['ess_min']:.0f}  "
              f"{'收敛' if rows[-1]['converged'] else '未收敛'}")

    ok = all(r["converged"] for r in rows)
    print(f"\n门槛：{'通过（3/3 收敛）' if ok else '不通过 —— 残差不作重复性分析用'}")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(group=list(GROUP), budget=H.BUDGET, sigma_pct=H.SIGMA_PCT,
                       branch=H.BRANCH, layers=H.LAYERS, seed=0,
                       rhat_max_thr=H.RHAT_MAX, gate_passed=ok, rows=rows),
                  fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
