"""M-B-SIM-01 等价性校验：模拟谱上的 voigt 基线 acf 是否也 ≥0.70。

PROTOCOL §1 用「voigt 基线 acf ∈ [0.730, 0.941] ⇒ 相对项 `voigt−acf ≥ 0.20` 在
`acf ≤ 0.50` 时必然成立」把 M-b 化简成绝对值判据。这一步在**模拟谱**上复核该前提：
若模拟谱的 voigt 基线也 ≥0.70，则化简成立；否则主实验的触发率需改用相对口径重算。

只读 `generated/`，不看 `truth/`；复用 `blind_results/` 里已有的 nested acf。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evidence", "nested_topology"))

import bm_experiments as E   # noqa: E402
import m4_holdout2 as H      # noqa: E402

SIM = os.path.join("out", "mb_sim01")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)
    ids = sorted(x[:-5] for x in os.listdir(os.path.join(SIM, "blind_results"))
                 if x.endswith(".json"))
    step = max(1, len(ids) // a.n)
    pick = ids[::step][:a.n]
    rows = []
    for sid in pick:
        f, Z = H.R.load_spectrum(os.path.join(SIM, "generated", sid + ".csv"))
        sg = (H.SIGMA_PCT / 100.0 * np.abs(Z), H.SIGMA_PCT / 100.0 * np.abs(Z))
        s, mc = H.B.fit(f, Z, K=1, alpha=0.5, sigma=sg, zeta_fixed=1.0, seed=0,
                        progress=False, **H.BUDGET)
        p = dict(R_inf=float(np.median(s["R_inf"])), W=float(np.median(s["W"])),
                 pi=np.median(s["pi"], axis=0), tau=np.median(s["tau"], axis=0),
                 phi=np.median(s["phi"], axis=0), zeta=np.median(s["zeta"], axis=0))
        res = H.B.model_z(f, p, include_L=False) - Z
        va = float(max(E.acf_max(res.real), E.acf_max(res.imag)))
        nested = json.load(open(os.path.join(SIM, "blind_results", sid + ".json"),
                                encoding="utf-8"))["acf"]
        rows.append(dict(sim_id=sid, voigt_acf=va, nested_acf=float(nested),
                         delta=va - float(nested)))
        print(f"{sid}  voigt {va:.3f}  nested {nested:.3f}  差 {va - nested:+.3f}", flush=True)

    va = np.array([r["voigt_acf"] for r in rows])
    nd = np.array([r["nested_acf"] for r in rows])
    dl = np.array([r["delta"] for r in rows])
    ok_low = bool(np.all(va >= 0.70))
    fail = [(r["sim_id"], r["nested_acf"], r["delta"]) for r in rows
            if r["nested_acf"] <= 0.50 and r["delta"] < 0.20]
    out = dict(n=len(rows), rows=rows,
               voigt_min=float(va.min()), voigt_max=float(va.max()),
               all_voigt_ge_070=ok_low, rel_clause_counterexamples=fail,
               equivalence_holds=bool(ok_low and not fail))
    print(f"\nvoigt acf 范围 {va.min():.3f}–{va.max():.3f}；全部 ≥0.70: {ok_low}")
    print(f"相对项在 acf≤0.50 时的反例数：{len(fail)}")
    print("等价性成立：" + str(out["equivalence_holds"]))
    p = os.path.join(SIM, "summary", "voigt_check.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写 " + p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
