# -*- coding: utf-8 -*-
"""§18 跑后补充诊断：H 组注入的高频偏差去哪了？

全量结果显示 H 组 acf 与 G 组几乎相同（`Δacf ≈ 0`），且 M1 **未剔点**
（`dropped_hz=[]`）。两种可能要分开：

  (a) 偏差被拟合参数（尤其 `L = R_ad·tau_ad` 的高频感抗支路）吸收；
  (b) 3/50 个端点在加权拟合里权重太小，本来就推不动 lag1–5 自相关。

本脚本用 §17 已验证的确定性多起点 LS（与 MCMC 13/14 一致，秒级）对同一对
G/H 谱各拟合一次 C4，逐参数比相对变化，并报最高 3 点的相对残差。
**只读、不改 §18 的任何判读规则、不改判定**；标 `exploratory`。

    python evidence/nested_topology/m4_synth_hf_absorb.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m4_four_circuits as FC      # noqa: E402
import m4_synth_probe as P         # noqa: E402

OUT = os.path.join("out", "m4_synth_hf_absorb.json")
KEYS = ("Rs", "R1", "R2", "R3", "Q1", "n1", "Q2", "n2", "L")


def fit(c: dict) -> dict:
    f, Z = P.make_observed(c)
    sg = 0.02 * np.abs(Z)
    w_ref = 2.0 * np.pi * np.sqrt(f.min() * f.max())
    b = FC.fit_one("C4", f, Z, sg, w_ref, n_starts=8)
    p = b["params"]
    Zm = FC._z_c4(f, p, w_ref)
    rel = np.abs(Zm - Z) / np.abs(Z)
    top = np.argsort(f)[-3:]
    return dict(params={k: float(p[k]) for k in KEYS},
                hf_top_rel=[float(rel[j]) for j in top],
                hf_top_mean=float(np.mean(rel[top])),
                cost=float(b["cost"]))


def main() -> int:
    cases = P.build_cases()
    out = []
    print("H 组机制诊断（LS，秒级；不改 §18 判读）", flush=True)
    for tag in ("T2a", "T2b", "T2c"):
        g = next(c for c in cases if c["group"] == "G" and c["truth"] == tag
                 and c["seed"] == 0)
        h = next(c for c in cases if c["group"] == "H" and c["truth"] == tag
                 and c["seed"] == 0)
        rg, rh = fit(g), fit(h)
        dL = 100 * (rh["params"]["L"] / rg["params"]["L"] - 1)
        row = dict(truth=tag, G=rg, H=rh,
                   pct_change={k: 100 * (rh["params"][k] / rg["params"][k] - 1)
                               for k in KEYS})
        out.append(row)
        print(f"  {tag}: L {rg['params']['L']:.1f} → {rh['params']['L']:.1f} H "
              f"({dL:+.1f}%)  最高3点相对残差 {rg['hf_top_mean']:.3f} → "
              f"{rh['hf_top_mean']:.3f}")
        biggest = sorted(row["pct_change"].items(), key=lambda kv: -abs(kv[1]))[:3]
        print("     参数变化最大：", ", ".join(f"{k} {v:+.1f}%" for k, v in biggest))
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(dict(status="exploratory", note="§18 跑后补充诊断，不改判读规则",
                       rows=out), fh, ensure_ascii=False, indent=1)
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
