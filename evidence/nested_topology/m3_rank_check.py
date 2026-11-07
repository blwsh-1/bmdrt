# -*- coding: utf-8 -*-
"""R1/R2 拆分：结构性奇异还是实践性不可辨识？（`BM_TOP_M3_PLAN.md` §19.2）

三件事，都是**噪声无关**或**线性化**的量，不需要跑 MCMC：

1. **数值秩**：噪声无关 Jacobian `J = ∂Z/∂θ`（θ 为 log 参数）的奇异值与条件数。
   `rank < n_params` → 结构性不可辨识（映射本身丢了一维）；
   `rank == n_params` 但 `σ_min` 极小 → 只是当前噪声水平下不可辨识。
2. **条件化 sd**：`sqrt(uᵀ(JᵀJ)⁻¹u)`，u 为指定方向。与"边际 sd" `1/‖J u‖`
   对比，可以看出共线造成的放大倍数。
3. **拆分比例 r = R1/(R1+R2)** 的 delta 法 sd：`sd_r = r(1−r)·sd_log(u)`，
   `u = e_R1 − e_R2`。这是 M3.5 的主终点（预注册在 §20.4）。

用法：
    python m3_rank_check.py            # A/B 两组 + 逐点相对口径对照
    python m3_rank_check.py --json x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_top_gen as G                                    # noqa: E402
import m3_design_check as D                               # noqa: E402

F = np.logspace(-2, 5, 61)
KS = ("R1", "R2", "Q1", "n1", "Q2", "n2", "Rs", "R_ad", "tau_ad")
I1, I2 = KS.index("R1"), KS.index("R2")
BASE = dict(D.BASE)


def jac(p: dict) -> np.ndarray:
    cols = D.jacobian(p, F)
    return np.column_stack([cols[k] for k in KS])


def rank_block(tag: str, p: dict) -> dict:
    J = jac(p)
    s = np.linalg.svd(J, compute_uv=False)
    tol = s[0] * max(J.shape) * np.finfo(float).eps
    rank = int((s > tol).sum())
    _, _, Vt = np.linalg.svd(J, full_matrices=False)
    v = Vt[-1]
    top = np.argsort(-np.abs(v))[:4]
    print(f"{tag:<24}σmax={s[0]:.3g}  σmin={s[-1]:.3g}  cond={s[0]/s[-1]:.3g}  "
          f"rank={rank}/{len(KS)}  (tol={tol:.1g})")
    print(f"{'':<24}最弱方向：" + "，".join(f"{KS[i]} {v[i]:+.2f}" for i in top))
    return {"tag": tag, "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
            "cond": float(s[0] / s[-1]), "rank": rank, "n_params": len(KS),
            "weakest_direction": {KS[i]: float(v[i]) for i in top}}


def split_block(tag: str, p: dict, sigma=None, rel: bool = False) -> dict:
    """拆分方向的条件化 sd 与 r 的 delta 法区间。"""
    z0 = G.gen_parallel_L2(F, p)
    J = jac(p)
    w = (np.concatenate([0.02 * np.abs(z0)] * 2) if rel
         else np.full(2 * F.size, float(sigma)))
    Jw = J / w[:, None]
    cov = np.linalg.pinv(Jw.T @ Jw, rcond=1e-14)
    u = np.zeros(len(KS))
    u[I1], u[I2] = 1.0, -1.0
    sd_log = float(np.sqrt(max(u @ cov @ u, 0.0)))
    marg = float(1.0 / np.linalg.norm(Jw @ u))
    r = p["R1"] / (p["R1"] + p["R2"])
    sd_r = r * (1 - r) * sd_log
    lo, hi = max(r - 1.645 * sd_r, 0.0), min(r + 1.645 * sd_r, 1.0)
    print(f"{tag:<24}sd_log={sd_log:>8.2f}  边际={marg:>7.3f}  "
          f"sd_r={sd_r:>7.3f}  90% r∈[{lo:.3f}, {hi:.3f}]  (真值 {r:.3f})")
    return {"tag": tag, "sd_log": sd_log, "marginal_sd": marg,
            "r_truth": float(r), "sd_r": float(sd_r),
            "r_ci90": [float(lo), float(hi)], "rel_weight": rel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    print("== 噪声无关 Jacobian 的秩（区分结构性 / 实践性）==")
    ranks = [rank_block("A  R_ad=3000", BASE),
             rank_block("B  R_ad=300", dict(BASE, R_ad=300.0))]

    print("\n== 拆分方向（R1 vs R2）的条件化不确定性 ==")
    splits = [split_block("A  固定 σ=61.0", BASE, 61.0),
              split_block("B  固定 σ=17.8", dict(BASE, R_ad=300.0), 17.8),
              split_block("A  逐点相对 2%|Z| *", BASE, rel=True),
              split_block("B  逐点相对 2%|Z| *", dict(BASE, R_ad=300.0), rel=True)]

    print("\n* 逐点相对是另一种噪声条件，仅作 M3.5 的预期上限，不与 A/B 组比较。")
    print("  线性化：σ 取真值谱 |Z_true|（Fisher 在真值处求值）；"
          "M3.5 的拟合用 |Z_obs|，二者相差 O(噪声)，对权重是二阶效应。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"f_grid": [float(F[0]), float(F[-1]), F.size],
                        "params": list(KS), "rank": ranks, "split": splits},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
