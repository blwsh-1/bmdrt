# -*- coding: utf-8 -*-
"""R1/R2 拆分：结构性奇异还是实践性不可辨识？（`BM_TOP_M3_PLAN.md` §19.2）

三件事，都是**噪声无关**或**线性化**的量，不需要跑 MCMC：

1. **数值秩**：噪声无关 Jacobian `J = ∂Z/∂θ`（θ 为 log 参数）的奇异值与条件数。
   `rank < n_params` → 结构性不可辨识（映射本身丢了一维）；
   `rank == n_params` 但 `σ_min` 极小 → 只是当前噪声水平下不可辨识。

   **读法限定**：这是**一个真值点、一条频栅、一个参数化**下的有限差分 Jacobian，
   结论只能是"该点邻域局部满列秩"。它**不能**证明整个参数空间处处无结构性
   不可辨识（映射在大范围内可能折叠）。要谈全局得另做扫描，本脚本不做。
2. **条件化 sd**：`sqrt(uᵀ(JᵀJ)⁻¹u)`，u 为指定方向。与"边际 sd" `1/‖J u‖`
   对比，可以看出共线造成的放大倍数。
3. **拆分比例 r = R1/(R1+R2)** 的 delta 法 sd：`sd_r = r(1−r)·sd_log(u)`，
   `u = e_R1 − e_R2`。这是 M3.5 的主终点（预注册在 §20.4）。

关于参数尺度（审查意见，§20.7）
--------------------------------
本脚本的 Jacobian **本来就是**对 log 参数求导：`J_j = ∂Z/∂log θ_j`，
即 `J_θ·diag(θ)`（`m3_design_check.jacobian` 用乘性扰动 `θ·e^{±h}` 实现）。
所以各列统一是 **Ω**，已经消掉了"Ω / H / s / 无量纲"混排的问题。

但仍有两点必须说清：

· 原始奇异值的单位是 **Ω**，它是"把 Z 推动多少欧姆"，**不能**当作跨参数可
  比较的物理量读；条件数也随"哪些参数取 log"这一选择变化。
· 因此另报一个**列归一化**版本 `J_norm = J·diag(1/‖J_j‖)`：无量纲、奇异值
  落在 `(0, √n]`，条件数就是标准的共线性条件数（不随单位缩放变）。秩用
  **相对阈值** `σ_max·max(m,n)·eps` 与 `1e-6/1e-8/1e-10` 四档一起报。

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


def norm_block(tag: str, p: dict) -> dict:
    """列归一化 Jacobian 的奇异值 / 相对秩 / 最弱方向（§20.7）。

    `J` 的列已经是 `∂Z/∂log θ_j`（单位 Ω）。归一化只做一件事：把每列除成
    单位长度，于是 `σ` 无量纲、`cond` 是共线性条件数，**不随参数单位变**。
    归一化前后秩必然相同（非零列缩放不改变秩），变的是"多小算零"这个门槛
    和条件数的可解释性——所以秩在这里用**相对阈值**报。

    坐标系统（上一轮这里算错过，本轮修正）：`v` 是 `J_norm` 的右奇异向量，
    活在**归一化系数空间**；要跟"原始 J 的最弱方向"比夹角，先把 `v` 映射回
    log 参数空间 `u_log ∝ D⁻¹v`（`D = diag(‖J_j‖)`，因为 `J_norm·v = J·D⁻¹v`），
    再与**原始 J 的右奇异向量** `v_raw`（本来就在 log 参数空间）点乘。
    拿 `v` 直接点 `u_log` 是混用两组基底，那个 cos 没有意义。
    """
    J = jac(p)
    cn = np.linalg.norm(J, axis=0)
    Jn = J / cn
    s = np.linalg.svd(Jn, compute_uv=False)
    eps = np.finfo(float).eps
    tol = s[0] * max(Jn.shape) * eps
    ranks = {f"1e-{d}": int((s / s[0] > 10.0 ** -d).sum())
             for d in (6, 8, 10, 12)}
    _, _, Vt = np.linalg.svd(Jn, full_matrices=False)
    v = Vt[-1]                                 # 归一化坐标（系数空间）的最弱方向
    # 换算到 log 参数坐标：Jn·v = J·(D⁻¹v)，所以"归一化后最弱"对应的参数方向 ∝ D⁻¹v
    u_log = v / cn
    u_log /= np.linalg.norm(u_log)
    # 夹角必须**在同一坐标系**里算：u_log 在 log 参数空间，所以对照量是原始 J
    # 的最弱右奇异向量（也在 log 参数空间）。直接拿 v（归一化坐标）点乘 u_log
    # 是把两组基底混在一起，那个 cos 值没有意义（上一轮的报告值即为此错误）。
    _, _, Vt_raw = np.linalg.svd(J, full_matrices=False)
    v_raw = Vt_raw[-1]
    cos_ang = float(abs(v_raw @ u_log))
    top = np.argsort(-np.abs(v))[:4]
    top_raw = np.argsort(-np.abs(v_raw))[:4]
    spread = np.log10(cn.max() / cn.min())
    print(f"{tag:<24}σmax={s[0]:.3f}  σmin={s[-1]:.3e}  cond={s[0]/s[-1]:.2e}   "
          f"rank={ranks['1e-12']}/{len(KS)}  (rel tol {tol:.1e})")
    print(f"{'':<24}相对阈值秩 1e-6:{ranks['1e-6']}  1e-8:{ranks['1e-8']}  "
          f"1e-10:{ranks['1e-10']}  1e-12:{ranks['1e-12']}；"
          f"列范数跨 {spread:.1f} dec（{cn.min():.3g}…{cn.max():.3g} Ω）")
    print(f"{'':<24}最弱方向（系数坐标，**不是**参数载荷）："
          + "，".join(f"{KS[i]} {v[i]:+.2f}" for i in top))
    print(f"{'':<24}同一方向的参数载荷（log 坐标）："
          + "，".join(f"{KS[i]} {u_log[i]:+.2f}" for i in top))
    print(f"{'':<24}原始 J 的最弱方向（log 坐标）："
          + "，".join(f"{KS[i]} {v_raw[i]:+.2f}" for i in top_raw)
          + f"；cos(原始, 归一化) = {cos_ang:.4f}")
    return {"tag": tag, "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
            "cond": float(s[0] / s[-1]), "rank_by_threshold": ranks,
            "rel_tol": float(tol), "col_norm_min": float(cn.min()),
            "col_norm_max": float(cn.max()), "col_norm_decades": float(spread),
            "weakest_direction_norm": {KS[i]: float(v[i]) for i in top},
            "weakest_direction_log": {KS[i]: float(u_log[i]) for i in top},
            "weakest_direction_raw": {KS[i]: float(v_raw[i]) for i in top_raw},
            "cos_raw_vs_normalized": cos_ang,
            "cos_note": "两向量均在 log 参数坐标；v_norm 需先乘 D⁻¹ 再归一化"}


def scaling_check(p: dict) -> dict:
    """数值核对 `J_log = J_θ·diag(θ)`（审查意见里的那条恒等式）。

    `m3_design_check.jacobian` 用乘性扰动 `θ_j·e^{±h}`，得到的就是
    `∂Z/∂log θ_j`。把它与"绝对扰动的 ∂Z/∂θ_j"乘上 `θ_j` 对照，可以确认
    **没有**哪个参数被漏掉缩放（`n1/n2` 本就在 (0.5,1)，乘不乘 1 都一样，
    所以它们不参与这项比对）。

    两种写法用的扰动点不同（`θ(1±h)` vs `θ·e^{±h}`），差异是有限差分的
    `O(h²)` 项，量级 `h²/6 ≈ 1.7e-7`——不是模型误差。
    """
    f = F
    worst = 0.0
    for k in p:
        if k in ("n1", "n2"):
            continue
        pp, pm = dict(p), dict(p)
        pp[k], pm[k] = p[k] * (1 + 1e-3), p[k] * (1 - 1e-3)
        d_abs = (D.spectrum(pp, f) - D.spectrum(pm, f)) / (2e-3 * p[k])
        j_abs = np.concatenate([d_abs.real, d_abs.imag]) * p[k]
        j_log = jac(p)[:, KS.index(k)]
        worst = max(worst, float(np.max(np.abs(j_abs - j_log))
                                 / np.max(np.abs(j_log))))
    print(f"[尺度恒等式] J_log vs J_θ·θ（7 个正参数）：max rel err {worst:.2e}"
          f"（预期 O(h²)≈1.7e-7，非模型误差）")
    return {"max_rel_err": worst, "expected_o_h2": 1.7e-7}


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
    # delta 法在 sd_log ≫ 1 时已经越界：此时 r 的区间只是把 [0,1] 撑满，
    # 不能当 90% 置信区间读。这条要在表里标出来，否则会被当成"区间很宽"的证据。
    valid = sd_log < 0.5
    print(f"{tag:<24}sd_log={sd_log:>8.2f}  边际={marg:>7.3f}  "
          f"sd_r={sd_r:>7.3f}  90% r∈[{lo:.3f}, {hi:.3f}]  (真值 {r:.3f})"
          + ("" if valid else "  †"))
    return {"tag": tag, "sd_log": sd_log, "marginal_sd": marg,
            "r_truth": float(r), "sd_r": float(sd_r),
            "r_ci90": [float(lo), float(hi)], "rel_weight": rel,
            "delta_method_valid": bool(valid)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    print("== 噪声无关 Jacobian 的秩（区分结构性 / 实践性）==")
    print("   列 = ∂Z/∂log θ（单位 Ω）；σ 的单位因此是 Ω，不可当跨参数的物理量读")
    print("   结论范围：真值点邻域的**局部**满列秩，不是全局可辨识性证明")
    ranks = [rank_block("A  R_ad=3000", BASE),
             rank_block("B  R_ad=300", dict(BASE, R_ad=300.0))]
    scaling = scaling_check(BASE)

    print("\n== 列归一化 Jacobian（无量纲；条件数=共线性条件数，§20.7）==")
    norms = [norm_block("A  R_ad=3000", BASE),
             norm_block("B  R_ad=300", dict(BASE, R_ad=300.0))]

    print("\n== 拆分方向（R1 vs R2）的条件化不确定性 ==")
    splits = [split_block("A  固定 σ=61.0", BASE, 61.0),
              split_block("B  固定 σ=17.8", dict(BASE, R_ad=300.0), 17.8),
              split_block("A  逐点相对 2%|Z| *", BASE, rel=True),
              split_block("B  逐点相对 2%|Z| *", dict(BASE, R_ad=300.0), rel=True)]

    print("\n* 逐点相对是另一种噪声条件，仅作 M3.5 的预期上限，不与 A/B 组比较。")
    print("  线性化：σ 取真值谱 |Z_true|（Fisher 在真值处求值）；"
          "M3.5 的拟合用 |Z_obs|，二者相差 O(噪声)，对权重是二阶效应。")
    print("  † delta 法越界（sd_log≫1）：此时 r 的 90% 区间只是把 [0,1] 撑满，"
          "应读作『拆分不受约束』，\n    而不是『90% 置信区间有这么宽』。"
          "线性化只在 sd_log 小的时候才有定量意义。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"f_grid": [float(F[0]), float(F[-1]), F.size],
                        "params": list(KS), "rank": ranks, "rank_normalized": norms,
                        "scaling_check": scaling, "split": splits},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
