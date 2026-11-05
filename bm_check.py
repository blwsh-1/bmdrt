# -*- coding: utf-8 -*-
"""合成数据端到端验收：跑一遍就给出 PASS / FAIL 清单。

    python bm_check.py            # 默认严格设置（约 15 分钟）
    python bm_check.py fast       # 快速冒烟（约 1 分钟，只验证流程能跑通）
    python bm_check.py nofit      # 只做数学自检，不采样（秒级）

分两部分：

  A. 数学自检（不依赖采样，秒级）
     A1 stick_pi 归一，且 π **不保证递减**
     A2 HN 的 DRT 归一化 ∫γ dlnτ = 1（多组 φ/ζ）
     A3 ζ=1 时 HN-DRT 与 Cole-Cole 闭式一致
     A4 φζ>1 时 γ 出负值（量化）
     A5 模型正演与解析 ZARC 一致
     A6 跨样本分量匹配能还原被打乱的编号
     A7 单样本 DRT 积分 = W（权重守恒）

  B. 拟合端到端（K=2，合成 2×ZARC + R∞，σ=0.2 Ω 已知）
     B1 收敛：全部参数的 rank-normalized split R̂ / bulk / tail ESS
     B2 分量：匹配后的面积与峰位是否命中真值
     B3 后验预测检验：覆盖率是否接近 0.90
     B4 拟合残差 rms 是否接近 σ√2

判据分两档，避免"只报好看的"：
  · 严格判据 R̂<1.01 且 bulk/tail ESS>400（Vehtari 2021 的推荐值）
  · 实用判据 R̂<1.05 且 ESS>100（够做中位数与 90% 区间的底线）
最终结论以**实用判据**为准，严格判据单独列出。
"""

from __future__ import annotations

import sys
import time

import numpy as np

import bm_core as B
import bm_experiments as E

OK, NG, SK = "PASS", "FAIL", "SKIP"
_rows: list[tuple[str, str, str]] = []
_FAST = False


def check(name: str, ok: bool, detail: str = ""):
    _rows.append((name, OK if ok else NG, detail))
    print(f"  [{OK if ok else NG}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def check_or_skip(name: str, ok: bool, detail: str = ""):
    """冒烟模式下只打印不判定：300 步的链本来就不会收敛，判 FAIL 是误导。"""
    if _FAST:
        _rows.append((name, SK, detail))
        print(f"  [{SK}] {name}" + (f"   {detail}" if detail else ""))
        return True
    return check(name, ok, detail)


# ==========================================================================
# A. 数学自检
# ==========================================================================

def part_a():
    print("\n" + "=" * 70)
    print("A. 数学自检（不依赖采样）")
    print("=" * 70)

    # ---------------- A1 stick-breaking ----------------
    print("\nA1 stick-breaking")
    v = np.array([0.3, 0.9])
    pi = B.stick_pi(v)
    check("π 之和为 1", abs(pi.sum() - 1.0) < 1e-12, f"Σπ={pi.sum():.12f}")
    check("π **不保证递减**（v=[0.3,0.9] → π 递增）", pi[1] > pi[0],
          f"π={np.round(pi,4)}")
    # K=10 随机抽一批，验证归一
    rng = np.random.default_rng(0)
    vv = rng.random((2000, 9))
    pis = np.array([B.stick_pi(x) for x in vv])
    check("K=10 批量归一误差 < 1e-12",
          float(np.abs(pis.sum(axis=1) - 1).max()) < 1e-12,
          f"max|Σπ−1|={np.abs(pis.sum(axis=1)-1).max():.2e}")

    # ---------------- A2 HN-DRT 归一化 ----------------
    print("\nA2 HN 的 DRT 归一化 ∫γ dlnτ = 1")
    ln0 = np.log(1.0 / (2 * np.pi * 100.0))
    grid = np.linspace(ln0 - 40, ln0 + 40, 400001)
    hn1 = lambda p, z: B._hn_drt_unit(grid, np.array([ln0]), np.array([p]),
                                      np.array([z]))[:, 0]
    cc1 = lambda p: B._cole_drt_unit(grid, np.array([ln0]), np.array([p]))[:, 0]
    worst = 0.0
    for phi, ze in [(0.5, 1.0), (0.85, 1.0), (0.95, 1.0), (0.999, 1.0),
                    (0.6, 0.5), (0.85, 0.6), (0.85, 1.8), (0.999, 0.5)]:
        g = hn1(phi, ze)
        s = np.trapezoid(g, grid)
        worst = max(worst, abs(s - 1.0))
        print(f"    φ={phi:<5} ζ={ze:<4} ∫γdlnτ={s:.8f}  γ_min={g.min():+.2e}")
    check("多组 (φ,ζ) 下积分与 1 的偏差 < 1e-3", worst < 1e-3,
          f"max|∫γ−1|={worst:.2e}")

    # φ=1 是 Dirac 极限：γ 在 τ0 处有 1/|1−q|^ζ 型奇点，网格积分必然发散。
    # 这不是公式错——理想 Debye（φ=1,ζ=1）的 DRT 本来就是 τ0 处的一个 δ。
    # 模型的 φ 支撑是**开区间** (0.5,1)，取不到 1，所以不影响使用；
    # 但要知道：一旦有人把上界改成 1，`drt_band` 会在网格点上炸出 inf。
    print("\nA2b φ=1 是 Dirac 极限（预期发散，非 bug）")
    g1 = hn1(1.0, 1.0)
    check("φ=1 时 γ 在 τ0 处出现奇点（max γ 极大）", g1.max() > 1e6,
          f"max γ={g1.max():.3e}，∫={np.trapezoid(g1, grid)}")
    check("模型 φ 上界为 1 且为开区间（故取不到 φ=1）",
          "Uniform(phi_lo, 1.0)" in __import__("inspect").getsource(B.make_model),
          "make_model 里 phi ~ Uniform(phi_lo, 1.0)，端点概率为 0")

    # ---------------- A3 ζ=1 退化到 Cole-Cole ----------------
    print("\nA3 ζ=1 时与 Cole-Cole 闭式一致")
    d = 0.0
    for phi in (0.5, 0.7, 0.85, 0.95):
        a = hn1(phi, 1.0)
        b = cc1(phi)
        d = max(d, float(np.abs(a - b).max()))
    check("HN(ζ=1) 与 Cole-Cole 最大差 < 1e-10", d < 1e-10, f"max diff={d:.2e}")

    # ---------------- A4 φζ>1 出现负值 ----------------
    print("\nA4 φζ>1 时 γ 出现负值")
    bad = hn1(0.9, 1.2)
    check("φ=0.9, ζ=1.2 时 γ_min < 0", bad.min() < 0, f"γ_min={bad.min():+.3e}")
    check("此时积分仍为 1（负值不影响归一）",
          abs(np.trapezoid(bad, grid) - 1.0) < 1e-3,
          f"∫γ={np.trapezoid(bad, grid):.6f}")

    # ---------------- A5 DRT 积分重建 Z（与正演自洽） ----------------
    # 这是对"DRT 公式到底对不对"最直接的检验：把 γ 代回
    #     Z(ω) = ∫ γ(lnτ)/(1+jωτ) dlnτ
    # 看能不能重建出 _hn_z 的阻抗。若公式把 ζ 漏了（例如退回 Cole-Cole），
    # 这一步在 ζ≠1 时会立刻对不上。
    print("\nA5 由 γ 积分重建 Z，与 _hn_z 对比（含 ζ≠1）")
    ff = np.logspace(-1, 4, 12)
    tau0 = 1.0 / (2 * np.pi * 100.0)
    grid2 = np.linspace(np.log(tau0) - 60, np.log(tau0) + 60, 400001)
    tau2 = np.exp(grid2)
    Kk = 1.0 / (1.0 + 1j * np.outer(2 * np.pi * ff, tau2))
    for phi, ze in ((0.85, 1.0), (0.9, 1.5), (0.6, 0.5), (0.95, 2.0)):
        g = B._hn_drt_unit(grid2, np.array([np.log(tau0)]),
                           np.array([phi]), np.array([ze]))[:, 0]
        Zint = np.trapezoid(g[None, :] * Kk, grid2, axis=1)
        Zdir = B._hn_z(ff, np.array([1.0]), np.array([tau0]),
                       np.array([phi]), np.array([ze]))[:, 0]
        err = float(np.max(np.abs(Zint - Zdir)) / np.max(np.abs(Zdir)))
        check(f"φ={phi}, ζ={ze}（φζ={phi * ze:.2f}）重建 Z 相对误差 < 1e-6",
              err < 1e-6, f"{err:.2e}")

    # ---------------- A6 正演一致性 ----------------
    print("\nA6 模型正演 vs 解析 ZARC")
    f, Zc = B.true_spectrum()
    Zm = B.model_z(f, B.truth_dict(), include_L=False)
    e = float(np.abs(Zm - Zc).max())
    check("model_z(真值) 与 zarc 解析式最大差 < 1e-10 Ω", e < 1e-10, f"max|ΔZ|={e:.2e}")

    # ---------------- A6 分量匹配 ----------------
    print("\nA7 跨样本分量匹配（还原被打乱的编号）")
    rng = np.random.default_rng(1)
    n, K = 400, 3
    tau_t = 1.0 / (2 * np.pi * np.array([357.0, 20.0, 2.0]))
    w_t = np.array([33.0, 15.8, 4.0])
    phi_t = np.array([0.85, 0.90, 0.80])
    s = {k: np.empty((n, K)) for k in ("tau", "w", "phi", "zeta", "pi")}
    for i in range(n):
        p = rng.permutation(K)
        s["tau"][i] = tau_t[p] * np.exp(0.05 * rng.standard_normal(K))
        s["w"][i] = w_t[p] * np.exp(0.10 * rng.standard_normal(K))
        s["phi"][i] = phi_t[p]
        s["zeta"][i] = 1.0
        s["pi"][i] = s["w"][i] / s["w"][i].sum()
    m = B.match_components(s)
    f0m = np.median(m["f0"], axis=0)
    wm = np.median(m["w"], axis=0)
    # 列顺序可整体置换：按 f0 降序对齐后再比
    o = np.argsort(-f0m)
    err_f0 = np.abs(f0m[o] - np.array([357.0, 20.0, 2.0])) / np.array([357.0, 20.0, 2.0])
    err_w = np.abs(wm[o] - w_t) / w_t
    uniq = [len(set(np.round(m["phi"][:, j], 4))) for j in range(K)]
    check("每列 φ 单一（说明指派正确，未被混列）", all(u == 1 for u in uniq),
          f"各列唯一值个数={uniq}")
    check("f0 相对误差 < 2%", float(err_f0.max()) < 0.02,
          f"max={err_f0.max()*100:.2f}%  {np.round(f0m[o],2)}")
    check("w 相对误差 < 5%", float(err_w.max()) < 0.05,
          f"max={err_w.max()*100:.2f}%  {np.round(wm[o],2)}")

    # ---------------- A7 单样本 DRT 积分 = W ----------------
    print("\nA8 单样本 DRT 积分 = Σw = W（权重守恒）")
    w_ = np.array([33.0, 15.8])
    lt = np.log(1.0 / (2 * np.pi * np.array([357.0, 20.0])))
    gg = np.zeros_like(grid)
    for k in range(2):
        gg += w_[k] * B._hn_drt_unit(grid, np.array([lt[k]]), np.array([0.85]),
                                     np.array([1.0]))[:, 0]
    integ = float(np.trapezoid(gg, grid))
    check("∫γdlnτ 与 Σw 偏差 < 0.5%", abs(integ - w_.sum()) / w_.sum() < 5e-3,
          f"∫={integ:.4f} Ω, Σw={w_.sum():.1f} Ω, 偏差={abs(integ-w_.sum())/w_.sum()*100:.2f}%")


# ==========================================================================
# B. 拟合端到端
# ==========================================================================

def part_b(fast: bool = False):
    print("\n" + "=" * 70)
    print("B. 拟合端到端（K=2，合成 2×ZARC + R∞，σ=0.2 Ω 已知）")
    print("=" * 70)

    f, Z = B.noisy()
    W_true = float(sum(e["R"] for e in B.ZARC))
    R_true = np.array([e["R"] for e in B.ZARC], float)
    f0_true = np.array([e["f0"] for e in B.ZARC], float)
    print(f"数据 {f.size} 点 {f.min():.3g}~{f.max():.3g} Hz，"
          f"R∞={B.R_INF} Ω，W={W_true} Ω，真值 f0={f0_true}")

    if fast:
        kw = dict(n_warmup=300, n_samples=300, target_accept=0.95, max_tree_depth=10)
        print("\n**冒烟模式**：只验证流程能跑通，B 段一律 SKIP 不判定——")
        print("  300 步的链本来就不会收敛，判 FAIL 没有意义。要判定请不带参数运行。")
    else:
        kw = dict(n_warmup=3000, n_samples=1500, target_accept=0.995, max_tree_depth=13)
    print(f"采样设置 {kw}，3 链，seed=0")

    t0 = time.time()
    s, mc = B.fit(f, Z, K=2, alpha=0.5, sigma=(B.SIGMA, B.SIGMA), num_chains=3,
                  seed=0, progress=False, extra_fields=("num_steps",), **kw)
    dt = time.time() - t0
    ch = mc.get_samples(group_by_chain=True)
    ndiv = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    print(f"用时 {dt:.0f}s，divergences {ndiv}/{3*kw['n_samples']}")
    tds = E.tree_depth_stats(mc, kw["max_tree_depth"])
    if tds:
        print(f"树深度 中位 {tds['med']:.0f}，p95 {tds['p95']:.0f}，"
              f"触顶 {tds['frac_cap']*100:.1f}%")

    # ---------------- B1 收敛 ----------------
    print("\nB1 全部参数收敛诊断（rank-normalized split R̂ / bulk / tail ESS）")
    rows = E.full_param_diag(ch)
    E.print_param_diag(rows, indent="  ")
    rmax = max(r["rhat"] for r in rows)
    bmin = min(r["bulk"] for r in rows)
    tmin = min(r["tail"] for r in rows)
    check_or_skip("严格判据：R̂<1.01 且 bulk/tail ESS>400",
          rmax < 1.01 and bmin > 400 and tmin > 400,
          f"R̂max={rmax:.3f}  bulk_min={bmin:.0f}  tail_min={tmin:.0f}")
    check_or_skip("实用判据：R̂<1.05 且 ESS>100",
          rmax < 1.05 and bmin > 100 and tmin > 100,
          f"R̂max={rmax:.3f}  bulk_min={bmin:.0f}  tail_min={tmin:.0f}")

    # ---------------- B2 分量 ----------------
    print("\nB2 分量估计（跨样本匹配后）")
    mm = B.match_components(s)
    f0m = np.median(mm["f0"], axis=0)
    wm = np.median(mm["w"], axis=0)
    o = np.argsort(-f0m)
    print("    匹配后 f0 = " + " / ".join(f"{v:.1f}" for v in f0m[o]) + " Hz")
    print("    匹配后 w  = " + " / ".join(f"{v:.2f}" for v in wm[o]) + " Ω")
    print("    真值   f0 = " + " / ".join(f"{v:.1f}" for v in f0_true) + " Hz")
    print("    真值   w  = " + " / ".join(f"{v:.2f}" for v in R_true) + " Ω")
    err_w = np.abs(wm[o] - R_true) / R_true
    err_f = np.abs(f0m[o] - f0_true) / f0_true
    check_or_skip("面积相对误差 < 5%", float(err_w.max()) < 0.05,
          f"max={err_w.max()*100:.2f}%")

    # 峰位分两条：ζ 自由的后验只**记录**，不判真假；ζ=1 固定的确定性最优才判。
    # 原因见 bm_ident.py：ζ≠1 时 DRT 峰不在 τ0 上，(τ0,ζ) 之间有弱约束方向，
    # 峰位会被拉走而阻抗几乎不变。用"后验峰位"当正确性判据会误判成代码错。
    print(f"\n    [记录] ζ 自由时后验峰位误差 {err_f.max()*100:.1f}%"
          f"（不作为判据，原因见 bm_ident.py 的 C2/C3 对照）")
    import bm_ident
    rmap = bm_ident.map_fit(f, Z, zfix=1.0)
    err_m = np.abs(rmap["f0"] - f0_true) / f0_true
    check_or_skip("峰位相对误差 < 5%（ζ=1 固定，确定性 MAP）", float(err_m.max()) < 0.05,
          f"max={err_m.max()*100:.2f}%  f0={np.round(rmap['f0'],1)}")

    print("\n    对照：comp_table（按平均权重排序，未做跨样本匹配）")
    for r in B.comp_table(s):
        print(f"      f={r['f_Hz']:7.1f} Hz  w={r['area']:6.2f} "
              f"[{r['area_lo']:6.2f},{r['area_hi']:6.2f}] Ω  "
              f"φ={r['phi']:.3f}  ζ={r['zeta']:.2f}  "
              f"φζ>1 占比={r['frac_neg']*100:.1f}%")

    # ---------------- B3 PPC ----------------
    print("\nB3 后验预测检验（真·PPC，含噪声重抽）")
    pp = B.posterior_predictive_check(s, f, Z, include_L=False, n_rep=300, level=0.90)
    print(f"    覆盖率 实部={pp['cover_re']:.3f}  虚部={pp['cover_im']:.3f}  "
          f"（目标 0.90）  复制数据 rms={pp['rms_med']:.4f} Ω")
    check_or_skip("实部覆盖率 ∈ [0.80, 0.98]", 0.80 <= pp["cover_re"] <= 0.98,
          f"{pp['cover_re']:.3f}")
    check_or_skip("虚部覆盖率 ∈ [0.80, 0.98]", 0.80 <= pp["cover_im"] <= 0.98,
          f"{pp['cover_im']:.3f}")

    # ---------------- B4 残差 ----------------
    print("\nB4 拟合残差")
    rs = B.fit_residual_stats(s, f, Z, include_L=False)
    exp = B.SIGMA * np.sqrt(2)
    print(f"    rms 中位数={rs['rms_med']:.4f} Ω，相对={rs['rel_med']*100:.2f}%，"
          f"期望≈σ√2={exp:.4f} Ω")
    check_or_skip("残差 rms 在 σ√2 的 ±30% 内", abs(rs["rms_med"] - exp) / exp < 0.30,
          f"{rs['rms_med']:.4f} vs {exp:.4f}")

    # ---------------- 附加：DRT 曲线 ----------------
    print("\n附加：后验 DRT 曲线（仅可视化，负值占比已标出）")
    import warnings
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        grid, med, lo, hi = B.drt_band(s)
    for w in wlist:
        print(f"    警告: {w.message}")
    print(f"    网格 {len(grid)} 点，γ 中位数范围 "
          f"[{med.min():.3f}, {med.max():.3f}]，积分={np.trapezoid(med, grid):.2f} Ω"
          f"（对照 W={W_true}，中位曲线的积分≠W 属正常：非线性）")

    return s


# ==========================================================================
# 汇总
# ==========================================================================

def summary():
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    npass = sum(1 for _, v, _ in _rows if v == OK)
    nskip = sum(1 for _, v, _ in _rows if v == SK)
    print(f"{'检查项':<46}{'结果':>6}")
    for nm, v, _ in _rows:
        print(f"{nm:<46}{v:>6}")
    if nskip:
        print(f"\n{npass} 通过 / {len(_rows)-npass-nskip} 失败 / {nskip} 跳过（冒烟模式）")
    else:
        print(f"\n{npass}/{len(_rows)} 通过")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"
    _FAST = (cmd == "fast")
    part_a()
    if cmd == "nofit":
        summary()
    else:
        part_b(fast=_FAST)
        summary()
