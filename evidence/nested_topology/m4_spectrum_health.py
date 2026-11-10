#!/usr/bin/env python
"""谱级健康度诊断：两条「只记账、不裁决」的几何量。

对应 `docs/REFS_DIGEST.md` §2.L（书 p.57 稳定性反查）与 §2.E（低频漂移）。
两者都在**不拟合、不采样**的前提下算，因此可以独立于 M4 的选择器使用，
也可以拿来解释「这条真实谱的低频形状值不值得看」。

## 一、`stability_shape()` —— 书 p.57 唯一那条从谱形反查测量合法性的规则

    若 `R_p > 0`、**全谱无感抗**、且 `R_p < R_t` → 需检验稳定性条件。

判定顺序（顺序本身是本工具唯一的判断，不要调换）：
  1. 存在感抗（`Im > IND_TOL·max|Z|`）→ `not_applicable`。**规则前提不成立**，
     此时既不能说通过也不能说不通过。
  2. 低频实部显著为负（`R_p < −NEG_TOL·max|Z|`）→ `unverified`。
  3. `R_p` 比实部弧顶低 `END_TOL·弧高` 以上 → `unverified`。
  4. 否则 `ok`。

三条实现口径，每条都是踩过坑之后定的：
  · **只看正虚部**。用 `|Im|` 会把每个容抗弧都算成「有感抗」（第一版实测全谱
    25–46%，四条合成谱全被读成有感抗，规则一次都没触发）。
  · **`R_p < R_t` 用端点比较，不用逐步单调**。按 `np.diff` 逐步判，1% 噪声就能在
    某一步造出超过容差的回落，把纯 voigt 真值误报成 `unverified`（实测 σ=1% 即触发）。
  · **`R_p ≤ 0` 要带相对容差**。超低阻样品（`archive/2EMC*`，`max|Z|≈0.05 Ω`）
    的噪声足以把低频实部推到 −0.01 Ω，硬判「越界」是假阳性。

## 二、`drift_roughness()` —— 低频段的粗糙度

`bm_kk.py` 自己的构造表写着「低频漂移 0.10R → 残差仅 1.5%，判 ok」，即 KK 对这个
量级的漂移无感。这里换一个不依赖 KK 的量：**低频相邻点相对跳变的中位数**，
再除以**中频段同一统计量**做基准，得 `ratio`。

口径：
  · `ratio ≈ 1` → 低频与中频一样平滑；
  · `ratio ≫ 1` → 低频显著比中频粗糙，该段形状不足以支撑拓扑结论；
  · `ratio < 1` → 低频反而更平滑（在 8Li 组实测出现）。

**为什么用中频做基准而不是绝对阈值**：18 条 AZ31 的绝对值 `jmax` 分布是连续的
（0.086→0.551，无断点），绝对阈值只能凭空切一刀；`ratio` 把「这条谱自己有多平」
除掉之后才可比。

## 用法

    python evidence/nested_topology/m4_spectrum_health.py            # 18 条 AZ31
    python evidence/nested_topology/m4_spectrum_health.py archive    # 指定目录
    python evidence/nested_topology/m4_spectrum_health.py --synth     # 加四类合成谱
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

IND_TOL = 0.05      # 感抗阈值：相对 `max|Z|`，只作用于**正的**虚部
END_TOL = 0.10      # `R_p < R_t` 容差：相对弧高
NEG_TOL = 0.05      # 负实部容差：相对 `max|Z|`
N_LOW = 8           # 低频段点数
N_MID = 12          # 中频段起点（跳过高频端）


def _order(f, Z):
    o = np.argsort(np.asarray(f, float))[::-1]      # 高频在前
    return np.asarray(f, float)[o], np.asarray(Z, complex)[o]


def stability_shape(f, Z, *, ind_tol: float = IND_TOL,
                    end_tol: float = END_TOL, neg_tol: float = NEG_TOL) -> dict:
    """书 p.57 的谱形反查。返回 `verdict ∈ {ok, unverified, not_applicable}`。"""
    f, Z = _order(f, Z)
    re = Z.real
    zmax = float(np.max(np.abs(Z))) or 1.0

    i_max = int(np.argmax(re))
    r_peak = float(re[i_max])
    rp = float(re[-1])                  # 最低频实部 ≈ R_p 的下界
    rs = float(re[0])
    arc = r_peak - rs

    depth = Z.imag / zmax               # 只看正虚部
    has_ind = bool(np.any(depth > ind_tol))
    i_ind = int(np.argmax(depth > ind_tol)) if has_ind else 0

    gap = (r_peak - rp) / max(arc, 1e-12)
    out = dict(rp=rp, rs=rs, r_peak=r_peak, arc=float(arc), gap=float(gap),
               f_peak=float(f[i_max]), has_ind=has_ind,
               ind_depth=float(depth.max()),
               ind_f=float(f[i_ind]) if has_ind else float("nan"))

    if has_ind:
        out.update(verdict="not_applicable",
                   why=f"存在感抗（Im {depth[i_ind]:.1%}·max|Z| @ {f[i_ind]:.3g} Hz），"
                       "规则前提「无感抗」不成立")
    elif rp < -neg_tol * zmax:
        out.update(verdict="unverified",
                   why=f"低频实部 {rp:.3g} < −{neg_tol:.0%}·max|Z|（{-neg_tol * zmax:.3g}），实部越界")
    elif gap > end_tol:
        out.update(verdict="unverified",
                   why=f"R_p={rp:.3g} 比弧顶 {r_peak:.3g}(@{f[i_max]:.3g} Hz)"
                       f" 低 {gap:.0%}·弧高 → R_p<R_t，且无感抗")
    else:
        out.update(verdict="ok",
                   why=f"R_p ≥ 弧顶（差 {gap:.0%}·弧高），无感抗")
    return out


def drift_roughness(f, Z, *, n_low: int = N_LOW, n_mid: int = N_MID) -> dict:
    """低频相邻点相对跳变，除以中频同量做基准。返回连续量，**无线判决**。"""
    f, Z = _order(f, Z)
    re = Z.real

    def _jm(seg):
        d = np.abs(np.diff(seg))
        s = np.maximum(np.abs(seg[:-1]), 1e-12)
        return float(np.median(d / s)) if d.size else float("nan")

    k = min(n_low, max(2, len(re) // 3))
    low = _jm(re[-k:])
    mid = _jm(re[n_mid:n_mid + k])
    ratio = low / mid if mid and np.isfinite(mid) and mid > 0 else float("nan")
    return dict(low=low, mid=mid, ratio=float(ratio),
                jmax=float(np.max(np.abs(np.diff(re[-k:])) /
                                  np.maximum(np.abs(re[-k:-1]), 1e-12))))


def _synth(name: str, noise: float = 0.01, seed: int = 1):
    import bm_core as B
    import bm_top_gen as G
    f = G.real_grid()
    if name == "voigt_true":
        fv, Zv = B.true_spectrum(f)
        return fv, G.add_noise(Zv, noise, seed=seed)
    if name == "nested_true":
        return f, G.add_noise(G.gen_parallel_L2(f, dict(G.REAL)), noise, seed=seed)
    d = G.gen_mixed_boundary(name, f=f, noise=noise, seed=seed)
    return d["f"], d["Z"]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want_synth = "--synth" in argv
    dirs = [a for a in argv if not a.startswith("--")] or ["out/_m6_samples/z_csv"]

    import bm_real as R

    print(f"IND_TOL={IND_TOL:.0%}·max|Z|（只看正虚部）  END_TOL={END_TOL:.0%}·弧高  "
          f"NEG_TOL={NEG_TOL:.0%}·max|Z|  N_LOW={N_LOW}\n")

    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.csv")))
    print(f"{'file':30s} {'stability':16s} {'R_p':>9s} {'R_peak':>9s} "
          f"{'Im+':>6s} {'drift':>6s}")
    print("-" * 82)
    cnt, ratios = {}, []
    for p in files:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                f, Z = R.load_spectrum(p)
        except BaseException as e:
            print(f"{os.path.basename(p):30s} 载入失败 {type(e).__name__}")
            continue
        s = stability_shape(f, Z)
        d = drift_roughness(f, Z)
        cnt[s["verdict"]] = cnt.get(s["verdict"], 0) + 1
        ratios.append((os.path.basename(p), d["ratio"]))
        print(f"{os.path.basename(p):30s} {s['verdict']:16s} {s['rp']:9.3g} "
              f"{s['r_peak']:9.3g} {s['ind_depth']:6.1%} {d['ratio']:6.2f}")
    print("-" * 82)
    print(f"stability 分布：{cnt}")

    ok = [(n, r) for n, r in ratios if np.isfinite(r)]
    if ok:
        rs = np.array([r for _, r in ok])
        print(f"drift ratio：n={len(rs)}  中位={np.median(rs):.2f}  "
              f"范围={rs.min():.2f}–{rs.max():.2f}")

    if want_synth:
        print("\n四类合成谱（真值合法；无 L 的应 ok，含 L 的应 not_applicable）")
        print("-" * 82)
        for noise in (0.01, 0.03):
            print(f"-- σ={noise:.0%}·max|Z| --")
            for nm in ("voigt_true", "nested_true",
                       "loop_out_of_window", "arcs_merged"):
                f, Z = _synth(nm, noise=noise)
                s = stability_shape(f, Z)
                d = drift_roughness(f, Z)
                print(f"{nm:22s} {s['verdict']:16s} drift={d['ratio']:5.2f}  {s['why']}")


if __name__ == "__main__":
    main()
