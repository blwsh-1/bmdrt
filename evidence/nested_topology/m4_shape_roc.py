"""用合成谱大规模标定「voigt 形态判据」的假阳性率与灵敏度。

背景：`m4_shape_criterion.py` 的两条排除性判据（Re 上升 >8%·max|Z|、
低频 Im >5%·|Z|）只在**4 个**手工用例上标定过，而真实谱 18 条 AZ31 是同一
材料的 Li 梯度（`independence_status` 全 `inferred`），不能当独立样本。
本脚本用参数扫描补这个缺口。

四类真值
--------
1. `voigt`     ：裸 voigt（R_inf + K 个 HN/ZARC），K=1..3 —— 判据**不该**排除
2. `voigt_ad`  ：voigt + 吸附支路 `Z_ad=R·jωτ/(1+jωτ)` —— 生产的 `--adsorption`
3. `npar`      ：nested parallel 支路（R∥L），L=1,2 —— 判据**该**排除
4. `nser`      ：nested series 支路（R+L），L=1,2 —— 判据**该**排除

**第 2 类是本判据的已知软肋**：`Z_ad` 的实部 `R(ωτ)²/(1+(ωτ)²)` 在 ω→0
时 →0，与 nested 的 RL 支路一样把低频实部拉回；其虚部恒正，也与低频感抗
同向。所以「裸 voigt 能排除」不等于「voigt 族能排除」。必须单独报出来。

输出：每类的命中率、阈值扫描（找特异度 ≥95% 下灵敏度最高的工作点）。
零拟合成本，只做正演 + 形态检查。

用法
----
    python evidence/nested_topology/m4_shape_roc.py            # 默认 120/类
    python evidence/nested_topology/m4_shape_roc.py 300        # 每类 300 条
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bm_top_gen as G          # noqa: E402

#: 判据的工作阈值（与 m4_shape_criterion.py 一致）
UP_DEFAULT = 0.08
IM_DEFAULT = 0.05

#: 噪声档（σ = frac·max|Z|，实虚独立，与 bm_core 的似然同口径）
SIGMAS = (0.005, 0.01, 0.02, 0.05)

#: 频栅：AZ31 同款 + 两档密度，检查判据对采样点数是否敏感
GRIDS = (("AZ31", 50), ("AZ31", 30), ("AZ31", 80))


def voigt_spec(f, R_inf, ws, taus, phis):
    """裸 voigt：R_inf + Σ w/(1+(jωτ)^φ)，ζ=1（`zeta="1.0"` 的生产默认）。"""
    z = np.full(len(f), float(R_inf), complex)
    w_ax = 2 * np.pi * np.asarray(f, float)
    for w, tau, phi in zip(ws, taus, phis):
        z = z + w / (1.0 + (1j * w_ax * tau) ** phi)
    return z


def voigt_ad_spec(f, R_inf, ws, taus, phis, R_ad, tau_ad):
    """voigt + 吸附支路（`bm_core.model_z` 的 `Rad`/`tau_ad` 项）。"""
    w_ax = 2 * np.pi * np.asarray(f, float)
    jt = 1j * w_ax * tau_ad
    return voigt_spec(f, R_inf, ws, taus, phis) + R_ad * jt / (1.0 + jt)


def rand_voigt(rng, K):
    """随机一对 (R_inf, w_k, tau_k, phi_k)，量级参照 BASE/REAL。"""
    lu = lambda lo, hi: float(10.0 ** rng.uniform(lo, hi))
    ws = [lu(0.5, 3.5) for _ in range(K)]
    taus = [lu(-2.0, 2.0) for _ in range(K)]
    phis = [float(rng.uniform(0.6, 1.0)) for _ in range(K)]
    return lu(-0.5, 1.7), ws, taus, phis


def measure(f, Z):
    """返回两条判据的统计量（与 m4_shape_criterion.analyze 同口径）。"""
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    scale = float(np.max(np.abs(Z)))
    dre = np.diff(Z.real)
    up_max = float(dre[dre > 0].max() / scale) if (dre > 0).any() else 0.0
    return up_max, float(Z.imag[0] / scale)


def make_set(kind: str, n: int, rng):
    """生成 n 条该类谱，覆盖 SIGMAS × GRIDS，返回 [(up_max, im_low, sigma, grid)]。"""
    out = []
    for i in range(n):
        sig = SIGMAS[i % len(SIGMAS)]
        gname, gpts = GRIDS[(i // len(SIGMAS)) % len(GRIDS)]
        f = G.real_grid(gname, gpts)
        if kind in ("voigt", "voigt_ad"):
            K = 1 + i % 3
            R_inf, ws, taus, phis = rand_voigt(rng, K)
            Z = (voigt_spec(f, R_inf, ws, taus, phis) if kind == "voigt" else
                 voigt_ad_spec(f, R_inf, ws, taus, phis,
                               R_ad=10.0 ** rng.uniform(1.5, 3.5),
                               tau_ad=10.0 ** rng.uniform(-2.0, 2.0)))
        else:
            branch = "parallel" if kind == "npar" else "series"
            p = G._rand_params(rng, layers=1 + i % 2, branch=branch)
            Z = G.gen(f, p, branch=branch, layers=1 + i % 2)
        Z = G.add_noise(Z, sig, seed=1000 + i)
        up, im = measure(f, Z)
        out.append((up, im, sig, f"{gname}{gpts}"))
    return out


def rates(sets, up_th, im_th):
    """在给定阈值下算各类的「判为排除」比例。"""
    return {k: float(np.mean([(u > up_th) or (m > im_th) for u, m, _, _ in v]))
            for k, v in sets.items()}


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 120
    kinds = ("voigt", "voigt_ad", "npar", "nser")
    # 每类独立 rng：共用一个会因调用顺序改变而生成不同样本，前后不可比
    sets = {k: make_set(k, n, np.random.default_rng(20261110 + 7 * i))
            for i, k in enumerate(kinds)}
    print(f"每类 {n} 条（噪声 {SIGMAS} × 网格 {[g for _, g in GRIDS]}）\n")

    print("=== 默认阈值（Re >8% / Im >5%）===")
    r = rates(sets, UP_DEFAULT, IM_DEFAULT)
    print(f"{'类':<10}{'判排除比例':>10}   应为")
    for k, label in (("voigt", "不该排除"), ("voigt_ad", "不该排除"),
                     ("npar", "该排除"), ("nser", "该排除")):
        print(f"{k:<10}{r[k]:>10.1%}   {label}")

    print("\n=== 按噪声档拆开（Re>8% / Im>5%）===")
    print(f"{'σ':>7}" + "".join(f"{k:>10}" for k in kinds))
    for s in SIGMAS:
        row = "".join(f"{np.mean([(u > UP_DEFAULT) or (m > IM_DEFAULT) for u, m, sg, _ in sets[k] if sg == s]):>10.1%}"
                      for k in kinds)
        print(f"{s:>6.1%}" + row)

    print("\n=== 阈值扫描：特异度(裸voigt不被排除) vs 灵敏度(nested 被排除) ===")
    print(f"{'Re阈值':>7}{'Im阈值':>8}{'特异度':>9}{'灵敏度':>9}   备注")
    best = None
    for up_th in (0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20):
        for im_th in (0.01, 0.03, 0.05, 0.08, 0.12):
            r = rates(sets, up_th, im_th)
            spec = 1 - r["voigt"]
            sens = 0.5 * (r["npar"] + r["nser"])
            # 生产实际会开的 ad 组：把它当"不该排除"计入更保守的特异度
            spec_ad = 1 - r["voigt_ad"]
            if spec >= 0.95 and (best is None or sens > best[2]):
                best = (up_th, im_th, sens, spec, spec_ad)
            print(f"{up_th:>7.0%}{im_th:>8.0%}{spec:>9.1%}{sens:>9.1%}"
                  f"   ad组特异度 {spec_ad:.0%}" +
                  ("  ← 当前默认" if (up_th, im_th) == (UP_DEFAULT, IM_DEFAULT) else ""))

    if best:
        print(f"\n建议工作点：Re>{best[0]:.0%} / Im>{best[1]:.0%} → "
              f"特异度 {best[3]:.1%}、灵敏度 {best[2]:.1%}、"
              f"含 ad 组特异度 {best[4]:.1%}")
    else:
        print("\n没有工作点能让裸 voigt 特异度 ≥95% —— 判据不成立，需重设。")

    # ---- 噪声归一化对照：若把分母从 max|Z| 换成局部噪声 σ，能否救 ----
    # 这里用**真值 σ**（扫描时已知），所以是乐观上界：真实谱必须自己估 σ̂，
    # 估不准就退化回绝对值口径。只用来判断「该不该继续修」，不是可用结果。
    print("\n=== 噪声归一化对照（分母换成真值 σ，乐观上界）===")
    print(f"{'上升/σ':>9}{'Im>5%':>8}{'裸voigt':>9}{'voigt+ad':>10}"
          f"{'npar':>8}{'nser':>8}")
    for snr_th in (3.0, 4.0, 5.0, 6.0, 8.0):
        row = []
        for k in kinds:
            row.append(np.mean([(u / sg > snr_th) or (m > IM_DEFAULT)
                                for u, m, sg, _ in sets[k]]))
        print(f"{snr_th:>9.0f}{IM_DEFAULT:>8.0%}" + "".join(f"{v:>9.0%}" for v in row))
    print("读法：『裸voigt』列是误判率（越低越好），后两列是检出率（越高越好）。")
    print("      ad 组即使归一化也高误判 → 判据对 `--adsorption` 开启的 voigt 无效。")
    print("\n注意：本扫描的对象是**设计分布**，不是实测谱。它回答"
          "「在这组参数范围里判据错多少」，\n不能代替真实谱验证，也不能据此"
          "给族级结论升档。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
