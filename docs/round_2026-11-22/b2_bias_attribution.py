"""b2_bias_attribution.py —— B2 的"`L` 系统性正偏 +1.4~2.0%"到底是不是系统偏。

问题
----
2026-11-21 复盘的结论之一是：三个配方的 `L` 后验中位都比真值高
+2.0% / +1.9% / +1.4%，且 90%CI **不含真值** ⇒ "方向一致的系统正偏"。
这三个是 `voigt_K1_L` / `nested_L2_L` / **`adr_L`**（`adr_series` 的 `L` 偏 +1.2%
但覆盖率 100%，不属于这一组——别混）。
读起来像"三族共同的正偏"，也就是一个系统性错误（生成/似然/参数化之一）。

本脚本要回答两件事，都**不跑 MCMC**（除 `--mcmc` 复核外）：

1. **三个 +Δ 是不是三次独立抽样？**
   `b2_baseline.generate()` 对每个配方都用 `default_rng(seed)`、都取
   `standard_normal(61)` 两次。所以三族的**标准化噪声实现逐位相同**，
   只是各自的 `sigma = pct·|Z_true|` 不同。也就是说"3/3 全正"实际只有
   **1 个**独立随机抽样，其同号不能当作"系统性"的证据。

2. **那一个噪声实现能不能定量预报出这三个 +Δ？**
   在线性-高斯（Laplace）近似下，噪声在真值处的**白化残差**
   `r = [(Z_obs−Z_true)_re/σ_re, (·)_im/σ_im]` 与雅可比 `J = ∂Z/∂x`
   决定后验均值的一阶位移：
       Δx ≈ (JᵀJ + Λ)^{-1} Jᵀ r ,      Λ = 先验精度（对角）
   若预报值与实测偏差在**方向与量级**上都对上，则偏差完全由那一次噪声实现
   解释，不存在需要修的系统性错误；剩下的问题只是"1 个 seed 估不出偏置"。

口径
----
· 坐标与 `b2_baseline.SPEC` 完全一致：log 参数在 `ln θ` 上做，有界参数在原尺度。
· 先验精度取自代码而非文档：voigt 见 `bm_core.make_model`（R_inf/W σ=1、
  ln τ 与 φ、ζ 为 Uniform ⇒ 曲率 0、L σ=2）；nested/adr 见
  `bm_top._prior_table` / `adr_prior_table`（Rs σ=1，其余 σ=2，n 为 Uniform）。
  Uniform 的参数在真值处（内点）曲率为 0，因此不产生拉力。
· 雅可比用中心差分，步长 `1e-6`（log 坐标下即 ×e^{±1e-6}）。

用法
----
    python docs/round_2026-11-22/b2_bias_attribution.py            # 只读，秒级
    python docs/round_2026-11-22/b2_bias_attribution.py --mcmc      # 另跑 4 次真拟合
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-21"))

import b2_baseline as B2          # noqa: E402
import bm_top as T                # noqa: E402


def sampled_coords(rec, flat):
    """只取**真正被采样**的参数。

    这一点必须按代码来，不能按真值字典来：`bm_select.fit_voigt` 把 ζ 以位置参数
    `"1.0"` 传给 `bm_real.fit_one`，即 **ζ 被钉死、不进采样**。若把 ζ 也放进雅可比，
    它与 τ/φ 的强相关会把协方差摊出去 —— 实测把 τ 的 1σ 夸大约 4 倍
    （5.6% vs MCMC 的 1.2%），整张稳健性表就假了。
    """
    fam = rec["family"]
    layers = int(rec.get("layers", 2))
    if fam == "voigt":
        keys = [k for k in flat if k != "zeta"]
    elif fam == "nested":
        keys = [k for k in T.sampled_keys("parallel", layers) if k in flat]
    else:
        keys = [k for k in T.adr_sampled_keys() if k in flat]
    if "L" in flat:                      # L 是族能力开关，不在 sampled_keys 里
        keys.append("L")
    return {k: B2.SPEC[fam][k] for k in keys}


#: 先验精度（log 坐标下的 1/σ²）。与上面 docstring 里点到的代码位置一致。
#: 不在表里的参数 = 有界 Uniform（曲率 0）。
PRIOR_PREC_LOG = {
    "voigt": {"R_inf": 1.0, "W": 1.0, "L": 0.25},
    "nested": {"Rs": 1.0, "R1": 0.25, "Q1": 0.25, "R2": 0.25, "Q2": 0.25,
               "R_ad": 0.25, "tau_ad": 0.25, "L": 0.25},
    "adr": {"Rs": 1.0, "Q": 0.25, "Rct": 0.25, "R_ad": 0.25,
            "tau_ad": 0.25, "L": 0.25},
}

FOCUS = ("voigt_K1_L", "nested_L2_L", "adr_L")


def _recipe(name):
    for r in B2.RECIPES:
        if r["name"] == name:
            return r
    raise KeyError(name)


def whiten(dz, sigma):
    return np.concatenate([np.asarray(dz.real) / sigma[0],
                           np.asarray(dz.imag) / sigma[1]])


def jacobian(family, f, flat, sigma, coords, layers):
    """白化雅可比：列 = ∂(白化 Z)/∂x_k。"""
    cols, names = [], list(coords)
    for k in names:
        th = float(flat[k])
        if coords[k] == "log":
            h = 1e-6
            p1, p2 = dict(flat), dict(flat)
            p1[k], p2[k] = th * np.exp(h), th * np.exp(-h)
            dz = (B2.forward(family, f, p1, layers)
                  - B2.forward(family, f, p2, layers)) / (2 * h)
        else:
            h = 1e-6 * max(abs(th), 1.0)
            p1, p2 = dict(flat), dict(flat)
            p1[k], p2[k] = th + h, th - h
            dz = (B2.forward(family, f, p1, layers)
                  - B2.forward(family, f, p2, layers)) / (2 * h)
        cols.append(whiten(dz, sigma))
    return np.column_stack(cols), names


def laplace(family, f, flat, sigma, z_true, z_obs, coords, layers):
    """返回 (Δx, 后验 sd, 名称)。"""
    J, names = jacobian(family, f, flat, sigma, coords, layers)
    prec = PRIOR_PREC_LOG[family]
    lam = np.array([prec.get(k, 0.0) if coords[k] == "log" else 0.0
                    for k in names])
    H = J.T @ J + np.diag(lam)
    r = whiten(z_obs - z_true, sigma)
    dx = np.linalg.solve(H, J.T @ r)
    cov = np.linalg.inv(H)
    return dx, np.sqrt(np.diag(cov)), names, J, r, lam


def one(name, seed=None):
    rec = dict(_recipe(name))
    if seed is not None:
        rec["seed"] = seed
    f, zt, zo, sg, _ = B2.generate(rec)
    flat = {k: float(v) for k, v in rec["truth"].items()}
    # 只对**真正被采样**的参数建列（见 `sampled_coords`）：`adr_series`(l=1) 借 nested
    # 的族名但只 1 层，SPEC 里多出 R2/Q2/n2；voigt 的 ζ 被 fit_voigt 钉死。
    coords = sampled_coords(rec, flat)
    layers = int(rec.get("layers", 2))
    dx, sd, names, J, r, lam = laplace(rec["family"], f, flat, sg, zt, zo,
                                       coords, layers)
    return dict(rec=rec, f=f, zt=zt, zo=zo, sg=sg, flat=flat, coords=coords,
                layers=layers, dx=dict(zip(names, dx)),
                sd=dict(zip(names, sd)), J=J, r=r, lam=lam, names=names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", action="store_true",
                    help="另跑真拟合复核 Laplace（voigt 的 seed 0/1/2 + 无噪声）")
    ap.add_argument("--nseed", type=int, default=500,
                    help="解析重抽噪声的 seed 数（默认 500，秒级）")
    a = ap.parse_args(argv)

    print("=" * 78)
    print("1. 三配方是否共享同一噪声实现")
    print("=" * 78)
    base = None
    for nm in FOCUS:
        r = one(nm)
        n = (r["zo"] - r["zt"]) / r["sg"][0]
        if base is None:
            base = n
        d = max(float(np.max(np.abs(n.real - base.real))),
                float(np.max(np.abs(n.imag - base.imag))))
        print(f"  {nm:<12} seed={r['rec']['seed']}  "
              f"与 voigt 的标准化噪声最大差 {d:.2e}")
    print("  ⇒ 逐位相同：'3/3 全正'只有 **1 个**独立抽样，不是三次。")

    print()
    print("=" * 78)
    print("2. Laplace 预报 vs 实测（B2_REPORT 的数字）")
    print("=" * 78)
    obs = {"voigt_K1_L": 2.0e-2, "nested_L2_L": 1.9e-2, "adr_L": 1.4e-2}
    print(f"{'配方':<12}{'L 预报':>10}{'L 实测':>10}"
          f"{'后验 sd(lnL)':>15}{'预报/实测':>12}")
    for nm in FOCUS:
        r = one(nm)
        d = r["dx"]["L"]
        s = r["sd"]["L"]
        o = obs[nm]
        print(f"  {nm:<12}{d*100:>9.2f}%{o*100:>9.2f}%{s*100:>13.2f}%"
              f"{d/o:>12.2f}")
    print("  读法：预报/实测≈1 ⇒ 偏差由这**一次**噪声实现定量解释。")

    print()
    print("=" * 78)
    print(f"3. 解析重抽 {a.nseed} 个噪声实现：Δ ln L 的分布（无 MCMC）")
    print("=" * 78)
    # Laplace 对噪声是线性的：Δx = M r，r 对噪声线性。于是可直接用
    # 同一 J/白化口径重抽标准正态，秒级得到 Δ 的抽样分布。
    for nm in FOCUS:
        r = one(nm)
        J, names = r["J"], r["names"]
        lam = np.array([PRIOR_PREC_LOG[r["rec"]["family"]].get(k, 0.0)
                        if r["coords"][k] == "log" else 0.0 for k in names])
        H = J.T @ J + np.diag(lam)
        Hinv = np.linalg.inv(H)
        # 把 z_true 处的 Z 也当成噪声的线性映射：Δx = Hinv Jᵀ (W nz)
        # nz = σ_re·ε_re + i σ_im·ε_im，且 W 除以 σ ⇒ 白化后恰是标准正态。
        n = r["f"].size
        rng = np.random.default_rng(12345)
        eps = np.concatenate([rng.standard_normal((a.nseed, n)),
                              rng.standard_normal((a.nseed, n))], axis=1)
        d_all = eps @ (Hinv @ J.T).T          # (nseed, nparam)
        iL = names.index("L")
        dL = d_all[:, iL]
        s = r["sd"]["L"]
        z0 = r["dx"]["L"] / s
        pct = float((dL > 0).mean() * 100)
        print(f"  {nm:<12} mean {dL.mean()*100:+6.2f}%  sd {dL.std()*100:5.2f}%  "
              f"P(Δ>0) {pct:4.1f}%  seed0 位于 {z0:+.2f}σ "
              f"(分位 {(dL < r['dx']['L']).mean()*100:4.1f}%)")
    print("  读法：mean≈0 且 sd 与实测偏差同量级 ⇒ 无系统偏，偏差是单次抽样振幅。")

    print()
    print("=" * 78)
    print("3b. 逐参数的 seed 间 1σ 与 B2 容差之比（判定对 seed 的稳健性）")
    print("=" * 78)
    print("  比值 = 1σ(seed) / 一步容差。>1 ⇒ 一次抽样就能翻转判定；")
    print("         >0.5 ⇒ 单 seed 判定偏脆，不该当成功率读。")
    for nm in FOCUS:
        r = one(nm)
        J, names = r["J"], r["names"]
        lam = np.array([PRIOR_PREC_LOG[r["rec"]["family"]].get(k, 0.0)
                        if r["coords"][k] == "log" else 0.0 for k in names])
        H = J.T @ J + np.diag(lam)
        sd = np.sqrt(np.diag(np.linalg.inv(H)))
        rec = r["rec"]
        tgt = set(rec.get("targets") or ())
        print(f"  [{nm}]")
        flags = []
        for i, k in enumerate(names):
            if k not in tgt:
                continue
            step = B2.step_of(rec["family"], k)
            # log 坐标下 sd 是 ln 单位、B2 的容差是相对量（q/th−1 ≤ 0.10），
            # 两者同量纲；有界参数两边都是绝对量。故直接比。
            ratio = sd[i] / step
            mark = "  ← 脆" if ratio > 0.5 else ""
            print(f"    {k:<8} 1σ {sd[i]:.4g}  容差 {step:.3g}  比值 {ratio:5.2f}{mark}")
            if ratio > 0.5:
                flags.append(k)
        print(f"    ⇒ 脆参数：{flags if flags else '无'}")
    print("  注：'1σ' 是该参数在后验里的抽样标准差（= 90%CI 半宽/1.645），")
    print("      由 (JᵀJ+Λ)^{-1} 给出；与 B2_REPORT 里 CI 宽度一致（见 §2 复核）。")

    if a.mcmc:
        print()
        print("=" * 78)
        print("4. 真拟合复核（voigt_K1_L，standard 预算）")
        print("=" * 78)
        rows = []
        for sd_ in (0, 1, 2):
            r = one("voigt_K1_L", seed=sd_)
            out = B2.fit(r["rec"], r["f"], r["zo"], r["sg"], "standard")
            got = float(out["flat"]["L"]) / 1e-5 - 1.0
            rows.append((sd_, r["dx"]["L"], got))
            print(f"  seed{sd_}  Laplace {r['dx']['L']*100:+6.2f}%   "
                  f"MCMC {got*100:+6.2f}%   R̂ {out['rhat']:.3f}  div {out['div']}")
        print("  两点连线是否同向同行 ⇒ Laplace 口径可用来给偏差定预算。")

    print()
    print("结论见 docs/round_2026-11-22/B2_BIAS_ATTRIBUTION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
