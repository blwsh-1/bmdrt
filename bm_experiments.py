# -*- coding: utf-8 -*-
"""BM-DRT 的全部实验脚本，用第一个命令行参数选择：

    python bm_experiments.py diag      # K=3 的 3 链完整诊断（含图）
    python bm_experiments.py isolate   # K=2 隔离实验：ζ 固定 vs 自由 vs 更严采样
    python bm_experiments.py ablate    # K=3 采样消融：ta / depth / warmup 逐个开
    python bm_experiments.py redig [npz]  # 对已保存样本重算诊断（不重跑采样）
    python bm_experiments.py scan      # 训练库 k 命中率扫描

依赖 bm_core.py（模型 + 合成数据）。

两个方法学要点
--------------
1) 混合模型的分量编号在样本间会互换（τ 没有排序约束），直接对原始编号算 R-hat
   会把**标签交换**误报成"链没收敛"（实测 ln_tau 的传统 R-hat 因此高达 31.6）。
   所以同时报两种：原始编号（暴露交换的严重程度）、按样本内 π 降序重排。

2) 收敛判据用 **rank-normalized split R̂ + bulk/tail ESS**（Vehtari 2021），
   不用 numpyro 自带的传统 gelman_rubin/effective_sample_size。实测同一份
   样本上，传统 R̂ 给 R_inf=1.044（看着合格），rank-normalized split R̂ 给
   1.202（不合格）、bulk ESS 只有 11.5——**旧口径偏乐观**。
"""

from __future__ import annotations

import ast as pyast
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import bm_core as B

# numpyro 只在**采样与收敛诊断**时需要。`paired_loo.py` 这类纯后处理脚本
# 要 import bm_real（→ 这里），没有 numpyro 时不该整条链都 import 失败。
# 真正用到 diag 的函数在缺库时会自己报错，别把失败提前到 import 阶段。
try:
    from numpyro import diagnostics as diag
except ImportError:                     # pragma: no cover
    diag = None


# --------------------------------------------------------------------------
# 共用诊断
# --------------------------------------------------------------------------

# P2-6（二轮审查）：扩散参数（Rw / Rwo+tauD）也是被采样的参数，逐参数
# R̂/ESS 表必须包含——否则扩散项慢混合会被「逐参数表全绿」掩盖。
PARAMS = ("R_inf", "W", "v", "ln_tau", "phi", "zeta", "L",
          "Rw", "Rwo", "tauD")


def with_zeta(ch, value: float = 1.0):
    """ζ 被钉死时 `get_samples(group_by_chain=True)` 里没有 `zeta` 键，补成常数。

    `B.fit` 已经给返回的 `s` 补过；但直接拿 `mc.get_samples(group_by_chain=True)`
    的地方（链级诊断）还需要再补一次。
    """
    if "zeta" in ch or "ln_tau" not in ch:
        return ch
    out = dict(ch)
    K = np.asarray(ch["ln_tau"]).shape[-1]
    out["zeta"] = np.full(np.asarray(ch["ln_tau"]).shape, float(value))
    return out


def per_elem_rhat(ch, name):
    """(nchain, nsample, ...) → 逐元素 R-hat / ESS（传统 gelman_rubin，留作对照）。"""
    if diag is None:
        raise ImportError("per_elem_rhat 需要 numpyro.diagnostics；"
                          "纯后处理脚本不要调它。")
    x = np.asarray(ch[name], float)
    shp = x.shape[2:]
    n = int(np.prod(shp)) if shp else 1
    xf = x.reshape(x.shape[0], x.shape[1], n)
    rh = np.array([float(diag.gelman_rubin(xf[:, :, i])) for i in range(n)])
    es = np.array([float(diag.effective_sample_size(xf[:, :, i])) for i in range(n)])
    return rh, es


def _rank3(a):
    """(nch, ns) 或 (nch, ns, n) → (rhat, bulk, tail)，逐元素、rank-normalized + split。

    为什么不用 numpyro 的 gelman_rubin / effective_sample_size：
      · gelman_rubin 是**未拆分**的经典 R-hat，对链内漂移不敏感，
        且对重尾/偏态后验会低估问题。Vehtari et al. (2021) 的
        rank-normalized split-R̂ 先对池化样本做秩变换（压掉重尾），
        再把每条链切成前后两半（暴露链内漂移），是现在的通行标准。
      · ESS 也要分 bulk（对均值/中位数敏感）与 tail（对 5%/95% 分位敏感）。
        本问题里"峰面积的可信区间"依赖尾部分位，只看总体 ESS 会漏。

    实现走 arviz（0.23 起 rhat 默认 method="rank" 且自带 split）。
    """
    import arviz as az

    a = np.asarray(a, float)
    if a.ndim == 2:
        a = a[:, :, None]
    n = a.shape[2]
    pack = {f"x{i}": a[:, :, i] for i in range(n)}
    rh = az.rhat(pack)
    eb = az.ess(pack, method="bulk")
    et = az.ess(pack, method="tail")
    f = lambda ds: np.array([float(np.asarray(ds[f"x{i}"].values)) for i in range(n)])
    return f(rh), f(eb), f(et)


def rank_diag(ch):
    """全部参数的 rank-normalized split R̂ / bulk ESS / tail ESS。"""
    out = {}
    for nm in PARAMS:
        if nm not in ch:
            continue
        r, b, t = _rank3(np.asarray(ch[nm], float))
        out[nm] = {"rhat": r, "bulk": b, "tail": t}
    return out


def full_param_diag(ch, rd=None):
    """全部参数的收敛诊断，两种口径并排。

    审阅意见（优先级 2 第 5 条）：分量面积是 A_k = W·π_k，所以 W 与 v
    都不是"辅助参数"，必须单独看；只报排序后的 A 和 logf0 不够。
    """
    rd = rank_diag(ch) if rd is None else rd
    rows = []
    for nm in PARAMS:
        if nm not in ch:
            continue
        gr, es = per_elem_rhat(ch, nm)       # 传统口径
        d = rd[nm]
        rows.append({
            "name": nm, "n": len(gr),
            "gr_max": float(np.max(gr)), "ess_med": float(np.median(es)),
            "rhat": float(np.nanmax(d["rhat"])),
            "bulk": float(np.nanmin(d["bulk"])),
            "tail": float(np.nanmin(d["tail"])),
        })
    return rows


def print_param_diag(rows, indent="  "):
    """R̂<1.01、bulk/tail ESS>400 是 Vehtari 建议的判据，这里一并标出。"""
    print(f"{indent}{'参数':<8}{'元素':>4}{'R̂(rank)':>10}{'bulk':>8}{'tail':>8}"
          f"{'GR(旧)':>9}{'ESS(旧)':>9}  判定")
    for r in rows:
        ok = (r["rhat"] < 1.01) and (r["bulk"] > 400) and (r["tail"] > 400)
        flag = "OK" if ok else ("R̂" if r["rhat"] >= 1.01 else "ESS")
        print(f"{indent}{r['name']:<8}{r['n']:>4}{r['rhat']:>10.3f}"
              f"{r['bulk']:>8.1f}{r['tail']:>8.1f}"
              f"{r['gr_max']:>9.3f}{r['ess_med']:>9.1f}  {flag}")


def tree_depth_stats(mc, max_depth):
    """树深度触顶率。需要 fit(..., extra_fields=("num_steps",))。

    NUTS 每次迭代的步数是 2^depth−1 量级；如果大量迭代停在 max_tree_depth，
    说明后验几何让轨迹一直"跑不完"，是**调参不足**的直接证据（与 R̂ 独立）。
    """
    ef = mc.get_extra_fields()
    if "num_steps" not in ef:
        return None
    n = np.asarray(ef["num_steps"], float).ravel()
    # ceil(log2(n))：n 恰为 2^d 时深度是 d，不是 d+1；否则 512 步(=2^9) 会被
    # 误判成触顶 10，frac_cap 系统性虚高（审查 P2-6）。
    td = np.ceil(np.log2(np.maximum(n, 1.0))).astype(int)
    return {"med": float(np.median(td)), "p95": float(np.percentile(td, 95)),
            "frac_cap": float(np.mean(td >= max_depth)), "max_depth": max_depth,
            "n": int(n.size)}



def acf_max(resid, lags=(1, 2, 3, 4, 5)):
    """残差自相关的最大绝对值（滞后 1–5）。白噪声时 |ρ|≲1.96/√n。

    这是**缺项探针**：rms 贴到噪声底只能说明"总误差小"，不能说明
    "没有结构"。模型表达不了的东西会以低频振荡的形式留在残差里，
    表现为相邻点同号——acf 抓的就是这个。
    """
    x = np.asarray(resid, float)
    x = x - x.mean()
    dd = float(np.dot(x, x))
    if dd <= 0:
        return 0.0
    return max(abs(float(np.dot(x[:-L], x[L:])) / dd) for L in lags if L < x.size)

def chain_pi(ch):
    """从 v 逐链逐样本重建 π，返回 (nch, ns, K)。K=1 时没有 v，π 恒为 1。"""
    if "v" not in ch:
        sh = np.asarray(ch["W"]).shape
        return np.ones(sh + (1,))
    v = np.asarray(ch["v"], float)
    nc, ns = v.shape[0], v.shape[1]
    return np.array([[B.stick_pi(v[c, i]) for i in range(ns)] for c in range(nc)])


def sorted_stats(ch):
    """按样本内 π 降序重排后，返回每列的 (rhat, bulk, tail) 以及重排后的 A / logf0。"""
    pich = chain_pi(ch)
    K = pich.shape[-1]
    o = np.argsort(-pich, axis=2)
    A = np.take_along_axis(np.asarray(ch["W"])[:, :, None] * pich, o, axis=2)
    lf = np.log10(1 / (2 * np.pi * np.take_along_axis(
        np.exp(np.asarray(ch["ln_tau"])), o, axis=2)))
    out = {}
    for nm, arr in (("A", A), ("logf0", lf)):
        r, b, t = _rank3(arr)
        out[nm] = [(float(r[j]), float(b[j]), float(t[j])) for j in range(K)]
    return out, A, lf, pich


def z_chain_diag(ch, f, n_samp: int = 200):
    """**不依赖分量编号**的收敛诊断：在固定频率上比较各链的 Z(f)。

    分量编号、排序、匹配都可能引入伪影；而最终关心的量是 Z(f) 本身。
    逐频率对实部/虚部算 R-hat 与 ESS，任何链间不一致都会直接暴露。

    返回 dict：{'max_rhat_re', 'max_rhat_im', 'min_bulk_re', 'min_tail_re', ...}
    """
    ch = with_zeta(ch)
    pich = chain_pi(ch)
    nc, ns = pich.shape[0], pich.shape[1]
    W = np.asarray(ch["W"], float)
    Ri = np.asarray(ch["R_inf"], float)
    tau = np.exp(np.asarray(ch["ln_tau"], float))
    phi = np.asarray(ch["phi"], float)
    zeta = np.asarray(ch["zeta"], float)
    # 扩散项：链级样本里是 Rw（Ω）与 Rwo/tauD，model_z 要的是 Aw 或 Rwo+tauD。
    # 不补进来，diffusion="Ws"/"Wo" 时 Z(f) 诊断衡量的是不含扩散的曲线，
    # 扩散参数慢混合会被"严格通过"掩盖（审查 P2-7）。
    has_rw = "Rw" in ch
    has_rwo = "Rwo" in ch and "tauD" in ch
    Rw = np.asarray(ch["Rw"], float) if has_rw else None
    Rwo = np.asarray(ch["Rwo"], float) if has_rwo else None
    tauD = np.asarray(ch["tauD"], float) if has_rwo else None
    # P2-6（二轮审查）：链里有 L 就按 include_L=True 重建，否则含电感模型
    # 的高频跨链诊断失真（高频感抗被硬塞进 HN 分量）。
    has_L = "L" in ch
    L = np.asarray(ch["L"], float) if has_L else None
    omega_ref = 2 * np.pi * np.sqrt(np.min(f) * np.max(f))
    idx = np.linspace(0, ns - 1, min(n_samp, ns)).astype(int)
    nf = len(f)
    Zr = np.empty((nc, len(idx), nf))
    Zi = np.empty((nc, len(idx), nf))
    for c in range(nc):
        for a, i in enumerate(idx):
            p = {"R_inf": Ri[c, i], "W": W[c, i], "pi": pich[c, i],
                 "tau": tau[c, i], "phi": phi[c, i], "zeta": zeta[c, i]}
            if has_rw:
                p["Aw"] = Rw[c, i] * np.sqrt(omega_ref)
            if has_rwo:
                p["Rwo"] = Rwo[c, i]
                p["tauD"] = tauD[c, i]
            if has_L:
                p["L"] = L[c, i]
            Zc = B.model_z(f, p, include_L=has_L)
            Zr[c, a] = Zc.real
            Zi[c, a] = Zc.imag
    rr, br, tr = _rank3(Zr)
    ri, bi, ti = _rank3(Zi)
    return {"max_rhat_re": float(rr.max()), "max_rhat_im": float(ri.max()),
            "min_bulk_re": float(br.min()), "min_tail_re": float(tr.min()),
            "min_bulk_im": float(bi.min()), "min_tail_im": float(ti.min()),
            "rhat_re": rr, "rhat_im": ri}


# --------------------------------------------------------------------------
# diag：K=3 的 3 链完整诊断
# --------------------------------------------------------------------------

def run_diag(K=3, alpha=0.5, nw=1500, ns=1500, nch=3, ta=0.9, depth=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, Z = B.noisy()
    print(f"数据: {f.size} 点，σ={B.SIGMA} Ω，真值 ZARC={B.ZARC}，R∞={B.R_INF}")
    print(f"设置: K={K} alpha={alpha} warmup={nw} samples={ns} "
          f"chains={nch} target_accept={ta} max_tree_depth={depth}")
    t0 = time.time()
    s, mc = B.fit(f, Z, K=K, alpha=alpha, sigma=(B.SIGMA, B.SIGMA),
                  n_warmup=nw, n_samples=ns, num_chains=nch, seed=0,
                  progress=False, target_accept=ta, max_tree_depth=depth,
                  extra_fields=("num_steps",))
    dt = time.time() - t0
    ch = mc.get_samples(group_by_chain=True)
    ndiv = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    ntot = nch * ns

    print(f"\n=== 采样 ===\n{nch} 链 × ({nw} warmup + {ns} samples)，用时 {dt:.0f}s")
    print(f"divergences: {ndiv} / {ntot}  ({ndiv/ntot*100:.2f}%)")
    tds = tree_depth_stats(mc, depth)
    if tds:
        print(f"树深度: 中位 {tds['med']:.0f}，p95 {tds['p95']:.0f}，"
              f"触顶(≥{depth})比例 {tds['frac_cap']*100:.1f}%  "
              f"（>0 说明轨迹被截断，需提高 max_tree_depth 或 target_accept）")

    print("\n=== 全部参数收敛诊断（rank-normalized split R̂ / bulk / tail ESS）===")
    rd = rank_diag(ch)
    print_param_diag(full_param_diag(ch, rd))
    print("  判据：R̂<1.01 且 bulk/tail ESS>400；GR/ESS(旧) 列是传统 gelman_rubin，仅作对照")

    st, A_s, lf_s, _ = sorted_stats(ch)
    print("\n=== 排序后（按样本内 π 降序重排）===")
    print(f"{'参数':<8}" + "".join(f"{f'c{j+1}':>17}" for j in range(K)))
    for nm in ("A", "logf0"):
        print(f"{nm:<8}" + "".join(f"{f'{a:.2f}/{b:.0f}/{t:.0f}':>17}"
                                   for a, b, t in st[nm]))
    print("  格式 R̂/bulk/tail")

    zd = z_chain_diag(ch, f)
    print("\n=== 总阻抗 Z(f) 的跨链诊断（不依赖分量编号）===")
    print(f"  实部  max R̂={zd['max_rhat_re']:.3f}  "
          f"min bulk={zd['min_bulk_re']:.0f}  min tail={zd['min_tail_re']:.0f}")
    print(f"  虚部  max R̂={zd['max_rhat_im']:.3f}  "
          f"min bulk={zd['min_bulk_im']:.0f}  min tail={zd['min_tail_im']:.0f}")
    print("  （这一项最贴近最终关心的量；它收敛才算真的收敛）")

    pi = s["pi"]
    order = np.argsort(-pi, axis=1)
    tau = np.take_along_axis(s["tau"], order, axis=1)
    A = np.take_along_axis(s["w"], order, axis=1)
    phi = np.take_along_axis(s["phi"], order, axis=1)
    zeta = np.take_along_axis(s["zeta"], order, axis=1)
    f0 = 1.0 / (2 * np.pi * tau)

    print("\n=== 分量（按样本内 π 降序，中位 [5%,95%]）===")
    for j in range(K):
        print(f"  comp{j+1}: A={np.median(A[:,j]):6.2f} "
              f"[{np.percentile(A[:,j],5):6.2f},{np.percentile(A[:,j],95):6.2f}] Ω  "
              f"f0={np.median(f0[:,j]):8.2f} Hz  "
              f"phi={np.median(phi[:,j]):.3f}  zeta={np.median(zeta[:,j]):.2f}")
    print(f"  真值: A=33.00/15.80 Ω，f0=357/20 Hz，phi=0.85/0.90，zeta=1")
    print(f"\nk(π>0.01) = {B.k_dist(s)}")
    ne = B.n_eff(s)
    print(f"N_eff = {np.median(ne):.2f} [{np.percentile(ne,5):.2f}, {np.percentile(ne,95):.2f}]")
    print("注意：k_dist 的 0.01 阈值是人为的；N_eff 下界恒为 1，上界才随 K 增大")

    # ------------------------------ 图 ------------------------------
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.28)
    cols = ["#c0392b", "#2471a3", "#27ae60", "#8e44ad", "#d68910"]

    for i, nm in enumerate(("R_inf", "W")):
        ax = fig.add_subplot(gs[0, i])
        vv = np.asarray(ch[nm])
        for c in range(nch):
            ax.plot(vv[c], lw=0.5, color=cols[c], alpha=0.8)
        ax.set_title(f"trace {nm}", fontsize=10)
        ax.set_xlabel("sample")
    ax = fig.add_subplot(gs[0, 2])
    for j in range(K):
        ax.plot(np.log10(f0[:, j]), lw=0.5, color=cols[j], alpha=0.7, label=f"c{j+1}")
    ax.set_title("trace log10(f0) [sorted by pi]", fontsize=10)
    ax.legend(fontsize=7, ncol=K)

    for j in range(K):
        ax = fig.add_subplot(gs[1, j])
        ax.hist(A[:, j], bins=60, color=cols[j], alpha=0.75, density=True)
        if j < 2:
            ax.axvline(33.0 if j == 0 else 15.8, color="k", ls="--", lw=1, label="truth")
            ax.legend(fontsize=7)
        ax.set_title(f"area A{j+1} (med {np.median(A[:,j]):.1f} Ω)", fontsize=10)
        ax.set_xlabel("Ω")

    for j in range(K):
        ax = fig.add_subplot(gs[2, j])
        ax.hist(np.log10(f0[:, j]), bins=60, color=cols[j], alpha=0.75, density=True)
        if j < 2:
            ax.axvline(np.log10(357.0 if j == 0 else 20.0), color="k", ls="--", lw=1)
        ax.set_title(f"peak f0 comp{j+1}", fontsize=10)
        ax.set_xlabel("log10 f0 / Hz")

    ax = fig.add_subplot(gs[3, 0])
    ax.scatter(A[:, 2] if K > 2 else A[:, 1],
               np.log10(f0[:, 2] if K > 2 else f0[:, 1]), s=2, c=cols[2], alpha=0.25)
    ax.set_xlabel("A_last / Ω"); ax.set_ylabel("log10 f0")
    ax.set_title("joint A vs f0 (last comp)", fontsize=10)

    ax = fig.add_subplot(gs[3, 1])
    ax.scatter(np.log10(f0[:, 0]), np.log10(f0[:, 1]), s=2, c="#555", alpha=0.25)
    ax.set_xlabel("log10 f0 comp1"); ax.set_ylabel("log10 f0 comp2")
    ax.set_title("comp1 vs comp2 peak (label swap?)", fontsize=10)

    ax = fig.add_subplot(gs[3, 2])
    for c in range(nch):
        ax.hist(np.asarray(ch["W"])[c], bins=40, histtype="step",
                color=cols[c], lw=1, label=f"chain{c+1}")
    ax.axvline(48.8, color="k", ls="--", lw=1)
    ax.set_title("W per chain", fontsize=10); ax.legend(fontsize=7)

    fig.suptitle(f"BM-DRT K={K} alpha={alpha} {nch} chains div={ndiv}/{ntot}", fontsize=12)
    out = "figures/bm_k3_diag.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n图已保存: {out}")


# --------------------------------------------------------------------------
# 统一的单次拟合报告（isolate / ablate 共用）
# --------------------------------------------------------------------------

def report_fit(tag, s, mc, dt, f, K, max_depth, nch, ns, save=None):
    ch = mc.get_samples(group_by_chain=True)
    nd = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    rd = rank_diag(ch)
    st, A, lf, pich = sorted_stats(ch)
    zd = z_chain_diag(ch, f)
    print(f"\n--- {tag}  ({dt:.0f}s) ---")
    print(f"  divergences {nd}/{nch*ns} ({nd/(nch*ns)*100:.2f}%)")
    tds = tree_depth_stats(mc, max_depth)
    if tds:
        print(f"  树深度 中位 {tds['med']:.0f} p95 {tds['p95']:.0f} "
              f"触顶 {tds['frac_cap']*100:.1f}%")
    print_param_diag(full_param_diag(ch, rd), indent="  ")
    for nm in ("A", "logf0"):
        print(f"  sorted {nm:6s}" +
              "".join(f"  c{j+1}:{a:.2f}/{b:.0f}/{t:.0f}"
                      for j, (a, b, t) in enumerate(st[nm])))
    print(f"  Z(f) 跨链: 实部 R̂={zd['max_rhat_re']:.3f} bulk={zd['min_bulk_re']:.0f} "
          f"tail={zd['min_tail_re']:.0f} | 虚部 R̂={zd['max_rhat_im']:.3f} "
          f"bulk={zd['min_bulk_im']:.0f} tail={zd['min_tail_im']:.0f}")
    print(f"  后验: A=" + "/".join(f"{np.median(A[:,:,j]):.1f}" for j in range(K)) +
          " Ω  f0=" + "/".join(f"{np.median(10**lf[:,:,j]):.0f}" for j in range(K)) + " Hz")
    print("  真值 A=33.0/15.8 Ω, f0=357/20 Hz, phi=0.85/0.90, zeta=1")
    if save:
        d = {k: np.asarray(v) for k, v in ch.items()}
        ef = mc.get_extra_fields()
        if "num_steps" in ef:
            d["num_steps"] = np.asarray(ef["num_steps"])
        np.savez(save, **d)
        print(f"  样本已存 {save}")


# --------------------------------------------------------------------------
# isolate：K=2 的隔离实验（只改一个因素）
# --------------------------------------------------------------------------

def run_isolate(K=2, nw=1500, ns=1500, nch=3):
    f, Z = B.noisy()
    base = dict(K=K, alpha=0.5, sigma=(0.2, 0.2), n_warmup=nw, n_samples=ns,
                num_chains=nch, seed=0, progress=False, extra_fields=("num_steps",))

    cfgs = [
        ("A  ζ free, ta=0.9, depth10", dict()),
        ("B  ζ=1 fixed, ta=0.9, depth10", dict(zeta_fixed=1.0)),
        ("C  ζ free, ta=0.995, depth13, warmup3000",
         dict(n_warmup=3000, target_accept=0.995, max_tree_depth=13)),
    ]
    for tag, extra in cfgs:
        kw = dict(base); kw.update(extra)
        md = kw.get("max_tree_depth", 10)
        t0 = time.time()
        s, mc = B.fit(f, Z, **kw)
        report_fit(tag, s, mc, time.time() - t0, f, K, md, nch, ns)


# --------------------------------------------------------------------------
# ablate：采样消融（K=3，只改一个因素 + 全开）
# --------------------------------------------------------------------------

def run_ablate(K=3, ns=1000, nch=3, base_warmup=1000, cfgs=None):
    """审阅意见（优先级 2 第 8 条）：把"更严采样"拆成三个旋钮单独看。

    之前只做了 K=2 的"全开"对照，无法判断 K=3 的失败该归因于哪个旋钮。
    这里固定 warmup/ns/chains/seed，只动 target_accept 与 max_tree_depth。
    """
    f, Z = B.noisy()
    base = dict(K=K, alpha=0.5, sigma=(B.SIGMA, B.SIGMA), n_warmup=base_warmup,
                n_samples=ns, num_chains=nch, seed=0, progress=False,
                extra_fields=("num_steps",))
    if cfgs is None:
        cfgs = [
            ("0  baseline ta=0.9  depth10 warmup1000", dict()),
            ("1  ta=0.995 only", dict(target_accept=0.995)),
            ("2  depth13 only", dict(max_tree_depth=13)),
            ("3  ta=0.995 + depth13", dict(target_accept=0.995, max_tree_depth=13)),
            ("4  ta=0.995 + depth13 + warmup3000",
             dict(target_accept=0.995, max_tree_depth=13, n_warmup=3000)),
        ]
    print(f"K={K} ns={ns} chains={nch} base_warmup={base_warmup}")
    print("判据：R̂<1.01 且 bulk/tail ESS>400（1500 有效样本才有意义）")
    for i, (tag, extra) in enumerate(cfgs):
        kw = dict(base); kw.update(extra)
        md = kw.get("max_tree_depth", 10)
        t0 = time.time()
        s, mc = B.fit(f, Z, **kw)
        report_fit(tag, s, mc, time.time() - t0, f, K, md, nch, ns,
                   save=f"bm_ablate_{i}.npz")
        ne = B.n_eff(s)
        print(f"  N_eff={np.median(ne):.2f}  k_dist={B.k_dist(s)}")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# redig：对已保存的样本重算诊断（不重跑采样）
# --------------------------------------------------------------------------

def run_redig(npz="bm_k3_samples.npz", f=None):
    """从 npz 里已存的 (chain, sample, ...) 样本重算全部诊断。

    bm_k3_samples.npz 是严格设置（ta=0.995, depth13, warmup3000, ns=1500）的
    3 链结果，所以可以零成本把"全部参数 + rank-normalized R̂/bulk/tail"补齐。
    树深度不在 npz 里（当时没存 num_steps），该项只能重跑。
    """
    d = np.load(npz)
    ch = {k: d[k] for k in d.files}
    K = np.asarray(ch["ln_tau"]).shape[-1]
    if f is None:
        f, _ = B.noisy()
    print(f"来源 {npz}：{np.asarray(ch['R_inf']).shape[0]} 链 × "
          f"{np.asarray(ch['R_inf']).shape[1]} 样本，K={K}")
    print("\n=== 全部参数收敛诊断（rank-normalized split R̂ / bulk / tail ESS）===")
    print_param_diag(full_param_diag(ch))
    st, A, lf, _ = sorted_stats(ch)
    print("\n=== 排序后（按样本内 π 降序重排）===")
    print(f"{'参数':<8}" + "".join(f"{f'c{j+1}':>17}" for j in range(K)))
    for nm in ("A", "logf0"):
        print(f"{nm:<8}" + "".join(f"{f'{a:.2f}/{b:.0f}/{t:.0f}':>17}"
                                   for a, b, t in st[nm]))
    print("  格式 R̂/bulk/tail")
    zd = z_chain_diag(ch, f)
    print("\n=== Z(f) 跨链 ===")
    print(f"  实部 max R̂={zd['max_rhat_re']:.3f} min bulk={zd['min_bulk_re']:.0f} "
          f"min tail={zd['min_tail_re']:.0f}")
    print(f"  虚部 max R̂={zd['max_rhat_im']:.3f} min bulk={zd['min_bulk_im']:.0f} "
          f"min tail={zd['min_tail_im']:.0f}")


# --------------------------------------------------------------------------
# scan：训练库 k 命中率
# --------------------------------------------------------------------------

def run_scan(path="data/train_data.csv", per_class=2, K=5, alpha=0.5, seed=0):
    import pandas as pd

    d = pd.read_csv(path)
    labels = [l for l in d["Circuit"].unique()
              if l.count("RCPE") >= 1 and set(l.split("-")) <= {"RCPE", "R", "L"}]
    labels = sorted(labels, key=lambda l: l.count("RCPE"))[:4]
    print(f"标签: {labels}\n每类 {per_class} 条，K={K}, alpha={alpha}\n")

    rng = np.random.default_rng(seed)
    conf = defaultdict(Counter)
    hit = tot = 0
    for lab in labels:
        sub = d[d["Circuit"] == lab]
        idx = rng.choice(len(sub), size=min(per_class, len(sub)), replace=False)
        for j in idx:
            r = sub.iloc[j]
            f = np.array(pyast.literal_eval(r["freq"]), float)
            Z = np.array(pyast.literal_eval(r["Z"]), complex)
            t0 = time.time()
            try:
                s, _ = B.fit(f, Z, K=K, alpha=alpha, sigma=None, include_L=False,
                             n_warmup=800, n_samples=700, progress=False)
            except Exception as e:
                print(f"  {lab} 失败: {type(e).__name__}: {e}")
                continue
            kd = B.k_dist(s)
            kmode = max(kd, key=kd.get)
            t = lab.count("RCPE")
            conf[t][kmode] += 1
            tot += 1
            hit += int(kmode == t)
            res = B.residual_stats(s, f, Z, include_L=False)
            print(f"  {lab:16s} 真值 {t}  k众数 {kmode} {kd}  "
                  f"N_eff={np.median(B.n_eff(s)):.2f}  "
                  f"rms={res['rms_med']:.4f} ({res['rel_med']*100:.2f}%)  "
                  f"{time.time()-t0:.0f}s")

    print(f"\n命中 {hit}/{tot}")
    ks = sorted({k for c in conf.values() for k in c} | set(conf))
    print("真\\BM " + "".join(f"{k:>6d}" for k in ks))
    for t in sorted(conf):
        print(f"{t:>6d} " + "".join(f"{conf[t][k]:>6d}" for k in ks))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if cmd == "diag":
        # python bm_experiments.py diag [warmup] [target_accept] [max_tree_depth]
        kw = {}
        if len(sys.argv) > 2:
            kw["nw"] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kw["ta"] = float(sys.argv[3])
        if len(sys.argv) > 4:
            kw["depth"] = int(sys.argv[4])
        run_diag(**kw)
    elif cmd == "redig":
        run_redig(sys.argv[2] if len(sys.argv) > 2 else "bm_k3_samples.npz")
    else:
        {"isolate": run_isolate, "scan": run_scan, "ablate": run_ablate}[cmd]()
