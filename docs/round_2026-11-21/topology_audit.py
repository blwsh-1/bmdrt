"""拓扑条件审计（只读正演）：每个族在"一步容差"尺度上到底有几个可辨识方向。

为什么算这个
------------
`B2_REPORT.md` 里 `adr` 的 `Rct`/`R_ad`/`tau_ad` 三个参数区间跨一个数量级
（1.9–7.7x），加预算也没好转。这提示问题不在采样、也不在参数值，而在**结构**：
单节点三支路并联时，三条支路的贡献可以在同一频段互相顶替。

与其逐个试参数，不如直接量结构本身：把 `Z(x)` 对每个参数求导，按"一步容差"
归一化，再按 sigma 归一化，得到设计矩阵 `A`。`A` 的奇异值就是"该方向上
一个容差步能产生几个 sigma 的信号"。

    A[:, k] = (∂Z/∂x_k) · δ_k / σ        x_k = log θ（log 参数）或 θ（有界参数）
    s_i     = A 的奇异值                 s_i >= 1 ⇒ 该方向在噪声之上可辨

`p` 个参数而只有 `r` 个 `s_i >= 1` ⇒ 有 `p−r` 个组合无论如何调都测不出。
这一步完全不需要 MCMC。

口径与边界
----------
1. **局部量**：在给定真值点求导，不是全局。真值换一个位置，条件数会变。
2. 与 `b2_baseline.py` 的规格 A 同一把尺子：`δ_k` 相同、`σ = pct·|Z|` 相同，
   所以 `s_i >= 1` 与 B2 的"可见"是同一判据。
3. 噪声按其频率形状进 `A`（`perpoint` 口径）。若换 `free`（常数 σ），
   高频段的 s 会整体塌掉——这本身就是 `NOISE_SCALE_PROBE.md` §9 的结论。
4. 参数按 `log` 坐标求导，所以 s 的含义是"**相对**变化一个容差步"。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/topology_audit.py
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import bm_top as T             # noqa: E402
import spectrum_metrics as SM  # noqa: E402
import b2_baseline as B2       # noqa: E402  （复用 FREQ / SPEC / forward / step_of）

OUT = os.path.join(REPO, "out", "topology_audit")
PCT = 1.0                      # 与 B2 正对照同一噪声档
S_MIN = 1.0                    # 可辨方向阈值（= 一个容差步产生 1σ 信号）
COS_HI = 0.90                  # 共线告警阈值


# ---------------------------------------------------------------------------
# 候选元素：顶层并联电容 Cp（只在研究侧实现，不进生产）
# ---------------------------------------------------------------------------
def add_shunt_cp(z, f, cp):
    w = 2 * np.pi * np.asarray(f, float)
    y = 1.0 / np.asarray(z, complex) + 1j * w * cp
    return 1.0 / y


CONFIGS = [
    dict(key="voigt_K1", family="voigt", note="串联 K=1 HN + L",
         truth=dict(R_inf=100.0, W=100.0, tau=1.0 / (2 * np.pi * 1e3),
                    phi=0.9, zeta=1.0, L=10e-6)),
    dict(key="voigt_K2", family="voigt", K=2, note="串联 K=2 HN + L",
         truth=dict(R_inf=100.0, W=130.0,
                    tau=(1.0 / (2 * np.pi * 1e3), 1.0 / (2 * np.pi * 3e4)),
                    phi=(0.9, 0.8), zeta=(1.0, 1.0), L=10e-6)),
    dict(key="voigt_K1_cp", family="voigt", extra="cp",
         note="voigt_K1 + 顶层并联 Cp（研究侧候选元素）",
         truth=dict(R_inf=100.0, W=100.0, tau=1.0 / (2 * np.pi * 1e3),
                    phi=0.9, zeta=1.0, L=10e-6, Cp=1.0e-10)),
    dict(key="nested_L1", family="nested", layers=1, note="单层 R1+CPE ∥ … + L",
         truth=dict(Rs=50.0, R1=100.0, Q1=2000.0, n1=0.9,
                    R_ad=0.5, tau_ad=1e-2, L=10e-6)),
    dict(key="nested_L2", family="nested", layers=2, note="双层 + L（B2 正对照）",
         truth=dict(B2.NESTED_TRUTH)),
    # 对照：把 R_ad 真值设成 100 欧（真有低频感抗回路，AZ31 那类），看那 2 个
    # 缺口是不是"因为真值里 R_ad≈0 才冗余"——否则不能把 nested 判成结构冗余。
    dict(key="nested_L2_ad", family="nested", layers=2,
         note="双层 + L + 有意义的 R_ad=100（对照）",
         truth=dict(B2.NESTED_TRUTH, R_ad=100.0, tau_ad=1e-2)),
    # 机制检验：把 CPE 指数从 0.85 拉到 0.6（远离 ω^-1）。若 cond 大幅下降，
    # 说明 adr 的退化来自"CPE 的分数幂律顶替 Z_ad 的 1/(jωτ) 色散"。
    dict(key="adr_n06", family="adr", note="同上但 n=0.6（机制对照）",
         truth=dict(Rs=50.0, Q=100.0, n=0.6, Rct=100.0,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
    # 若把吸附支路从"与 Rct 并联"改成"与 Rct 串联"（Randles A 型），
    # 结构就是 nested_L1 + R_ad。这条用来验证 adr 的修法是否真能解退化。
    dict(key="nested_L1_ad", family="nested", layers=1,
         note="Rs+[CPE∥(R1+Z_ad)]+L —— adr 的串联改法",
         truth=dict(Rs=50.0, R1=100.0, Q1=100.0, n1=0.85,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
    # 第三条路：保留 adr 的并联物理，只把 Rct 固定（外部直流测量给）。
    # p 从 7 降到 6 —— 若一维缺口正好来自 1/Rct+1/R_ad，这一步应当补齐。
    dict(key="adr_fixRct", family="adr", drop=("Rct",),
         note="adr 但固定 Rct（并联物理不变）",
         truth=dict(Rs=50.0, Q=100.0, n=0.85, Rct=100.0,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
    dict(key="adr", family="adr", note="单节点三支路 Rs+[CPE∥Rct∥Z_ad] + L",
         truth=dict(Rs=50.0, Q=100.0, n=0.85, Rct=100.0,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
]


def fwd(cfg, f, flat):
    """按配置正演。voigt 的 tau/phi/zeta 允许是标量（K=1）或元组（K≥2）。"""
    fam = cfg["family"]
    p = dict(flat)
    if fam == "voigt":
        K = int(cfg.get("K", 1))
        for k in ("tau", "phi", "zeta"):
            v = p[k]
            p[k] = np.array([v] if np.isscalar(v) else list(v), float)
        p["pi"] = np.ones(K) / K if K > 1 else np.array([1.0])
        z = B2.C.model_z(f, p, include_L=True)
        if cfg.get("extra") == "cp":            # 研究侧候选元素，不进生产
            z = add_shunt_cp(z, f, float(p["Cp"]))
        return z
    if fam == "nested":
        return T.z_nested(f, p, branch="parallel", layers=int(cfg["layers"]),
                          include_L=True)
    if fam == "adr":
        return T.z_adr(f, p, include_L=True)
    raise ValueError(fam)


def keys_of(cfg):
    """本配置真正被采样的参数（顺序与采样一致）。`drop` 模拟"固定某个参数"。"""
    return tuple(k for k in _keys_of(cfg) if k not in cfg.get("drop", ()))


def _keys_of(cfg):
    fam = cfg["family"]
    if fam == "voigt":
        K = int(cfg.get("K", 1))
        base = ["R_inf", "W"]
        base += ["tau"] if K == 1 else ["tau%d" % i for i in range(K)]
        base += ["phi"] if K == 1 else ["phi%d" % i for i in range(K)]
        base += ["zeta"] if K == 1 else ["zeta%d" % i for i in range(K)]
        out = base + ["L"]
        if cfg.get("extra") == "cp":
            out = out + ["Cp"]
        return tuple(out)
    if fam == "nested":
        return T.sampled_keys("parallel", int(cfg["layers"]), None, True)
    return T.adr_sampled_keys(None, True)


def get(flat, key):
    """支持 tau0/tau1 这类按分量展开的键。"""
    if key in flat:
        return float(flat[key]) if np.isscalar(flat[key]) else flat[key]
    for stem in ("tau", "phi", "zeta"):
        if key.startswith(stem) and key[len(stem):].isdigit():
            return float(flat[stem][int(key[len(stem):])])
    raise KeyError(key)


def set_(flat, key, val):
    q = dict(flat)
    for stem in ("tau", "phi", "zeta"):
        if key.startswith(stem) and key[len(stem):].isdigit():
            i = int(key[len(stem):])
            v = list(flat[stem]) if not np.isscalar(flat[stem]) else [flat[stem]]
            v[i] = val
            q[stem] = tuple(v)
            return q
    q[key] = val
    return q


def stem_of(key: str) -> str:
    """`tau0/tau1` 这类按分量展开的键折回 `tau`。

    不能直接 `rstrip("0123456789")`：`n1`/`n2` 本身就是真键，会被削成 `n`。
    """
    m = re.match(r"^(tau|phi|zeta)\d$", key)
    return m.group(1) if m else key


EXTRA_SPEC = {"Cp": "log"}


def coord_of(family, key):
    stem = stem_of(key)
    if stem in EXTRA_SPEC:
        return EXTRA_SPEC[stem]
    return B2.SPEC[family][stem]


def step_(family, key):
    stem = stem_of(key)
    if stem in EXTRA_SPEC:
        return 0.10
    return B2.step_of(family, stem)


def audit(cfg, pct=PCT):
    f = B2.FREQ.copy()
    flat = cfg["truth"]
    z0 = fwd(cfg, f, flat)
    sg = SM.sigma_perpoint(z0, pct)[0]
    keys = keys_of(cfg)
    cols, steps = [], []
    for k in keys:
        if stem_of(k) not in flat:
            continue
        x0 = get(flat, k)
        d = step_(cfg["family"], k)
        h = 1e-3 * d
        zp = fwd(cfg, f, set_(flat, k, x0 * (1 + h) if coord_of(cfg["family"], k) == "log"
                              else x0 + h))
        zm = fwd(cfg, f, set_(flat, k, x0 * (1 - h) if coord_of(cfg["family"], k) == "log"
                              else x0 - h))
        dz = (zp - zm) / (2 * h) * d          # 每个容差步引起的 ΔZ
        cols.append(np.concatenate([dz.real, dz.imag]))
        steps.append(d)
    A = np.array(cols).T / np.concatenate([sg, sg])[:, None]   # (2n, p)
    sv = np.linalg.svd(A, compute_uv=False)
    # 列归一化后的两两 |cos|（共线性）
    nrm = np.linalg.norm(A, axis=0)
    An = A / np.where(nrm > 0, nrm, 1.0)
    C = np.abs(An.T @ An)
    np.fill_diagonal(C, 0.0)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if C[i, j] >= COS_HI:
                pairs.append(dict(a=keys[i], b=keys[j], cos=float(C[i, j])))
    # 频带归属：每个参数灵敏度最大的频点，及 HF(>1e4)/LF(<10) 的能量占比
    f_peak, hf_share, lf_share = {}, {}, {}
    nf = f.size
    for i, k in enumerate(keys):
        # A 的行是 [Re(2n) ; Im(2n)]，按分量合成回 |dZ|（每容差步，单位 sigma）
        s = np.sqrt(A[:nf, i] ** 2 + A[nf:, i] ** 2)
        j = int(np.argmax(s))
        f_peak[k] = float(f[j])
        tot = float(np.sum(s ** 2)) or 1.0
        hf_share[k] = float(np.sum(s[f > 1e4] ** 2) / tot)
        lf_share[k] = float(np.sum(s[f < 10] ** 2) / tot)
    # VIF：列归一化后 (A'A)^{-1} 的对角元。1 = 与其余列正交；
    # 越大表示该参数被其余参数"顶替"得越厉害。用 pinv 承接奇异情形。
    AtA = An.T @ An
    vif = np.diag(np.linalg.pinv(AtA))
    # 最弱方向（最小奇异值的右奇异向量）上谁在动 —— 三向退化时两两 cos 抓不到，
    # 这一步才能点名"哪几个参数一起漂"。
    _, _, Vt = np.linalg.svd(A)
    vmin = Vt[-1]
    part = {k: float(abs(vmin[i])) for i, k in enumerate(keys)}
    group = sorted([k for k, v in part.items() if v >= 0.30],
                   key=lambda k: -part[k])
    r_eff = int(np.sum(sv >= S_MIN))
    return dict(key=cfg["key"], family=cfg["family"], note=cfg["note"],
                n_points=int(f.size), pct=pct,
                zmax=float(np.max(np.abs(z0))),
                params=list(keys), steps=steps,
                singular_values=[float(v) for v in sv],
                s_max=float(sv[0]), s_min=float(sv[-1]),
                cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else None,
                r_eff=r_eff, p=len(keys), deficit=len(keys) - r_eff,
                collinear=pairs,
                vif={k: float(vif[i]) for i, k in enumerate(keys)},
                vif_max=float(np.max(vif)),
                weak_direction=part, degenerate_group=group,
                f_peak=f_peak, hf_share=hf_share, lf_share=lf_share,
                A_norm=[[float(v) for v in row] for row in An],
                sigma_profile=dict(pct=pct,
                                   sigma_min=float(sg.min()),
                                   sigma_max=float(sg.max())))


def cp_detectability(pct_list=(0.1, 0.5, 1.0, 2.0)):
    """顶层并联 Cp 在什么噪声水平下才可辨。

    做法：取 voigt_K1 真值，加一个 `Cp` 使 1 MHz 处 `ωC` 与该处 `1/|Z|` 同量级
    （`Cp = 0.5·|Y(f_max)|/ω_max`），再算 `ΔZ/σ` 在 HF 段的峰值。
    """
    f = B2.FREQ.copy()
    flat = CONFIGS[0]["truth"]
    z0 = fwd(CONFIGS[0], f, flat)
    y_hf = abs(1.0 / z0[-1]) / (2 * np.pi * f[-1])
    rows = []
    for pct in pct_list:
        sg = SM.sigma_perpoint(z0, pct)[0]
        # snr 与 Cp 成正比（小扰动下线性），故一次采样即可外推阈值
        cp_ref = 0.01 * y_hf
        d = np.abs(add_shunt_cp(z0, f, cp_ref) - z0) / sg
        snr = float(np.max(d[f > 1e4]))
        rows.append(dict(pct=pct, snr_at_1pct_frac=snr,
                         frac_min=float(0.01 / snr) if snr > 0 else None,
                         cp_min_F=float(0.01 / snr * y_hf) if snr > 0 else None,
                         f_at=float(f[int(np.argmax(np.where(f > 1e4, d, 0)))])))
    return dict(y_hf_per_omega=float(y_hf), rows=rows)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(res["configs"])
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig1, axs = plt.subplots(nrow, ncol, figsize=(13.2, 3.0 * nrow),
                             squeeze=False)
    for ax, r in zip(axs.ravel(), res["configs"]):
        f = B2.FREQ
        A = np.array(r["A_norm"]).T        # (p, 2n)
        nf = f.size
        mag = np.sqrt(A[:, :nf] ** 2 + A[:, nf:] ** 2)
        for i, k in enumerate(r["params"]):
            ax.semilogx(f, mag[i], lw=1.4, label=k)
        ax.axhline(1.0, color="#D85A30", ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e1)          # 统一量程，否则 1e-8 的曲线把其余压平
        ax.set_title("%s  p=%d  r_eff=%d  cond=%.3g"
                     % (r["key"], r["p"], r["r_eff"], r["cond"] or -1),
                     fontsize=9)
        ax.set_ylabel("|dZ per tolerance step| / sigma", fontsize=7)
        ax.legend(fontsize=6, ncol=2, frameon=False, loc="lower center")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel("f / Hz", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT, "fig_audit_sensitivity.png"), dpi=150)
    plt.close(fig1)

    # 奇异值谱：退化是"组合方向"上的，单参数灵敏度图里看不出来。
    fig2, ax = plt.subplots(figsize=(7.6, 4.4))
    for r in res["configs"]:
        sv = r["singular_values"]
        ax.plot(np.arange(1, len(sv) + 1), sv, "o-", ms=4, lw=1.3,
                label="%s (r_eff %d/%d, cond %.3g)"
                      % (r["key"], r["r_eff"], r["p"], r["cond"] or -1))
    ax.axhline(S_MIN, color="#D85A30", ls="--", lw=1.2)
    ax.text(0.4, S_MIN * 1.35, "detectability threshold (1 sigma per tolerance step)",
            color="#D85A30", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index i")
    ax.set_ylabel("s_i  (sigma per one tolerance step)")
    ax.set_title("Topology conditioning: how many independent directions the data can see",
                 fontsize=10)
    ax.legend(fontsize=6.6, frameon=False, loc="lower left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "fig_audit_spectrum.png"), dpi=150)
    plt.close(fig2)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = dict(pct=PCT, s_min=S_MIN, cos_hi=COS_HI,
               configs=[audit(c) for c in CONFIGS],
               cp=cp_detectability())
    print("%-13s %2s %5s %10s %7s %9s  %s"
          % ("config", "p", "r_eff", "cond", "s_min", "max VIF", "退化组(最弱方向)"))
    for r in res["configs"]:
        print("%-13s %2d %5d %10.1f %7.2f %9.1f  %s"
              % (r["key"], r["p"], r["r_eff"], r["cond"] or -1, r["s_min"],
                 r["vif_max"], ", ".join(r["degenerate_group"]) or "-"))
        v = sorted(r["vif"].items(), key=lambda x: -x[1])[:3]
        print("              VIF top3: %s"
              % ", ".join("%s=%.1f" % (k, x) for k, x in v))
    print("\nCp 单独可辨阈值（voigt_K1 真值，1MHz 处 |Y|/ω = %.3g F）:"
          % res["cp"]["y_hf_per_omega"])
    print("  %-6s %-14s %-12s %s"
          % ("pct", "snr@Cp=1%%·Yhf", "frac_min", "Cp_min / F"))
    for x in res["cp"]["rows"]:
        print("  %-6.2f %-14.3f %-12.4g %s"
              % (x["pct"], x["snr_at_1pct_frac"], x["frac_min"], x["cp_min_F"]))
    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, allow_nan=False)
    fig(res)
    print("\n->", os.path.join(OUT, "audit.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
