# -*- coding: utf-8 -*-
"""bm_corr —— 全参数后验相关矩阵 + 置信区间表（"别只看误差"的那一层证据）。

为什么需要它
------------
手动拟合实践者的核心提醒是：**参数相关性、置信区间、残差图要一起看**。
边际 90% 区间会漏掉简并——两个参数一起动、拟合一样好，各自区间都很窄，
实际谁都没定住。简并只存在于联合分布里，所以必须算相关矩阵。

两个必须守住的口径
------------------
1. **先对齐分量编号再算相关。** 混合模型的编号在样本间自由交换
   （见 `bm_core.match_components` 的说明）。直接按列算，会把"样本 1 的
   第 1 个峰"和"样本 2 的第 1 个峰"（其实是另一个峰）配成一对，
   得到的是**假相关**——看起来很强，实际是标签跳变。
2. **尺度参数取对数。** `R_inf / W / w / τ / σ / L / R_W / τ_D` 都是正量，
   后验近对数正态。在对数尺度上算 Pearson，等价于在"数量级"上算相关，
   不会被少数大值主导。`φ / ζ` 是形状参数，留在原尺度。

常数参数（例如 `--zeta 1.0` 时 ζ 被钉死）**不参与矩阵**，单列进 `fixed`。
不能让它以 `std=0 → corr=NaN` 的形式混在矩阵里——那会被读成"没有相关"，
而真相是"这个参数根本没在动"。

用法
----
    from bm_corr import param_corr, format_report
    pc = param_corr(s, include_L=True)
    print(format_report(pc))

    # 从已有结果重算（不重跑拟合）：
    python bm_corr.py out/x/real_x_samples.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

#: 强相关 / 中等相关的默认阈值。0.9 以上基本就是"只能定出一个组合"。
STRONG = 0.90
MEDIUM = 0.70


# --------------------------------------------------------------------------
# 取列
# --------------------------------------------------------------------------

def _columns(s: dict, include_L, mm=None):
    """把后验样本摊成 (names, X, is_log, dropped)。

    `mm` 是 `bm_core.match_components()` 的输出；传了就按对齐后的编号取列，
    没传就退回原始编号（**只适用于 K=1 或确认没有标签跳变的情况**）。
    """
    n = int(np.asarray(s["R_inf"]).ravel().size)
    cols: list[tuple[str, np.ndarray, bool]] = []

    def add(name: str, arr, log: bool = True):
        if arr is None:
            return
        a = np.asarray(arr, float).ravel()
        if a.size != n or not np.all(np.isfinite(a)):
            return
        if log:
            if np.any(a <= 0):
                return
            cols.append((f"ln {name}", np.log(a), True))
        else:
            cols.append((name, a, False))

    add("R_inf", s.get("R_inf"))
    # **W 不进矩阵**：W = Σ_k w_k，与分量幅值完全冗余。同时放进去会得到
    # 恒等于 +1.00 的伪相关（K=1 时 w_1 ≡ W，逐样本完全相等），
    # 那是参数化的必然结果，不是物理简并。总量与 R_inf 的补偿关系
    # 在 param_corr() 里用派生量单独算（见 _degenerate_wtot）。
    add("s_re", s.get("s_re"))
    add("s_im", s.get("s_im"))
    if include_L:
        add("L", s.get("L"))
    add("Rw", s.get("Rw"))
    add("Rwo", s.get("Rwo"))
    add("tauD", s.get("tauD"))
    # 吸附 RL（M4）：Rad 与 R_inf 的补偿方向（Z_ad = R − R/(1+jωτ) 的常数
    # 部分被 R_inf 吸收）是已知简并，必须进矩阵才看得见（M3 标定实测拐点
    # 居中时 corr(Rad,R_inf) 强负、R_ad 报出 -90% 的假值）。
    add("R_ad", s.get("Rad"))
    add("tau_ad", s.get("tau_ad"))

    # ---- 分量参数 ----
    if mm is not None:
        tau, w, phi, zeta = mm["tau"], mm["w"], mm["phi"], mm["zeta"]
        f0 = mm.get("f0")
        if f0 is None:
            f0 = 1.0 / (2 * np.pi * np.asarray(tau, float))
    else:
        tau = s.get("tau")
        w = s.get("w")
        phi = s.get("phi")
        zeta = s.get("zeta")
        f0 = (1.0 / (2 * np.pi * np.asarray(tau, float))
              if tau is not None else None)
    if tau is not None and w is not None and f0 is not None:
        # 与 report() 的分量编号一致：按峰位从高到低（高频在前）。
        order = np.argsort(-np.median(np.asarray(f0, float), axis=0))
        for j, idx in enumerate(order, 1):
            add(f"τ_{j}", np.asarray(tau, float)[:, idx])
            add(f"w_{j}", np.asarray(w, float)[:, idx])
            if phi is not None:
                add(f"φ_{j}", np.asarray(phi, float)[:, idx], log=False)
            if zeta is not None:
                add(f"ζ_{j}", np.asarray(zeta, float)[:, idx], log=False)

    if not cols:
        return [], np.zeros((0, 0)), [], n

    names = [c[0] for c in cols]
    is_log = [c[2] for c in cols]
    X = np.column_stack([c[1] for c in cols])
    ok = np.all(np.isfinite(X), axis=1)
    return names, X[ok], is_log, int((~ok).sum())


# --------------------------------------------------------------------------
# 主函数
# --------------------------------------------------------------------------

def param_corr(s: dict, include_L=None, mm=None, strong: float = STRONG,
               medium: float = MEDIUM) -> dict:
    """算全参数相关矩阵 + 每个参数的 90% 区间。

    返回（全部 JSON 可序列化，矩阵四舍五入到 3 位）：
        names      参与矩阵的参数名（已剔除常数参数）
        corr       对称相关矩阵，list[list]，对角 = 1
        summary    [{name, q05, q50, q95, sd, log}]，**原尺度**的分位数
        fixed      [{name, value}]：没参与矩阵的常数参数
        strong     [{a, b, rho, why}]，|ρ| ≥ strong
        medium     [{a, b, rho, why}]，medium ≤ |ρ| < strong
        n_draw / n_dropped / strong_tol / medium_tol / note
    """
    if include_L is None:
        include_L = "L" in s

    names, X, is_log, n_drop = _columns(s, include_L, mm=mm)
    out = dict(names=[], corr=[], summary=[], fixed=[], strong=[], medium=[],
               n_draw=int(X.shape[0]), n_dropped=n_drop,
               strong_tol=float(strong), medium_tol=float(medium), note="")

    if not names or X.shape[0] < 10:
        out["note"] = ("样本不足或没有可用参数（n_draw="
                       f"{X.shape[0]}），不计算相关矩阵")
        return out

    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    # 常数参数的判据：标准差相对量级可忽略。用 1e-12 而不是 0，
    # 因为 ζ 被钉死时是精确常数，但浮点求和后 std 可能是 1e-17。
    varying = sd > 1e-12 * np.maximum(1.0, np.abs(mu))
    fixed_idx = [i for i in range(len(names)) if not varying[i]]
    keep = [i for i in range(len(names)) if varying[i]]

    # 原尺度的分位数：对数尺度取的 ln 值要 exp 回去才是参数本身。
    q = np.percentile(X, [5, 50, 95], axis=0)
    for i in keep:
        if is_log[i]:
            vals = np.exp(q[:, i])
            # 对数尺度算出的 sd 不能直接放进原尺度表：直接用 exp(X) 的样本
            # 标准差，才是参数本身的离散度（审查 P2-4）。
            sd_orig = float(np.std(np.exp(X[:, i])))
        else:
            vals = q[:, i]
            sd_orig = float(sd[i])
        out["summary"].append(dict(
            name=names[i].replace("ln ", "", 1), q05=float(vals[0]),
            q50=float(vals[1]), q95=float(vals[2]),
            sd=sd_orig, log=bool(is_log[i])))
    for i in fixed_idx:
        v = X[0, i]
        out["fixed"].append(dict(
            name=names[i].replace("ln ", "", 1),
            value=float(np.exp(v) if is_log[i] else v)))

    if len(keep) < 2:
        out["note"] = "只有一个参数在动，没有可算的相关性"
        return out

    Xk = X[:, keep]
    kn = [names[i] for i in keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xk, rowvar=False)
    C = np.atleast_2d(C)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)

    out["names"] = kn
    out["corr"] = [[round(float(C[i, j]), 3) for j in range(len(kn))]
                   for i in range(len(kn))]

    pairs = []
    for i in range(len(kn)):
        for j in range(i + 1, len(kn)):
            rho = float(C[i, j])
            if abs(rho) < medium:
                continue
            pairs.append(dict(a=kn[i], b=kn[j], rho=rho,
                              why=_why(kn[i], kn[j], rho)))
    pairs.sort(key=lambda p: -abs(p["rho"]))
    out["strong"] = [p for p in pairs if abs(p["rho"]) >= strong]
    out["medium"] = [p for p in pairs if abs(p["rho"]) < strong]

    # 已知简并方向的定向检查：脊线 / R_inf-W / 噪声同源 / 扩散简并。
    deg = _degenerate(kn, C, strong)
    wt = _degenerate_wtot(s, strong)
    if wt:
        deg.append(wt)
    out["degenerate"] = deg
    if out["strong"]:
        out["note"] = (f"{len(out['strong'])} 对参数 |ρ|≥{strong:.2f}："
                       "这些参数只能定出一个组合，**不要单独引用其中任何一个**。")
    else:
        out["note"] = f"没有 |ρ|≥{strong:.2f} 的参数对，边际区间可以单独引用。"
    return out


def _why(a: str, b: str, rho: float) -> str:
    """给一对参数配一句"为什么相关"，只覆盖已知机制，其余留空。"""
    import re

    def kind(x):
        m = re.match(r"^(ln )?(τ|w|φ|ζ)_(\d+)$", x)
        if m:
            return (m.group(2), int(m.group(3)))
        # 非分量参数也要剥掉 "ln " 前缀，否则 "ln s_re"/"ln Rw"/"ln L"
        # 永远对不上下面的已知简并分支，解释退化成泛泛"正相关"（审查 P2-5）。
        return (x[3:] if x.startswith("ln ") else x, 0)

    ka, ia = kind(a)
    kb, ib = kind(b)
    sgn = "正" if rho > 0 else "负"

    if {ka, kb} == {"w", "τ"} and ia == ib:
        return ("R=A√τ 型脊线：幅值与位置只能定出一个组合，"
                "单独的 R 或 τ 都没有意义（HN↔Warburg 退化指纹）")
    if {ka, kb} == {"τ", "φ"} and ia == ib:
        return ("φ 与 τ 联合简并：峰宽和峰位一起动才能保持谱形"
                "（φζ 乘积受约束时，φ 变大 τ 必须跟着移）")
    if {ka, kb} == {"w", "φ"} and ia == ib:
        return ("幅值与峰宽互相补偿：峰加宽的同时压低，谱形近似不变")
    if {ka, kb} == {"R_inf", "W"}:
        return ("总串联电阻与极化幅值互相补偿：低频 |Z| 只约束二者之和，"
                "R_inf 偏大时 W 必然偏小")
    if {ka, kb} == {"s_re", "s_im"}:
        return ("两路噪声尺度同源（都随 |Z| 定标），一个偏大另一个跟着偏大")
    if {ka, kb} == {"Rw", "tauD"} or {ka, kb} == {"Rwo", "tauD"}:
        return ("扩散幅值与扩散时间常数在带内简并："
                "A_W=R_W/√τ_D 只定一个组合，两个参数各自不可辨识")
    if ka == "L" or kb == "L":
        return ("串联电感与高频段的 R_inf/容抗竞争："
                "只有高频点足够多、感抗尾明显时才能分开")
    if ka == "R_inf" and kb in ("w", "τ"):
        return ("R_inf 与低频分量的幅值互相补偿（低频端没走到直流极限时的典型简并）")
    if ka in ("w", "τ") and kb == "R_inf":
        return ("R_inf 与低频分量的幅值互相补偿（低频端没走到直流极限时的典型简并）")
    if ka in ("τ", "w") and kb in ("τ", "w") and ia != ib:
        return (f"{sgn}相关：两个分量的位置/幅值互相替换——"
                "典型的两时间常数贴近、DRT 也会并峰的情形")
    return f"{sgn}相关"


def _degenerate(kn, C, strong: float) -> list[dict]:
    """已知简并方向的定向检查（比"哪对最强"更有物理含义）。"""
    out = []
    idx = {n: i for i, n in enumerate(kn)}
    import re
    comp = {}
    for n in kn:
        m = re.match(r"^(ln )?(τ|w|φ|ζ)_(\d+)$", n)
        if m:
            comp.setdefault(int(m.group(3)), {})[m.group(2)] = n
    for k, d in sorted(comp.items()):
        if "w" in d and "τ" in d:
            r = float(C[idx[d["w"]], idx[d["τ"]]])
            if abs(r) >= strong:
                out.append(dict(kind="R-tau ridge", comp=k, rho=r,
                                text=(f"分量 #{k} 的 w 与 τ 相关 {r:+.2f}："
                                      "沿 R=A√τ 脊线简并，"
                                      "只能引用组合量，不能分别引用 R 和 τ")))
    return out


def _degenerate_wtot(s: dict, strong: float) -> dict | None:
    """R_inf 与**总极化幅值** W 的补偿关系（W 不进矩阵，这里单独算）。

    W = Σ_k w_k，所以它和分量幅值冗余，不能一起放进相关矩阵；但
    「R_inf 与总极化电阻互相补偿」是真实的物理简并：低频 |Z| 只约束
    二者之和，R_inf 偏大时 W 必然偏小。这条要单独报，否则会被
    "把 W 从矩阵里删掉" 一起删掉。
    """
    try:
        W = np.asarray(s["W"], float).ravel()
        R = np.asarray(s["R_inf"], float).ravel()
    except Exception:
        return None
    ok = np.isfinite(W) & np.isfinite(R) & (W > 0) & (R > 0)
    if ok.sum() < 10:
        return None
    lw, lr = np.log(W[ok]), np.log(R[ok])
    if lw.std() < 1e-12 or lr.std() < 1e-12:
        return None
    r = float(np.corrcoef(lr, lw)[0, 1])
    if not np.isfinite(r) or abs(r) < strong:
        return None
    return dict(kind="Rinf-Wtot", comp=0, rho=r,
                text=(f"R_inf 与总极化幅值 W=Σw 相关 {r:+.2f}："
                      "低频 |Z| 只约束二者之和。报极化电阻要用 Σw 或 R_inf+Σw，"
                      "不要单独报 R_inf；W 与分量幅值冗余，故不进相关矩阵"))


def compact(pc: dict) -> dict:
    """去掉矩阵本体，只留结论（写进逐 K 的中间结果时用）。"""
    if not pc:
        return {}
    return {k: v for k, v in pc.items() if k != "corr"}


def format_report(pc: dict, top: int = 10) -> str:
    """把 param_corr 结果打成几行可读文本。"""
    if not pc:
        return "  （没有参数相关矩阵）"
    lines = [f"  样本 {pc.get('n_draw', 0)} 个"
             + (f"（剔除 {pc['n_dropped']} 个含非有限值的样本）"
                if pc.get("n_dropped") else "")]
    if pc.get("fixed"):
        fx = "、".join(f"{f['name']}={f['value']:.3g}" for f in pc["fixed"])
        lines.append(f"  已钉死（不参与相关）：{fx}")
    st, md = pc.get("strong") or [], pc.get("medium") or []
    if st:
        lines.append(f"  **|ρ|≥{pc['strong_tol']:.2f} 的成对参数（只能定出组合）**")
        for p in st[:top]:
            lines.append(f"    {p['rho']:+.2f}  {p['a']} ↔ {p['b']}　{p['why']}")
    if md:
        lines.append(f"  |ρ| 在 {pc['medium_tol']:.2f}–{pc['strong_tol']:.2f} 的")
        for p in md[:top]:
            lines.append(f"    {p['rho']:+.2f}  {p['a']} ↔ {p['b']}")
    for d in (pc.get("degenerate") or []):
        lines.append(f"  · {d['text']}")
    if not st and not md:
        lines.append("  没有 |ρ|≥0.70 的参数对。")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI：从已有结果重算（不重跑拟合）
# --------------------------------------------------------------------------

def _from_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 0
    p = Path(argv[0])
    if not p.exists():
        print(f"找不到 {p}")
        return 1
    s = _from_npz(p)
    import bm_core as B
    mm = B.match_components(s) if "tau" in s else None
    pc = param_corr(s, include_L=("L" in s), mm=mm)
    print(f"=== {p.name} ===")
    print(format_report(pc))
    if "--matrix" in argv and pc.get("names"):
        names = pc["names"]
        w = max(len(n) for n in names) + 1
        print("\n  相关矩阵（下三角）")
        print(" " * (w + 1) + "".join(f"{i + 1:>7}" for i in range(len(names))))
        for i, n in enumerate(names):
            row = "".join(f"{pc['corr'][i][j]:>7.2f}" for j in range(i + 1))
            print(f"  {n:<{w}}{row}")
        for i, n in enumerate(names, 1):
            print(f"  {i:>2} = {n}")
    if "--json" in argv:
        print(json.dumps(pc, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
