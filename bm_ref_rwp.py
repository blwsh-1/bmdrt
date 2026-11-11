# -*- coding: utf-8 -*-
"""bm_ref_rwp.py —— **参考性 Rwp**（独立 LS 重拟合），只作跨工具对照。

硬边界（写在这里，也写进 JSON 的字段说明）：

1. **不参与**任何门禁、K 选择、参数推断与验收判定。生产的统计语义不变：
   生产链是贝叶斯参数推断（σ 进采样），它的目标函数**不是**最小化相对残差。
2. `rwp_ls_relative` 必须来自**独立的最小二乘重拟合**，不能拿 MCMC 后验参数
   代进去算一遍就冒称"LS 参考结果"——那是两件事，混起来会误导。
3. 只支持生产默认拓扑：`Rs + jωL + Σ HN(ζ=1)`。生产若开了 ζ 自由 / 扩散 /
   吸附，参考拓扑就变了，这里直接 `skipped` 并写明原因，不偷偷近似。

Rwp 的定义（跨工具比对必须用同一把尺子，故写死并原样落进 JSON）：

    sigma_i = pct * |Z_obs(f_i)|                     # 默认 pct = 0.02
    dZ_i    = Z_fit(f_i) - Z_obs(f_i)
    LS 目标 = sum_i ( |dZ_i|^2 / sigma_i^2 )
            = (1/pct^2) * sum_i ( |dZ_i|^2 / |Z_obs_i|^2 )
    Rwp[%]  = sqrt( mean_i ( |dZ_i|^2 / |Z_obs_i|^2 ) ) * 100

由上式可见**LS 目标与 Rwp 单调等价**：最小化 LS 就是最小化 Rwp。

三条容易踩的口径细节：
  · 每频点**等权**（不做 log 变换，不按点数归一）；
  · 实部与虚部**联合**进同一个 |dZ|^2，同一个 sigma_i —— 不是各算一个 Rwp 再平均；
  · 分母是 |Z_obs|（观测），不是 |Z_fit|；**不除 √2**。
    项目内部的 `rms_rel` 是 sqrt(mean over 2N of rel^2)，恰好等于 Rwp/√2，
    两者不能互比（R16/R17 的同一条教训）。
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["rwp", "z_from_params", "refit", "reference_block",
           "WEIGHT_DEFINITION", "selftest"]

# 落到 JSON 里的口径字符串：**改了公式必须同时改这句**，否则字段在说谎。
WEIGHT_DEFINITION = (
    "Rwp[%] = sqrt(mean_i(|Z_fit-Z_obs|^2/|Z_obs|^2))*100；"
    "LS 权重 sigma_i = 0.02*|Z_obs_i|（相对权重，逐点）；"
    "每频点等权，实部与虚部联合计入同一个 |dZ|^2；"
    "分母为 |Z_obs|，不除 sqrt(2)（与项目内部 rms_rel 相差 sqrt(2) 倍，不可互比）"
)

NOTE = ("参考指标：独立 LS + 逐点相对权重重新优化后的最优 Rwp，"
        "只用于与外部工具/别家软件对比；不参与门禁、K 选择、参数推断与验收判定。")

# 缺省参数：起点数、上限、每点百分比
STARTS = 8
MAX_NFEV = 4000
PCT = 2.0


# ---------------------------------------------------------------------------
# 阻抗与 Rwp
# ---------------------------------------------------------------------------

def z_from_params(f, *, R_inf, w, tau, phi, L=0.0, include_L=False):
    """`Rs + jωL + Σ w_i/[1+(jωτ_i)^φ_i]`，ζ 固定 1（与生产默认一致）。

    这里**独立实现**，不调 `bm_core.model_z`：口径要能被外部脚本照抄，
    同时 `selftest()` 负责和生产解析式对拍，防止两份实现悄悄漂移。
    """
    f = np.asarray(f, float)
    z = np.full(f.shape, float(R_inf), complex)
    if include_L and L:
        z = z + 1j * 2 * np.pi * f * float(L)
    w = np.atleast_1d(np.asarray(w, float))
    tau = np.atleast_1d(np.asarray(tau, float))
    phi = np.atleast_1d(np.asarray(phi, float))
    if w.size:
        x = np.maximum(2 * np.pi * f[:, None] * tau[None, :], 1e-300)
        mag = x ** phi[None, :]
        ang = phi[None, :] * (np.pi / 2)
        z = z + (w[None, :] / (1.0 + mag * (np.cos(ang) + 1j * np.sin(ang)))).sum(axis=1)
    return z


def rwp(f, Z, Zfit):
    """Rwp[%]，定义见模块 docstring（与 WEIGHT_DEFINITION 逐字对应）。"""
    Z = np.asarray(Z, complex)
    Zfit = np.asarray(Zfit, complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel2 = np.abs(Zfit - Z) ** 2 / np.abs(Z) ** 2
    if not np.all(np.isfinite(rel2)):
        return float("nan")
    return float(np.sqrt(np.mean(rel2)) * 100.0)


# ---------------------------------------------------------------------------
# 独立 LS 重拟合
# ---------------------------------------------------------------------------

def _pack(w, tau, phi, R_inf, L, include_L):
    """全部取 log10（φ 除外）→ 无约束优化下天然保正，边界只管量级。"""
    x = [np.log10(w), np.log10(tau), np.asarray(phi, float), [np.log10(R_inf)]]
    if include_L:
        x.append([np.log10(L)])
    return np.concatenate([np.atleast_1d(a) for a in x]).astype(float)


def _unpack(x, K, include_L):
    w = 10.0 ** np.asarray(x[:K], float)
    tau = 10.0 ** np.asarray(x[K:2 * K], float)
    phi = np.asarray(x[2 * K:3 * K], float)
    R_inf = float(10.0 ** x[3 * K])
    L = float(10.0 ** x[3 * K + 1]) if include_L else 0.0
    return dict(w=w, tau=tau, phi=phi, R_inf=R_inf, L=L)


def _bounds(K, f, zmax, include_L):
    tau_lo, tau_hi = 1.0 / (2 * np.pi * f.max()) / 1e3, 1.0 / (2 * np.pi * f.min()) * 1e3
    lo = np.concatenate([np.full(K, np.log10(1e-8 * zmax)),
                         np.full(K, np.log10(tau_lo)),
                         np.full(K, 0.50),
                         [np.log10(1e-8 * zmax)] + ([-12.0] if include_L else [])])
    hi = np.concatenate([np.full(K, np.log10(10.0 * zmax)),
                         np.full(K, np.log10(tau_hi)),
                         np.full(K, 1.00),
                         [np.log10(10.0 * zmax)] + ([-3.0] if include_L else [])])
    return lo, hi


def _starts(x_ref, K, f, zmax, include_L, n):
    """确定性多起点：参考解打头（保证 LS 不劣于生产解），其余铺满窗口。"""
    rng = np.random.default_rng(0)
    xs = [x_ref]
    tau_ref = x_ref[K:2 * K]
    # ① 参考解整体平移 τ（上一个/下一个十倍频）
    for shift in (-1.0, 1.0):
        v = x_ref.copy()
        v[K:2 * K] = np.clip(tau_ref + shift, -9, 4)
        xs.append(v)
    # ② 各分量在窗口内等间隔铺开
    tau_lo = np.log10(1.0 / (2 * np.pi * f.max()))
    tau_hi = np.log10(1.0 / (2 * np.pi * f.min()))
    grid = np.linspace(tau_lo, tau_hi, K)
    for frac in (0.3, 1.0):
        v = x_ref.copy()
        v[:K] = np.log10(frac * zmax)
        v[K:2 * K] = grid
        v[2 * K:3 * K] = 0.85
        xs.append(v)
    # ③ 参考解按 3 倍上下扰动 w / τ
    for fac in (1.0 / 3.0, 3.0):
        v = x_ref.copy()
        v[:K] = v[:K] + np.log10(fac)
        v[K:2 * K] = v[K:2 * K] + np.log10(fac)
        xs.append(v)
    # ④ 随机补足（固定 seed → 可复现）
    while len(xs) < n:
        v = x_ref.copy()
        v[:K] = rng.uniform(np.log10(0.01 * zmax), np.log10(zmax), K)
        v[K:2 * K] = rng.uniform(np.log10(1.0 / (2 * np.pi * f.max())),
                                 np.log10(1.0 / (2 * np.pi * f.min())), K)
        v[2 * K:3 * K] = rng.uniform(0.6, 0.95, K)
        xs.append(v)
    return xs[:max(1, n)]


def refit(f, Z, *, R_inf, w, tau, phi, L=0.0, include_L=False,
          pct=PCT, starts=STARTS):
    """从**参考参数**出发做独立加权 LS，返回最优 Rwp 与诊断。

    返回 dict：status / rwp_pct / n_starts / failed_starts / nfev / elapsed_s /
    at_bound（有多少参数贴在边界上，用于判断解是否被边界卡住）。
    """
    from scipy.optimize import least_squares

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    K = int(np.atleast_1d(w).size)
    zmax = float(np.max(np.abs(Z)))
    sig = (pct / 100.0) * np.abs(Z)

    x_ref = _pack(w, tau, phi, R_inf, L, include_L)
    lo, hi = _bounds(K, f, zmax, include_L)
    x_ref = np.clip(x_ref, lo + 1e-9, hi - 1e-9)

    def resid(x):
        p = _unpack(x, K, include_L)
        d = z_from_params(f, **p, include_L=include_L) - Z
        return np.concatenate([d.real / sig, d.imag / sig])

    t0 = time.time()
    best = None
    failed = 0
    nfev = 0
    for x0 in _starts(x_ref, K, f, zmax, include_L, starts):
        try:
            rr = least_squares(resid, np.clip(x0, lo, hi), bounds=(lo, hi),
                               max_nfev=MAX_NFEV, ftol=1e-11, xtol=1e-11,
                               gtol=1e-11, x_scale="jac")
        except Exception:
            failed += 1
            continue
        nfev += int(rr.nfev)
        if not np.all(np.isfinite(rr.fun)):
            failed += 1
            continue
        cost = float(np.sqrt(np.mean(rr.fun ** 2)))
        if best is None or cost < best[0]:
            best = (cost, rr.x)
    if best is None:
        return dict(status="failed", rwp_pct=None, n_starts=starts,
                    failed_starts=failed, nfev=nfev,
                    elapsed_s=round(time.time() - t0, 2), at_bound=None)

    x = best[1]
    p = _unpack(x, K, include_L)
    val = rwp(f, Z, z_from_params(f, **p, include_L=include_L))
    # 边界诊断：贴在 1e-6 相对距离内算"卡边界"，说明解可能被框住
    span = np.maximum(hi - lo, 1e-12)
    at_bound = int(np.sum((x - lo) / span < 1e-6) + np.sum((hi - x) / span < 1e-6))
    return dict(status="ok", rwp_pct=val, n_starts=starts, failed_starts=failed,
                nfev=nfev, elapsed_s=round(time.time() - t0, 2),
                at_bound=at_bound)


# ---------------------------------------------------------------------------
# 生产接线用的门面
# ---------------------------------------------------------------------------

def _skip_reason(res, zeta):
    """参考拓扑与生产不一致时，如实跳过（不近似、不硬套）。"""
    if str(zeta) == "free":
        return "生产拟合 ζ 自由（非 1.0），参考 LS 的拓扑会不同，不做参考重拟合"
    try:
        z = float(zeta)
    except (TypeError, ValueError):
        return f"无法解析 zeta={zeta!r}，跳过"
    if abs(z - 1.0) > 1e-9:
        return f"生产拟合 ζ={z:g}≠1，参考 LS 的拓扑会不同，不做参考重拟合"
    d = res.get("diffusion")
    if d not in (None, "", "none"):
        return f"生产拟合带扩散项（{d}），参考 LS 只覆盖基础 voigt 拓扑，不做"
    if res.get("adsorption") not in (None, "", "none"):
        return f"生产拟合带吸附项（{res.get('adsorption')}），参考 LS 未覆盖，不做"
    return None


def reference_block(f, Z, res, include_L, zeta, *, pct=PCT, starts=STARTS):
    """给 `save_out` 用：返回 `reference_rwp` 字典（只读、不影响任何判定）。

    `rwp_primary` 恒算（便宜，且它就是"生产参数换成 Rwp 这把尺子"）；
    `rwp_ls_relative` 只有拓扑一致时才独立重拟合。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    K = int(res["K"])
    w = np.atleast_1d(np.asarray(res["w"], float))
    tau = 1.0 / (2 * np.pi * np.atleast_1d(np.asarray(res["f0"], float)))
    phi = np.atleast_1d(np.asarray(res["phi"], float))
    R_inf = float(res["R_inf"])
    L = float(res["L"]) if (include_L and res.get("L") is not None) else 0.0

    zp = z_from_params(f, R_inf=R_inf, w=w, tau=tau, phi=phi, L=L,
                       include_L=include_L)
    block = {
        "rwp_primary": rwp(f, Z, zp),
        "rwp_ls_relative": None,
        "ls_relative_refit_performed": False,
        "ls_relative_refit_status": "not_performed",
        "ls_relative_weight_definition": WEIGHT_DEFINITION,
        "note": NOTE,
    }
    if w.size != K:
        block["ls_relative_refit_status"] = "skipped"
        block["reference_note"] = f"分量数不符（w={w.size}, K={K}），跳过"
        return block

    reason = _skip_reason(res, zeta)
    if reason:
        block["ls_relative_refit_status"] = "skipped"
        block["reference_note"] = reason
        return block

    out = refit(f, Z, R_inf=R_inf, w=w, tau=tau, phi=phi, L=L,
                include_L=include_L, pct=pct, starts=starts)
    block["ls_relative_refit_performed"] = True
    block["ls_relative_refit_status"] = out["status"]
    block["ls_relative_refit_starts"] = out["n_starts"]
    block["ls_relative_refit_failed_starts"] = out["failed_starts"]
    block["ls_relative_elapsed_s"] = out["elapsed_s"]
    if out["status"] != "ok":
        block["reference_note"] = "LS 重拟合未收敛，只报状态不报数"
        return block

    block["rwp_ls_relative"] = out["rwp_pct"]
    block["ls_relative_at_bound"] = out["at_bound"]
    # 参考解打头 → LS 最优点不可能比 primary 差；差了说明实现有问题，
    # 宁可标 failed 也不把可疑数字交出去。
    # 容差用**绝对下限**兜底：无噪声合成谱上 primary 可以是 0（或 1e-16），
    # 纯相对容差会把"数值噪声级差异"误判成实现失败。
    prim = block["rwp_primary"]
    if prim is not None and np.isfinite(prim) and out["rwp_pct"] > max(prim, 1e-6) * 1.001:
        block["ls_relative_refit_status"] = "failed"
        block["reference_note"] = (f"LS 结果 {out['rwp_pct']:.4f}% 反而劣于 "
                                   f"primary {prim:.4f}%，拒绝采用（实现自检失败）")
        block["rwp_ls_relative"] = None
    return block


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def selftest():
    """与生产解析式对拍 + Rwp 与内部 rms_rel 的 √2 关系，两条都必须是恒等。"""
    import bm_core as B

    f = np.logspace(-1, 5, 61)
    rng = np.random.default_rng(0)
    K = 2
    p = dict(R_inf=0.43, w=np.array([0.16, 1.03]),
             tau=np.array([1 / (2 * np.pi * 70.3), 1 / (2 * np.pi * 4.19)]),
             phi=np.array([0.92, 0.78]), L=1.7e-7)
    z1 = z_from_params(f, **p, include_L=True)
    z2 = B.model_z(f, dict(R_inf=p["R_inf"], W=float(p["w"].sum()),
                           pi=p["w"] / p["w"].sum(), tau=p["tau"],
                           phi=p["phi"], zeta=np.ones(K), L=p["L"]),
                   include_L=True)
    e_hn = float(np.max(np.abs(z1 - z2)) / np.max(np.abs(z2)))

    # Rwp 与内部 rms_rel：rms_rel = Rwp/√2
    Z = z1 * (1 + 0.01 * rng.standard_normal(f.size))
    d = (z1 - Z) / np.abs(Z)
    rms_rel = float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2)) * 100)
    r1 = rwp(f, Z, z1)
    e_rms = abs(r1 - rms_rel * np.sqrt(2))

    ok = (e_hn < 1e-12) and (e_rms < 1e-10)
    print(f"[selftest] 与 bm_core.model_z 最大相对差 {e_hn:.2e}（须 <1e-12）")
    print(f"[selftest] Rwp={r1:.6f}% vs rms_rel*√2={rms_rel * np.sqrt(2):.6f}%"
          f"（差 {e_rms:.2e}，须 <1e-10）")
    print("[selftest] " + ("通过" if ok else "失败"))
    return dict(hn=e_hn, rms=e_rms, ok=bool(ok))


if __name__ == "__main__":
    import json
    import sys

    if "--selftest" in sys.argv[1:]:
        sys.exit(0 if selftest()["ok"] else 1)
    print(json.dumps({"weight_definition": WEIGHT_DEFINITION}, ensure_ascii=False,
                     indent=2))
