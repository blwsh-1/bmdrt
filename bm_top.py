# -*- coding: utf-8 -*-
"""嵌套并联电路拓扑（M1：只做阻抗计算，不接采样器）。

**本模块不改 `bm_core.py` / `bm_ad.py` 的任何生产逻辑**；它只提供新拓扑的
正演与自检。接入采样器是 M2（届时薄封装进 `bm_families/` 注册表）。

拓扑（两层，Shahri et al. 2024 Table 1 circuit 1/4 的推广）
-----------------------------------------------------------
    Z = Rs + [ Y1 + 1/(R1 + [ Y2 + 1/(R2 + Z_branch) ]^-1)^-1 ]^-1
    其中 Y_i = 1/Z_CPE(Q_i, n_i) 是 CPE 导纳

展开就是：

    Z_inner = R2 + Z_branch
    Z_par2  = 1 / (1/Z_CPE2 + 1/Z_inner)
    Z_mid   = R1 + Z_par2
    Z       = Rs + 1 / (1/Z_CPE1 + 1/Z_mid)

感抗支路有两种接法（**两者不等价，必须分开建模**）
-------------------------------------------------
    branch="parallel"：R_ad ∥ L_ad，Z = R·jωτ/(1+jωτ)，τ = L/R
        ω→0 → 0（短路）；ω→∞ → R_ad
        = Shahri 2024 circuit 2/3，也是 `bm_ad.z_ad`
    branch="series"  ：R_L + L 串联，Z = R_L + jωL
        ω→0 → R_L（有限电阻）；ω→∞ → ∞（开路）
        = Triviño-Peláez et al. 2026（Electrochim. Acta）的感抗支路

**这是本项目与立项书 v1.2 §2.2 的一处修正**：立项书只写了 R∥L；文献 #3 用的是
R+L 串联。两者低频极限不同，合成标定与可辨识性必须分别验证。

物理校准（文献 #3）
-------------------
低频感抗环来自**吸附中间体表面覆盖度的有限时间弛豫与活性位再生**，
中频容性来自吸附中间体的形成/积累；二者都是界面动力学，**不需要扩散/传质**。
所以本拓扑里的感抗支路只该用 R、L（或 τ）两个参数，不引入 Warburg。

渐近极限（推导见 `_selftest`，数值核对见 `NESTED_TOPOLOGY_PROJECT.md` §2.3）
--------------------------------------------------------------------------
    branch="parallel"：ω→∞ → Rs；          ω→0 → Rs + R1 + R2
    branch="series"  ：ω→∞ → Rs；          ω→0 → Rs + R1 + R2 + R_L

高频端两者都趋于 Rs（CPE 短路）；低频端的差别全部来自支路本身。

CPE 参数化（避免直接采 Y）
--------------------------
`Y` 的量级跨 8 个数量级且量纲依赖 `n`。本模块改用**窗口内幅值**：

    Q ≡ |Z_CPE(ω_ref)| = 1/(Y·ω_ref^n)     ⇒  Z_CPE(ω) = Q·(ω_ref/ω)^n·e^(-j·nπ/2)

`ω_ref = 2π√(f_min·f_max)`（与扩散项同一参考频率）。这样 `Q` 与 `zmax` 同尺度，
先验可以直接写 `LogNormal(log(zmax), 2)`。

自检：`python bm_top.py`
"""

from __future__ import annotations

import numpy as np

try:                                   # jax 只在前向计算时需要
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
except Exception:                      # pragma: no cover
    jax, jnp = None, None


# --------------------------------------------------------------------------
# 元件
# --------------------------------------------------------------------------

def omega_ref(f) -> float:
    """窗口几何中心角频率 `2π√(f_min·f_max)`。"""
    f = np.asarray(f, float)
    return float(2.0 * np.pi * np.sqrt(f.min() * f.max()))


def z_cpe(f, Q, n, f_ref=None):
    """CPE，用**窗口内幅值** Q = |Z(ω_ref)| 参数化。

    `Z = 1/(Y (jω)^n)`，令 `Q = 1/(Y ω_ref^n)` 得
    `Z = Q·(ω_ref/ω)^n·exp(-j nπ/2)`。
    """
    f = np.asarray(f, float)
    w_ref = 2.0 * np.pi * (f_ref if f_ref is not None else np.sqrt(f.min() * f.max()))
    w = 2.0 * np.pi * f
    return Q * (w_ref / w) ** n * np.exp(-1j * n * np.pi / 2.0)


def j_z_cpe(omega, Q, n, w_ref):
    """jax 版 CPE（omega、w_ref 都是角频率）。"""
    return Q * (w_ref / omega) ** n * jnp.exp(-1j * n * jnp.pi / 2.0)


def z_rl_parallel(f, R, tau):
    """R∥L：`Z = R·jωτ/(1+jωτ)`。ω→0 → 0；ω→∞ → R。"""
    x = 2.0 * np.pi * np.asarray(f, float) * tau
    return R * (1j * x) / (1.0 + 1j * x)


def j_z_rl_parallel(omega, R, tau):
    x = omega * tau
    return R * (1j * x) / (1.0 + 1j * x)


def z_rl_series(f, R, L):
    """R+L 串联：`Z = R + jωL`。ω→0 → R；ω→∞ → ∞。"""
    return R + 1j * 2.0 * np.pi * np.asarray(f, float) * L


def j_z_rl_series(omega, R, L):
    return R + 1j * omega * L


# --------------------------------------------------------------------------
# 拓扑
# --------------------------------------------------------------------------

# 完整元数据表（含派生量与两种支路），**不是**采样清单；
# 采样参数请用 sampled_keys(branch, layers)。
ALL_PARAM_KEYS = ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2",
                  "R_ad", "tau_ad", "R_L", "tau_L", "L_ad")
PARAM_KEYS = ALL_PARAM_KEYS          # 向后兼容别名


def sampled_keys(branch: str = "parallel", layers: int = 2) -> tuple:
    """当前拓扑**真正需要采样**的参数。

    - `layers=1` 没有第二个并联节点 → 不采 `R2` / `Q2` / `n2`；
    - `branch="parallel"` 不采 `R_L` / `tau_L`，反之不采 `R_ad` / `tau_ad`；
    - `L_ad` 永远是派生量（`R_L·tau_L`），不采样。

    给未使用的参数也建先验，会引入纯由先验决定的无效维度，NUTS 会在那里空转。
    """
    if branch not in ("parallel", "series"):
        raise ValueError(f"branch 只能是 'parallel'/'series'，收到 {branch!r}")
    if layers not in (1, 2):
        raise ValueError(f"layers 只能是 1/2，收到 {layers!r}")
    keys = ["Rs", "R1", "Q1", "n1"]
    if layers >= 2:
        keys += ["R2", "Q2", "n2"]
    keys += ["R_ad", "tau_ad"] if branch == "parallel" else ["R_L", "tau_L"]
    return tuple(keys)


def z_nested(f, p: dict, f_ref: float | None = None,
             branch: str = "parallel", layers: int = 2):
    """嵌套并联阻抗。

    参数（dict）
    -----------
    Rs, R1, R2 : 串联电阻（Ω）
    Q1, n1     : 外层 CPE 的窗口幅值与指数
    Q2, n2     : 内层 CPE
    branch="parallel"：`R_ad`、`tau_ad`（τ = L_ad/R_ad）
    branch="series"  ：`R_L`、`L_ad`

    layers=1 时退化为 `Rs + 1/(1/CPE1 + 1/(R1 + Z_branch))`，用于 M3 的
    拓扑退化检查。
    """
    if branch not in ("parallel", "series"):
        raise ValueError(f"branch 只能是 'parallel'/'series'，收到 {branch!r}")
    if layers not in (1, 2):
        raise ValueError(f"layers 只能是 1/2，收到 {layers!r}")
    f = np.asarray(f, float)
    w_ref = 2.0 * np.pi * (f_ref if f_ref is not None else np.sqrt(f.min() * f.max()))

    zc1 = z_cpe(f, float(p["Q1"]), float(p["n1"]), f_ref=w_ref / (2 * np.pi))
    if branch == "parallel":
        zb = z_rl_parallel(f, float(p["R_ad"]), float(p["tau_ad"]))
    else:
        zb = z_rl_series(f, float(p["R_L"]), float(p["L_ad"]))

    if layers >= 2:
        z_inner = float(p["R2"]) + zb
        zc2 = z_cpe(f, float(p["Q2"]), float(p["n2"]), f_ref=w_ref / (2 * np.pi))
        z_par2 = 1.0 / (1.0 / zc2 + 1.0 / z_inner)
        z_mid = float(p["R1"]) + z_par2
    else:
        # 单层：R1 与支路直接串联，**不含 R2**（R2 只属于第二层并联节点）
        z_mid = float(p["R1"]) + zb
    z_par1 = 1.0 / (1.0 / zc1 + 1.0 / z_mid)
    return float(p["Rs"]) + z_par1


def j_z_nested(omega, p: dict, w_ref, branch: str = "parallel", layers: int = 2):
    """jax 版（omega、w_ref 为角频率）。"""
    if branch not in ("parallel", "series"):
        raise ValueError(f"branch 只能是 'parallel'/'series'，收到 {branch!r}")
    if layers not in (1, 2):
        raise ValueError(f"layers 只能是 1/2，收到 {layers!r}")
    zc1 = j_z_cpe(omega, p["Q1"], p["n1"], w_ref)
    if branch == "parallel":
        zb = j_z_rl_parallel(omega, p["R_ad"], p["tau_ad"])
    else:
        zb = j_z_rl_series(omega, p["R_L"], p["L_ad"])
    if layers >= 2:
        z_inner = p["R2"] + zb
        zc2 = j_z_cpe(omega, p["Q2"], p["n2"], w_ref)
        z_par2 = 1.0 / (1.0 / zc2 + 1.0 / z_inner)
        z_mid = p["R1"] + z_par2
    else:
        z_mid = p["R1"] + zb
    z_par1 = 1.0 / (1.0 / zc1 + 1.0 / z_mid)
    return p["Rs"] + z_par1


def param_spec():
    """参数名 → (含义, 单位, 先验建议)。M2/M3 直接用这张表，不在代码里散落。"""
    return {
        "Rs":   ("溶液/引线串联电阻", "Ω", "LogNormal(log(0.2·zmax), 1.0)"),
        "R1":   ("外层并联支路电阻", "Ω", "LogNormal(log(0.1·zmax), 2.0)"),
        "R2":   ("内层并联支路电阻", "Ω", "LogNormal(log(0.1·zmax), 2.0)"),
        "Q1":   ("外层 CPE 窗口幅值 |Z(ω_ref)|", "Ω", "LogNormal(log(zmax), 2.0)"),
        "n1":   ("外层 CPE 指数", "—", "Uniform(0.5, 1.0)"),
        "Q2":   ("内层 CPE 窗口幅值", "Ω", "LogNormal(log(zmax), 2.0)"),
        "n2":   ("内层 CPE 指数", "—", "Uniform(0.5, 1.0)"),
        "R_ad": ("吸附支路电阻（branch=parallel）", "Ω", "LogNormal(log(0.1·zmax), 2.0)"),
        "tau_ad": ("吸附支路弛豫时间", "s", "LogNormal(log(1/ω_ref), 2.0)"),
        "R_L":  ("感抗支路串联电阻（branch=series）", "Ω", "LogNormal(log(0.1·zmax), 2.0)"),
        "tau_L": ("感抗支路特征时间 L/R_L（branch=series）", "s", "LogNormal(log(1/ω_ref), 2.0)"),
        "L_ad": ("感抗支路电感 = R_L·tau_L（**派生量，不采样**）", "H", "—"),
    }


# --------------------------------------------------------------------------
# 自检（M2：数值正确性）
# --------------------------------------------------------------------------

def _selftest():
    print("=" * 72)
    print("bm_top 自检（M1 元件 + M2 数值正确性）")
    print("=" * 72)
    ok = True
    f = np.logspace(-2, 5, 61)
    w_ref = omega_ref(f)

    base = dict(Rs=10.0, R1=55.0, R2=600.0, Q1=200.0, n1=0.75,
                Q2=900.0, n2=0.85, R_ad=3000.0, tau_ad=1.6,
                R_L=800.0, tau_L=1.625, L_ad=1300.0)

    # 1) CPE 窗口参数化：|Z(ω_ref)| 必须等于 Q
    for Q, n in ((200.0, 0.75), (5.0, 0.95), (1e4, 0.55)):
        zr = z_cpe(np.array([np.sqrt(f.min() * f.max())]), Q, n)
        e = abs(abs(zr[0]) - Q) / Q
        print(f"[1] |Z_CPE(ω_ref)| vs Q：Q={Q:g}, n={n} → rel err {e:.2e}")
        ok &= e < 1e-12

    # 2) 逐节点展开与 z_nested 一致（两种支路）
    for branch in ("parallel", "series"):
        p = dict(base)
        zc1 = z_cpe(f, p["Q1"], p["n1"])
        zc2 = z_cpe(f, p["Q2"], p["n2"])
        zb = (z_rl_parallel(f, p["R_ad"], p["tau_ad"]) if branch == "parallel"
              else z_rl_series(f, p["R_L"], p["L_ad"]))
        z_manual = (p["Rs"] + 1.0 / (1.0 / zc1
                    + 1.0 / (p["R1"] + 1.0 / (1.0 / zc2 + 1.0 / (p["R2"] + zb)))))
        z_mod = z_nested(f, p, branch=branch)
        e = float(np.max(np.abs(z_mod - z_manual)) / np.max(np.abs(z_manual)))
        print(f"[2] {branch:<8} 逐节点 vs z_nested：rel err {e:.2e}")
        ok &= e < 1e-12

    # 3) 渐近极限（超宽频率，避免窗口截断误判）
    for branch, lim0, lim_inf in (("parallel", base["Rs"] + base["R1"] + base["R2"],
                                   base["Rs"]),
                                  ("series", base["Rs"] + base["R1"] + base["R2"]
                                   + base["R_L"], base["Rs"])):
        fw = np.array([1e-12, 1e12])
        z = z_nested(fw, base, branch=branch)
        e0 = abs(z[0].real - lim0) / abs(lim0)
        einf = abs(z[1].real - lim_inf) / abs(lim_inf)
        print(f"[3] {branch:<8} ω→0 → {z[0].real:.3f}（理论 {lim0:g}，err {e0:.1e}）"
              f"；ω→∞ → {z[1].real:.4f}（理论 {lim_inf:g}，err {einf:.1e}）")
        ok &= e0 < 1e-6 and einf < 1e-6
        ok &= abs(z[0].imag) < 1e-6 * abs(lim0) and abs(z[1].imag) < 1e-3 * abs(lim_inf)

    # 4) 无源性：全域 min Re ≥ 0
    for branch in ("parallel", "series"):
        z = z_nested(f, base, branch=branch)
        r = float(np.min(z.real))
        print(f"[4] {branch:<8} 无源性 min Re = {r:.3f} Ω（须 ≥ 0）")
        ok &= r >= 0.0

    # 5) 退化：n=1 的 CPE 等价于纯电容（|Z| ∝ 1/f）
    p = dict(base); p["n1"] = 1.0; p["n2"] = 1.0
    z = z_nested(f, p, branch="parallel")
    zc = z_cpe(f, p["Q1"], 1.0)
    e = float(np.max(np.abs(zc / z_cpe(np.array([np.sqrt(f.min() * f.max())]),
                                       p["Q1"], 1.0) - (f.min() * f.max()) ** 0.5 / f)))
    print(f"[5] n=1 时 CPE ∝ 1/f：rel err {e:.2e}")
    ok &= e < 1e-12

    # 6) 退化：layers=1 与手写单层一致
    p = dict(base)
    zc1 = z_cpe(f, p["Q1"], p["n1"])
    zb = z_rl_parallel(f, p["R_ad"], p["tau_ad"])
    z1 = p["Rs"] + 1.0 / (1.0 / zc1 + 1.0 / (p["R1"] + zb))
    e = float(np.max(np.abs(z_nested(f, p, layers=1) - z1)) / np.max(np.abs(z1)))
    print(f"[6] layers=1 与手写单层一致：rel err {e:.2e}")
    ok &= e < 1e-12

    # 7) 边界稳定性：参数跨量级、频率跨 18 dec，不得出现 nan/inf
    bad = 0
    for Q in (1e-3, 1e6):
        for n in (0.5, 1.0):
            for R2 in (1e-3, 1e6):
                for tau in (1e-6, 1e4):
                    pp = dict(base); pp.update(Q1=Q, Q2=Q, n1=n, n2=n,
                                               R2=R2, tau_ad=tau)
                    z = z_nested(np.logspace(-9, 9, 41), pp, branch="parallel")
                    if not np.all(np.isfinite(z)):
                        bad += 1
    print(f"[7] 边界扫描（Q/n/R2/tau 极端值 × 18 dec）：非有限结果 {bad} 个")
    ok &= bad == 0

    # 8) jax 与 numpy 一致
    if jnp is not None:
        for branch in ("parallel", "series"):
            zj = np.asarray(j_z_nested(jnp.asarray(2 * np.pi * f), base,
                                       jnp.asarray(w_ref), branch=branch))
            zn = z_nested(f, base, branch=branch)
            e = float(np.max(np.abs(zj - zn)) / np.max(np.abs(zn)))
            print(f"[8] {branch:<8} jax vs numpy：rel err {e:.2e}")
            ok &= e < 1e-12
        z0 = complex(np.asarray(j_z_nested(jnp.asarray(0.0), base,
                                           jnp.asarray(w_ref))))
        print(f"     ω=0 → {z0}（应为 Rs+R1+R2，有限）")
        ok &= np.isfinite(z0)
    else:
        print("[8] jax 不可用，跳过")

    # 9) sampled_keys：四种组合与预期 tuple 完整相等（新增参数时不会漏断言）
    print("[9] sampled_keys 分支/层数矩阵（完整 tuple 比较）：")
    expect = {
        ("parallel", 1): ("Rs", "R1", "Q1", "n1", "R_ad", "tau_ad"),
        ("parallel", 2): ("Rs", "R1", "Q1", "n1", "R2", "Q2", "n2",
                          "R_ad", "tau_ad"),
        ("series", 1): ("Rs", "R1", "Q1", "n1", "R_L", "tau_L"),
        ("series", 2): ("Rs", "R1", "Q1", "n1", "R2", "Q2", "n2",
                        "R_L", "tau_L"),
    }
    for (br, ly), exp in expect.items():
        ks = sampled_keys(br, ly)
        good = (ks == exp)
        print(f"    {br:<8} layers={ly} → {ks}" + ("" if good else f"  ≠ {exp}"))
        ok &= good

    # 10) init_values 口径：物理 → 无约束 → 反变换必须可往返
    #     传错量级不会报错（见 fit_top docstring），所以这条要机器守
    try:
        from numpyro.distributions.transforms import biject_to

        e_max = 0.0
        zmax_t, wref_t = 1000.0, 1.0
        for branch in ("parallel", "series"):
            for layers in (1, 2):
                tab = _prior_table(zmax_t, wref_t, branch, layers)
                init = physical_to_unconstrained(base, zmax=zmax_t, w_ref=wref_t,
                                                 branch=branch, layers=layers)
                for k, u in init.items():
                    back = float(np.asarray(biject_to(tab[k].support)(u)))
                    e_max = max(e_max, abs(back - base[k]) / max(abs(base[k]), 1e-30))
        print(f"[10] init_values 无约束往返（4 种拓扑）：max rel err {e_max:.2e}")
        ok &= e_max < 1e-12
    except ImportError as ex:          # 缺 numpyro 只跳过，不算失败
        print(f"[10] 无约束往返检查跳过（{type(ex).__name__}: {ex}）")
    except Exception as ex:            # 自己的转换写错了 → 必须失败，不能静默跳过
        print(f"[10] 无约束往返检查**失败**（{type(ex).__name__}: {ex}）")
        ok = False

    print("=" * 72)
    print("全部通过" if ok else "有失败项")
    return ok


# --------------------------------------------------------------------------
# 采样（M3 原型：单次固定 branch×layers，不做拓扑自动选择）
# --------------------------------------------------------------------------

LOG2PI = float(np.log(2.0 * np.pi))


def _loglik_chained(s, f, Z, branch, layers, sigma):
    """`(chain, draw, n_freq)` 的同频联合对数似然。

    `ll[c,d,j] = log p(Z'_j) + log p(Z''_j)` —— 同一频率的实虚部是**一个**
    留出单元，不能摊成 2N 个独立观测（与 `bm_real.loglik_matrix` 同一口径）。
    """
    keys = sampled_keys(branch, layers)
    ch, dr = s["Rs"].shape[:2]
    ll = np.empty((ch, dr, f.size))
    for c in range(ch):
        for d in range(dr):
            p = {k: float(s[k][c, d]) for k in keys}
            if branch == "series":
                p["L_ad"] = p["R_L"] * p["tau_L"]
            z = z_nested(f, p, branch=branch, layers=layers)
            if sigma is not None:
                sr, si = float(sigma[0]), float(sigma[1])
            else:
                sr, si = float(s["s_re"][c, d]), float(s["s_im"][c, d])
            lr = -0.5 * ((Z.real - z.real) / sr) ** 2 - np.log(sr) - 0.5 * LOG2PI
            li = -0.5 * ((Z.imag - z.imag) / si) ** 2 - np.log(si) - 0.5 * LOG2PI
            ll[c, d] = lr + li
    return ll


def _prior_table(zmax, w_ref, branch: str, layers: int) -> dict:
    """参数 → 先验分布（**单一来源**）。

    `fit_top` 的采样与 `physical_to_unconstrained` 的换算都读这张表，
    避免"文档里的先验"与"代码里的先验"各写一遍——v3 就吃过这个亏
    （`param_spec()` 曾把派生量 `L_ad` 写成独立先验）。

    顺序即 `sampled_keys` 的顺序，不含派生量。
    """
    import numpyro.distributions as dist

    t = {
        "Rs": dist.LogNormal(jnp.log(0.2 * zmax), 1.0),
        "R1": dist.LogNormal(jnp.log(0.1 * zmax), 2.0),
        "Q1": dist.LogNormal(jnp.log(zmax), 2.0),
        "n1": dist.Uniform(0.5, 1.0),
    }
    if layers >= 2:
        t["R2"] = dist.LogNormal(jnp.log(0.1 * zmax), 2.0)
        t["Q2"] = dist.LogNormal(jnp.log(zmax), 2.0)
        t["n2"] = dist.Uniform(0.5, 1.0)
    if branch == "parallel":
        t["R_ad"] = dist.LogNormal(jnp.log(0.1 * zmax), 2.0)
        t["tau_ad"] = dist.LogNormal(jnp.log(1.0 / w_ref), 2.0)
    else:
        t["R_L"] = dist.LogNormal(jnp.log(0.1 * zmax), 2.0)
        t["tau_L"] = dist.LogNormal(jnp.log(1.0 / w_ref), 2.0)
    return t


def physical_to_unconstrained(p_phys: dict, *, zmax: float, w_ref: float,
                              branch: str = "parallel", layers: int = 2) -> dict:
    """物理参数 → **无约束**空间，供 `fit_top(init_values=...)` 使用。

    NumPyro 的 `init_params` 期望的是无约束（unconstrained）初值：正参数是
    `log`，区间参数是 `logit` 型的伸缩变换。直接把手写的物理值（如
    `R_ad=3000`）传进去**不会报错**，但等价于给了一个极度偏心的起点
    （`exp(3000)`），warmup 会白白拉长，还可能直接把采样器卡死。

    `zmax` 与 `w_ref` 必须与 `fit_top` 内部用的一致：
    `zmax = max|Z|`、`w_ref = 2π√(f_min·f_max)`。

    例：
        >>> p = dict(Rs=10, R1=55, R2=600, Q1=200, n1=0.75,
        ...          Q2=900, n2=0.85, R_ad=3000, tau_ad=1.6)
        >>> init = physical_to_unconstrained(p, zmax=1000.0, w_ref=1.0)
    """
    from numpyro.distributions.transforms import biject_to

    tab = _prior_table(float(zmax), float(w_ref), branch, layers)
    out = {}
    for k in sampled_keys(branch, layers):
        if k not in p_phys:
            continue
        u = biject_to(tab[k].support).inv(np.asarray(float(p_phys[k])))
        out[k] = np.asarray(u, dtype=float).reshape(())
    missing = set(sampled_keys(branch, layers)) - set(out)
    if missing:
        raise ValueError(f"init_values 缺少参数：{sorted(missing)}")
    return out


def fit_top(f, Z, *, branch: str = "parallel", layers: int = 2,
            n_warmup: int = 600, n_samples: int = 1200, seed: int = 0,
            num_chains: int = 4, target_accept: float = 0.9,
            max_tree_depth: int = 10, sigma=None, sigma_floor: float = 0.0,
            init_values: dict | None = None, progress: bool = True) -> dict:
    """单次固定 `branch × layers` 的 NUTS 拟合（M3 研究性原型）。

    边界（审查批准的范围）
    ----------------------
    · 采样参数由 `sampled_keys(branch, layers)` 决定：**不采**派生量 `L_ad`，
      也不为未使用的分支/层数建先验（否则引入纯先验决定的无效维度）；
    · `series` 支路的 `L_ad = R_L·tau_L` 用 `numpyro.deterministic` 派生；
    · 噪声：`sigma=None` 时沿用 `bm_core` 的 `LogNormal(log 0.01·zmax, 1)` 与
      `sigma_floor` 语义；显式传 `(s_re, s_im)` 时固定不采（**绝对标准差**，
      不是相对 `|Z|` 的逐点值，见 `BM_TOP_M3_PLAN.md` §4.1）；
    · **不做**拓扑自动选择、跨族比较——等 M3 结论。

    `init_values`
    -------------
    直接进 NumPyro 的 `init_params`，因此必须是**无约束空间**的值（正参数取
    `log`、`n1/n2` 取 logit 型伸缩）。物理参数请先过
    `physical_to_unconstrained(...)`。语义写清楚是因为传错的代价是静默的：
    数值不会报错，只会让 warmup 无意义地变长。

    返回 `dict`，`s` 与 `fit_one` 同名；`pointwise_elpd` 是逐频率点的 LOO。
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    if f.size != Z.size or f.size < 8:
        raise ValueError("f 与 Z 长度需一致且不少于 8 点")
    if not np.all(np.isfinite(f)) or f.min() <= 0:
        raise ValueError("频率必须是有限正数")
    sampled_keys(branch, layers)                 # 提前校验 branch/layers

    def model(f_arr, Zr, Zi):
        zmax = jnp.max(jnp.sqrt(Zr ** 2 + Zi ** 2))
        w_ref = 2.0 * jnp.pi * jnp.sqrt(jnp.min(f_arr) * jnp.max(f_arr))
        omega = 2.0 * jnp.pi * f_arr
        p = {}
        for k, d in _prior_table(zmax, w_ref, branch, layers).items():
            p[k] = numpyro.sample(k, d)
        if branch == "series":
            p["L_ad"] = numpyro.deterministic("L_ad", p["R_L"] * p["tau_L"])

        Zt = j_z_nested(omega, p, w_ref, branch=branch, layers=layers)

        if sigma is None:
            if sigma_floor > 0:
                floor = sigma_floor * zmax
                b_re = numpyro.sample("s_re_base",
                                      dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                b_im = numpyro.sample("s_im_base",
                                      dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                s_re = numpyro.deterministic("s_re", b_re + floor)
                s_im = numpyro.deterministic("s_im", b_im + floor)
            else:
                s_re = numpyro.sample("s_re", dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
                s_im = numpyro.sample("s_im", dist.LogNormal(jnp.log(0.01 * zmax), 1.0))
        else:
            s_re = jnp.asarray(float(sigma[0]))
            s_im = jnp.asarray(float(sigma[1]))
        numpyro.sample("obs_re", dist.Normal(Zt.real, s_re), obs=Zr)
        numpyro.sample("obs_im", dist.Normal(Zt.imag, s_im), obs=Zi)

    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept,
                     max_tree_depth=max_tree_depth),
                num_warmup=n_warmup, num_samples=n_samples,
                num_chains=num_chains, progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed), f, Z.real, Z.imag,
             extra_fields=("diverging",), init_params=init_values)

    s_ch = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    if branch == "series" and "L_ad" not in s_ch:
        s_ch["L_ad"] = s_ch["R_L"] * s_ch["tau_L"]
    s = {k: v.reshape((-1,) + v.shape[2:]) for k, v in s_ch.items()}

    ll = _loglik_chained(s_ch, f, Z, branch, layers, sigma)
    import arviz as az
    # 注意：`from_dict` 的第一个位置参数是 posterior；log_likelihood 必须
    # 作为关键字传。`az.loo` 还需要 posterior 组来对齐 (chain, draw)，
    # 所以两样都要给。
    idata = az.from_dict(posterior=s_ch, log_likelihood={"obs": ll})
    loo = az.loo(idata)
    # arviz 0.23 的 ELPDData 是 pandas Series，字段用下标取（不是属性）
    pw = loo.get("loo_i", None)
    if pw is None:
        pw = loo.get("elpd_i", None)

    from numpyro.diagnostics import summary
    st = summary(s_ch, prob=0.9, group_by_chain=True)
    rhat = [float(np.nanmax(np.asarray(v["r_hat"]))) for v in st.values()]
    ess = [float(np.nanmin(np.asarray(v["n_eff"]))) for v in st.values()]
    div = mcmc.get_extra_fields().get("diverging")

    return {
        "s": s,
        "elpd": float(loo["elpd_loo"]),
        "elpd_se": float(loo["se"]),
        "pointwise_elpd": None if pw is None else np.asarray(pw, float),
        "pareto_k": np.asarray(loo["pareto_k"], float),
        "idata": idata,
        "sigma_re": (float(np.median(s["s_re"])) if "s_re" in s
                     else (float(sigma[0]) if sigma is not None else None)),
        "sigma_im": (float(np.median(s["s_im"])) if "s_im" in s
                     else (float(sigma[1]) if sigma is not None else None)),
        "divergences": int(np.sum(div)) if div is not None else None,
        "rhat_max": max(rhat) if rhat else None,
        "ess_min": min(ess) if ess else None,
        "branch": branch, "layers": layers,
        "f_ref": float(np.sqrt(f.min() * f.max())),
    }


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
