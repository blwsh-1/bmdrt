# -*- coding: utf-8 -*-
"""嵌套并联电路拓扑（M1 正演 + `fit_top()` 独立采样路径）。

**本模块不改 `bm_core.py` / `bm_ad.py` 的任何生产逻辑**，也不经
`bm_families/` 注册表——架构冻结未解除期间，`fit_top()` 是一条**并行的
独立采样路径**，只服务 M3 合成标定。M2「接注册表」因此记为**绕过**，
不是完成；`topology=voigt` 的生产行为逐位不变。

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

噪声接口（M3.5）
----------------
`fit_top(..., sigma=...)` 三种形态，校验在建模**之前**完成（`_check_sigma`）：

| 输入 | 含义 |
|---|---|
| `None` | 自由估计噪声（采 `s_re`/`s_im`，受 `sigma_floor` 约束） |
| `(σ_re, σ_im)` 标量对 | 实虚各自固定标量噪声（A/B 组口径） |
| `(2, n_freq)` 数组 | 实虚各自**逐频点**噪声（`rel-perpoint-2pct` 条件） |

`(n_freq,)` 单数组**不接受**（与标量对混淆，见 `_check_sigma`）；`sigma` 与
`sigma_floor` 互斥。标量与逐点共用同一段代码，标量路径的数值逐位未变
（自检 [12]、`evidence/nested_topology/m3_regress_scalar.py`）。

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
                                                 branch=branch, layers=layers,
                                                 sigma=(1.0, 1.0))
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

    # 11) init_values 接真实采样器：位点齐全 + 多链首维（静态审查提的两处边界）
    #     结构检查覆盖不到这两条——它们分别以 AssertionError / IndexError 出现，
    #     所以必须真跑一次 fit_top，而不是只看字典的键。
    try:
        import numpyro  # noqa: F401
        f11 = np.logspace(-2.0, 5.0, 21)
        Z11 = z_nested(f11, base, branch="parallel", layers=2)
        zmax11 = float(np.max(np.abs(Z11)))
        wref11 = float(2 * np.pi * np.sqrt(f11[0] * f11[-1]))
        sig11 = (0.02 * zmax11, 0.02 * zmax11)
        need_circuit = set(sampled_keys("parallel", 2))
        for nc, sg, tag in ((1, sig11, "固定 σ"), (4, sig11, "固定 σ"),
                            (1, None, "自由 σ"), (4, None, "自由 σ")):
            init11 = physical_to_unconstrained(base, zmax=zmax11, w_ref=wref11,
                                               sigma=sg, num_chains=nc)
            r11 = fit_top(f11, Z11, n_warmup=8, n_samples=8, seed=0,
                          num_chains=nc, sigma=sg, init_values=init11,
                          progress=False)
            need = need_circuit | (set() if sg is not None else {"s_re", "s_im"})
            miss = need - set(r11["s"])
            print(f"[11] {nc} 链 {tag}：初值 {len(init11)} 位点 → 后验 "
                  f"{len(r11['s'])} 位点，缺 {sorted(miss) if miss else '无'}")
            ok &= not miss
        # 负例：自由 σ 却只给电路初值，必须**报错**（不能静默给个错起点）
        stray = physical_to_unconstrained(base, zmax=zmax11, w_ref=wref11,
                                          sigma=sig11, num_chains=1)
        try:
            fit_top(f11, Z11, n_warmup=4, n_samples=4, seed=0, num_chains=1,
                    sigma=None, init_values=stray, progress=False)
            print("[11] 负例未触发（自由 σ + 缺噪声初值竟能跑）→ 记为失败")
            ok = False
        except Exception as ex:
            print(f"[11] 负例按预期失败：{type(ex).__name__}（自由 σ 缺噪声位点）")
    except ImportError as ex:          # 缺 numpyro 只跳过
        print(f"[11] 初值采样检查跳过（{type(ex).__name__}: {ex}）")
    except Exception as ex:            # 自己的换算写错 → 必须失败
        print(f"[11] 初值采样检查**失败**（{type(ex).__name__}: {ex}）")
        ok = False

    # 12) 逐点噪声接口（M3.5）：校验矩阵 / 手算 loglik / 模型真吃数组 / 标量路径不变
    #     四条都必须**在没有 MCMC 的情况下**可判：跑一次 NUTS 只能证明"没崩"，
    #     证明不了似然用的是数组而不是被拍平的标量。
    try:
        from numpyro.infer.util import log_density

        f12 = np.logspace(-2.0, 5.0, 31)
        n12 = f12.size
        Z12 = z_nested(f12, base, branch="parallel")
        sg12 = 0.02 * np.abs(Z12)                    # 逐点 σ，实虚同值（§20.1）
        Zp12 = Z12 + (0.5 - 0.25j)                   # 非零残差，免得起因掩盖公式错

        # 12a 接受 / 拒绝矩阵：一维非长度 2 必须显式拒绝，不能靠广播猜
        cases = [("None", None, 0.0, True),
                 ("标量对 tuple", (61.0, 61.0), 0.0, True),
                 ("标量对 1-D", np.array([61.0, 61.0]), 0.0, True),
                 ("(2,n) 数组", np.vstack([sg12, sg12]), 0.0, True),
                 ("(n,) 单数组", sg12, 0.0, False),
                 ("(2,n-1) 列数不符", np.vstack([sg12, sg12])[:, :-1], 0.0, False),
                 ("(3,n) 行数不符", np.ones((3, n12)), 0.0, False),
                 ("sigma+floor 同给", (61.0, 61.0), 0.01, False),
                 ("含 0 分量", (0.0, 61.0), 0.0, False)]
        bad = []
        for tag, sg, fl, want in cases:
            try:
                _check_sigma(f12, sg, fl)
                got = True
            except ValueError:
                got = False
            if got != want:
                bad.append(tag)
        print(f"[12a] sigma 校验矩阵 {len(cases)} 例："
              + ("全部符合预期" if not bad else f"不符合预期 {bad}"))
        ok &= not bad

        # 12b 逐点 loglik 手算对照（`_loglik_chained` 是 LOO/ELPD 的那条路径）
        s12 = {k: np.asarray([[[float(base[k])]]]) for k in sampled_keys("parallel", 2)}

        def _hand_ll(sig):
            return (-0.5 * ((Zp12.real - Z12.real) / sig) ** 2 - np.log(sig) - 0.5 * LOG2PI
                    - 0.5 * ((Zp12.imag - Z12.imag) / sig) ** 2 - np.log(sig) - 0.5 * LOG2PI)

        ll12 = _loglik_chained(s12, f12, Zp12, "parallel", 2, sg12, sg12)[0, 0]
        hand12 = _hand_ll(sg12)
        e12b = float(np.max(np.abs(ll12 - hand12)) / np.max(np.abs(hand12)))
        print(f"[12b] 逐点 loglik vs 手算（{n12} 点）：max rel err {e12b:.2e}")
        ok &= e12b < 1e-12

        # 12c 模型真的吃了数组：obs 位点的 scale 是 (n_freq,)，且 logp 之差等于手算之差
        #     `log_density` 的 `params` 走**约束空间**（不施加 transform），
        #     所以这里直接给物理值；两个模型的先验相同 → logp 之差只剩似然
        par12 = {k: float(base[k]) for k in sampled_keys("parallel", 2)}
        args12 = (f12, Zp12.real, Zp12.imag)
        lp_s, _ = log_density(_make_model("parallel", 2, "scalar", 61.0, 61.0),
                              args12, {}, par12)
        lp_v, tr_v = log_density(_make_model("parallel", 2, "perpoint", sg12, sg12),
                                 args12, {}, par12)
        scale_v = np.asarray(tr_v["obs_re"]["fn"].scale)
        loc_ok = np.allclose(np.asarray(tr_v["obs_re"]["fn"].loc), Z12.real,
                             rtol=0, atol=1e-9)
        scale_ok = scale_v.shape == (n12,) and np.array_equal(scale_v, sg12)
        d_model = float(lp_v - lp_s)
        d_hand = float(np.sum(_hand_ll(sg12) - _hand_ll(61.0)))
        e12c = abs(d_model - d_hand) / max(abs(d_hand), 1.0)
        print(f"[12c] 模型 obs scale shape={scale_v.shape}（须 ({n12},)）、"
              f"逐位相等={scale_ok}；logp 差 vs 手算差 rel err {e12c:.2e}")
        ok &= bool(scale_ok) and bool(loc_ok) and e12c < 1e-10

        # 12d 标量路径不变：常数数组必须与标量给出**逐位相同**的 loglik 与 logp
        ll_c = _loglik_chained(s12, f12, Zp12, "parallel", 2,
                               np.full(n12, 61.0), np.full(n12, 61.0))[0, 0]
        ll_s = _loglik_chained(s12, f12, Zp12, "parallel", 2, 61.0, 61.0)[0, 0]
        lp_c, _ = log_density(_make_model("parallel", 2, "perpoint",
                                          np.full(n12, 61.0), np.full(n12, 61.0)),
                              args12, {}, par12)
        same_ll = np.array_equal(ll_c, ll_s)
        same_lp = float(lp_c) == float(lp_s)
        print(f"[12d] 标量 vs 常数数组：loglik 逐位相等={same_ll}、"
              f"logp 逐位相等={same_lp}")
        ok &= bool(same_ll) and bool(same_lp)
    except ImportError as ex:          # 缺 numpyro 只跳过
        print(f"[12] 逐点噪声检查跳过（{type(ex).__name__}: {ex}）")
    except Exception as ex:            # 接口或似然写错 → 必须失败
        print(f"[12] 逐点噪声检查**失败**（{type(ex).__name__}: {ex}）")
        ok = False

    print("=" * 72)
    print("全部通过" if ok else "有失败项")
    return ok


# --------------------------------------------------------------------------
# 采样（M3 原型：单次固定 branch×layers，不做拓扑自动选择）
# --------------------------------------------------------------------------

LOG2PI = float(np.log(2.0 * np.pi))


def _check_sigma(f, sigma, sigma_floor: float = 0.0):
    """校验并归一化噪声输入。**必须在进入 NumPyro 模型之前调用**。

    返回 `(mode, s_re, s_im)`：

    | 输入 | mode | s_re / s_im |
    |---|---|---|
    | `None` | `"free"` | `None`（噪声是被采样的位点） |
    | 标量对 `(σ_re, σ_im)` | `"scalar"` | 两个 `float` |
    | `(2, n_freq)` 数组 | `"perpoint"` | 两个 `(n_freq,)` 数组 |

    三条硬约束（都是**显式报错**，不靠广播猜）：

    · `(n_freq,)` **不接受**。单数组会与"标量对"混淆（`n_freq == 2` 时更会静默
      通过），所以要求逐点噪声必须写成 `(2, n_freq)`；一维且长度不为 2 的输入
      直接拒绝并提示 reshape。
    · `sigma` 与 `sigma_floor` **互斥**：固定 σ 时 floor 无意义，静默忽略会让
      "我加了 floor"的调用方以为噪声下界生效。
    · 全部分量必须有限且严格为正（`Normal` 的 `scale` 不接受 0）。

    `f` 可以是 `None`（`physical_to_unconstrained` 没有频率），此时跳过长度校验。
    """
    if sigma is None:
        return ("free", None, None)
    if float(sigma_floor) > 0.0:
        raise ValueError(
            "sigma 与 sigma_floor 同时给出：固定噪声（含逐点）时 sigma_floor 无意义。"
            "固定噪声请只传 sigma；要噪声下界请用 sigma=None + sigma_floor")

    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(
                f"sigma 长度 {arr.size} 的一维输入有歧义：逐点噪声必须写成 "
                f"(2, n_freq) 的二维数组（第 0 行实部、第 1 行虚部），"
                f"标量对才是长度 2 的一维输入。收到 shape={arr.shape}")
        arr = arr.reshape(2, 1)          # 标量对，下面按标量返回
        kind = "scalar"
    elif arr.ndim == 2 and arr.shape[0] == 2:
        kind = "perpoint"
    else:
        raise ValueError(
            f"sigma 只接受 (σ_re, σ_im) 标量对或 (2, n_freq) 数组，"
            f"收到 shape={arr.shape}")

    if kind == "perpoint":
        n = arr.shape[1]
        if f is not None and n != np.asarray(f).size:
            raise ValueError(
                f"逐点 sigma 的列数 {n} 与频点数 {np.asarray(f).size} 不一致")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        bad = int(np.sum(~np.isfinite(arr)) + np.sum(arr <= 0.0))
        raise ValueError(f"sigma 必须全为有限正数，检测到 {bad} 个非法分量")

    if kind == "scalar":
        return ("scalar", float(arr[0, 0]), float(arr[1, 0]))
    return ("perpoint", arr[0], arr[1])


def _loglik_chained(s, f, Z, branch, layers, sig_re, sig_im):
    """`(chain, draw, n_freq)` 的同频联合对数似然。

    `ll[c,d,j] = log p(Z'_j) + log p(Z''_j)` —— 同一频率的实虚部是**一个**
    留出单元，不能摊成 2N 个独立观测（与 `bm_real.loglik_matrix` 同一口径）。

    `sig_re`/`sig_im` 是 `_check_sigma` 归一化后的结果：`None` 表示自由噪声
    （逐 draw 从 `s["s_re"]`/`s["s_im"]` 取），否则是**常数或 `(n_freq,)` 数组**
    —— 后者走 numpy 广播，不需要另写一份似然。
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
            if sig_re is None:
                sr, si = float(s["s_re"][c, d]), float(s["s_im"][c, d])
            else:
                sr, si = sig_re, sig_im
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
                              branch: str = "parallel", layers: int = 2,
                              sigma=None, sigma_floor: float = 0.0,
                              num_chains: int = 1) -> dict:
    """物理参数 → **无约束**空间，供 `fit_top(init_values=...)` 使用。

    NumPyro 的 `init_params` 期望的是无约束（unconstrained）初值：正参数是
    `log`，区间参数是 `logit` 型的伸缩变换。直接把手写的物理值（如
    `R_ad=3000`）传进去**不会报错**，但等价于给了一个极度偏心的起点
    （`exp(3000)`），warmup 会白白拉长，还可能直接把采样器卡死。

    `zmax` 与 `w_ref` 必须与 `fit_top` 内部用的一致：
    `zmax = max|Z|`、`w_ref = 2π√(f_min·f_max)`。

    **返回的字典必须覆盖全部潜在采样位点**，否则 NumPyro 不会替你补：
    缺 `s_re` 时 `potential_fn` 直接断言失败。因此：

    · `sigma=(s_re, s_im)` 固定噪声 → 只有电路参数（噪声是常数、不采）；
    · `sigma=None` 自由噪声 → 另加 `s_re`/`s_im`（`sigma_floor>0` 时是
      `s_re_base`/`s_im_base`，因为模型采样的是 `base`，`s = base + floor`）。
      初值取 `p_phys["s_re"]`/`["s_im"]`，缺省 `0.01·zmax`（先验中位）。

    **`num_chains>1` 时必须给每个位点形状 `(num_chains,)`**：NumPyro 只做
    "首维等于链数"的检查（`mcmc.run` 里那句 raise），标量会先在
    `jnp.shape(x)[0]` 上抛 `IndexError`。本函数按 `num_chains` 广播。

    例：
        >>> p = dict(Rs=10, R1=55, R2=600, Q1=200, n1=0.75,
        ...          Q2=900, n2=0.85, R_ad=3000, tau_ad=1.6)
        >>> init = physical_to_unconstrained(p, zmax=1000.0, w_ref=1.0,
        ...                                  sigma=(20.0, 20.0), num_chains=4)
    """
    from numpyro.distributions.transforms import biject_to

    _check_sigma(None, sigma, sigma_floor)      # 提前报错（与 fit_top 同一套校验）
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

    if sigma is None:
        import numpyro.distributions as dist

        floor = float(sigma_floor) * float(zmax)
        s0 = 0.01 * float(zmax)          # 与先验 LogNormal(log 0.01·zmax, 1) 的中位一致
        s_re = float(p_phys.get("s_re", s0))
        s_im = float(p_phys.get("s_im", s0))
        sup = dist.LogNormal(0.0, 1.0).support      # 与先验同支撑：正数
        if sigma_floor > 0:
            for name, s in (("s_re_base", s_re), ("s_im_base", s_im)):
                if s - floor <= 0.0:
                    raise ValueError(
                        f"{name} 的物理初值 {s:.4g} 不大于 floor "
                        f"{floor:.4g}（= sigma_floor·zmax），无法换算到无约束空间")
                out[name] = np.asarray(biject_to(sup).inv(
                    np.asarray(s - floor)), dtype=float).reshape(())
        else:
            for name, s in (("s_re", s_re), ("s_im", s_im)):
                if s <= 0.0:
                    raise ValueError(f"{name} 的物理初值必须为正，收到 {s:.4g}")
                out[name] = np.asarray(biject_to(sup).inv(np.asarray(s)),
                                       dtype=float).reshape(())

    if int(num_chains) > 1:
        out = {k: np.repeat(np.asarray(v, dtype=float).reshape(1), int(num_chains))
               for k, v in out.items()}
    return out


def _make_model(branch: str, layers: int, sig_mode: str,
                sig_re=None, sig_im=None, sigma_floor: float = 0.0):
    """构造 NumPyro 模型。噪声用 `_check_sigma` **归一化后**的输入。

    逐点 σ 与标量 σ 走**同一段代码**：`jnp.asarray` 对标量给 0 维、对
    `(n_freq,)` 给向量，`dist.Normal(loc, scale)` 与标准化残差都自然广播。
    所以标量路径的数值与引入逐点 σ 之前**逐位相同**（自检 [12] 守这条）。

    独立成函数只为一件事：自检能在**不跑 MCMC** 的情况下拿到 `obs` 位点的
    `scale`，验证模型真的吃了数组（而不是在追踪时被 `float()` 拍平）。
    """
    import numpyro                        # 局部导入：numpyro 仍是可选依赖
    import numpyro.distributions as dist

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

        if sig_mode == "free":
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
            # 标量 → 0 维；逐点 → (n_freq,)。两种情况都不采噪声位点。
            s_re = jnp.asarray(sig_re)
            s_im = jnp.asarray(sig_im)
        numpyro.sample("obs_re", dist.Normal(Zt.real, s_re), obs=Zr)
        numpyro.sample("obs_im", dist.Normal(Zt.imag, s_im), obs=Zi)

    return model


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
      `sigma_floor` 语义；显式传 `(s_re, s_im)` 时固定不采（**绝对标准差**,
      见 `BM_TOP_M3_PLAN.md` §4.1）；传 `(2, n_freq)` 数组时为**逐点**固定
      噪声（M3.5 的 `rel-perpoint-2pct` 条件，口径见 §20.1）；
    · **不做**拓扑自动选择、跨族比较——等 M3 结论。

    `sigma` 的三种形态见 `_check_sigma`：`None` / 标量对 / `(2, n_freq)`。
    一维输入只当标量对解释，`(n_freq,)` **会被拒绝**（否则与标量对混淆）。
    `sigma` 与 `sigma_floor` 互斥，校验在构造模型**之前**完成。

    `init_values`
    -------------
    直接进 NumPyro 的 `init_params`，因此必须是**无约束空间**的值（正参数取
    `log`、`n1/n2` 取 logit 型伸缩）。物理参数请先过
    `physical_to_unconstrained(...)`。语义写清楚是因为传错的代价是静默的：
    数值不会报错，只会让 warmup 无意义地变长。

    两点硬约束（NumPyro 0.19 的实际行为，自检 [11] 守着）：

    · **位点必须齐全**——`sigma=None` 时还有 `s_re`/`s_im`（`sigma_floor>0`
      时是 `s_re_base`/`s_im_base`）；漏了会以 `AssertionError` 失败，不是静默；
    · **`num_chains>1` 时每位点首维须等于链数**——标量会抛 `IndexError`。
      `physical_to_unconstrained(..., num_chains=n, sigma=...)` 会一并处理。

    返回 `dict`，`s` 与 `fit_one` 同名；`pointwise_elpd` 是逐频率点的 LOO。
    `sigma_mode` 记录噪声口径；逐点模式下 `sigma_re`/`sigma_im` 为 `None`
    （没有单一标量可报），数组另存 `sigma_re_perpoint`/`sigma_im_perpoint`。
    """
    from numpyro.infer import MCMC, NUTS

    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    if f.size != Z.size or f.size < 8:
        raise ValueError("f 与 Z 长度需一致且不少于 8 点")
    if not np.all(np.isfinite(f)) or f.min() <= 0:
        raise ValueError("频率必须是有限正数")
    sampled_keys(branch, layers)                 # 提前校验 branch/layers
    sig_mode, sig_re, sig_im = _check_sigma(f, sigma, sigma_floor)

    model = _make_model(branch, layers, sig_mode, sig_re, sig_im,
                        float(sigma_floor))

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

    ll = _loglik_chained(s_ch, f, Z, branch, layers, sig_re, sig_im)
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
    # 逐参数记录：只报 max 时看不出是哪个参数在拖后腿（地形问题通常集中在 1–2 个）
    rhat_map = {k: float(np.nanmax(np.asarray(v["r_hat"]))) for k, v in st.items()}
    ess_map = {k: float(np.nanmin(np.asarray(v["n_eff"]))) for k, v in st.items()}
    rhat, ess = list(rhat_map.values()), list(ess_map.values())
    div = mcmc.get_extra_fields().get("diverging")

    return {
        "s": s,
        "elpd": float(loo["elpd_loo"]),
        "elpd_se": float(loo["se"]),
        "pointwise_elpd": None if pw is None else np.asarray(pw, float),
        "pareto_k": np.asarray(loo["pareto_k"], float),
        "idata": idata,
        "sigma_mode": sig_mode,
        "sigma_re": (float(np.median(s["s_re"])) if "s_re" in s
                     else (sig_re if sig_mode == "scalar" else None)),
        "sigma_im": (float(np.median(s["s_im"])) if "s_im" in s
                     else (sig_im if sig_mode == "scalar" else None)),
        # 逐点口径没有单一标量可报：数组原样带走，别在返回值里悄悄取中位
        "sigma_re_perpoint": (None if sig_mode != "perpoint"
                              else np.asarray(sig_re, float)),
        "sigma_im_perpoint": (None if sig_mode != "perpoint"
                              else np.asarray(sig_im, float)),
        "divergences": int(np.sum(div)) if div is not None else None,
        "rhat_max": max(rhat) if rhat else None,
        "ess_min": min(ess) if ess else None,
        "rhat_by_param": rhat_map,
        "ess_by_param": ess_map,
        "branch": branch, "layers": layers,
        "f_ref": float(np.sqrt(f.min() * f.max())),
    }


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
