"""两入口共用的残差指标、逐点噪声与 ELPD 可比性判定。

**为什么有这个模块**（2026-11-16 收敛）
--------------------------------------
`nested_fit.py`（单族研究接口）与 `circuit_runner.py`（跨族候选调度）各自抄了一份
`metrics()`、各自写死了 2%·|Z|、各自定义了一套预算表。实测后果：

· `quick` 档同名不同值（nested_fit 用 `bm_select.QUICK` 800/400/10，
  runner 用 300/300/8）——看档名会误以为两次结果同预算；
· 同一份 `σ_j = 2%·|Z_j|`，`nested_fit` 标 `sigma_from_data=True`（elpd 不可作
  预测证据），runner 走 `bm_select.fit_nested` 不传该开关从而标 `False`。

两者**采样内核本来就是同一个** `bm_top.fit_top`（探针
`docs/round_2026-11-16/entry_equiv_probe.py`：standard 档 13 个实参逐一相同）。
所以该收敛的不是拟合路径，而是外围口径：噪声定义、指标计算、预算解析、结果解释。

边界
----
纯函数。不 import `bm_top`/`bm_core`/`bm_real`/`bm_select`，不跑拟合、不读文件、
不写产物。只有 `bm_experiments.acf_max` 一个数值依赖（残差自相关，§14 同式）。

三层语义必须分开（混写会让弱理由掩盖强理由）
--------------------------------------------
| 概念 | 问的是什么 | 本模块的字段 |
|---|---|---|
| σ 泄漏 | `σ_j` 是否含**被留出点自身**的观测值 | `sigma_from_data` / `sigma_leak` |
| 先验尺度依赖 | 先验尺度 `zmax=max|Z|` 是否由本批观测算出、留出 j 时未重算（R21） | `prior_scale_data_dependent` |
| 候选间是否同噪声 | 各候选是否共享**同一组**固定 σ | `noise_shared_across_candidates` |

`σ_j = 2%·|Z_j^obs|` 同时满足「由观测算出」与「外部规定的相对误差模型」**两条**：
它既是预先规定的误差模型（2% 是外部给定），又确实含被预测点自身的幅值。
对 ELPD 而言起作用的是**后者**——留出第 j 点时 `σ_j` 无处可算，似然尺度用了要预测
的那个数，算出来不是 `p(Z_j|Z_{-j})`。这与「σ 由残差反向估计」不是一回事，结论
却相同：**单个 `elpd_loo` 不能当样本外预测证据**。

反过来，共享同一组 σ 让 `Δelpd` 的差异来自模型本身而非噪声口径，这一点是
**成立**的；但 R21 的先验尺度依赖仍在，所以 `Δelpd` 只能作**候选间相对参考**，
不能写成标准留一预测性能。两条结论独立，不许用其中一条替代另一条。
"""
from __future__ import annotations

import numpy as np

import bm_experiments as E

#: §13/§14 验收口径的**唯一来源**（2000/1000/2 链 + target_accept=0.98）。
#:
#: 注意 `bm_select.BUDGETS["standard"]` 是**另一套**：2000/1000/2 + target_accept=0.9
#: （§9.4.4 的 A2 档）。两者同名不同值，历史上混用过。本模块只提供下面这一份，
#: 谁要 §9.4.4 的口径请显式去 `bm_select.BUDGETS` 取，不许再抄第二份常量。
STANDARD = dict(n_warmup=2000, n_samples=1000, num_chains=2,
                target_accept=0.98, max_tree_depth=10)

#: 预算的**五个核心数值键**。档名（`"quick"`）不是预算，任何产物都必须落这五项。
BUDGET_KEYS = ("n_warmup", "n_samples", "num_chains",
               "target_accept", "max_tree_depth")

#: 逐点相对噪声的默认百分比（2%），与 §14 / M3.5 `rel-perpoint-2pct` 同值。
SIGMA_PCT = 2.0


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------
def budget_actual(budget: dict, *, tier: str | None = None) -> dict:
    """把预算 dict 归一成「五个数值 + 档名 + 是否与其他入口共享」。

    `tier_shared=False` 是给 `quick` 档用的：两个入口的 `quick` **数值不同**
    （800/400/10 vs 300/300/8）。产物里必须能看出这一点，否则拿着
    `budget="quick"` 去比对两次结果，会以为预算相同。

    缺键直接报错——不允许产物里出现「档名有了、数值没有」的预算块。
    """
    miss = [k for k in BUDGET_KEYS if k not in budget]
    if miss:
        raise ValueError(f"预算缺键 {miss}；必须落 {list(BUDGET_KEYS)} 五项数值")
    out = {k: budget[k] for k in BUDGET_KEYS}
    out["tier"] = tier
    out["tier_shared_across_entries"] = (tier == "standard")
    return out


# ---------------------------------------------------------------------------
# 逐点噪声
# ---------------------------------------------------------------------------
def sigma_perpoint(Z, pct: float = SIGMA_PCT) -> np.ndarray:
    """`σ_j = (pct/100)·|Z_j|`，实虚共用同一 σ，返回 `(2, n_freq)`。

    `pct` 的单位是**百分数**（`2.0` 表示 2%，不是 0.02）。写成比例会在似然上
    差 100 倍而不报错，所以这里直接按百分数收，并校验取值范围。

    边界（都显式报错，不悄悄加地板）：
    · `|Z_j| = 0` ⇒ `σ_j = 0` ⇒ `Normal` 的 scale 非法。逐点口径**不提供**
      sigma floor（`bm_top._check_sigma` 里两者互斥），所以只能拒绝；
    · `|Z_j|` 非有限 ⇒ 拒绝；
    · `0 < pct < 100` ⇒ 越界拒绝（100% 以上没有物理意义，多半是把比例当百分数传了）。

    返回二维数组而不是 `(arr, arr)` tuple：`bm_top._check_sigma` 两种写法等价，
    但数组能避免调用方各自决定容器类型（差分测试里 tuple/ndarray 会被误报成差异）。
    """
    p = float(pct)
    if not np.isfinite(p) or p <= 0.0 or p >= 100.0:
        raise ValueError(
            f"sigma pct 必须是 (0, 100) 内的百分数（2.0 表示 2%），收到 {pct!r}")
    Z = np.asarray(Z, complex)
    if Z.ndim != 1 or Z.size == 0:
        raise ValueError(f"Z 必须是一维非空数组，收到 shape={Z.shape}")
    a = np.abs(Z)
    if not np.all(np.isfinite(a)):
        raise ValueError("|Z| 含非有限值，无法定义逐点噪声")
    if not np.all(a > 0.0):
        n0 = int(np.sum(a <= 0.0))
        raise ValueError(
            f"|Z| 含 {n0} 个零或负值 ⇒ σ_j=0 会让 Normal 的 scale 非法；"
            "逐点口径不接受 sigma floor，请先清理该频点")
    s = p / 100.0 * a
    return np.stack([s, s])


# ---------------------------------------------------------------------------
# 残差指标（超集：两个入口的字段都在，公式逐字未变）
# ---------------------------------------------------------------------------
def metrics(f, Z, Zm) -> dict:
    """§14 `_metrics` 同式，字段取两入口的**并集**。

    原 `circuit_runner.metrics` 给 5 个键（`rms_rel`/`bands`/`acf`/`acf_re`/
    `acf_im`/`acf_mag`），原 `nested_fit.metrics` 给 13 个键
    （多 `rms_re`/`rms_im`/`rel_median`/`rel_max`/`rel_max_at_Hz`/`bands_pct`/
    `top_hz`/`top_rel_pct`）。这里全给，**每个字段的公式、分母、频段边界与原来
    逐字相同**：

    · `rms_rel = 100·√mean(concat(d.real, d.imag)²)`，`d = (Z_fit − Z_obs)/|Z_obs|`
      —— 不是 `Rwp`，数值上是 `Rwp/√2`；
    · `rms_re`/`rms_im` 是**绝对**残差（Ω），不归一；
    · 频段固定 `[0.01,1) / [1,100) / [100,1e5)`，取 `d` 的 rms×100；
    · `acf = max(acf_max(Re res), acf_max(Im res))`，尺度不变，只响应频率相关结构。

    `bands` 与 `bands_pct` 是**同值别名**：两个入口的历史字段名不同，合并时不能
    任删一个（会让现有产物读取方 KeyError），所以并列给出。

    `|Z_obs| = 0` 时 `d` 会是 inf/nan —— 原实现同样如此（上游 `load_spectrum`
    已排除零阻抗点），这里**不新增保护**，以免悄悄改变历史数值。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    Zm = np.asarray(Zm, complex)
    if not (f.shape == Z.shape == Zm.shape) or f.ndim != 1:
        raise ValueError(
            f"f/Z/Zm 形状需一致且为一维：{f.shape}/{Z.shape}/{Zm.shape}")
    if f.size == 0:
        raise ValueError("空谱")

    w = np.abs(Z)
    res = Zm - Z
    rel = np.abs(res) / w
    d = res / w

    bands = []
    for lo, hi in ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d[m]) ** 2))) * 100
                     if m.any() else None)

    i_top = int(np.argmax(f))
    acf_re = float(E.acf_max(res.real))
    acf_im = float(E.acf_max(res.imag))
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        rms_re=float(np.sqrt(np.mean(res.real ** 2))),
        rms_im=float(np.sqrt(np.mean(res.imag ** 2))),
        rel_median=float(np.median(rel)) * 100,
        rel_max=float(np.max(rel)) * 100,
        rel_max_at_Hz=float(f[int(np.argmax(rel))]),
        bands=bands,
        bands_pct=bands,
        acf=float(max(acf_re, acf_im)),
        acf_re=acf_re,
        acf_im=acf_im,
        acf_mag=float(E.acf_max(rel)),
        top_hz=float(f[i_top]),
        top_rel_pct=float(rel[i_top]) * 100,
    )


# ---------------------------------------------------------------------------
# ELPD 可比性判定（三层语义分开，不合并成一个 bool）
# ---------------------------------------------------------------------------
def elpd_validity(*, likelihood: str, sigma_from_data: bool | None = None) -> dict:
    """这批 `elpd_loo` 能不能当样本外预测证据、能到什么程度。

    两个**独立**的理由，必须分层报（合并必有一个被掩盖）：

    L1 `sigma_leak` —— σ 含被留出点自身
        `perpoint` 口径下 `σ_j = 2%·|Z_j^obs|`，留出第 j 点时 `σ_j` 无从计算，
        似然尺度用了要预测的那个数 ⇒ `elpd_loo` 不是 `p(Z_j|Z_{-j})`。
        `free` 口径下 σ 是被采样的位点，不含该点观测 ⇒ 这一层**不成立**。

    L2 `prior_scale_data_dependent` —— 先验尺度 `zmax=max|Z|` 由本批观测算出，
        留出 j 点时未重算（**R21**）。两个口径**都成立**，与 L1 无关。

    因此：
    · `valid_for_predictive_comparison`（与 `bm_top.fit_top` 返回的同义）只回答
      L1 —— `perpoint` ⇒ False，`free` ⇒ True。**不要**把它当成「可以当验收证据」；
    · `comparison_scope` 回答的是「`Δelpd` 能用到什么程度」：L2 恒成立 ⇒ 恒为
      `"relative_only"`，即候选间相对参考，不得写成标准留一预测性能；
    · `noise_shared_across_candidates` 只说明「各候选是不是同一组固定 σ」。共享
      让 `Δelpd` 的差异来自模型而非噪声口径，但**不能**因此把 `Δelpd` 升级成
      样本外证据（L2 仍在）。

    `sigma_from_data` 允许调用方显式给（如 `nested_fit` 走 `fit_top` 时已知）；
    不给则按 `likelihood` 推导。两者不一致时以调用方显式值为准，但会记
    `derived_mismatch=True`，让「口径说是 perpoint、标注却写 False」这种事
    不会静默通过。
    """
    if likelihood not in ("perpoint", "free"):
        raise ValueError(f"未知 likelihood={likelihood!r}（可用 perpoint / free）")
    derived = (likelihood == "perpoint")
    if sigma_from_data is None:
        sfd, mismatch = derived, False
    else:
        sfd = bool(sigma_from_data)
        mismatch = (sfd != derived)

    reasons = []
    if sfd:
        reasons.append(
            "L1 σ_j=2%·|Z_j^obs| 含被留出点自身：留出 j 后 σ_j 无处可算，"
            "elpd_loo 不是 p(Z_j|Z_-j)")
    reasons.append(
        "L2 先验尺度 zmax=max|Z| 由本批观测算出、留出 j 点时未重算（R21）："
        "Δelpd 只能作候选间相对参考")

    return dict(
        likelihood=likelihood,
        sigma_from_data=sfd,
        sigma_leak=sfd,
        prior_scale_data_dependent=True,            # R21 恒成立
        valid_for_predictive_comparison=not sfd,    # 与 bm_top 同义（只答 L1）
        comparison_scope="relative_only",           # L2 恒成立
        noise_shared_across_candidates=True,        # 调用方可覆盖（见 runner）
        derived_mismatch=mismatch,
        reason="；".join(reasons),
    )


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _selftest() -> int:
    ok, total, bad = 0, 0, []

    def c(name, cond):
        nonlocal ok, total
        total += 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if cond:
            ok += 1
        else:
            bad.append(name)

    def raises(fn):
        try:
            fn()
        except Exception:                        # noqa: BLE001
            return True
        return False

    # ---- sigma_perpoint ----
    f = np.logspace(-2, 5, 20)
    Z = 10 + 100 / (1 + 1j * 2 * np.pi * f * 1e-3)
    s = sigma_perpoint(Z, 2.0)
    c("sigma 形状 (2,n)", s.shape == (2, Z.size))
    c("sigma 实虚同值", np.array_equal(s[0], s[1]))
    c("sigma 严格 = pct/100·|Z|", np.allclose(s[0], 0.02 * np.abs(Z), rtol=0, atol=0))
    c("pct 是百分数：2.0 是 0.02 的 100 倍",
      abs(s[0][0] / sigma_perpoint(Z, 0.02)[0][0] - 100.0) < 1e-9)
    c("pct=0 拒绝", raises(lambda: sigma_perpoint(Z, 0.0)))
    c("pct<1 的比例写法被拒（0.02 当百分数=0.02%）… 实为合法但极小",
      np.allclose(sigma_perpoint(Z, 0.02)[0], 0.0002 * np.abs(Z)))
    c("pct=100 拒绝", raises(lambda: sigma_perpoint(Z, 100.0)))
    c("pct 非有限拒绝", raises(lambda: sigma_perpoint(Z, float("nan"))))
    c("|Z|=0 拒绝", raises(lambda: sigma_perpoint(np.array([1 + 0j, 0 + 0j]))))
    c("|Z| 非有限拒绝", raises(lambda: sigma_perpoint(np.array([1 + 0j, np.inf + 0j]))))
    c("Z 二维拒绝", raises(lambda: sigma_perpoint(np.ones((2, 3), complex))))

    # ---- metrics ----
    Zm = Z * (1 + 1e-3)
    m = metrics(f, Z, Zm)
    need_runner = {"rms_rel", "bands", "acf", "acf_re", "acf_im", "acf_mag"}
    need_nested = {"rms_rel", "rms_re", "rms_im", "rel_median", "rel_max",
                   "rel_max_at_Hz", "bands_pct", "acf", "acf_re", "acf_im",
                   "acf_mag", "top_hz", "top_rel_pct"}
    c("runner 原 5 键全在", need_runner <= set(m))
    c("nested 原 13 键全在", need_nested <= set(m))
    c("bands 与 bands_pct 同值", m["bands"] == m["bands_pct"])
    c("acf = max(acf_re, acf_im)", m["acf"] == max(m["acf_re"], m["acf_im"]))
    c("rms_rel 对 0.1% 比例偏差 ≈ 0.0707%（÷√2）",
      abs(metrics(f, Z, Z * (1 + 1e-3))["rms_rel"] - 0.0707107) < 1e-5)
    c("频段边界 [0.01,1)/[1,100)/[100,1e5) 各段非空",
      all(x is not None for x in m["bands"]))
    c("top_hz = 最高频", m["top_hz"] == f.max())
    c("形状不一致拒绝", raises(lambda: metrics(f, Z, Zm[:-1])))
    c("空谱拒绝", raises(lambda: metrics(np.array([]), np.array([]), np.array([]))))

    # 逐字段数值回归：手工算一遍核心式
    res = Zm - Z
    d = res / np.abs(Z)
    c("rms_rel 手工式一致",
      abs(m["rms_rel"] - float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100) < 1e-12)
    c("rms_re 是绝对残差（Ω）",
      abs(m["rms_re"] - float(np.sqrt(np.mean(res.real ** 2)))) < 1e-12)
    c("rel_median 是百分数",
      abs(m["rel_median"] - float(np.median(np.abs(res) / np.abs(Z))) * 100) < 1e-12)

    # ---- elpd_validity ----
    vp = elpd_validity(likelihood="perpoint")
    vf = elpd_validity(likelihood="free")
    c("perpoint ⇒ sigma_leak=True", vp["sigma_leak"] is True)
    c("perpoint ⇒ valid=False", vp["valid_for_predictive_comparison"] is False)
    c("free ⇒ sigma_leak=False", vf["sigma_leak"] is False)
    c("free ⇒ valid=True（只答 L1）", vf["valid_for_predictive_comparison"] is True)
    c("两口径都带 R21 层", vp["prior_scale_data_dependent"] and vf["prior_scale_data_dependent"])
    c("两口径 comparison_scope 都是 relative_only",
      vp["comparison_scope"] == vf["comparison_scope"] == "relative_only")
    c("perpoint 理由含 L1+L2", "L1" in vp["reason"] and "L2" in vp["reason"])
    c("free 理由只含 L2", "L1" not in vf["reason"] and "L2" in vf["reason"])
    c("显式 sigma_from_data 覆盖推导并记 mismatch",
      elpd_validity(likelihood="perpoint", sigma_from_data=False)["derived_mismatch"] is True)
    c("显式值与推导一致时不记 mismatch",
      elpd_validity(likelihood="perpoint", sigma_from_data=True)["derived_mismatch"] is False)
    c("未知 likelihood 拒绝", raises(lambda: elpd_validity(likelihood="scalar")))

    # ---- 预算 ----
    b = budget_actual(dict(STANDARD), tier="standard")
    c("standard 五项齐全", all(k in b for k in BUDGET_KEYS))
    c("standard 标 tier_shared=True", b["tier_shared_across_entries"] is True)
    c("quick 标 tier_shared=False",
      budget_actual(dict(STANDARD), tier="quick")["tier_shared_across_entries"] is False)
    c("缺键拒绝", raises(lambda: budget_actual({"n_warmup": 1}, tier="quick")))
    c("STANDARD 与 §14 冻结档逐项相同",
      STANDARD == dict(n_warmup=2000, n_samples=1000, num_chains=2,
                       target_accept=0.98, max_tree_depth=10))

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
