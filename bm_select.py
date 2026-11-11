#!/usr/bin/env python
"""M4 跨族选择器（骨架）：voigt 族（`bm_core`）vs nested 族（`bm_top`）。

本文件的范围
------------
· 输入有两条路：
  - **合成谱**（`--cases`，默认）：`bm_core.true_spectrum`（voigt 真值）+
    `bm_top_gen` 四个拓扑与三个边界用例。自检与预注册实验都走这条。
  - **真实谱**（`--real GLOB`）：**只做可表示性诊断，不拟合**。秒级返回
    「voigt 族在结构上能否表示这条谱」，用于决定某批谱该不该进跨族比较。
    真实谱上跑完整选择（要拟合、要几分钟一条）目前**没有**入口。
· **不改 `bm_real.py` / `bm_top.py` / `bmdrt.py`**。接生产路径（`topology=`
  开关）要单独批；届时建议先跑 `--real`，voigt 不可表示的谱直接免掉比较。
· 两族噪声口径统一为 `sigma=None + sigma_floor`（立项书 R3）。`sigma=None`
  在两侧都落到 `LogNormal(log 0.01·zmax, 1)` 的同一先验上；只有同一似然
  定义下的 elpd 才可比。固定 σ 或逐点 σ 会破坏这条，不要在这种地方图省事。

判定规则（**跑之前写死，跑完不许改**）
--------------------------------------
逐点 LOO 的配对差：
    Δelpd = Σ_i ( elpd_i^nested − elpd_i^voigt )
    dse   = sqrt( n · Var(逐点差) )            # 与 arviz.compare 同口径

    切到 nested 需**同时**满足 `Δelpd ≥ SWITCH_DELTA` 且 `Δelpd/dse ≥ SWITCH_Z`；
    切到 voigt 同理反向；否则 `indistinguishable`。

    另有一个终态 `voigt_inadmissible`：当谱里的感抗落在 voigt 族**当前能力**
    （`VOIGT_CAPS`）覆盖不到的位置时，「否则会判并列」的那一档改报它。
    它不替代 `nested`/`voigt`——Δ 显著时照常报，正向信息不丢。
    依据：`docs/REFS_DIGEST.md` §2.I ⑥（串联 `jωL` 只覆盖高频，低频感抗需
    `adsorption`；AZ31 实测低频 `Im` 比 `ωL` 上界大 7–9 个数量级）。

验收口径（与上面这套判定分开，别混）
------------------------------------
· **门禁 = 零错切**：nested 真值被判 voigt、或 voigt 真值被判 nested，一次都不许有；
· **命中数只报数**，不设门槛。`indistinguishable`（"证据不足"）**不算错切**，
  按 `THREE_LAYER_SCOPE.md` §2 的合规输出是**「并列」**，不是"默认用 voigt"；
· 判定必须在多个 seed 上一致才给结论，否则记 `unstable`——实测同一个用例
  换一条随机流就能从 `indistinguishable` 变成 `voigt`（见 `select_across_families`）。
  **根因已查明**：这类翻转来自「族内最优候选被采样质量门禁剔除、族内拿次优
  候选参赛」，不是某一族更能解释。现在由 `_family_state` 把该情形整族记
  `degraded` 并输出并列（§9.4.1），不再硬算 Δ。

`indistinguishable` 的读法只有一句：**本窗本噪声下两族不可分**。
既不能说"两族等价"，也不能说"nested 不成立"。但档位要分来源：由配对
LOO 算出 `|Δ|<门槛` 的并列属 **B 档**；由 `degraded`（候选质量剔除改变了
族内最优、不做配对）产生的并列属 **A 档**（重跑/加迭代可解），不得记成
"数据本身分不开"。见 §9.4.1。

v1.5 缺口修补（`NESTED_TOPOLOGY_PROJECT.md` §9.4.4，预注册先于运行）
------------------------------------------------------------------
· **guard 清单已统一**：`guard_state()` 对两族出同一张 `GUARD_NAMES` 表，
  值域 `ok / triggered / not_evaluable / not_applicable`。算不出的（nested
  侧 tail-ESS、树深触顶**记 `not_applicable`（结构性不提供，跳过门禁）；
  该有却缺的（R̂=NaN、div 缺失）记 `not_evaluable` 并按 fail-closed 拒绝**。A 类进门禁
  需 `--strict-guards`；**B 类（参数简并 / 幅值 / 拐点）只记账不裁决**——
  把"参数定不住"接进族级门禁，等于重犯 §9.4.2 那个错。
· **候选池可结构对称**：`--pair-mode per-branch` 把 nested 拆成每条支路
  一组（各 2 个候选），分别与 voigt 的 2 个候选配对，严格 2 vs 2；族级
  结论只在两组一致时成立。默认 `pooled` 保持与 A2 逐位可比。
· **族内重跑**：`--retry-degraded N` 在降级时把**整批**两族候选统一换到
  `standard` 档重跑（换 seed）。整批而非只补失败者，避免组内出现两种
  采样预算造出新的不对称。重跑只改预算，不动 `SWITCH_*` / 用例 / 噪声。
· **仍在的缺口**：族内选 K/L 与跨族比较共用同一份 LOO，`Δelpd` 系统性
  偏乐观；`fit_top` 不给 tail-ESS 与树深，这两项只能记 `not_applicable`
  （诊断能力缺失，不等于"通过"）。**voigt 侧的能力开关已接线**（2026-11-10）：
  `include_L`（串联 jωL，高频）与 `adsorption`（R∥L，低频）都可开，**默认仍全关**
  以保与 A2/B2 逐位可比；打开任一项即构成**能力臂**，须按 §9.4.11 单独预注册、
  与基线分开报数。**残留的不对称**是 nested 的 `L_ad=R_L·tau_L` 关不掉——
  nested 永远带一条感抗支路，voigt 默认不带。
  **族级结论要分两级说（2026-11-10 定，别合并成一句）**：
    · **「哪一族能表示这条谱」**属**模型类**问题，可以指认 → 记 **B-缺项**
      （输出 `voigt_inadmissible`）。AZ31 18/18 条全部落在这一档
      （+ `archive/` 12 条，真实谱合计 30/30）。
    · **「哪一族更合理 / 机制是什么」**要外部证据 → 仍 **C 档 / 探索性**。
      即使 voigt 被排除，也**不能**推出「nested 是对的」。
  这正是 `THREE_LAYER_SCOPE.md` §2.1 的两个子类：B-缺项解决的是候选集完整性，
  不是机制正确性。
· nested 族 `L`（layers）**不是** DRT 峰数，固定扫 1..2，不做 DRT 预判。
· **nested 族的定义含两条支路**（`parallel`=`R∥L`、`series`=`R+L`，见 v1.3 文献
  校准）：族内候选是 `支路 × L` 的笛卡尔积，按 elpd 取优。选到哪条支路属
  **族内**结果，不参与族级门禁——"支路选错"要参数级证据才能判，不在本文件范围。

用法
----
    python bm_select.py                  # 默认自检（5 seed × 2 支路，B2 实测约 42 min）
    python bm_select.py --seeds 3 --branches parallel   # 11-09 首轮旧口径
    python bm_select.py --pair-mode per-branch --strict-guards  # 2v2 对称 + A 类 guard 进门禁
    python bm_select.py --dry            # 只打印用例，不拟合
    python bm_select.py --dry --json out/dry.json   # 干跑结果也落盘

运行时长的实测口径：B2（4 例 × 5 seed、quick、**无重跑**）合计约 42 min，
最慢单例 `loop_out_of_window` 988 s。别再按"25–35 min"做预算。

`--retry-degraded` 目前**不可用**：B1（standard 2000/1000）与 B1'（light
1200/600）两次实测都停滞在重跑阶段（单例 >50 min、父进程空等且 CPU 长期
低负载），11-10 两次中止。已排除的两个假说：换预算触发 JAX 重编译（实测
draws 400→600 只多 0.6 s）、多链本身不可用（生产 3 链跑完 171 条谱）。
根因未定，默认 0；查明前不要开，也不要照抄成示例命令。
"""

from __future__ import annotations

import argparse
import contextlib
import glob as _glob
import inspect
import io
import os
import sys
import time

import numpy as np

import bm_core as B
import bm_real as R
import bm_top as T
import bm_top_gen as G

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

#: 两族的采样预算（quick 档）。与 `bmdrt.py --quick` 的 800/400/2 同源，
#: 但显式写在这里，避免随生产默认值漂移。
QUICK = dict(n_warmup=800, n_samples=400, num_chains=2, progress=False,
             target_accept=0.9, max_tree_depth=10, extra_fields=("num_steps",))

#: 两族共用的噪声语义（R3）：σ 由后验采样，只加地板。
SIGMA_FLOOR = 0.0

#: 切换阈值（nats）与显著性倍数。改这两个数等于改验收标准，须先登记。
SWITCH_DELTA = 4.0
SWITCH_Z = 2.0

#: 跨族 `Δelpd` 的口径标注（**R21**，2026-11-11 补）。
#: 两族的先验尺度 `zmax = max|Z|` 都是**由本批观测算出**的，而逐点 PSIS-LOO
#: 留出第 j 点时没有重算 `zmax` → 严格说 `elpd_loo` 不是 `p(Z_j|Z_{-j})`。
#: 处置：不改实现（改成非观测尺度会让 §13/§14/§18/§19 全部数值作废），
#: 而是**每一条跨族判定都带这个标记**，报告里不得把 `Δelpd` 写成标准留一预测。
#: 敏感性见 `docs/round_2026-11-11/m4_zmax_scale_probe.py`。
ELPD_PRIOR_SCALE = "data_dependent_zmax"
ELPD_PRIOR_SCALE_NOTE = (
    "两族先验尺度 zmax=max|Z| 由本批观测算出，逐点 LOO 未按留出点重算 → "
    "Δelpd 只作族间**相对**参考，不是标准留一预测性能（R21）")

#: 采样质量门槛（两侧同等）。超过即认为该候选的 elpd 不可用于比较。
RHAT_MAX = 1.05
DIV_FRAC_MAX = 0.02

#: 与 `bm_real.grade` / `bm_grade.GUARD_CLASS` 同源的护栏阈值
BAD_K_MAX = 0.10          # Pareto-k>0.7 的点占比
ESS_MIN = 100.0           # grade() 的「可用」线
TREE_CAP_MAX = 0.05       # 树深触顶占比
CORR_STRONG = 0.95        # 立项书 §9.4 的结构简并护栏
BOUND_EPS = 0.01          # 参数距 Uniform 先验边界的相对宽度

#: 分量「未定位」判据（§9.4.17.7 / 风险 R15）。两把尺子互相独立，
#: **任一命中**即判该分量没被数据定住：
#:   · 时间常数 5–95% 跨 `> COMP_UNRES_DECADE`：窗口共 5 decade，
#:     定不到 2 decade 的「时间常数」不是一个独立的弛豫；
#:   · 幅值 `w_i/W < COMP_UNRES_AMP`：低于既有检测下限的量级
#:     （`SNR_im≈15` → 6.7%，见 `WS_CRITERION_PLAN.md`），拟合上看不见。
#: 阈值只写这一次，判据实现见 `unresolved_components()`。
#:
#: **与 R̂ 互补，不能互相替代**（`m4_compempty_check.py --real` 实测）：
#: 同一配置 seed0 上 R̂=1.008（合格）而排序后 τ 跨 2.14 decade（本护栏命中）；
#: seed4 上 R̂=1.64（命中）而 τ 跨 1.88 decade（本护栏未命中，因为两条链各自
#: 把次分量放在**不同的紧模态**里，排序后反而不摊开）。两条都要报。
COMP_UNRES_AMP = 0.05
COMP_UNRES_DECADE = 2.0

#: 采样预算档。`quick` 必须逐位等于 A2 的 800/400/2，否则前后对比失效。
#: `standard` 是 §9.4.4 预注册档，数值不改；`light` 是 11-10 B1 实测后发现
#: standard 在刚性谱上代价失控（单例 >50 min）而加的轻量档，属工程配置。
BUDGETS = {
    "quick": QUICK,
    "light": dict(QUICK, n_warmup=1200, n_samples=600),
    "standard": dict(QUICK, n_warmup=2000, n_samples=1000),
}
PAIR_MODES = ("pooled", "per-branch")
RETRY_SEED_STEP = 1000    # 重跑换随机流，不复用原 seed

#: 两族**同一张** guard 清单。值只能是 ok / triggered / not_evaluable /
#: not_applicable —— 算不出来就显式写 not_evaluable，绝不填 ok 冒充通过。
GUARD_A = ("rhat", "div_frac", "ess_bulk", "ess_tail", "bad_k", "loo_ok", "tree_cap")
GUARD_B = ("param_at_bound", "param_degenerate", "loop_amp", "loop_corner",
           "comp_unresolved")
GUARD_NAMES = GUARD_A + GUARD_B

#: 族内「降配参赛」判据（v1.4 修实现缺陷，见 §9.4.1）：被质量门禁剔除的候选，
#: 若其 elpd 比该族**实际参赛**的最优候选还高出这么多 nats，说明这次剔除改变了
#: 族内最优——此时整族记 `degraded`，不许拿次优候选去和另一族配对。
#: 与 `SWITCH_DELTA` 数值相同但含义不同，**不要合并成一个常数**：
#: 前者管「族内比较基准是否被污染」，后者管「跨族差是否够显著」。
DEGRADE_MARGIN = 4.0

#: nested 族的支路。"parallel" = `R∥L`（低频极限 0）；"series" = `R+L` 串联
#: （Triviño-Peláez 2026 用的形态，低频极限 `R_L` 而非 0）。
#: **族级判定必须覆盖整族**，所以两支路都扫；`branch` 只在族内报告，
#: 不参与族级门禁（选错支路是族内的分支问题，不是把族认反）。
BRANCH = "parallel"
BRANCHES = ("parallel", "series")


def _as_branches(branches) -> tuple[str, ...]:
    """把 `str` / 序列规整成支路元组，顺带校验拼写。"""
    b = (branches,) if isinstance(branches, str) else tuple(branches)
    bad = [x for x in b if x not in BRANCHES]
    if bad or not b:
        raise ValueError(f"支路只能是 {BRANCHES}，收到 {branches!r}")
    return b


#: voigt 族**当前实际启用**的元件能力。**定义在 `fit_voigt` 之后**（由它的签名
#: 默认值推导，不手写），因为它是 `voigt_inadmissible` 判据的唯一依据。
#: 改成推导而不是写死常量：手写常量与签名漂移时**不会报错**，只会让判据描述一支
#: 并不存在的族，输出静静地错（2026-11-10 记过同类问题）。

#: 谱「有感抗」的相对阈值：`Im Z > IND_TOL_SPEC·max|Z|`，**只看正虚部**。
#: 与 `evidence/nested_topology/m4_spectrum_health.py` 同口径（那边有两处注释
#: 记了为什么不能看 `|Im|`：容抗弧的 `Im` 本来就是负的，用绝对值会把每个弧都
#: 算成有感抗，实测 25–46% 全命中）。
IND_TOL_SPEC = 0.05


def _ind_freqs(f, Z, tol: float = IND_TOL_SPEC) -> np.ndarray:
    """出现感抗（正虚部超阈值）的频点。无感抗时返回空数组。"""
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    zmax = float(np.max(np.abs(Z))) or 1.0
    return f[Z.imag / zmax > tol]


#: 「调用方没传这个参数」的哨兵。不能拿 `None` 当哨兵，因为 `None` 本身是
#: `adsorption` 的一个**合法取值**（= 不用吸附支路）。
_UNSET = object()


def voigt_can_represent(f, Z, *, include_L=None,
                        adsorption=_UNSET) -> tuple[bool, str]:
    """voigt 族在**结构上**能否表示这条谱。

    依据 `docs/REFS_DIGEST.md` §2.I ⑥ 的两条定量结论：
      · 串联 `jωL` 只能产生**高频**感抗（贡献 `ωL` 随频率单调升，0.01 Hz 处
        对 `L=1e-6` 只有 6e-8 Ω，而 AZ31 实测低频 `Im` 是 29–1123 Ω）；
      · `R∥L`（`adsorption="RL"`）才是**低频**感抗（峰在 `1/(2πτ)`）。

    所以「有感抗就判不可表示」太粗，必须分频段——否则将来打开 `include_L`
    之后，只有高频仪器感抗的谱会被误判为不可用。

    划分点取谱的**几何中频** `sqrt(f_min·f_max)`。它只是个分界，不是物理阈值；
    对 AZ31 的 0.01–1e5 Hz 窗口是 31.6 Hz，实测低频感抗（0.01–1.4 Hz）与高频
    仪器段（5.2k–27k Hz）分得很开，落在两侧。

    返回 `(能否表示, 原因)`。`原因` 在能表示时为空串。

    **`True` 的确切含义（2026-11-10 收紧，别读多）**：只表示「该族在**结构上**
    有产生这种感抗的元件」，**不表示"能贴合"**。反例已经测出来了：低频感抗段在
    `adsorption="RL"` 下会判 `True`，但把 `R∥L` 挂在 voigt 的最外层串联上，
    `noise=0` 对 nested 谱的残差只从 44.0% 降到 40.7%，Δ 完全没动
    （`NESTED_TOPOLOGY_PROJECT.md` §9.4.12）。根因是**接点**：nested 的 `R∥L`
    在第二个并联节点内（`C2 ∥ (R2+Z_ad)`），voigt 没有那个节点。

    所以本函数回答的是「在不在候选空间」，不是「拟合得好不好」。后者要看 elpd /
    残差，且按 `THREE_LAYER_SCOPE.md` 最多记 B 档。
    """
    include_L = VOIGT_CAPS["include_L"] if include_L is None else include_L
    if adsorption is _UNSET:
        adsorption = VOIGT_CAPS["adsorption"]

    fi = _ind_freqs(f, Z)
    if fi.size == 0:
        return True, ""

    f = np.asarray(f, float)
    split = float(np.sqrt(float(np.min(f)) * float(np.max(f))))
    lo, hi = fi[fi <= split], fi[fi > split]
    need = []
    if lo.size and not adsorption:
        # 措辞要点：**不要**写成「需 adsorption=RL」。2026-11-10 的 B 臂实测
        # （`NESTED_TOPOLOGY_PROJECT.md` §9.4.12）证明：把 `R∥L` 挂在最外层串联
        # 只把 noise=0 残差从 44.0% 降到 40.7%，Δ 一点没动——真正缺的是**接点**，
        # 而那个接点就是 nested 的定义。写「需 adsorption」会让人去开一个开了也没用的开关。
        need.append(f"低频感抗 {lo.min():.3g}–{lo.max():.3g} Hz（{lo.size} 点）"
                    "需把 R∥L 放进第二并联节点（= 改族定义；顶层 adsorption 不足以表示）")
    if hi.size and not include_L:
        need.append(f"高频感抗 {hi.min():.3g}–{hi.max():.3g} Hz（{hi.size} 点）"
                    "需 include_L")
    if need:
        return False, "；".join(need)
    return True, ""


# --------------------------------------------------------------------------
# 两族拟合（统一的返回形状）
# --------------------------------------------------------------------------

def fit_voigt(f, Z, K: int, *, seed: int = 0, include_L: bool = False,
              adsorption: str | None = None,
              sigma_floor: float = SIGMA_FLOOR, kw: dict | None = None) -> dict:
    """voigt 族：`bm_real.fit_one`（K 个串联 HN + 可选 L / 吸附支路）。

    `include_L` / `adsorption` 是**族能力开关**，默认全关（= A2/B2 基线口径）。
    打开任一项都会扩大 voigt 族的候选空间，因此**必须与新臂一起预注册**
    （`NESTED_TOPOLOGY_PROJECT.md` §9.4.11），不能和基线混着报。

    两个开关的物理分工（`docs/REFS_DIGEST.md` §2.I ⑥）：
      · `include_L=True` → 串联 `jωL`，只产生**高频**感抗（仪器伪影）；
      · `adsorption="RL"` → `R·jωτ/(1+jωτ)`，产生**低频**感抗（吸附弛豫），
        元件与 nested 族 `parallel` 支路完全相同。
    """
    kw = dict(QUICK if kw is None else kw)
    r = R.fit_one(f, Z, K, include_L, "1.0", seed=seed, kw=kw,
                  sigma_floor=float(sigma_floor), adsorption=adsorption)
    pt = r.get("pointwise_elpd")
    pt = None if pt is None else np.asarray(pt, float)
    caps = _caps_suffix(include_L, adsorption)
    # 分量是否定位：跟着候选一起返回，**不改任何判定**（§9.4.18.2 默认路径逐位不变）。
    # 放进这里而不是 `_select_one`，是为了让 candidate_trace / JSON 里每个候选都自带
    # `k_eff`，事后不用重跑 MCMC 才能解释「K=2 里到底几个分量被定住了」。
    ec = unresolved_components((r or {}).get("s") or {})
    comp_state = ("not_evaluable" if ec is None else
                  "not_applicable" if ec["k"] < 2 else
                  "triggered" if ec["unresolved"] else "ok")
    return dict(family="voigt", dof=K, branch=None, label=f"K={K}{caps}",
                include_L=bool(include_L), adsorption=adsorption,
                elpd=float(r["elpd"]), se=float(r["elpd_se"]), pointwise=pt,
                div=int(r.get("ndiv") or 0), rhat=r.get("rhat_max"),
                grade=r.get("grade"), ok=_usable(pt, r["elpd"]), raw=r,
                k_eff=(None if ec is None else int(ec["n_resolved"])),
                comp_state=comp_state,
                comp_span=(None if ec is None else list(ec["span_dec"])),
                comp_amp=(None if ec is None else list(ec["amp_frac"])))


def _caps_suffix(include_L: bool, adsorption: str | None) -> str:
    """候选标签上标出族能力，避免「K=2」在开启能力的臂里被误读成基线。"""
    s = ""
    if adsorption:
        s += f"+{adsorption}"
    if include_L:
        s += "+L"
    return s


def fit_nested(f, Z, layers: int, *, branch: str = BRANCH, seed: int = 0,
               sigma_floor: float = SIGMA_FLOOR, kw: dict | None = None) -> dict:
    """nested 族：`bm_top.fit_top`（R∥L 支路嵌在两层 R∥CPE 里）。"""
    kw = dict(QUICK if kw is None else kw)
    o = T.fit_top(f, Z, branch=branch, layers=layers, seed=seed,
                  n_warmup=int(kw["n_warmup"]), n_samples=int(kw["n_samples"]),
                  num_chains=int(kw["num_chains"]),
                  target_accept=float(kw["target_accept"]),
                  max_tree_depth=int(kw["max_tree_depth"]),
                  sigma=None, sigma_floor=float(sigma_floor), progress=False)
    pt = o.get("pointwise_elpd")
    pt = None if pt is None else np.asarray(pt, float)
    return dict(family="nested", dof=layers, branch=branch,
                label=f"{branch}/L={layers}",
                elpd=float(o["elpd"]), se=float(o["elpd_se"]), pointwise=pt,
                div=int(o.get("divergences") or 0), rhat=o.get("rhat_max"),
                grade=None, ok=_usable(pt, o["elpd"]), raw=o)


def _usable(pt, elpd) -> bool:
    return pt is not None and np.isfinite(float(elpd)) and pt.size > 0


#: voigt 族的**默认**元件能力，由 `fit_voigt` 的签名默认值推导（见文件上方注释）。
#: `voigt_can_represent()` 在不传参时用的就是这份。
_FIT_VOIGT_SIG = inspect.signature(fit_voigt).parameters
VOIGT_CAPS = dict(include_L=_FIT_VOIGT_SIG["include_L"].default,
                  adsorption=_FIT_VOIGT_SIG["adsorption"].default)


def _finite(x) -> bool:
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _corr_pairs(s: dict) -> list[tuple]:
    """后验里 |ρ|>CORR_STRONG 的参数对（只看标量列）。

    `L_ad` 在 series 支路是 `R_L·tau_L` 的 deterministic 派生量，必然与两者
    强相关，属**定义式**不是简并，故显式排除；噪声位点 `s_*` 同样不参与。
    """
    arr, n0 = {}, None
    for k, v in (s or {}).items():
        if k.startswith("s_") or k == "L_ad":
            continue
        a = np.asarray(v, float).ravel()
        if a.size < 2:
            continue
        if n0 is None:
            n0 = a.size
        if a.size == n0 and np.isfinite(a).all() and np.std(a) > 0:
            arr[k] = a
    if len(arr) < 2:
        return []
    keys = list(arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(np.stack([arr[k] for k in keys]))
    return [(keys[i], keys[j], round(float(C[i, j]), 3))
            for i in range(len(keys)) for j in range(i + 1, len(keys))
            if np.isfinite(C[i, j]) and abs(C[i, j]) > CORR_STRONG]


def _uniform_bound_frac(s: dict) -> float | None:
    """`n1/n2`（Uniform(0.5,1)）贴先验边界的样本占比。"""
    near = total = 0
    for k, v in (s or {}).items():
        if not k.startswith("n"):
            continue
        a = np.asarray(v, float).ravel()
        if a.size < 2 or a.min() < 0.5 - 1e-6 or a.max() > 1.0 + 1e-6:
            continue
        total += a.size
        near += int(np.sum((a < 0.5 + BOUND_EPS * 0.5) |
                           (a > 1.0 - BOUND_EPS * 0.5)))
    return near / total if total else None


def unresolved_components(s: dict, *, amp: float = COMP_UNRES_AMP,
                          decade: float = COMP_UNRES_DECADE) -> dict | None:
    """voigt 族后验里「时间常数或幅值没被数据定住」的分量（§9.4.17 / R15）。

    返回 `dict(k, span_dec, amp_frac, unresolved, n_resolved)`，槽位按排序统计量
    对齐；`None` = 后验里没有可判字段（**不是**「都定住了」）。

    **必须用排序统计量。** 混合模型的编号在样本间自由交换（K=2 实测列排序
    一致率 0.504），直接对 `s['tau'][:, i]` 取分位会得到「列跨两个模态」的
    假象：本谱 K=2+L 的列跨 2.35 decade，而把它排序后只有一个槽跨 1.88 decade。
    更要紧的是**列中位会严重误导**——同一配置里次分量列中位 `w/W`=0.018，
    排序后（每样本取最小幅值再取中位）是 **0.156**，差了近 10 倍。曾据此
    把该分量判成「空载」，被排序统计量直接否掉（见 §9.4.17.5 的更正）。

    阈值为什么这么定：时间常数用 **2 decade**（窗口共 5 decade，定不到 2 decade
    不能算独立弛豫）；幅值用 **5%**（`SNR_im≈15` 的检测下限 6.7% 同量级）。
    幅值这把尺子在本批 7 条 2EMC 谱上一次也没触发过（所有分量 ≥15%），
    保留它是为了防止「幅值近零、τ 又恰好紧」的退化样本冒充一个弛豫。
    """
    tau, w = s.get("tau"), s.get("w")
    if tau is None or w is None:
        return None
    tau = np.atleast_2d(np.asarray(tau, float))
    w = np.atleast_2d(np.asarray(w, float))
    if tau.shape != w.shape or tau.shape[1] < 1:
        return None
    if not (np.isfinite(tau).all() and np.isfinite(w).all()):
        return None
    lt = np.sort(np.log(tau), axis=1)                    # 每个样本内按 τ 升序
    ws = np.sort(w, axis=1)                              # 幅值升序：[:, 0] 最小
    tot = np.maximum(ws.sum(axis=1), 1e-300)
    k = lt.shape[1]
    span, frac = [], []
    for j in range(k):
        lo, hi = np.percentile(lt[:, j], [5, 95])
        span.append(float((hi - lo) / np.log(10.0)))     # decade
        frac.append(float(np.median(ws[:, j] / tot)))
    bad = [j for j in range(k) if span[j] > decade or frac[j] < amp]
    return dict(k=k, span_dec=span, amp_frac=frac, unresolved=bad,
                n_resolved=k - len(bad))


def guard_state(c: dict, n_draw: int, f_max: float | None = None) -> dict:
    """两族**同一张** guard 清单，只记账（A 类之外的项不进门禁）。

    返回值域：`ok` / `triggered` / `not_evaluable` / `not_applicable`，
    四者不能混：
      · `not_evaluable` = **该有却缺**（R̂=NaN、div 缺失、LOO 点缺失），
        门禁按 fail-closed 拒绝。把"没算出来"读成"很干净"是最隐蔽的假通过。
      · `not_applicable` = **结构性不提供 / 该族根本没有**：nested 侧
        `fit_top` 不返回 tail-ESS 与树深，voigt 侧无环支路。跳过门禁，
        既不算通过也不算拒绝，但必须在报表里列出来。
    """
    raw = c.get("raw") or {}
    is_v = c.get("family") == "voigt"
    g = dict.fromkeys(GUARD_NAMES, "not_evaluable")

    # ---- A 类：采样/LOO 健康（与 bm_grade.GUARD_CLASS 的 A 档同源）----
    rhat = c.get("rhat")
    g["rhat"] = ("not_evaluable" if not _finite(rhat) else
                 "triggered" if float(rhat) > RHAT_MAX else "ok")
    # `div` 缺失/非有限时不能折成 0 发散「通过」——那是把「没算出来」读成
    # 「很干净」。这里是 R̂=NaN 之外的第二条静默放行路径。
    dv = c.get("div")
    g["div_frac"] = ("not_evaluable" if not _finite(dv) else
                     "triggered" if float(dv) / max(n_draw, 1) > DIV_FRAC_MAX
                     else "ok")
    g["loo_ok"] = "ok" if c.get("ok", False) else "triggered"

    if is_v:
        b, t = raw.get("bulk_min"), raw.get("tail_min")
        g["ess_bulk"] = ("not_evaluable" if not _finite(b) else
                         "triggered" if float(b) <= ESS_MIN else "ok")
        g["ess_tail"] = ("not_evaluable" if not _finite(t) else
                         "triggered" if float(t) <= ESS_MIN else "ok")
        bk = raw.get("bad_k")
        g["bad_k"] = ("not_evaluable" if not _finite(bk) else
                      "triggered" if float(bk) > BAD_K_MAX else "ok")
        cap = (raw.get("tds") or {}).get("frac_cap")
        g["tree_cap"] = ("not_evaluable" if not _finite(cap) else
                         "triggered" if float(cap) >= TREE_CAP_MAX else "ok")
    else:
        e = raw.get("ess_min")
        # numpyro summary 的 n_eff ≈ bulk ESS；与 voigt 的 bulk_min 口径近似，
        # 但**不是同一个统计量**，报表里要带这条注记。
        g["ess_bulk"] = ("not_evaluable" if not _finite(e) else
                         "triggered" if float(e) <= ESS_MIN else "ok")
        # `fit_top` **不返回** tail-ESS 与树深触顶统计（返回 dict 里根本没有
        # `tail_min` / `tds` 这两个键）→ 属**结构性不提供**，不是本次数值
        # 算失败。两者必须分开：`not_applicable` 跳过门禁，
        # `not_evaluable` 才是「该有却缺」、按 fail-closed 拒绝。
        # 混成一类会让 strict 模式把整个 nested 族永久判死（B2 实测 80/80
        # 候选都命中），选择器就再也输出不了 nested。
        g["ess_tail"] = "not_applicable"
        g["tree_cap"] = "not_applicable"
        pk = raw.get("pareto_k")
        if pk is None:
            g["bad_k"] = "not_evaluable"
        else:
            frac = float(np.mean(np.asarray(pk, float) > 0.7))
            g["bad_k"] = "triggered" if frac > BAD_K_MAX else "ok"

    # ---- B 类：结构/参数可辨识性，**只记账** ----
    s = raw.get("s") or {}
    bf = _uniform_bound_frac(s)
    g["param_at_bound"] = ("not_evaluable" if bf is None else
                           "triggered" if bf > 0.05 else "ok")
    g["param_degenerate"] = ("triggered" if _corr_pairs(s) else "ok")

    if is_v:
        g["loop_amp"] = g["loop_corner"] = "not_applicable"
        # 分量未定位：只有 K≥2 才有「多余自由度」可判；K=1 记 not_applicable 而
        # 不是 ok——「不适用」和「通过了」不能混（§9.4.17.4 的同一条纪律）。
        ec = unresolved_components(s)
        if ec is None:
            g["comp_unresolved"] = "not_evaluable"
        elif ec["k"] < 2:
            g["comp_unresolved"] = "not_applicable"
        else:
            g["comp_unresolved"] = "triggered" if ec["unresolved"] else "ok"
    else:
        # nested 无 stick-breaking 权重，「冗余分量」的判据口径未定（M5）；
        # 结构性不提供 ≠ 通过，故 not_applicable 而不是 ok。
        g["comp_unresolved"] = "not_applicable"
        sig = raw.get("sigma_im") or raw.get("sigma_re")
        R = next((np.asarray(s[k], float).ravel() for k in ("R_ad", "R_L")
                  if k in s), None)
        tau = next((np.asarray(s[k], float).ravel() for k in ("tau_ad", "tau_L")
                    if k in s), None)
        g["loop_amp"] = ("not_evaluable" if R is None or not _finite(sig) else
                         "triggered" if float(np.mean(R < 3 * float(sig))) > 0.5
                         else "ok")
        if tau is None or f_max is None:
            g["loop_corner"] = "not_evaluable"
        else:
            f_ad = 1.0 / (2 * np.pi * np.maximum(tau, 1e-300))
            g["loop_corner"] = ("triggered" if float(np.mean(f_ad > f_max)) > 0.5
                                else "ok")
    return g


def _quality(c: dict, n_draw: int, strict: bool = False) -> tuple[bool, str]:
    """两侧同等的采样可用性判断——**唯一真值来源是 `guard_state()`**。

    旧实现自己另写一套判断，与 `guard_state` 不一致：`R̂=NaN` 时
    `float(nan) > RHAT_MAX` 为 False，于是**静默通过**（单链的 R̂ 恒 NaN，
    这条路径是现实的），而同一份输入在 `guard_state` 里被判
    `not_evaluable`。现在只做投影，不另立标准。

    默认只取 `rhat` / `div_frac`（A2 口径，保证 11-09 以来可比）；
    `strict=True` 追加 `ess_bulk` / `ess_tail` / `bad_k` / `tree_cap`。

    状态语义（四者不能混）：
      · `ok`             → 通过
      · `triggered`      → 拒绝
      · `not_evaluable`  → **拒绝**（该有却缺：R̂=NaN、div 缺失、LOO 点缺失）
      · `not_applicable` → 跳过（该实现结构性没有这个量，既不通过也不拒绝）
    """
    if not c.get("ok", False):
        return False, "LOO 不可用"
    gs = guard_state(c, n_draw)
    keys = (["rhat", "div_frac"] +
            (["ess_bulk", "ess_tail", "bad_k", "tree_cap"] if strict else []))
    bad, unknown = [], []
    for k in keys:
        st = gs.get(k, "not_evaluable")
        if st == "triggered":
            # R̂ / div 保留原来的数值格式，历史日志可以逐字对照
            if k == "rhat":
                bad.append(f"R̂={float(c['rhat']):.3f}>{RHAT_MAX}")
            elif k == "div_frac":
                frac = float(c["div"]) / max(n_draw, 1)
                bad.append(f"div={c['div']}({frac:.1%})")
            else:
                bad.append(f"guard:{k}")
        elif st == "not_evaluable":
            unknown.append("R̂" if k == "rhat" else
                           ("div" if k == "div_frac" else k))
    if bad:
        return False, "; ".join(bad)
    if unknown:
        return False, "指标缺失:" + ",".join(unknown)
    return True, ""


def _refit(c: dict, f, Z, *, seed: int, kw: dict,
           sigma_floor: float = SIGMA_FLOOR) -> dict:
    """按候选自身的 `spec`（K 或 branch/L）重拟合，用于族内重跑。

    重跑必须**原样复现该候选的族能力**（`include_L` / `adsorption`），否则整批
    重跑会在同一组里混进两种候选空间——那比「只补失败者」更糟：它造出的不对称
    是结构性的，不是采样量级的。
    """
    if c["family"] == "voigt":
        return fit_voigt(f, Z, c["dof"], seed=seed, kw=kw,
                         include_L=bool(c.get("include_L", VOIGT_CAPS["include_L"])),
                         adsorption=c.get("adsorption", VOIGT_CAPS["adsorption"]),
                         sigma_floor=sigma_floor)
    return fit_nested(f, Z, c["dof"], branch=c["branch"], seed=seed,
                      sigma_floor=sigma_floor, kw=kw)


def _family_state(cands: list[dict], n_draw: int,
                  strict: bool = False,
                  gate_b: bool = False) -> tuple[dict | None, bool, str]:
    """族内取最优，同时判断**采样失败有没有污染该族的比较基准**。

    返回 `(best, degraded, why)`。旧实现只做「剔除不合格 → 取剩余里 elpd 最大」，
    这在候选池大小不对称时会制造跨族偏差：例如 voigt 的 K=2 因 R̂=1.84 被剔除后，
    族内只剩 K=1 参赛，nested 却仍满配参赛，于是 Δelpd 被凭空抬高 65 nats，
    把一条纯 voigt 谱判成 nested（实测 11-09 A 轮 `voigt_true` seed=5）。

    修法不改阈值、不改用例：只要「被剔除者的 elpd 高于参赛最优者 + DEGRADE_MARGIN」，
    就认为该族基准不可信，整族 `degraded`，最终输出**并列**（合规输出），
    而不是硬造一个 Δ。整族无可用候选时同样记 degraded，不再抛异常。

    `gate_b=True`（`--gate-comp`，§9.4.18）：把 `comp_unresolved` 也当不合格。
    **注意后果**：未定位的 K=2 往往 elpd 更高，于是 `DEGRADE_MARGIN` 会立刻生效、
    整族记 degraded → 输出并列。这是 fail-closed 的**设计意图**，不是 bug；
    但它会让 voigt 的命中数下降，所以必须单列清楚代价。
    """
    q = {c["label"]: _quality(c, n_draw, strict) for c in cands}
    ok = [c for c in cands if q[c["label"]][0]]
    if gate_b:
        blocked, ok2 = [], []
        for c in ok:
            st = c.get("comp_state")
            if st in ("triggered", "not_evaluable"):
                blocked.append(c)
            else:
                ok2.append(c)
        for c in blocked:
            q[c["label"]] = (False, f"comp_unresolved={c.get('comp_state')}"
                                    f"（k_eff={c.get('k_eff')}/{c.get('dof')}）")
        ok = ok2
    if not ok:
        why = "; ".join(f"{c['label']}→{q[c['label']][1]}" for c in cands)
        return None, True, f"族内无可用候选（{why}）"
    best = max(ok, key=lambda c: c["elpd"])
    lost = [c for c in cands
            if not q[c["label"]][0] and c["elpd"] > best["elpd"] + DEGRADE_MARGIN]
    if lost:
        why = "; ".join(
            f"{c['label']} elpd={c['elpd']:.2f}>参赛最优{best['label']} "
            f"{best['elpd']:.2f}（{q[c['label']][1]}）" for c in lost)
        return best, True, f"降配参赛：{why}"
    return best, False, ""


# --------------------------------------------------------------------------
# 配对比较
# --------------------------------------------------------------------------

def paired_diff(pt_a, pt_b, f=None) -> tuple[float, float]:
    """`Σ差` 与 `sqrt(n·Var(逐点差))`。

    两族必须在**同一批频率点**上：长度相等不等于网格相同，所以传 `f` 时
    额外校验「逐点 LOO 与频率网格一一对应」。只比长度会漏掉「两套同为 50
    点的不同网格」——那时配对差逐点错位，数值看着正常但毫无意义。
    """
    a = np.asarray(pt_a, float)
    b = np.asarray(pt_b, float)
    if a.size != b.size:
        raise ValueError(f"逐点 LOO 长度不一致（{a.size} vs {b.size}）：不同 f 网格不能配对")
    if f is not None and np.asarray(f, float).size != a.size:
        raise ValueError(f"逐点 LOO 长度 {a.size} 与频率网格 "
                         f"{np.asarray(f).size} 不一致：无法确认逐点对齐")
    d = a - b
    n = d.size
    var = float(np.var(d, ddof=1)) if n > 1 else float("nan")
    return float(d.sum()), float(np.sqrt(n * var))


def _compare_group(V: list[dict], N: list[dict], n_draw: int,
                   strict: bool, f=None, voigt_ok: bool = True,
                   voigt_why: str = "", gate_b: bool = False) -> dict:
    """一组「voigt 候选 vs nested 候选」的配对判定（2 vs 2 或 2 vs 4）。

    `voigt_ok=False`（见 `voigt_can_represent`）时，**只把「否则会判并列」的那一档**
    升级成 `voigt_inadmissible`：
      · Δ 显著指向 nested → 仍报 `nested`（正向信息要保留，不能因为 voigt 缺席就丢掉）；
      · Δ 显著指向 voigt → 仍报 `voigt`（这种情形本身可疑，留给上层报警，不静默改判）；
      · Δ 不显著 → `voigt_inadmissible`：**并列的原因是 voigt 结构上不在候选空间**，
        不是"两族等价"。这两句的可操作后果不同：前者是**B-缺项**（可指认，指向换族
        定义或排除该族，**加采样无效**），后者才是 C 档。
        注意「voigt 被排除」≠「nested 是对的」——机制仍无外部证据。
    """
    v, dv, why_v = _family_state(V, n_draw, strict, gate_b=gate_b)
    n, dn, why_n = _family_state(N, n_draw, strict, gate_b=gate_b)
    why = "; ".join(x for x in (why_v, why_n) if x)
    kn = _k_note(v)
    # R21：每一条判定都带「先验尺度数据依赖」标记，别让下游把 Δ 当标准 LOO
    stamp = dict(elpd_prior_scale=ELPD_PRIOR_SCALE,
                 elpd_caveat=ELPD_PRIOR_SCALE_NOTE)
    if dv or dn or v is None or n is None:
        # 比较基准被采样失败污染 → **不做配对**，输出合规的「并列」。
        return dict(decision="indistinguishable", d_elpd=None, dse=None, z=None,
                    voigt=v, nested=n, degraded=True,
                    degraded_why=why or "族内无可用候选", reason="", note="",
                    k_note=kn, **stamp)
    d_elpd, dse = paired_diff(n["pointwise"], v["pointwise"], f=f)
    # dse 不可用（n=1、逐点差方差恰为 0）时**两个方向都不能判**。
    # 旧实现把 z 设成 +inf：正向 Δ≥4 会无条件切 nested（绕过显著性门槛），
    # 反向 -z=-inf 又永远切不动 voigt——同一份数据两个方向不对称。
    if not _finite(dse) or float(dse) <= 0:
        return dict(decision="indistinguishable", d_elpd=None, dse=None, z=None,
                    voigt=v, nested=n, degraded=False, degraded_why="",
                    reason=f"dse 不可用（{dse!r}）：逐点差方差为 0 或非有限，"
                           "无法做显著性判定，两方向一律并列", note="",
                    k_note=kn, **stamp)
    z = d_elpd / dse
    sw = (d_elpd >= SWITCH_DELTA and z >= SWITCH_Z)
    bk = (-d_elpd >= SWITCH_DELTA and -z >= SWITCH_Z)
    if sw:
        dec, why = "nested", ""
    elif bk:
        dec, why = "voigt", ""
    elif not voigt_ok:
        dec, why = "voigt_inadmissible", f"voigt 族结构上不可用：{voigt_why}"
    else:
        dec, why = "indistinguishable", ""
    # `reason` 专指「**算不出**」（dse 不可用之类），会进 `_health.non_evaluable`；
    # `voigt_inadmissible` 是**算得出但意义不同**，必须放 `note`，否则会被读成
    # 「这次没算出来」。两者混一起是 §9.4.6 记过的同类错误。
    return dict(decision=dec, d_elpd=d_elpd, dse=dse, z=z, voigt=v, nested=n,
                degraded=False, degraded_why="",
                reason="", note=why if dec == "voigt_inadmissible" else "",
                k_note=kn, **stamp)


def _k_note(v: dict | None) -> str:
    """「K 是 elpd 选的，但分量没全定位」的提示（只记账，不改决策）。

    §9.4.18：`k_eff < K` 说明多出来的那几个分量没有被数据定住。
    这条**不能**替代 `decision`，也不能塞进 `reason`（那是「算不出」的字段）；
    它是给报告用的一句话，`k_eff` 的数值在候选里。
    """
    if not v or v.get("family") != "voigt":
        return ""
    k, ke = v.get("dof"), v.get("k_eff")
    if ke is None or k is None or int(ke) >= int(k):
        return ""
    return (f"K={k} 由 elpd 选出，但只有 {ke} 个分量被定位"
            f"（comp_unresolved={v.get('comp_state')}）→ K 不得读成时间常数个数")


def _select_one(f, Z, *, krange=(1, 2), layers=(1, 2), branches=BRANCHES,
                seed: int = 0, sigma_floor: float = SIGMA_FLOOR,
                kw: dict | None = None, trace: bool = False,
                pair_mode: str = "pooled", retry: int = 0,
                strict: bool = False, retry_kw: dict | None = None,
                include_L: bool | None = None,
                adsorption=_UNSET, gate_b: bool = False) -> dict:
    """单个 seed 的完整判定。

    `include_L` / `adsorption` 是 **voigt 族的元件能力**，默认取 `VOIGT_CAPS`
    （= 基线口径，两项全关）。打开它们等于扩大 voigt 的候选空间，**必须作为
    新臂单独预注册**（§9.4.11）并与基线分开报数，不能混成一批。

    为什么要显式传：`voigt_inadmissible` 判据与 `fit_voigt` 的实际调用必须是
    **同一套能力**。判据说「表示不了」而拟合其实开了那个开关，输出就会自相矛盾。

    `pair_mode="per-branch"` 时把 nested 拆成每条支路一组（`L=1,2`，2 个候选），
    分别与 voigt 的 `K=1,2`（2 个候选）配对 → **候选池严格 2 vs 2**；
    族级结论只在两组给出同一判定时成立，否则 `unstable`（不做多数表决）。

    `retry>0` 时，若任一组的比较基准被采样质量污染，则**整批** voigt+nested
    候选统一换到 `standard` 档、换 seed 重跑。整批重跑而不是只补失败者，
    是为了不让同一组里出现 quick / standard 两种采样预算（那会造出新的不对称）。
    """
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"pair_mode 只能是 {PAIR_MODES}，收到 {pair_mode!r}")
    kw = dict(QUICK if kw is None else kw)
    rkw = dict(BUDGETS["standard"] if retry_kw is None else retry_kw)
    n_draw = int(kw["num_chains"]) * int(kw["n_samples"])
    bs = _as_branches(branches)
    # 族能力先归一化到具体值：`None` 对 include_L 是「取默认」，对 adsorption
    # 不是合法取值（合法的是 None=关 / "RL"=开），所以这里统一用 `_UNSET` 区分。
    if include_L is None:
        include_L = bool(VOIGT_CAPS["include_L"])
    if adsorption is _UNSET:
        adsorption = VOIGT_CAPS["adsorption"]
    # 谱形级判据，与采样无关，算一次即可。它决定「否则会判并列」的那一档要不要
    # 升级成 `voigt_inadmissible`；不影响是否做配对，也不动 SWITCH_*。
    # **必须用这次运行的真实能力算**，否则判据与拟合会各说各话。
    voigt_ok, voigt_why = voigt_can_represent(f, Z, include_L=include_L,
                                              adsorption=adsorption)

    V = [fit_voigt(f, Z, K, seed=seed, sigma_floor=sigma_floor, kw=kw,
                   include_L=include_L, adsorption=adsorption) for K in krange]
    N = [fit_nested(f, Z, L, branch=br, seed=seed,
                    sigma_floor=sigma_floor, kw=kw)
         for br in bs for L in layers]
    if trace:
        print("    " + " | ".join(f"{c['label']} elpd={c['elpd']:.1f}"
                                  f"(R̂{c['rhat']:.2f},div{c['div']})" for c in V + N))

    def evaluate(Vc, Nc, nd):
        groups = ([("pooled", Nc)] if pair_mode == "pooled" else
                  [(br, [c for c in Nc if c["branch"] == br]) for br in bs])
        return [dict(group=name, **_compare_group(Vc, sub, nd, strict, f,
                                                  voigt_ok=voigt_ok,
                                                  voigt_why=voigt_why,
                                                  gate_b=gate_b))
                for name, sub in groups]

    gres = evaluate(V, N, n_draw)
    eff_seed = int(seed)
    retry_log = []
    for attempt in range(1, int(retry) + 1):
        if not any(g["degraded"] for g in gres):
            break
        before = [f"{g['group']}:{g['degraded_why']}" for g in gres if g["degraded"]]
        s2 = int(seed) + RETRY_SEED_STEP * attempt
        eff_seed = s2
        V = [_refit(c, f, Z, seed=s2, kw=rkw, sigma_floor=sigma_floor) for c in V]
        N = [_refit(c, f, Z, seed=s2, kw=rkw, sigma_floor=sigma_floor) for c in N]
        n_draw = int(rkw["num_chains"]) * int(rkw["n_samples"])
        gres = evaluate(V, N, n_draw)
        after = [f"{g['group']}:{g['degraded_why']}" for g in gres if g["degraded"]]
        retry_log.append(dict(attempt=attempt, seed=s2,
                              warmup=int(rkw["n_warmup"]),
                              samples=int(rkw["n_samples"]),
                              before=before, after=after,
                              rescued=len(after) < len(before)))

    decisions = [g["decision"] for g in gres]
    decision = decisions[0] if len(set(decisions)) == 1 else "unstable"
    degraded = any(g["degraded"] for g in gres)
    why = "; ".join(f"{g['group']}: {g['degraded_why']}"
                    for g in gres if g["degraded"])
    # 「算不出」与「降级」分开报：前者不是采样失败，不能记成 degraded，
    # 但同样不能拿某一组的 Δ 冒充整族差值。
    non_eval = "; ".join(f"{g['group']}: {g['reason']}"
                         for g in gres if g.get("reason"))
    ns = [g["nested"] for g in gres if g["nested"] is not None]
    nbest = max(ns, key=lambda c: c["elpd"]) if ns else None
    first = gres[0]
    # 只要有一组降级或不可算，就不展示 Δ：per-branch 下 first["d_elpd"] 只
    # 代表 parallel 那一组，打印成整族配对差会被误读。逐组值留在 groups 里。
    no_delta = (decision == "unstable" or degraded
                or any(g.get("reason") for g in gres))
    return dict(seed=seed, effective_seed=eff_seed, decision=decision,
                d_elpd=None if no_delta else first["d_elpd"],
                dse=None if no_delta else first["dse"],
                z=None if no_delta else first["z"],
                degraded=degraded, degraded_why=why,
                non_evaluable_why=non_eval,
                voigt=first["voigt"], nested=nbest,
                all_voigt=V, all_nested=N, n_draw=n_draw,
                voigt_ok=voigt_ok, voigt_note=voigt_why,
                voigt_caps=dict(include_L=bool(include_L), adsorption=adsorption),
                pair_mode=pair_mode, strict=strict,
                groups=[{k: v for k, v in g.items() if k != "voigt" and k != "nested"}
                        for g in gres],
                retry_log=retry_log)


def select_across_families(f, Z, *, n_seeds: int = 3, **kw) -> dict:
    """多 seed 聚合的跨族判定。

    为什么要多 seed：**同一个用例换一条随机流就能翻转判定**。实测
    `voigt_true` 在 seed=1 时 Δelpd=+0.47（不可分），换一次采样 nested 的
    `L=2` 直接 R̂=17.3/div=51 被判不可用，Δ 变成 −27.2（选 voigt）。
    单 seed 的判定是采样噪声，不是结论。

    `decision` 只在**所有 seed 一致**时给出；否则记 `unstable`，并且
    `decisions` 里保留逐 seed 结果，不做多数表决（3 个 seed 的多数表决没有
    统计意义，反而会掩盖不稳定）。
    """
    runs = [_select_one(f, Z, seed=int(kw.get("seed", 0)) + i,
                        **{k: v for k, v in kw.items() if k != "seed"})
            for i in range(max(1, int(n_seeds)))]
    decisions = [r["decision"] for r in runs]
    decision = decisions[0] if len(set(decisions)) == 1 else "unstable"

    best = runs[0] if decision != "unstable" else None
    bs = _as_branches(kw.get("branches", BRANCHES))
    pm = kw.get("pair_mode", "pooled")
    # 本次运行**实际**用的 voigt 能力（不是模块默认）。注意 `_UNSET` 是哨兵，
    # 不能直接当 `adsorption=None` 显示。
    _il = kw.get("include_L", VOIGT_CAPS["include_L"])
    _il = bool(VOIGT_CAPS["include_L"] if _il is None else _il)
    _ad = kw.get("adsorption", VOIGT_CAPS["adsorption"])
    _ad = VOIGT_CAPS["adsorption"] if _ad is _UNSET else _ad
    _caps = _caps_suffix(_il, _ad) or "（基线，两项全关）"
    notes = [
        "合成谱：真值来自生成器，不是实测谱",
        "两族 σ 均由后验采样（sigma=None），未固定；elpd 无跨口径可比性",
        "族内选 K/L 与跨族比较共用同一份 LOO，Δelpd 偏乐观",
        "两族共用同一张 guard 清单（GUARD_NAMES）；`not_evaluable` 记「该有却缺」"
        "并按 fail-closed 拒绝，`not_applicable` 记「该实现结构性不提供」并跳过门禁，"
        "两者不混。B 类 guard 属参数可辨识性，**只记账不进族级门禁**",
        (f"配对模式 {pm}：nested 按支路拆成 2 候选一组，与 voigt 2 候选严格对称"
         if pm == "per-branch" else
         f"配对模式 {pm}：nested 各支路合并成 {2 * len(bs)} 候选与 voigt 2 候选比，"
         "候选池大小不对称（本模式只为与 A2 逐位可比而保留）"),
        "「不可算」（dse 不可用、指标缺失）与「降级」（采样失败）分开记；"
        "两者都只输出并列，都不能挑某个 seed 当结论",
        f"nested 族扫了支路 {list(bs)}：选中哪条支路属**族内**结果，"
        "不参与族级门禁（选错支路 ≠ 把族认反）",
        "**族级判定 ≠ 参数可识别**：判成 nested 只说明该族预测更好，"
        "不代表 R_ad/tau_ad 或 R1/R2 可分开（M3 §19.2 已证 R1/R2 共线）",
        f"voigt 族能力 {_caps}（include_L={_il}, adsorption={_ad!r}）："
        "谱里出现该能力覆盖不到的感抗时，「否则会判并列」的那一档输出 "
        "voigt_inadmissible（族不在候选空间），不是「两族等价」。"
        "依据见 docs/REFS_DIGEST.md §2.I ⑥；能力臂的预注册见 §9.4.11",
    ]
    if kw.get("retry", 0):
        notes.append(f"族内重跑 {kw['retry']} 次（standard 档、整批换 seed）；"
                     "重跑只改采样预算，不改判定阈值与用例")
    if kw.get("strict", False):
        notes.append("strict-guards 开启：ESS / Pareto-k 等 A 类 guard 已接入可用性门禁")
    if decision == "indistinguishable":
        notes.insert(0, "不可分 ≠ 两族等价，也 ≠ nested 不成立（B 档）")
    if decision == "unstable":
        notes.insert(0, f"{len(set(decisions))} 个 seed 给出不同判定 "
                        f"{decisions}：该谱在本预算下**不可稳定分族**，"
                        "不许挑其中一个 seed 当结论")
    deg = [r["seed"] for r in runs if r["degraded"]]
    if deg:
        notes.insert(0, f"seed {deg} 出现族内降配参赛（采样质量剔除改变了族内最优），"
                        "这些 seed 不做配对、直接输出并列；详见 runs[*].degraded_why")

    out = dict(decision=decision, decisions=decisions, n_seeds=len(runs),
               degraded_seeds=deg, thresholds=dict(delta=SWITCH_DELTA, z=SWITCH_Z),
               voigt_caps=dict(include_L=_il, adsorption=_ad),
               not_=notes, runs=runs)
    if best is not None:
        out.update(d_elpd=best["d_elpd"], dse=best["dse"], z=best["z"],
                   voigt=best["voigt"], nested=best["nested"],
                   all_voigt=best["all_voigt"], all_nested=best["all_nested"])
    else:                       # 不稳定时把这些字段留 None，别让人误读成"平均"
        out.update(d_elpd=None, dse=None, z=None, voigt=None, nested=None,
                   all_voigt=None, all_nested=None)
    return out


# --------------------------------------------------------------------------
# 自检：四类合成谱
# --------------------------------------------------------------------------

def _cases(noise: float = 0.01, seed: int = 1) -> list[tuple]:
    """四类用例：`(tag, expect, note, f, Z)`。

    **修正记录（第一次跑完就改，一并留档）**
    初版把 `loop_out_of_window` 与 `arcs_merged` 的期望写成"不得选 nested"。
    跑出来两条都是 nested（Δelpd=88.5 / 282.7），核对后确认**是初版期望写错**：

    · 两例的真值都是 nested。`REAL` 的 `R_ad=3000`、`tau_ad=1.6 s` →
      `f_ad=0.0995 Hz`，本来就在窗内；`arcs_merged` 只改 `Q1/n1/n2`，
      环原封不动（`L_ad=4800 H`）；
    · `loop_out_of_window` 把 `f_ad` 推到 0.003 Hz（窗外），但 `L_ad=1.6e5 H`，
      窗内留的是**大幅值感抗尾**——voigt 不带 L 项时根本没有这个形状。

    所以"该选 nested"是对的。初版把**参数级不可识别**（`R_ad`/`tau_ad` 定不住、
    `R1/R2` 只能报和）错当成了**族级不该选**；这条区分现在写进
    `select_across_families` 的 `not_`，不再用族级判定冒充参数结论。
    """
    f = G.real_grid()
    fv, Zv = B.true_spectrum(f)
    cases = [("voigt_true", "voigt",
              "真值即 voigt（两弧）；nested 若也能贴合 1% 噪声，说明两族结构重叠",
              fv, G.add_noise(Zv, noise, seed=seed))]

    Zn = G.gen_parallel_L2(f, dict(G.REAL))
    cases.append(("nested_true", "nested", "真值 nested，环在窗内（f_ad=0.0995 Hz）",
                  f, G.add_noise(Zn, noise, seed=seed)))

    for c, note in (
            ("loop_out_of_window",
             "真值 nested；环推出窗外，但窗内剩 82% |Z(f_min)| 的感抗尾"),
            ("arcs_merged",
             "真值 nested 且环仍在窗内；R1/R2 共线是**参数级**限制，不影响族级")):
        d = G.gen_mixed_boundary(c, f=f, noise=noise, seed=seed)
        cases.append((c, "nested", note, d["f"], d["Z"]))
    return cases


def _slim_runs(runs: list[dict], f_max: float | None = None,
               strict: bool = False) -> list[dict]:
    """压成可序列化明细：候选质量 + **两族同一张 guard 清单** + 重跑记录。

    不落 posterior / idata 大对象；`not_evaluable` 原样保留，方便复核缺口。
    """
    fin = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 3)
    out = []
    for r in runs:
        nd = r.get("n_draw") or QUICK["num_chains"] * QUICK["n_samples"]
        cands = []
        for c in r["all_voigt"] + r["all_nested"]:
            usable, why = _quality(c, nd, strict)
            cands.append(dict(family=c["family"], label=c["label"],
                              dof=c.get("dof"),
                              elpd=round(c["elpd"], 2), rhat=fin(c["rhat"]),
                              div=c["div"], usable=bool(usable), why=why,
                              # K 与「实际定位到几个分量」分开记（§9.4.18）：
                              # `k_eff` 只对 voigt 有定义，nested 记 None。
                              k_eff=c.get("k_eff"),
                              comp_state=c.get("comp_state"),
                              comp_span=[fin(x) for x in (c.get("comp_span") or [])] or None,
                              comp_amp=[fin(x) for x in (c.get("comp_amp") or [])] or None,
                              guards=guard_state(c, nd, f_max)))
        out.append(dict(seed=r["seed"],
                        effective_seed=r.get("effective_seed", r["seed"]),
                        decision=r["decision"],
                        d_elpd=fin(r["d_elpd"]), dse=fin(r["dse"]), z=fin(r["z"]),
                        degraded=r["degraded"], degraded_why=r["degraded_why"],
                        non_evaluable_why=r.get("non_evaluable_why", ""),
                        n_draw=nd, pair_mode=r.get("pair_mode"),
                        groups=r.get("groups"), retry_log=r.get("retry_log"),
                        candidates=cands))
    return out


def guard_summary(rows: list[dict]) -> dict:
    """两族 guard 状态的汇总计数，用来量化还剩多少 `not_evaluable`。"""
    acc = {fam: {s: 0 for s in ("ok", "triggered", "not_evaluable",
                                "not_applicable")}
           for fam in ("voigt", "nested")}
    for row in rows:
        for tr in row.get("candidate_trace") or []:
            for c in tr.get("candidates") or []:
                for name, val in (c.get("guards") or {}).items():
                    if name in GUARD_A:       # 只汇总 A 类：B 类本来就不进门禁
                        acc[c["family"]][val] = acc[c["family"]].get(val, 0) + 1
    return acc


def _health(rows: list[dict]) -> dict:
    """把「门禁」与「健康度」拆开：rc 只回答有没有错切，健康度回答能不能用。

    B2 的 rc=0 同时带着 8/20 降级与 2 例 unstable。混成单一退出码会让下游
    （CI、报告、别人引用）把「没认反族」读成「这个选择器可以用了」。
    """
    wrong = sum(int(r.get("wrong") or 0) for r in rows)
    return dict(
        cases=len(rows),
        n_eval=sum(len(r.get("decisions") or []) for r in rows),
        wrong=wrong,
        degraded=sum(len(r.get("degraded_seeds") or []) for r in rows),
        unstable_cases=sum(1 for r in rows if r.get("decision") == "unstable"),
        voigt_inadmissible=sum(1 for r in rows
                               if r.get("decision") == "voigt_inadmissible"),
        non_evaluable=sum(1 for r in rows
                          for tr in (r.get("candidate_trace") or [])
                          for g in (tr.get("groups") or []) if g.get("reason")),
        gate_ok=(wrong == 0),
        note="rc 只代表错切，不代表通过；判据是否可用看降级/unstable/不可算",
    )


def _verdict(expect: str, decisions: list[str]) -> tuple[bool, int, int]:
    """返回 `(是否无错切, 命中数, 错切数)`。

    这里的口径按预注册那句「**零错误切换**」落实，分两件事报，不能混：

    · **错切**（门禁）：nested 真值被判 voigt，或 voigt 真值被判 nested。
      这两种都是把族认反了，必须为 0。
    · **命中**（只报数，不设门槛）：判到正确族的 seed 数。判成
      `indistinguishable` **不算错切**——按 `THREE_LAYER_SCOPE.md` §2，
      证据不足时的合规输出是「并列」，不是「默认用 voigt」。
    · **`voigt_inadmissible`** 同样两者都不算：它说的是「voigt 结构上不在候选
      空间」，比「并列」更弱，更不能计作命中。若真值就是 voigt 却出现这个终态，
      说明 `voigt_can_represent` 的判据误伤，该查 `IND_TOL_SPEC` 与能力常量。

    初版把"必须正向识别"当成验收条件，与预注册的"零错误切换"不一致，
    会让"证据不足"被记成错误；现按上表分开。
    """
    if expect == "voigt":
        return ("nested" not in decisions), decisions.count("voigt"), decisions.count("nested")
    return ("voigt" not in decisions), decisions.count("nested"), decisions.count("voigt")


def _selftest(noise: float = 0.01, seed: int = 1, n_seeds: int = 3,
              branches=(BRANCH,), pair_mode: str = "pooled",
              retry: int = 0, strict: bool = False,
              budget: str = "quick",
              retry_budget: str = "standard",
              include_L=None, adsorption=_UNSET,
              gate_b: bool = False) -> tuple[int, list[dict]]:
    # `VOIGT_CAPS` 已改为**由 `fit_voigt` 签名推导**（见 `_usable` 之后），
    # 所以「常量与签名漂移」在结构上不可能发生。剩下的唯一风险是
    # **参数加了但没转发给 `R.fit_one`**——那时判据说「表示不了」、拟合其实也没开，
    # 两者恰好一致，错会藏起来。下面用源码做一次廉价核对。
    _sv = inspect.signature(fit_voigt).parameters
    assert _sv["include_L"].default == VOIGT_CAPS["include_L"], (
        f"VOIGT_CAPS.include_L={VOIGT_CAPS['include_L']} 与 "
        f"fit_voigt 默认 {_sv['include_L'].default} 不一致")
    assert _sv["adsorption"].default == VOIGT_CAPS["adsorption"], (
        f"VOIGT_CAPS.adsorption={VOIGT_CAPS['adsorption']} 与 "
        f"fit_voigt 默认 {_sv['adsorption'].default} 不一致")
    assert "adsorption=adsorption" in inspect.getsource(fit_voigt), (
        "fit_voigt 收了 adsorption 参数但没转发给 R.fit_one：族能力开关会静默失效，"
        "voigt_inadmissible 判据也会跟着一起错")

    cases = _cases(noise=noise, seed=seed)
    bs = _as_branches(branches)
    print(f"四类合成谱，σ={noise:g}·max|Z|，seed={seed}×{n_seeds}，"
          f"判定阈值 Δelpd≥{SWITCH_DELTA:g} 且 z≥{SWITCH_Z:g}（全 seed 一致才给结论）")
    print(f"nested 族支路 = {list(bs)}（{len(bs) * 2} 个候选）；"
          "选到哪条支路属族内结果，不计入错切。")
    print(f"配对模式={pair_mode}  基础预算={budget}({BUDGETS[budget]['n_warmup']}/"
          f"{BUDGETS[budget]['n_samples']}/{BUDGETS[budget]['num_chains']})  "
          f"族内重跑={retry}({retry_budget})  strict-guards={strict}  "
          f"gate-comp={gate_b}")
    _il_eff = bool(VOIGT_CAPS["include_L"] if include_L is None else include_L)
    _ad_eff = VOIGT_CAPS["adsorption"] if adsorption is _UNSET else adsorption
    print(f"voigt 族能力 {_caps_suffix(_il_eff, _ad_eff) or '（基线，两项全关）'}"
          "  ← 能力臂要与基线**分开报数**，不能混成一批（§9.4.11）")
    print("门禁 = 错切数（必须 0）；命中数只报数。判成 indistinguishable 按「并列」处理，不算错。")
    print(f"{'case':<22}{'n':>4}{'Δelpd':>10}{'dse':>8}{'z':>8}"
          f"{'voigt':>9}{'nested':>13}  {'decision':<18}{'expect':<9}{'命中':>7}{'错切':>5}")
    bad = 0
    rows = []
    for tag, expect, note, f, Z in cases:
        t0 = time.time()
        r = select_across_families(f, Z, seed=seed, n_seeds=n_seeds,
                                   branches=bs, pair_mode=pair_mode,
                                   retry=retry, strict=strict,
                                   include_L=_il_eff, adsorption=_ad_eff,
                                   kw=dict(BUDGETS[budget]),
                                   retry_kw=dict(BUDGETS[retry_budget]),
                                   gate_b=gate_b)
        ok, hits, wrong = _verdict(expect, r["decisions"])
        bad += (wrong > 0)
        # 逐 seed 的 nested 选中标签 → 支路分布（族内信息，只报数）
        nlab = [x["nested"]["label"] if x["nested"] else "-" for x in r["runs"]]
        rnd = lambda v: None if v is None else round(v, 2)
        rows.append(dict(case=tag, expect=expect, note=note,
                         voigt_caps=r["voigt_caps"],
                         decision=r["decision"], decisions=r["decisions"],
                         d_elpd=r["d_elpd"], dse=r["dse"], z=r["z"],
                         voigt=None if r["voigt"] is None else r["voigt"]["label"],
                         nested=None if r["nested"] is None else r["nested"]["label"],
                         nested_labels=nlab,
                         degraded_seeds=r["degraded_seeds"],
                         retry_rescued=sum(1 for x in r["runs"]
                                           for q in x.get("retry_log") or []
                                           if q.get("rescued")),
                         candidate_trace=_slim_runs(r["runs"], float(f.max()), strict),
                         d_per_seed=[rnd(x["d_elpd"]) for x in r["runs"]],
                         z_per_seed=[rnd(x["z"]) for x in r["runs"]],
                         hits=int(hits), wrong=int(wrong), ok=bool(ok),
                         elapsed_s=round(time.time() - t0, 1)))
        fmt = lambda x, w=10, p=2: ("-" if x is None else f"{x:>{w}.{p}f}")
        print(f"{tag:<22}{f.size:>4}{fmt(r['d_elpd'])}{fmt(r['dse'],8)}"
              f"{fmt(r['z'],8)}{(r['voigt']['label'] if r['voigt'] else '-'):>9}"
              f"{(r['nested']['label'] if r['nested'] else '-'):>13}  "
              f"{r['decision']:<18}{expect:<9}{f'{hits}/{n_seeds}':>7}{wrong:>5}"
              f"{'' if ok else '  ✗错切'}")
        def _seed_txt(dec, x, y, zz):
            num = "Δ=不做配对" if x is None else f"Δ={x:+.2f},z={y:+.2f}"
            return f"{dec}({num}){(' [降级]' if zz else '')}"

        print(f"{'':<22}逐 seed: " + " | ".join(
            _seed_txt(d, x, y, zz["degraded"]) for d, x, y, zz in
            zip(r["decisions"], rows[-1]["d_per_seed"],
                rows[-1]["z_per_seed"], r["runs"]))
            + f"   支路={[s.split('/')[0] for s in nlab]}")
        if r["degraded_seeds"]:
            for x in r["runs"]:
                if x["degraded"]:
                    print(f"{'':<22}↳ seed={x['seed']} 降级原因：{x['degraded_why']}")

    print()
    if bad:
        print(f"**有 {bad} 条出现错切（把族认反）**：先查 guard 与似然口径，"
              "不要靠调阈值掩盖。")
    else:
        print(f"零错切（{len(cases)} 例 × {n_seeds} seed 全无认反）。"
              "这只说明「在这套生成分布上不认反族」，不说明真实谱能切对。")
    unstable = [r["case"] for r in rows if r["decision"] == "unstable"]
    if unstable:
        print(f"其中 {len(unstable)} 例 seed 间不稳定：{unstable} → 输出口径是「并列」，"
              "不得取任一 seed 的结果当结论。")
    deg_total = sum(len(r["degraded_seeds"]) for r in rows)
    rescued = sum(r["retry_rescued"] for r in rows)
    print(f"降级 seed 合计 {deg_total}/{len(cases) * n_seeds}"
          f"（A2 基线 7/20）；族内重跑救回 {rescued} 次。")
    if h_inad := [r["case"] for r in rows
                  if r["decision"] == "voigt_inadmissible"]:
        print(f"voigt_inadmissible {len(h_inad)} 例：{h_inad} → "
              "并列的原因是 **voigt 族结构上表示不了这条谱**（B-缺项），不是两族等价；"
              "处理方向是换族定义或排除该族，**加采样无效**。"
              "另注：开 adsorption/include_L 也补不上——元件接点不同，见 §9.4.12。")
    for fam, d in guard_summary(rows).items():
        print(f"  {fam:<7}A 类 guard：ok={d['ok']} triggered={d['triggered']} "
              f"not_evaluable={d['not_evaluable']} not_applicable={d['not_applicable']}")
    # 分量定位（B 类，§9.4.18）：K 与 k_eff 必须分开报，二者相等才算「K 用满了」。
    comp: dict[str, int] = {}
    under: set[str] = set()
    for r in rows:
        for tr in r.get("candidate_trace") or []:
            for c in tr.get("candidates") or []:
                if c["family"] != "voigt":
                    continue
                st = c.get("comp_state")
                if st:
                    comp[st] = comp.get(st, 0) + 1
                ke, kk = c.get("k_eff"), c.get("dof")
                if ke is not None and kk and int(ke) < int(kk):
                    under.add(f"{r['case']}:{c['label']}(k_eff={ke})")
    if comp:
        print("  voigt 候选 comp_unresolved：" +
              "  ".join(f"{k}={v}" for k, v in sorted(comp.items())) +
              ("" if gate_b else "   ← 只记账，未进门禁"))
    if under:
        print(f"  **K 未用满**（k_eff<K，K 不得读成时间常数个数）："
              f"{sorted(under)}")
        if not gate_b:
            print("    这些候选仍可参赛；要它影响判定请显式加 --gate-comp"
                  "（§9.4.18，代价=降配护栏会把整族打成并列）")
    h = _health(rows)
    print(f"rc={1 if bad else 0} 只代表「有无错切」，**不代表通过**。健康度："
          f"降级 {h['degraded']}/{h['n_eval']}、unstable {h['unstable_cases']}/"
          f"{h['cases']} 例、dse 不可算 {h['non_evaluable']} 次。")
    print("提醒：elpd 只作方向筛选，不作严格结论；B 类 guard 只记账不进门禁。")
    return (1 if bad else 0), rows


# --------------------------------------------------------------------------
# 真实谱：可表示性诊断（秒级，不拟合）
# --------------------------------------------------------------------------

def real_diagnose(paths, *, include_L=None, adsorption=_UNSET,
                  verbose: bool = True) -> list[dict]:
    """对一批真实谱逐条判断 **voigt 族在结构上能否表示它**。

    只做谱形判据（`voigt_can_represent`），**不拟合、不采样**，所以是秒级。
    这回答的是「这条谱该不该放进跨族比较」，不是「哪族更好」——后者要拟合，
    且按 `THREE_LAYER_SCOPE.md` 最多记 B 档。

    之所以单独做这个入口：选择器原先只能跑 `_cases()` 的四个合成用例，
    对用户手上的谱没有入口。而真实谱的结论**恰恰不需要拟合**就能给出最要紧
    的那一半——哪一族根本进不了候选空间（B-缺项）。
    """
    include_L = VOIGT_CAPS["include_L"] if include_L is None else include_L
    if adsorption is _UNSET:
        adsorption = VOIGT_CAPS["adsorption"]

    rows = []
    for path in paths:
        try:
            # `load_spectrum` 会往 stdout 打「读入 N 点」，表格里不需要
            with contextlib.redirect_stdout(io.StringIO()):
                f, Z = R.load_spectrum(path)
        except (Exception, SystemExit) as e:          # noqa: BLE001
            # P2-14（二轮审查）：原来是 except BaseException——把 Ctrl-C /
            # SystemExit 一起吞成"读取失败"继续循环，批量诊断无法中止。
            # SystemExit（load_spectrum 的文件级报错）按主链路约定记失败
            # 继续；KeyboardInterrupt 仍向上传播。
            rows.append(dict(path=path, name=os.path.basename(path),
                             ok=False, why=f"读取失败：{type(e).__name__}: {e}",
                             n=0, f_min=float("nan"), f_max=float("nan"),
                             im_depth=float("nan"), n_ind=0,
                             n_ind_low=0, n_ind_high=0, split=float("nan")))
            continue

        f = np.asarray(f, float)
        Z = np.asarray(Z, complex)
        zmax = float(np.max(np.abs(Z))) or 1.0
        im_depth = float(np.max(Z.imag) / zmax)          # 只看正虚部
        fi = _ind_freqs(f, Z)
        split = float(np.sqrt(f.min() * f.max()))
        lo = int(np.sum(fi <= split))
        hi = int(fi.size - lo)
        ok, why = voigt_can_represent(f, Z, include_L=include_L,
                                      adsorption=adsorption)
        rows.append(dict(path=path, name=os.path.basename(path), ok=bool(ok),
                         why=why, n=int(f.size), f_min=float(f.min()),
                         f_max=float(f.max()), im_depth=im_depth,
                         n_ind=int(fi.size), n_ind_low=lo, n_ind_high=hi,
                         split=split))

    if verbose:
        def _tag(r):
            """一行能读完的判定标签。长理由放到汇总里，逐行重复会刷屏。"""
            if r["n"] == 0:
                return "读取失败"
            if r["ok"]:
                return "可"
            lo, hi = r["n_ind_low"] > 0, r["n_ind_high"] > 0
            return "低+高" if (lo and hi) else ("仅低" if lo else "仅高")

        caps = _caps_suffix(include_L, adsorption) or "（基线，两项全关）"
        print(f"voigt 族能力 = {caps}；感抗按几何中频 sqrt(f_min·f_max) 分低/高两段")
        print(f"{'file':<28}{'N':>4}  {'f/Hz':<13}{'Im+':>7}  {'低/高':<7}判定")
        for r in rows:
            if r["n"] == 0:
                print(f"{r['name']:<28}{'-':>4}  {'-':<13}{'-':>7}  {'-':<7}{r['why']}")
                continue
            fr = f"{r['f_min']:.3g}–{r['f_max']:.3g}"
            print(f"{r['name']:<28}{r['n']:>4}  {fr:<13}{r['im_depth']:>7.1%}  "
                  f"{r['n_ind_low']:>2}/{r['n_ind_high']:<4}  {_tag(r)}")

        cat: dict[str, int] = {}
        for r in rows:
            k = _tag(r)
            cat[k] = cat.get(k, 0) + 1
        print("\n判定分布：" + "；".join(f"{k} {v}/{len(rows)}"
                                     for k, v in sorted(cat.items())))
        if cat.get("仅低") or cat.get("低+高"):
            print("  仅低/低+高 = **低频感抗** → 需把 R∥L 放进第二并联节点"
                  "（= 改族定义；顶层 adsorption 不足以表示，§9.4.12）")
        if cat.get("仅高") or cat.get("低+高"):
            print("  仅高/低+高 = **高频仪器感抗** → 需 include_L，或按 A 条预裁那几点")
        if rows and not any(r["ok"] for r in rows):
            print("全部不可表示 → 全落在 B-缺项：**不是「两族等价」，是 voigt 不在候选"
                  "空间**；处理方向是换族定义或排除该族，加采样无效。")
        print("注：本条只看谱形，不拟合。nested 族两条支路对上述谱结构上均可用。")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="M4 跨族选择器（voigt vs nested）")
    ap.add_argument("--dry", action="store_true", help="只打印用例，不拟合")
    ap.add_argument("--real", default=None, metavar="GLOB",
                    help="真实谱诊断：对匹配的文件只判「voigt 能否表示这类谱」"
                         "（秒级、不拟合）。可逗号分隔多个 glob，例如 "
                         "'out/_m6_samples/z_csv/*.csv,archive/*.csv'")
    ap.add_argument("--noise", type=float, default=0.01, help="噪声 σ / max|Z|")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5,
                    help="每例跑几个 seed（<2 时判定会随采样流翻转，不建议）")
    ap.add_argument("--branches", default="parallel,series",
                    help="nested 族扫哪些支路，逗号分隔（parallel / series）")
    ap.add_argument("--pair-mode", choices=PAIR_MODES, default="pooled",
                    help="per-branch = nested 按支路拆成 2 候选一组，与 voigt 2 候选严格对称")
    ap.add_argument("--retry-degraded", type=int, default=0,
                    help="降级时整批换到 standard 预算重跑的次数（A 档动作）")
    ap.add_argument("--strict-guards", action="store_true",
                    help="把 ESS/Pareto-k 等 A 类 guard 也接进可用性门禁")
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="quick",
                    help="基础采样预算档（quick 与 A2 逐位可比）")
    ap.add_argument("--retry-budget", choices=sorted(BUDGETS), default="standard",
                    help="族内重跑用的预算档；light=1200/600，standard=2000/1000")
    ap.add_argument("--voigt-include-L", action="store_true",
                    help="voigt 族启用串联 jωL（只覆盖高频仪器感抗）。"
                         "属**族能力臂**，须与基线分开报数（§9.4.11）")
    ap.add_argument("--voigt-adsorption", choices=("none", "RL"), default="none",
                    help="voigt 族启用吸附支路 R·jωτ/(1+jωτ)（覆盖低频感抗）。"
                         "同上，属能力臂")
    ap.add_argument("--gate-comp", action="store_true",
                    help="把分量未定位（comp_unresolved）接进族内择优。"
                         "**会**触发 DEGRADE_MARGIN → 输出并列，代价见 §9.4.18.2")
    ap.add_argument("--json", default=None, help="把自检结果写到该路径")
    a = ap.parse_args(argv)
    branches = _as_branches([x.strip() for x in a.branches.split(",") if x.strip()])
    v_il = bool(a.voigt_include_L)
    v_ad = None if a.voigt_adsorption == "none" else a.voigt_adsorption
    v_caps = _caps_suffix(v_il, v_ad) or "（基线，两项全关）"

    if a.real:
        pats = [x.strip() for x in a.real.split(",") if x.strip()]
        paths, miss = [], []
        for pat in pats:
            hit = sorted(_glob.glob(pat))
            if hit:
                paths.extend(hit)
            else:
                miss.append(pat)
        if miss:
            print(f"未匹配到文件：{miss}", file=sys.stderr)
        if not paths:
            print("没有任何文件可诊断（检查 --real 的 glob）", file=sys.stderr)
            return 2
        rows = real_diagnose(paths, include_L=v_il, adsorption=v_ad)
        if a.json:
            import json
            with open(a.json, "w", encoding="utf-8") as fp:
                json.dump(dict(real=True, n=len(rows),
                               voigt_caps=dict(include_L=v_il, adsorption=v_ad),
                               rows=rows), fp, ensure_ascii=False, indent=1)
            print(f"诊断结果已写入 {a.json}")
        return 0

    if a.dry:
        cases = _cases(a.noise, a.seed)
        for tag, expect, note, f, Z in cases:
            print(f"{tag:<22}{f.size:>4} 点  {f.min():.3g}–{f.max():.3g} Hz  "
                  f"|Z|max={np.abs(Z).max():.3g}  期望={expect}\n"
                  f"{'':<22}↳ {note}")
        print(f"nested 支路 = {list(branches)}；每例候选数 = "
              f"{2 + 2 * len(branches)}")
        print(f"pair_mode={a.pair_mode}  budget={a.budget}  "
              f"retry={a.retry_degraded}({a.retry_budget})  "
              f"strict_guards={a.strict_guards}  voigt_caps={v_caps}")
        print("per-branch 下每组 2(voigt) vs 2(nested)；pooled 下 2 vs "
              f"{2 * len(branches)}")
        if a.json:                       # 干跑也落盘：旧版这里静默不产出文件
            import json
            with open(a.json, "w", encoding="utf-8") as fp:
                json.dump(dict(dry=True, noise=a.noise, seed=a.seed,
                               branches=list(branches), pair_mode=a.pair_mode,
                               budget=a.budget, retry=a.retry_degraded,
                               retry_budget=a.retry_budget,
                               strict_guards=a.strict_guards,
                               voigt_caps=dict(include_L=v_il, adsorption=v_ad),
                               guard_names=dict(A=list(GUARD_A), B=list(GUARD_B)),
                               thresholds=dict(delta=SWITCH_DELTA, z=SWITCH_Z),
                               cases=[dict(case=t, expect=e, note=n,
                                           n_points=int(f.size),
                                           f_min=float(f.min()),
                                           f_max=float(f.max()),
                                           absZ_max=float(np.abs(Z).max()))
                                      for t, e, n, f, Z in cases]),
                          fp, ensure_ascii=False, indent=1)
            print(f"干跑结果已写入 {a.json}")
        return 0

    rc, rows = _selftest(a.noise, a.seed, a.seeds, branches=branches,
                         pair_mode=a.pair_mode, retry=a.retry_degraded,
                         strict=a.strict_guards, budget=a.budget,
                         retry_budget=a.retry_budget,
                         include_L=v_il, adsorption=v_ad,
                         gate_b=bool(a.gate_comp))
    if a.json:
        import json
        with open(a.json, "w", encoding="utf-8") as fp:
            json.dump(dict(noise=a.noise, seed=a.seed, n_seeds=a.seeds, rc=rc,
                           branches=list(branches), pair_mode=a.pair_mode,
                           budget=a.budget, retry=a.retry_degraded,
                           retry_budget=a.retry_budget,
                           strict_guards=a.strict_guards,
                           gate_comp=bool(a.gate_comp),
                           voigt_caps=dict(include_L=v_il, adsorption=v_ad),
                           guard_names=dict(A=list(GUARD_A), B=list(GUARD_B)),
                           guard_summary=guard_summary(rows),
                           health=_health(rows),
                           thresholds=dict(delta=SWITCH_DELTA, z=SWITCH_Z),
                           rows=rows), fp, ensure_ascii=False, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
