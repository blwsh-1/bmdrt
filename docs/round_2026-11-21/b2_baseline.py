"""B2 基准：按规格 A/B/C 造谱，再用**与构造一致**的 sigma 检验参数能否恢复。

为什么需要它
------------
21 条测试把 B2（参数恢复）、B3（族识别）、B4（采样）混成一个 rms 看，所以
"合成 0.08%" 既不能说明参数恢复、也不能说明族识别（`BASELINE_SPEC.md` §5）。
本脚本把 B2 单独做成可执行判据：造一条**已知真值**的谱 -> 用同族拟合 ->
逐参数报偏差。没有它，后面所有"模型改进了"的判断都只能靠 rms 猜。

预注册判据（跑之前冻结，跑完不改）
----------------------------------
规格 A（可辨识）：对每个采样参数 k，按其**自然尺度**扰动一步 δ_k
    log 参数：×1.10；有界参数：+0.10·(high−low)
    要求 `snr_k = max_f |Z(θ⊕δ_k) − Z(θ)| / σ(f) >= 1`

    与 `BASELINE_SPEC.md` §3 规格 A **等价**，不是放宽：对电感
    `ΔZ = 0.1·ωL`，故 `snr>=1 ⇔ 0.1·ωL >= σ ⇔ ωL >= 10σ`，正是原文。
    统一成"扰动一步"的形式只是为了让 n/phi/zeta 这类无量纲参数也能算。

规格 B（sigma 与数据匹配）：构造噪声与拟合 sigma 用**同一 pct**，
    且 `0.5 <= pct <= 5`（百分数）。依据：`BASELINE_SPEC.md` §3 B。

规格 C（表达力）：真值由**同族**正演生成 ⇒ 天然在族内；
    族识别另由 B3 检查（`--b3`，默认关，见文末"未做的部分"）。

B2（参数恢复）：每个"可见"参数（规格 A 通过）满足 `|θ̂ − θ| <= δ_k`
    —— 与规格 A 同一把尺子。全部可见参数同时满足才算过。

B4（采样）：`rhat <= 1.05` 且 `div <= 5`（div 1–5 记 `minor_div`，
    按层 0 口径算通过，见 `LAYER0_SUMMARY.md`）。

**不可见参数（snr < 1）不参与 B2**，标 `unresolved_by_design`。理由：模型族
自带的冗余维度（如 nested 的 `R_ad`）本来就不该被这个真值约束，把它算作
失败等于惩罚"族按设计工作"。

边界（必须与结论一起引用）
--------------------------
1. 只测**同族**恢复。族选错时的表现不在本脚本范围（那是 B3）。
2. 噪声是复高斯、实虚等方差；真实仪器噪声可能相关、非高斯。
3. `freq` 网格固定 61 点 1 Hz–1 MHz；换窗口会改变 tau 的可见性。
4. 每个配方只跑 1 个 seed。单次结果不能当成功率。
5. 先验贴合度（prior z）只在 nested/adr 上算（`bm_top._prior_table` 是唯一
   有表的族）；voigt 的 5 条先验没有表可读，故标 `n/a`，**不是**"通过了"。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/b2_baseline.py
    ... --budget quick --force          # 试跑
    ... --new-only 2                    # 分批（断点续跑）
    ... --b3                            # 额外跑族识别（quick/perpoint）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import bm_core as C            # noqa: E402
import bm_select as SEL        # noqa: E402
import bm_top as T             # noqa: E402
import circuit_runner as RU    # noqa: E402
import spectrum_metrics as SM  # noqa: E402

OUT = os.path.join(REPO, "out", "b2_baseline")


def state_path(budget: str) -> str:
    """`standard` 是本轮主产物；其余档单独存，避免互相覆盖。"""
    return os.path.join(OUT, "b2.json" if budget == "standard"
                        else "b2_%s.json" % budget)


#: **harness 本地**预算，不是生产档。唯一用途是把 B4 失败分成"预算不够"与
#: "结构不可辨"（`STD_RECHECK.md` 的同一做法）。`budget_actual` 会记下五项数值，
#: 所以不与 `circuit_runner.BUDGETS` 的任何档混报。
DEEP = dict(n_warmup=4000, n_samples=2000, num_chains=2,
            target_accept=0.97, max_tree_depth=11, extra_fields=("num_steps",))

BUDGETS_LOCAL = {"quick": RU.BUDGETS["quick"], "standard": RU.BUDGETS["standard"],
                 "deep": DEEP}

REPORT = os.path.join(HERE, "B2_REPORT.md")

#: 频率窗：1 Hz – 1 MHz，61 点（与 21 条测试的谱长同量级）。
FREQ = np.logspace(0.0, 6.0, 61)

#: 规格 B 的百分数下限/上限（`sigma_perpoint` 收百分数）。
PCT_LO, PCT_HI = 0.5, 5.0

#: 90% 区间相对宽度上限 `(q95−q05)/|median|`。超过即 `degenerate_interval`：
#: 点估计落在容差内也不算恢复。**这条是看到首轮结果之后补的**（见 `RULE_NOTE`），
#: 方向是**更严**，且对已存结果统一重算，见 `rescore_row`。
CI_WIDTH_MAX = 1.0

RULE = 2
RULE_NOTE = (
    "v1 只判点估计落在一步容差内。首轮 adr 出现 "
    "`R_ad` 偏 +11%（超 10%）而 90%CI=[55,906] —— 点估计本身无意义，"
    "被判成\"恢复失败\"会误导（真因是参数退化，不是采样或偏差）。"
    f"v2 增加 `ci_rel_width <= {CI_WIDTH_MAX:.1f}` 的必要条件，并对所有已存结果重算，"
    "不重拟合、不改 sigma、不改容差。")

#: 有界参数的先验支撑宽度 —— 决定"扰动一步"是多少。
BOUNDED_WIDTH = {"phi": 0.5,            # bm_core.make_model: U(0.5, 1.0)
                 "zeta": 1.7,           # bm_core.make_model: U(0.3, 2.0)
                 "n1": 0.5, "n2": 0.5,  # bm_top._prior_table: U(0.5, 1.0)
                 "n": 0.5}              # bm_top.adr_prior_table: U(0.5, 1.0)

#: 参数 -> (坐标, 该坐标下"一步"的大小)。log 坐标的一步 = ×1.10。
SPEC = {
    "voigt": {"R_inf": "log", "W": "log", "tau": "log",
              "phi": "abs", "zeta": "abs", "L": "log"},
    "nested": {"Rs": "log", "R1": "log", "Q1": "log", "n1": "abs",
               "R2": "log", "Q2": "log", "n2": "abs",
               "R_ad": "log", "tau_ad": "log", "L": "log"},
    "adr": {"Rs": "log", "Q": "log", "n": "abs", "Rct": "log",
            "R_ad": "log", "tau_ad": "log", "L": "log"},
}

#: 与 `circuit_runner._pointwise_median_z` 对齐的 cfg（只读这些键）。
CFG = {"voigt": dict(K=1, include_L=True, diffusion=None),
       "nested": dict(layers=2, branch="parallel", include_L=True,
                      diffusion=None),
       "adr": dict(include_L=True, diffusion=None)}

#: 配方。`neg_*` 是负对照：应当在**拟合之前**被规格检查拦下。
#: nested 的共享真值。**为什么 Q1=2000 而不像直觉那样 100**：外层 CPE 与整个
#: 内层支路并联，频率越高它越"短路"。要让内层弧（`Q2`/`n2`）在窗口内可见，
#: 必须让外层拐点在内层拐点**之上**，即 `(Q2/R2)^(1/n2) < (Q1/R1)^(1/n1)`。
#: 初版 `Q1=100` 直接导致 `Q2/n2` 的 snr≈0.5（被外层 CPE 罩住）——这是该族
#: 的结构性质，不是配错参数，见 `B2_REPORT.md` §4。
NESTED_TRUTH = dict(Rs=50.0, R1=100.0, Q1=2000.0, n1=0.9,
                    R2=200.0, Q2=60.0, n2=0.85,
                    R_ad=0.5, tau_ad=1e-2, L=10e-6)

#: 正对照统一用 1% 噪声：规格 B 允许 0.5–5%，1% 下所有非 R_ad 参数 snr>=3。
#: 与 `runner` 的 perpoint（2%）不同 —— 只影响 B3 的权重整体缩放，见 §5。
PCT_POS = 1.0

#: 配方必须**自己声明要测哪些参数**（`targets`）。
#: 规格 A 只对这些 target 判定；未声明的参数记 `unresolved_by_design` 并排除出
#: B2。理由：模型族自带的冗余维度（nested 的 `R_ad`/`tau_ad`）本来就不该被这
#: 个真值约束，把它算成"规格失败"等于惩罚"族按设计工作"。
NESTED_TARGETS = ("Rs", "R1", "Q1", "n1", "R2", "Q2", "n2", "L")

RECIPES = [
    dict(name="voigt_K1_L", family="voigt", pct=PCT_POS, seed=0,
         targets=("R_inf", "W", "tau", "phi", "zeta", "L"),
         desc="BASELINE_SPEC 4 合格配方：R_inf=100, HN(100R,1kHz,phi=.9), L=10uH",
         truth=dict(R_inf=100.0, W=100.0, tau=1.0 / (2 * np.pi * 1e3),
                    phi=0.9, zeta=1.0, L=10e-6)),
    dict(name="nested_L2_L", family="nested", pct=PCT_POS, seed=0,
         targets=NESTED_TARGETS,
         desc="nested 两层 + 顶层 L=10uH；R_ad 设 0.5 欧（声明不测，应判 unresolved）",
         truth=dict(NESTED_TRUTH)),
    dict(name="adr_L", family="adr", pct=PCT_POS, seed=0,
         targets=("Rs", "Q", "n", "Rct", "R_ad", "tau_ad", "L"),
         desc="adr 单节点三支路 + 顶层 L=10uH",
         truth=dict(Rs=50.0, Q=100.0, n=0.85, Rct=100.0,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
    # adr 的"串联改法"检验：参数值与 `adr_L` 逐一对应，只把吸附支路从
    # "与 Rct 并联"改成"与 Rct 串联"。结构就是现成的 `nested/L=1`，
    # **不需要新增族**。`TOPOLOGY_AUDIT.md` §3.2 预测它 cond 21.2、7/7。
    dict(name="adr_series", family="nested", layers=1, pct=PCT_POS, seed=0,
         targets=("Rs", "R1", "Q1", "n1", "R_ad", "tau_ad", "L"),
         desc="adr 串联改法 = nested/L=1（Rs+[CPE∥(R1+Z_ad)]+L）",
         truth=dict(Rs=50.0, R1=100.0, Q1=100.0, n1=0.85,
                    R_ad=100.0, tau_ad=1e-3, L=10e-6)),
    dict(name="neg_L_drowned", family="nested", pct=PCT_POS, seed=0,
         expect="reject", targets=NESTED_TARGETS,
         desc="负对照 1：同一条谱只把 L 换成 24.9nH（照抄 ecm_05687）⇒ 规格 A 该失败",
         truth=dict(NESTED_TRUTH, L=24.9e-9)),
    dict(name="neg_sigma_too_tight", family="nested", pct=1.8e-5, seed=0,
         expect="reject", targets=NESTED_TARGETS,
         desc="负对照 2：sigma 只 1.8e-5%（照抄 ecm_05687）⇒ 规格 B 该失败",
         truth=dict(NESTED_TRUTH)),
]


# ---------------------------------------------------------------------------
# 正演与参数扰动
# ---------------------------------------------------------------------------
def forward(family: str, f, flat: dict, layers: int = 2):
    """扁平参数 -> Z(f)。三个族各自的入口都来自生产代码，不重写公式。"""
    p = dict(flat)
    if family == "voigt":
        p.update(pi=np.array([1.0]), tau=np.array([flat["tau"]]),
                 phi=np.array([flat["phi"]]), zeta=np.array([flat["zeta"]]))
        return C.model_z(f, p, include_L=True)
    if family == "nested":
        return T.z_nested(f, p, branch="parallel", layers=layers,
                          include_L=True)
    if family == "adr":
        return T.z_adr(f, p, include_L=True)
    raise ValueError(family)


def step_of(family: str, key: str) -> float:
    """该参数"一步"的大小（log 坐标用乘数，有界坐标用绝对量）。"""
    if SPEC[family][key] == "log":
        return 0.10
    return 0.10 * BOUNDED_WIDTH[key]


def perturb(family: str, flat: dict, key: str):
    q = dict(flat)
    s = step_of(family, key)
    q[key] = flat[key] * (1.0 + s) if SPEC[family][key] == "log" else flat[key] + s
    return q, s


def visibility(family: str, f, flat: dict, sigma, layers: int = 2):
    """规格 A：逐参数算 max_f |Z(theta+one step) - Z(theta)| / sigma(f)。"""
    z0 = forward(family, f, flat, layers)
    rows = {}
    for k in SPEC[family]:
        if k not in flat:
            continue
        q, _ = perturb(family, flat, k)
        d = np.abs(forward(family, f, q, layers) - z0)
        rows[k] = dict(coordinate=SPEC[family][k],
                       step=float(step_of(family, k)),
                       snr=float(np.max(d / sigma)),
                       f_at=float(f[int(np.argmax(d / sigma))]))
    return rows


def parity_check(family: str, f, flat: dict, layers: int = 2) -> float | None:
    """B1 抽检：numpy 与 jax 正演是否一致（voigt 无 jax 入口，返回 None）。"""
    if family == "voigt":
        return None
    p = dict(flat)
    w_ref = 2 * np.pi * math.sqrt(f.min() * f.max())
    zn = forward(family, f, flat, layers)
    if family == "nested":
        zj = np.asarray(T.j_z_nested(2 * np.pi * f, p, w_ref, branch="parallel",
                                     layers=layers, include_L=True))
    else:
        zj = np.asarray(T.j_z_adr(2 * np.pi * f, p, w_ref, include_L=True))
    return float(np.max(np.abs(zj - zn)) / np.max(np.abs(zn)))


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def generate(rec: dict):
    f = FREQ.copy()
    fam = rec["family"]
    z_true = forward(fam, f, rec["truth"], int(rec.get("layers", 2)))
    pct = float(rec["pct"])
    sigma = SM.sigma_perpoint(z_true, pct)          # (2, n)
    rng = np.random.default_rng(int(rec["seed"]))
    nz = (sigma[0] * rng.standard_normal(f.size)
          + 1j * sigma[1] * rng.standard_normal(f.size))
    return f, z_true, z_true + nz, sigma, sigma[0]


# ---------------------------------------------------------------------------
# 拟合
# ---------------------------------------------------------------------------
def fit(rec: dict, f, z_obs, sigma, budget: str):
    fam = rec["family"]
    kw = dict(BUDGETS_LOCAL[budget])
    t0 = time.time()
    if fam == "voigt":
        c = SEL.fit_voigt(f, z_obs, 1, seed=int(rec["seed"]), include_L=True,
                          sigma_floor=0.0, kw=kw, sigma=sigma)
        raw, div = c["raw"], int(c["raw"]["ndiv"] or 0)
        p = RU._median_params_voigt(raw)
        flat = dict(R_inf=p["R_inf"], W=p["W"], tau=float(p["tau"][0]),
                    phi=float(p["phi"][0]), zeta=float(p["zeta"][0]),
                    L=p["L"])
        ci = dict(R_inf=raw["s"]["R_inf"], W=raw["s"]["W"],
                  tau=raw["s"]["tau"][:, 0], phi=raw["s"]["phi"][:, 0],
                  zeta=raw["s"]["zeta"][:, 0], L=raw["s"]["L"])
    elif fam == "nested":
        nl = int(rec.get("layers", 2))
        c = SEL.fit_nested(f, z_obs, nl, branch="parallel", seed=int(rec["seed"]),
                           sigma_floor=0.0, kw=kw, sigma=sigma,
                           sigma_from_data=True, include_L=True)
        raw, div = c["raw"], int(c["raw"]["divergences"] or 0)
        flat = RU._median_params_nested(raw, "parallel", nl, None, True)
        ci = raw["s"]
    else:
        c = SEL.fit_adr(f, z_obs, seed=int(rec["seed"]), sigma_floor=0.0, kw=kw,
                        sigma=sigma, sigma_from_data=True, include_L=True)
        raw, div = c["raw"], int(c["raw"]["divergences"] or 0)
        flat = RU._median_params_adr(raw, None, True)
        ci = raw["s"]
    cfg = dict(CFG[fam])
    if fam == "nested":
        cfg["layers"] = int(rec.get("layers", 2))
    zm = RU._pointwise_median_z(raw, fam, cfg, f)
    return dict(flat=flat, ci=ci, zm=zm, div=div,
                rhat=float(raw["rhat_max"]),
                ess_min=(None if raw.get("ess_min") is None
                         else float(raw["ess_min"])),
                seconds=round(time.time() - t0, 1), kw=kw)


def score(family: str, truth: dict, vis: dict, targets, samp, prior_z) -> dict:
    """B2：逐参数偏差 vs 一步；同时报 90% CI 是否覆盖真值。

    只有 `targets` 里的参数参与判定。其余参数照样记录（可见性、偏差），
    但标 `unresolved_by_design` 并排除在 B2 之外。
    """
    rows = {}
    for k, v in vis.items():
        th = float(truth[k])
        q = float(samp["flat"][k])
        coord, s = v["coordinate"], v["step"]
        err = (q / th - 1.0) if coord == "log" else (q - th)
        c = np.asarray(samp["ci"][k], float)
        lo, hi = float(np.percentile(c, 5)), float(np.percentile(c, 95))
        width = (hi - lo) / abs(q) if q else float("inf")
        rows[k] = dict(truth=th, median=q, coordinate=coord, step=s,
                       err=float(err), snr=v["snr"], f_at=v["f_at"],
                       target=bool(k in targets),
                       visible=bool(v["snr"] >= 1.0),
                       weak=bool(v["snr"] < 3.0),
                       within=bool(abs(err) <= s),
                       ci_rel_width=float(width),
                       degenerate=bool(width > CI_WIDTH_MAX),
                       within_strict=bool(abs(err) <= s and width <= CI_WIDTH_MAX),
                       ci90=[lo, hi], ci_covers=bool(lo <= th <= hi),
                       prior_z=prior_z.get(k))
    vis_t = {k: r for k, r in rows.items() if r["target"] and r["visible"]}
    b2 = bool(vis_t) and all(r["within_strict"] for r in vis_t.values())
    cov = [r["ci_covers"] for r in vis_t.values()]
    return dict(params=rows, b2_pass=b2,
                n_visible=len(vis_t),
                n_within=sum(1 for r in vis_t.values() if r["within_strict"]),
                n_degenerate=sum(1 for r in vis_t.values() if r["degenerate"]),
                ci_cover=float(np.mean(cov)) if cov else None)


def prior_z_of(family: str, flat: dict, f, zmax: float,
               layers: int = 2) -> dict:
    """先验贴合度：log 参数偏离先验中心几个 sigma。只有 nested/adr 有表。"""
    if family == "voigt":
        return {}
    w_ref = 2 * np.pi * math.sqrt(f.min() * f.max())
    tab = (T._prior_table(zmax, w_ref, "parallel", layers, None, True)
           if family == "nested" else T.adr_prior_table(zmax, w_ref, None, True))
    out = {}
    for k, v in flat.items():
        d = tab.get(k)
        if d is None:
            continue
        try:
            loc, sc = float(d.loc), float(d.scale)
            out[k] = float((math.log(v) - loc) / sc)
        except Exception:                              # Uniform 等无 loc/scale
            continue
    return out


def rescore_row(r: dict) -> dict:
    """按当前 `RULE` 从**已存**的 `ci90`/`median` 重算判定，不重拟合。

    存在理由：判据版本升级时（见 `RULE_NOTE`），旧结果必须用新规则重算，
    否则同一个表里两套规则并存。所需字段全都已落盘，所以重算是确定的。
    """
    if not r.get("b2"):
        return r
    ps = r["b2"]["params"]
    for p in ps.values():
        q, s = float(p["median"]), float(p["step"])
        lo, hi = p["ci90"]
        w = (float(hi) - float(lo)) / abs(q) if q else float("inf")
        p["ci_rel_width"] = w
        p["degenerate"] = bool(w > CI_WIDTH_MAX)
        p["within"] = bool(abs(p["err"]) <= s)
        p["within_strict"] = bool(p["within"] and not p["degenerate"])
    vis_t = {k: p for k, p in ps.items() if p.get("target", True) and p["visible"]}
    r["b2"]["n_visible"] = len(vis_t)
    r["b2"]["n_within"] = sum(1 for p in vis_t.values() if p["within_strict"])
    r["b2"]["n_degenerate"] = sum(1 for p in vis_t.values() if p["degenerate"])
    r["b2"]["b2_pass"] = bool(vis_t) and all(p["within_strict"]
                                             for p in vis_t.values())
    b4 = bool(r["b4"]["rhat"] <= 1.05 and r["b4"]["div"] <= 5)
    r["state"] = ("b2_pass" if (r["b2"]["b2_pass"] and b4)
                  else "b4_fail" if not b4 else "b2_fail")
    return r


# ---------------------------------------------------------------------------
# 单条主流程
# ---------------------------------------------------------------------------
def run_one(rec: dict, budget: str) -> dict:
    f, z_true, z_obs, sigma, s1d = generate(rec)
    lay = int(rec.get("layers", 2))
    vis = visibility(rec["family"], f, rec["truth"], sigma[0], lay)
    targets = tuple(rec["targets"])
    fam_ok = (PCT_LO <= float(rec["pct"]) <= PCT_HI)
    # 规格 A 只看 target：未声明的参数不参与判定，标 unresolved_by_design。
    inv = sorted([k for k in targets if vis[k]["snr"] < 1.0])
    untested = sorted([k for k in vis if k not in targets])
    spec_a = not inv
    rec_out = dict(name=rec["name"], family=rec["family"], desc=rec["desc"],
                   expect=rec.get("expect", "pass"), targets=list(targets),
                   untested=untested,
                   pct=float(rec["pct"]), seed=int(rec["seed"]),
                   n_points=int(f.size), freq_range=[float(f.min()), float(f.max())],
                   zmax=float(np.max(np.abs(z_true))),
                   sigma_median=float(np.median(sigma[0])),
                   spec=dict(A=spec_a, A_invisible=inv, B=fam_ok,
                             B_bounds=[PCT_LO, PCT_HI],
                             C="same_family_generated",
                             b1_parity=parity_check(rec["family"], f, rec["truth"], lay)),
                   visibility=vis,
                   truth={k: float(v) for k, v in rec["truth"].items()})
    # 规格不通过就**不拟合**：这正是负对照要证明的行为（省掉无效 MCMC）。
    if not (spec_a and fam_ok):
        rec_out.update(state="rejected_by_spec", fit=None, b2=None)
        print("  [reject] %-20s A=%s B=%s 不可见=%s"
              % (rec["name"], spec_a, fam_ok, ",".join(inv) or "-"))
        return rec_out

    print("  [fit]    %-20s %s 预算=%s ..." % (rec["name"], rec["family"], budget),
          end="", flush=True)
    s = fit(rec, f, z_obs, sigma, budget)
    b4 = bool(s["rhat"] <= 1.05 and s["div"] <= 5)
    pz = prior_z_of(rec["family"], s["flat"], f, float(np.max(np.abs(z_obs))), lay)
    sc = score(rec["family"], rec["truth"], vis, targets, s, pz)
    met = SM.metrics(f, z_obs, s["zm"])
    state = ("b2_pass" if sc["b2_pass"] and b4
             else "b4_fail" if not b4 else "b2_fail")
    print(" rhat=%.3f div=%d rms=%.2f%% b2=%s (%d/%d) %.0fs"
          % (s["rhat"], s["div"], met["rms_rel"], sc["b2_pass"],
             sc["n_within"], sc["n_visible"], s["seconds"]))
    rec_out.update(
        state=state, b2=sc,
        b4=dict(rhat=s["rhat"], div=s["div"], ess_min=s["ess_min"],
                pass_=b4, minor_div=bool(0 < s["div"] <= 5)),
        metrics=dict(rms_rel=float(met["rms_rel"])),
        seconds=s["seconds"], budget_actual=SM.budget_actual(s["kw"], tier=budget),
        noise=dict(noise_re=[float(v) for v in (z_obs - z_true).real],
                   noise_im=[float(v) for v in (z_obs - z_true).imag]),
        z_true=dict(re=[float(v) for v in z_true.real],
                    im=[float(v) for v in z_true.imag]),
        z_obs=dict(re=[float(v) for v in z_obs.real],
                   im=[float(v) for v in z_obs.imag]),
        z_fit=dict(re=[float(v) for v in s["zm"].real],
                   im=[float(v) for v in s["zm"].imag]))
    return rec_out


# ---------------------------------------------------------------------------
# B3（族识别，默认关）
# ---------------------------------------------------------------------------
def run_b3(rec: dict, budget: str = "quick") -> dict:
    """在 quick/perpoint 下跑全部候选，看 top-1 是否命中真值族。

    只在 `--b3` 时执行。**quick 的收敛不可信**（层 0 已证钝化 #1 反例），
    所以这里只读排名、不读 R̂，结论标 `provisional`。
    """
    f, z_true, z_obs, sigma, _ = generate(rec)
    cands = RU.build_candidates(include_L=True, budget_profile=budget)
    rows = []
    for c in cands:
        try:
            o = RU.fit_candidate(c, f, z_obs, seed=int(rec["seed"]),
                                 budget_profile=budget, likelihood="perpoint")
            rows.append(dict(id=c.id, family=c.topology_id,
                             cfg=dict(c.config), elpd=float(o["elpd"]),
                             rhat=o["rhat_max"], div=o["divergences"]))
        except Exception as e:                          # 单候选失败不该毁掉整条
            rows.append(dict(id=c.id, family=c.topology_id,
                             cfg=dict(c.config), error=repr(e)[:120]))
    ok = [r for r in rows if "elpd" in r]
    ok.sort(key=lambda r: -r["elpd"])
    top = ok[0] if ok else None
    return dict(n_candidates=len(rows), top1=(top or {}).get("id"),
                top1_family=(top or {}).get("family"),
                hit=bool(top and top["family"] == rec["family"]),
                provisional=True, budget=budget, ranking=ok)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def fmt_par(k: str, r: dict) -> tuple:
    if r["coordinate"] == "log":
        return "%s" % f"{r['truth']:.4g}", "%+.1f%%" % (100 * r["err"])
    return "%.3f" % r["truth"], "%+.3f" % r["err"]


def write_report(state: dict) -> None:
    L = ["# B2 参数恢复：可执行的基准判据",
         "",
         "> 由 `b2_baseline.py` 生成（幂等，数据变直接重跑）。判据在脚本头部预注册。",
         "> 规格来源 `BASELINE_SPEC.md` §3；层 0 口径来源 `LAYER0_SUMMARY.md`。",
         "",
         "## 0. 一句话",
         "",
         "**B2 = 造一条已知真值的谱，用同族 + 同 sigma 拟合，逐参数看能否回到真值；"
         "回不到就不许说模型改进。**",
         "",
         "## 1. 规格 A 与 B2 是同一把尺子",
         "",
         "规格 A 原文是 `Δ|Z|_k >= 10σ`，本脚本写成 `snr_k = |ΔZ|/σ >= 1`，其中 "
         "`ΔZ` 由**扰动一步** `δ_k` 产生（log 参数 ×1.10，有界参数 +0.10×支撑宽度）。",
         "对电感 `ΔZ = 0.1·ωL`，故 `snr>=1 ⇔ ωL >= 10σ` —— 与原文等价，不是放宽。",
         "统一成\"一步\"的形式之后，`n/phi/zeta` 这类无量纲参数也能算，且 B2 的容差"
         "直接就是同一个 `δ_k`。",
         "",
         "**判据 v%d**（`RULE`）：点估计落在一步内 **且** 90%% 区间相对宽度 "
         "`(q95−q05)/|median| <= %.1f`。后一条是看到首轮结果后补的，"
         "补的方向是**更严**，且所有已存结果都从落盘的区间重算（`rescore_row`），"
         "不重拟合、不改 sigma、不改容差。"
         % (RULE, CI_WIDTH_MAX),
         "",
         RULE_NOTE,
         "",
         "## 2. 总表",
         "",
         "| 配方 | 族 | 规格 A | 规格 B | 状态 | R̂ | div | rms% | B2 命中 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in state["rows"]:
        sp = r["spec"]
        a = "ok" if sp["A"] else "**fail**"
        b = "ok" if sp["B"] else "**fail**"
        if r["state"] == "rejected_by_spec":
            L.append("| `%s` | %s | %s | %s | `rejected_by_spec` | — | — | — | — |"
                     % (r["name"], r["family"], a, b))
            continue
        b2 = r["b2"]
        L.append("| `%s` | %s | %s | %s | `%s` | %.3f | %d | %.2f | %d/%d |"
                 % (r["name"], r["family"], a, b, r["state"], r["b4"]["rhat"],
                    r["b4"]["div"], r["metrics"]["rms_rel"],
                    b2["n_within"], b2["n_visible"]))
    L += ["", "## 3. 逐参数偏差", ""]
    for r in state["rows"]:
        if r["b2"] is None:
            L += ["### `%s` —— 被规格拦下" % r["name"], "",
                  "不可见（target 内）：%s；pct=%.3g（允许 %.1f–%.1f）；%s。"
                  % (", ".join("`%s`(snr=%.2f)" % (k, r["visibility"][k]["snr"])
                               for k in r["spec"]["A_invisible"]) or "无",
                     r["pct"], PCT_LO, PCT_HI,
                     "规格 A 未过 ⇒ 这条配方**不拟合**"
                     if not r["spec"]["A"] else "规格 B 未过 ⇒ **不拟合**"),
                  ""]
            continue
        L += ["### `%s` —— %s（%s）" % (r["name"], r["state"], r["desc"]), "",
              "| 参数 | 真值 | 后验中位 | 偏差 | 容差 | snr | 目标 | CI 宽度 | 覆盖 | prior z |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, p in r["b2"]["params"].items():
            tv, ev = fmt_par(k, p)
            tol = ("±%.0f%%" % (100 * p["step"]) if p["coordinate"] == "log"
                   else "±%.3f" % p["step"])
            if not p.get("target", True):
                flag = "不测"
            elif p["visible"]:
                flag = "是"
            else:
                flag = "**否**"
            w = p.get("ci_rel_width")
            wtxt = ("—" if w is None else
                    ("**%.1fx 退化**" % w if p.get("degenerate") else "%.2f" % w))
            L.append("| `%s` | %s | %s | %s | %s | %.2f | %s | %s | %s | %s |"
                     % (k, tv, ("%.4g" % p["median"]), ev, tol, p["snr"], flag,
                        wtxt, "是" if p["ci_covers"] else "否",
                        "—" if p["prior_z"] is None else "%.2f" % p["prior_z"]))
        det = [k for k, p in r["b2"]["params"].items() if not p.get("target", True)]
        L += ["", "CI 覆盖率（可见目标参数）：%s；目标参数中最低 snr %.2f。"
              % ("—" if r["b2"]["ci_cover"] is None
                 else "%.0f%%" % (100 * r["b2"]["ci_cover"]),
                 min([p["snr"] for p in r["b2"]["params"].values()
                      if p.get("target", True) and p["visible"]] or [float("nan")])),
              ("未声明为 target（`unresolved_by_design`）：%s —— 族自带冗余维度，"
               "不计入 B2。" % ", ".join("`%s`" % k for k in det)) if det else "",
              ("**R̂=%.3f 未达标 ⇒ 上面这 %d/%d 记 `non_evaluable`，不是通过**；"
               "是预算还是结构，看 §4 的 deep 档。" % (r["b4"]["rhat"],
                                                      r["b2"]["n_within"],
                                                      r["b2"]["n_visible"]))
              if not r["b4"]["pass_"] else "",
              ""]
    # 预算敏感性：只有存在 `b2_deep.json` 时才出这张表（不硬编码数字）。
    deep = state_path("deep")
    if os.path.exists(deep):
        d = {r["name"]: r for r in json.load(open(deep, encoding="utf-8"))["rows"]}
        L += ["## 4. 预算敏感性（`deep` = 4000+2000 / ta 0.97 / depth 11，"
              "harness 本地档，不是生产档）", "",
              "| 配方 | standard R̂ | deep R̂ | standard div | deep div | "
              "standard B2 | deep B2 |", "|---|---|---|---|---|---|---|"]
        for r in state["rows"]:
            q = d.get(r["name"])
            if not q or not q.get("b4") or not r.get("b4"):
                continue
            L.append("| `%s` | %.3f | %.3f | %d | %d | %d/%d | %d/%d |"
                     % (r["name"], r["b4"]["rhat"], q["b4"]["rhat"],
                        r["b4"]["div"], q["b4"]["div"],
                        r["b2"]["n_within"], r["b2"]["n_visible"],
                        q["b2"]["n_within"], q["b2"]["n_visible"]))
        rescued, stuck = [], []
        for r in state["rows"]:
            q = d.get(r["name"])
            if not q or not q.get("b4") or not r.get("b4"):
                continue
            if r["b4"]["pass_"]:
                continue
            (rescued if q["b4"]["pass_"] else stuck).append(r["name"])
        L += ["", "读法：R̂ 随预算下降 ⇒ 原来那条是 `budget_limited`（B4 可救，"
              "B2 只是暂缓）；R̂ 不降 ⇒ 结构不可辨，B2 永久 `non_evaluable`。"
              "**注意**：这一档不能拿来给 B2 判据加分——判据在 `standard` 上定，"
              "换预算只是定位失败原因。", ""]
        if rescued:
            L += ["- `budget_limited`（deep 下 B4 转通过）：%s。"
                  % ", ".join("`%s`" % n for n in rescued)]
        if stuck:
            L += ["- 加预算仍不过：%s ⇒ 不是预算问题。"
                  % ", ".join("`%s`" % n for n in stuck)]
        L += [""]
    # 跨配方一致的区间失配：单条不看，三条同向才报（避免把一次性噪声当结论）
    by_key = {}
    for r in state["rows"]:
        if not r.get("b2"):
            continue
        for k, p in r["b2"]["params"].items():
            if p.get("target", True) and p["visible"]:
                by_key.setdefault(k, []).append((r["name"], p))
    sysmiss = [(k, v) for k, v in by_key.items()
               if len(v) >= 3 and not any(p["ci_covers"] for _, p in v)]
    L += ["## 5. 结论只到哪",
          "",
          "- 负对照被拦在拟合之前 ⇒ 规格检查本身有效（它不是\"跑完再看\"的事后解释）。",
          "- `unresolved_by_design` 的参数（未声明 target 或 snr<1）**不计入** B2，"
          "也不计入失败。",
          "- B4 未通过时，B2 的数字标 `non_evaluable`，**不得单独引用**"
          "（层 0 的口径）。"]
    for k, v in sysmiss:
        L += ["- **跨 %d 个配方，`%s` 的 90%%CI 全部不含真值**：%s ⇒ 方向一致的"
              "系统偏移（不是随机失配）。但 1 seed 分不开\"噪声实现\"与\"模型偏置\"，"
              "要等多 seed 才能判。" % (
                  len(v), k, "、".join("`%s` %+.1f%%" % (n, 100 * p["err"])
                                       for n, p in v))]
    L += [
          "- 每个配方 1 个 seed、1 个频率窗、复高斯等方差噪声 ⇒ 只说明\"这条配方"
          "在该条件下能否恢复\"，不是成功率。",
          "- **B3 未做**（`--b3` 默认关）：quick 的排名不足以支撑族识别结论。",
          "",
          "## 6. 未做的部分与前置条件",
          "",
          "| 项 | 为什么没做 | 前置 |",
          "|---|---|---|",
          "| B3 族识别 | quick 的\"通过\"已被证明不可信（钝化 #1） | 定 B3 用哪个预算、"
          "以及 elpd 在 perpoint 下的可比性 |",
          "| 多 seed | 单次结果不能报成功率 | 每配方 ≥5 seed，成本 ×5 |",
          "| 换频率窗 | tau 的可见性随窗口变 | 至少两档窗口 |",
          "| voigt 的 prior z | `bm_core.make_model` 的 5 条先验没有表可读 | "
          "抽一张 `voigt_prior_table`（会动生产文件，需单独授权） |",
          ""]
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))


def fig(state: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in state["rows"] if r["b2"] is not None]
    if not ok:
        return
    fig1, axs = plt.subplots(len(ok), 2, figsize=(10.5, 3.1 * len(ok)),
                             squeeze=False)
    for i, r in enumerate(ok):
        f = np.logspace(0, 6, len(r["z_true"]["re"]))
        zt = np.array(r["z_true"]["re"]) + 1j * np.array(r["z_true"]["im"])
        zo = np.array(r["z_obs"]["re"]) + 1j * np.array(r["z_obs"]["im"])
        zf = np.array(r["z_fit"]["re"]) + 1j * np.array(r["z_fit"]["im"])
        a = axs[i][0]
        a.loglog(f, np.abs(zt), color="#222", lw=1.6, label="truth")
        a.loglog(f, np.abs(zo), "o", ms=2.6, color="#D85A30", alpha=.55,
                 label="obs (noisy)")
        a.loglog(f, np.abs(zf), "--", color="#378ADD", lw=1.6,
                 label="fit (rms %.2f%%)" % r["metrics"]["rms_rel"])
        a.set_xlabel("f / Hz"); a.set_ylabel("|Z| / ohm")
        a.set_title("%s  %s" % (r["name"], r["state"]), fontsize=10)
        a.legend(fontsize=7.5, frameon=False)
        b = axs[i][1]
        b.semilogx(f, 100 * (zo.real - zf.real) / np.abs(zo), "o-", ms=2.6,
                   lw=.8, color="#D85A30", label="Re resid")
        b.semilogx(f, 100 * (zo.imag - zf.imag) / np.abs(zo), "s-", ms=2.4,
                   lw=.8, color="#534AB7", label="Im resid")
        b.axhline(0, color="#999", lw=.8)
        b.set_xlabel("f / Hz"); b.set_ylabel("resid / % of |Z|")
        b.legend(fontsize=7.5, frameon=False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUT, "fig_b2_curves.png"), dpi=150)
    plt.close(fig1)

    # 偏差以**容差**为单位：1 = 判据阈值。用 symlog 承接"退化参数偏 2000%"那种
    # 量级，否则一条超长柱会把其余全部压成零（首版就是这样，图不可读）。
    rows = [(r, k, p) for r in ok for k, p in r["b2"]["params"].items()]
    fig2, ax = plt.subplots(figsize=(7.8, 0.32 * len(rows) + 1.9))
    labels, vals, cols = [], [], []
    for r, k, p in rows:
        labels.append("%s / %s" % (r["name"].replace("_", " "), k))
        vals.append(p["err"] / p["step"] if p["step"] else 0.0)
        if not p.get("target", True):
            cols.append("#B4B2A9")
        elif not p["visible"]:
            cols.append("#B4B2A9")
        elif p.get("degenerate"):
            cols.append("#EF9F27")
        elif p["within"]:
            cols.append("#1D9E75")
        else:
            cols.append("#D85A30")
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=.62)
    ax.axvline(0, color="#333", lw=.9)
    for s_ in (-1, 1):
        ax.axvline(s_, color="#D85A30", lw=.9, ls="--", alpha=.7)
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("deviation / tolerance   (1.0 = threshold, dashed)")
    ax.set_title("B2: green=pass, orange=interval degenerate, "
                 "red=out of tolerance, grey=not tested", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "fig_b2_params.png"), dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
def json_safe(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(float(v)) else None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="standard",
                    choices=["quick", "standard", "deep"])
    ap.add_argument("--force", action="store_true", help="重跑已完成的配方")
    ap.add_argument("--new-only", type=int, default=0, help="本次最多新跑几条")
    ap.add_argument("--b3", action="store_true", help="额外做族识别（quick）")
    ap.add_argument("--only", default="", help="逗号分隔的配方名")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    state = dict(rows=[], budget=a.budget, rule=RULE)
    STATE = state_path(a.budget)
    if os.path.exists(STATE):
        old = json.load(open(STATE, encoding="utf-8"))
        if not a.force and old.get("budget") == a.budget:
            state = old
    state["rule"] = RULE
    for r in state["rows"]:
        rescore_row(r)
    done = {r["name"]: r for r in state["rows"]}
    want = [r for r in RECIPES
            if not a.only or r["name"] in a.only.split(",")]

    new = 0
    for rec in want:
        if rec["name"] in done and not a.force and done[rec["name"]].get("state") \
                not in ("rejected_by_spec",):
            print("  [skip]   %-20s 已完成（%s）" % (rec["name"],
                                                  done[rec["name"]]["state"]))
            continue
        if a.new_only and new >= a.new_only:
            break
        out = run_one(rec, a.budget)
        if a.b3 and out["state"] != "rejected_by_spec":
            out["b3"] = run_b3(rec)
            print("           b3 top1=%s family=%s hit=%s"
                  % (out["b3"]["top1"], out["b3"]["top1_family"],
                     out["b3"]["hit"]))
        done[rec["name"]] = out
        new += 1
        state["rows"] = [rescore_row(done[r["name"]])
                         for r in RECIPES if r["name"] in done]
        with open(STATE, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(json_safe(state), fh, ensure_ascii=False, indent=1,
                      allow_nan=False)

    state["rows"] = [rescore_row(done[r["name"]])
                     for r in RECIPES if r["name"] in done]
    with open(STATE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(json_safe(state), fh, ensure_ascii=False, indent=1,
                  allow_nan=False)
    # 报告与图只为 `standard` 生成：它是本轮判据所在档。其他档只落 JSON，
    # 供报告里的"预算敏感性"一节自动读取——否则换档会覆盖掉主产物。
    if a.budget == "standard":
        write_report(state)
        fig(state)
        print("->", REPORT)

    print("-" * 78)
    for r in state["rows"]:
        if r["b2"] is None:
            print("%-22s %-22s A=%s B=%s" % (r["name"], r["state"],
                                             r["spec"]["A"], r["spec"]["B"]))
        else:
            print("%-22s %-12s rhat=%.3f div=%d b2=%s (%d/%d)"
                  % (r["name"], r["state"], r["b4"]["rhat"], r["b4"]["div"],
                     r["b2"]["b2_pass"], r["b2"]["n_within"], r["b2"]["n_visible"]))
    print("->", STATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
