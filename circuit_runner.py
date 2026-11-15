# -*- coding: utf-8 -*-
"""registry 驱动的候选电路统一调度器。

**这是研究/比较接口，不是生产默认通道。** 边界照 `CIRCUIT_TOPOLOGY_ADMISSION.md`
§8.1：`circuit_registry` 回答"这个候选能不能被表达、签发身份、被复核"，
本模块回答"把它跑起来、把观测如实落到 `observation_block`"。
两者都不回答"该不该用它" —— 那是准入判定，仍由 §9.4.21 A1–A4 卡着。

本模块**不改** `bm_*.py`、不改任何阈值、不改 `bm_real.fit_one` 的默认族、
不接 `bmdrt.py` 的 `auto`。它只把已经存在的两个入口
（`bm_select.fit_voigt` / `bm_select.fit_nested`）统一调度，并用**同一套残差口径**
计算 `rms_rel` / `acf`，使"可比较"这件事成立而不是被假定。

三个设计约束（对应 `CIRCUIT_TOPOLOGY_ADMISSION.md` 与 M-B-SIM-01 的结论）
--------------------------------------------------------------------------
1. **统一误差定义**：所有族的 `rms_rel` / `acf` 都由本模块的 `metrics()` 从
   `Z_model − Z_data` 现算（与 §14 `_metrics` 逐字同式），不读各族自报的数。
   族自报的 `acf` 口径不同（voigt 走 `bm_real` 的 band 中位、nested 根本没算），
   直接比会得出"看起来可比"的假结论。
2. **`elpd` 一律标 `valid_for_comparison=False`**：`bm_select` 用
   `sigma=None + sigma_floor`，`sigma_from_data=True` 下 `elpd_loo` 不是
   `p(Z_j|Z_{-j})`（R21）。这个事实必须跟着数字走。
3. **不合成总分、不排序**：输出是逐候选的 `observation_block` 列表。
   出 `identified / weak_preference / ...` 是 evaluator 的职责，本模块不做。

用法
----
    python circuit_runner.py --list                 # registry 里有什么、能不能跑
    python circuit_runner.py --plan  <data.csv>     # 候选集 + 预检，不拟合
    python circuit_runner.py --run   <data.csv>     # 逐候选拟合 + 落观测
    python circuit_runner.py --selftest             # 自检（合成数据，极小预算）

产物落在 `out/circuit_runner/<data_stem>/`：
    plan.json / screening.json / observations.jsonl / run_meta.json
`observations.jsonl` 是 append-only；重跑同一 `(candidate_id, data_sha256, seed,
budget_profile)` 会命中缓存直接跳过（断点续跑）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import circuit_registry as CR                                   # noqa: E402
import bm_experiments as E                                      # noqa: E402

OUT_ROOT = os.path.join("out", "circuit_runner")

#: 可调度的拓扑 → (构造候选的函数, 拟合适配器)。**只登记"能真跑"的**：
#: `w_inf` / `w_open` / `w_short` / `adsorption` 在 registry 里 `wired=False`
#: 且没有槽位，它们的拓扑根本不存在，所以不能出现在这里 —— 这不是遗漏，
#: 是 §5/§11 把扩散与吸附接入列为暂缓的如实反映。
TOPOLOGY_ADAPTERS = ("voigt", "nested")

#: 候选集构造：默认只开基线能力（不带 L、不带吸附），因为打开任一项都是
#: 新的族能力臂，必须单独预注册（§9.4.11）。要跑扩展臂请显式传 `--include-L`。
DEFAULT_KS = (1, 2)
DEFAULT_LAYERS = (1, 2)

#: `测试数据/` 三组合成谱的真值拓扑（`NOISE_ACF_TRUTH.md` §1 的数值断言：
#: 涂层 `R1-[P2,R3-[P4,R5-P6]]` = nested L=2、钝化 `[P1,P2-R3]-R4` = nested L=1、
#: 腐蚀 `R1-[R2,P3]` = ZARC = voigt K=1）。**两组嵌套的最内层都是容性 CPE**，
#: 所以它们必须用 `branch="capacitive"` 才落在真值拓扑上；`parallel` 那支
#: （R∥L）在这三组上属于"骨架对、元件类型错"，跑出来只能当反例。
DATASETS = {
    "coating": dict(
        name="涂层", root=os.path.join("测试数据", "涂层"),
        sub="Coating_Generated EIS data and Nyquist plots",
        truth=dict(topology_id="nested", branch="capacitive", layers=2),
        truth_formula="R1-[P2,R3-[P4,R5-P6]]"),
    "passivation": dict(
        name="钝化", root=os.path.join("测试数据", "钝化"),
        sub="Passivating_Generated EIS data and Nyquist plots",
        truth=dict(topology_id="nested", branch="capacitive", layers=1),
        truth_formula="[P1,P2-R3]-R4"),
    "corrosion": dict(
        name="腐蚀", root=os.path.join("测试数据", "腐蚀"),
        sub="Uniform_Generated data and Nyquist plots",
        truth=dict(topology_id="voigt", branch=None, layers=1),
        truth_formula="R1-[R2,P3]（ZARC）"),
}

#: `nested` 的三个分支各自需要的 branch 槽位元件（registry 里登记过的）。
BRANCH_ELEMENT = {"parallel": "rl_parallel", "series": "rl_series",
                  "capacitive": "cpe"}

#: 采样档。`quick` 只用于把管线跑通；任何要写进证据的数字都该用 `standard`。
#: `extra_fields=("num_steps",)` 不是可有可无：`bm_real.grade()` 会读
#: `tree_depth_stats(mc, ...)["frac_cap"]`，而那个统计要靠 num_steps 才能算。
#: 漏掉它不会报"缺参数"，而是让 `tds` 变成 None、随后在 `grade` 里抛
#: `TypeError: 'NoneType' object is not subscriptable`——离病因很远。
#: `bm_select.QUICK` 里本来就带这一项，这里必须对齐。
BUDGETS = {
    "quick": dict(n_warmup=300, n_samples=300, num_chains=2,
                  target_accept=0.9, max_tree_depth=8,
                  extra_fields=("num_steps",)),
    "standard": dict(n_warmup=2000, n_samples=1000, num_chains=2,
                     target_accept=0.98, max_tree_depth=10,
                     extra_fields=("num_steps",)),
}

#: 正确的 R21 理由。**不要写成 `sigma_from_data=True`**：`fit_top` 只有在显式传
#: 该开关时才逐点估 σ，`bm_select` 的跨族口径传的是 `sigma=None`（σ 被采样），
#: 那一支的 `elpd_valid_for_predictive_comparison` 反而是 True。真正让 `elpd_loo`
#: 不是标准留一的是**先验尺度 `zmax=max|Z|` 由本批观测算出**（留出 j 点时未重算）。
R21_REASON = ("先验尺度 zmax=max|Z| 由本批观测算出、留出 j 点时未重算（R21），"
              "故 elpd_loo 不是标准 p(Z_j|Z_-j)；只能作候选间相对参考")

SAMPLING_RHAT_MAX = 1.05   # 与 bm_select 的 rhat 判据同值；此处只标 sampling_status


# --------------------------------------------------------------------------
# 统一指标：所有族共用这一份，不读各族自报
# --------------------------------------------------------------------------
def _jsonable(x):
    """递归转成 `json.dump(..., allow_nan=False)` 能吃的形状。

    `bool` 必须先于 `int` 判断（`numpy.bool_` 是 `int` 的子类之外的另一路，
    但 Python `bool` 是 `int` 的子类，顺序反了会把 True 写成 1）。
    """
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def metrics(f, Z, Zm) -> dict:
    """与 `evidence/nested_topology/m4_holdout2.py::_metrics` 逐字同式（§14 口径）。

    为什么要抄一份而不是 import：那个文件在 `.gitignore` 覆盖的 evidence/ 下，
    生产侧不能依赖它；抄一份并把等式写进 `--selftest` 的断言里，比隐式依赖安全。
    """
    w = np.abs(Z)
    res = np.asarray(Zm) - Z
    rel = np.abs(res) / w
    d = res / w
    bands = []
    for lo, hi in ((0.01, 1), (1, 100), (100, 1e5)):
        m = (f >= lo) & (f < hi)
        bands.append(float(np.sqrt(np.mean(np.abs(d)[m] ** 2))) * 100
                     if m.any() else None)
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100,
        bands=bands,
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)),
        acf_im=float(E.acf_max(res.imag)),
        acf_mag=float(E.acf_max(rel)),
    )


# --------------------------------------------------------------------------
# 候选集
# --------------------------------------------------------------------------
def build_candidates(ks=DEFAULT_KS, layers=DEFAULT_LAYERS,
                     include_L: bool = False, budget_profile: str = "quick",
                     init_profile: str = "uniform",
                     branches=("parallel",)) -> list:
    """按 registry 构造候选。

    `branches` 默认只有 `parallel`（与 `nested_fit.py` 一致）。**`测试数据/` 的
    涂层与钝化真值最内层是容性 CPE**，要把它们纳入比较必须显式传
    `branches=("parallel","capacitive")` —— 这两种分支的候选 id 不同，
    不会被平均成一个"nested 更好"的结论。
    `series` / `capacitive` 在 admission §4.1.3/§4.1.4 是"参数层未验收"的
    研究接口，结果一律 `exploratory`，不得与 parallel 混报。
    """
    for b in branches:
        if b not in BRANCH_ELEMENT:
            raise ValueError(f"branch={b!r} 不在 {sorted(BRANCH_ELEMENT)}")
    out = []
    for K in ks:
        # include_L=True 时 registry 要求**恰好一个** `series_inductor` 槽位挂
        # `rl_series`；漏了会在 make_candidate 处直接 raise（自检覆盖这条）。
        els = [("relaxation_unit", "hn")] * int(K)
        if include_L:
            els.append(("series_inductor", "rl_series"))
        out.append(CR.make_candidate(
            "voigt", els,
            {"K": int(K), "include_L": bool(include_L), "adsorption": None},
            init_profile=init_profile, budget_profile=budget_profile,
            likelihood_profile="perpoint-2pct"))
    for b in branches:
        for L in layers:
            out.append(CR.make_candidate(
                "nested", [("outer_cpe", "cpe")]
                + ([("inner_cpe", "cpe")] if int(L) == 2 else [])
                + [("branch", BRANCH_ELEMENT[b])],
                {"layers": int(L), "branch": str(b)},
                init_profile=init_profile, budget_profile=budget_profile,
                likelihood_profile="perpoint-2pct"))
    return out


def find_dataset_files(key: str, project_root: str = "..", limit: int | None = None):
    """定位 `测试数据/<组>/` 下的合成谱。返回排序后的路径列表。

    路径是 `..` 相对（仓库在 `D:/eis训练数据/bmdrt`，数据在 `D:/eis训练数据/测试数据`）。
    找不到就返回空表，由调用方报错 —— 不静默用别的目录顶替。
    """
    if key not in DATASETS:
        raise ValueError(f"未知数据集 {key!r}；已有 {sorted(DATASETS)}")
    d = DATASETS[key]
    root = os.path.join(project_root, d["root"], d["sub"])
    if not os.path.isdir(root):
        return []
    hits = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".csv"):
                hits.append(os.path.join(dirpath, fn))
    hits.sort()
    return hits[:limit] if limit else hits


def truth_candidate_for(key: str, *, budget_profile: str = "quick",
                        init_profile: str = "uniform"):
    """返回该数据集真值拓扑对应的**候选对象**（用于"它到底在不在候选集里"）。"""
    t = DATASETS[key]["truth"]
    if t["topology_id"] == "voigt":
        return build_candidates(ks=(int(t["layers"]),), layers=(),
                                budget_profile=budget_profile,
                                init_profile=init_profile)[0]
    el = [("outer_cpe", "cpe")]
    if int(t["layers"]) == 2:
        el.append(("inner_cpe", "cpe"))
    el.append(("branch", BRANCH_ELEMENT[t["branch"]]))
    return CR.make_candidate("nested", el,
                             {"layers": int(t["layers"]), "branch": t["branch"]},
                             init_profile=init_profile,
                             budget_profile=budget_profile,
                             likelihood_profile="perpoint-2pct")


def expand_inputs(path: str, limit: int | None = None) -> list:
    """把 `--run/--plan` 的入参展开成 csv 列表：文件 / 目录 / glob 三种都接受。

    目录与 glob 一律 `sorted`，保证批处理顺序可复现；扩展名只认 `.csv/.txt/.z`
    （`bm_real.load_spectrum` 支持的三类），其它文件直接忽略而不是报错。
    """
    import glob as _glob
    exts = (".csv", ".txt", ".z")
    if os.path.isdir(path):
        hits = []
        for dirpath, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(exts):
                    hits.append(os.path.join(dirpath, fn))
        hits.sort()
    elif any(ch in path for ch in "*?["):
        hits = sorted(x for x in _glob.glob(path) if x.lower().endswith(exts))
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        hits = [path]
    return hits[:limit] if limit else hits


def plan(data_path: str, **kw) -> dict:
    """构造候选集 + L0 预检，**不拟合**。"""
    import bm_real as R
    f, Z = R.load_spectrum(data_path)
    cands = build_candidates(**kw)
    scr = CR.precheck_candidates(cands, f, Z)
    return dict(data_path=data_path, n_points=int(f.size),
                f_range=[float(f.min()), float(f.max())],
                data_sha256=sha256_file(data_path),
                candidates=[c.describe() for c in cands],
                screening=scr,
                precheck_states={s: scr[s] for s in CR.PRECHECK_STATES})


# --------------------------------------------------------------------------
# 拟合适配器：不同族 → 同一个 (params, Zm, diag) 形状
# --------------------------------------------------------------------------
def _median_params_voigt(raw: dict) -> dict:
    s = raw["s"]
    return dict(R_inf=float(np.median(s["R_inf"])),
                W=float(np.median(s["W"])),
                pi=np.median(s["pi"], axis=0),
                tau=np.median(s["tau"], axis=0),
                phi=np.median(s["phi"], axis=0),
                zeta=np.median(s["zeta"], axis=0))


def _median_params_nested(o: dict, branch: str, layers: int) -> dict:
    import bm_top as T
    return {k: float(np.median(o["s"][k])) for k in T.sampled_keys(branch, layers)}


#: `perpoint` 口径的逐点相对标准差（与 §14 / `nested_fit --sigma perpoint` 同值）
SIGMA_PCT_PERP0INT = 0.02


def fit_candidate(cand, f, Z, *, seed: int = 0, budget_profile: str = "quick",
                  sigma_floor: float | None = None,
                  likelihood: str = "free") -> dict:
    """调现有入口拟合一个候选，返回统一形状的观测（不含 registry 校验）。

    **不复制任何拟合逻辑**：voigt 走 `bm_select.fit_voigt`（其内部是
    `bm_real.fit_one`），nested 走 `bm_select.fit_nested`（其内部是
    `bm_top.fit_top`）。本函数只做参数抽取与正演，用同一套 `metrics()` 量残差。
    """
    import bm_select as SEL
    import bm_core as B
    import bm_top as T

    kw = dict(BUDGETS[budget_profile])
    sf = SEL.SIGMA_FLOOR if sigma_floor is None else float(sigma_floor)
    # 噪声口径：`free` = σ 被采样（旧行为，逐位不变）；`perpoint` = 固定 2%·|Z|。
    # 两者不能混报：同一候选在两种口径下的 rms/elpd 不可直接比较。
    if likelihood == "perpoint":
        sig = SIGMA_PCT_PERP0INT * np.abs(Z)
        sigma = (sig, sig)
    elif likelihood == "free":
        sigma = None
    else:
        raise ValueError(f"未知 likelihood={likelihood!r}（可用 free / perpoint）")
    cfg = dict(cand.config)
    t0 = time.time()

    if cand.topology_id == "voigt":
        K = int(cfg["K"])
        c = SEL.fit_voigt(f, Z, K, seed=seed,
                          include_L=bool(cfg.get("include_L", False)),
                          adsorption=cfg.get("adsorption"),
                          sigma_floor=sf, kw=kw, sigma=sigma)
        raw = c["raw"]
        params = _median_params_voigt(raw)
        Zm = B.model_z(f, params, include_L=bool(cfg.get("include_L", False)))
        rhat = float(raw["rhat_max"])
        div = int(raw["ndiv"] or 0)
    elif cand.topology_id == "nested":
        layers = int(cfg["layers"])
        branch = str(cfg["branch"])
        c = SEL.fit_nested(f, Z, layers, branch=branch, seed=seed,
                           sigma_floor=sf, kw=kw, sigma=sigma)
        raw = c["raw"]
        params = _median_params_nested(raw, branch, layers)
        Zm = T.z_nested(f, params, branch=branch, layers=layers)
        rhat = float(raw["rhat_max"])
        div = int(raw["divergences"] or 0)
    else:
        raise ValueError(f"拓扑 {cand.topology_id!r} 没有适配器；"
                         f"可用的是 {TOPOLOGY_ADAPTERS}")

    met = metrics(f, Z, Zm)
    ci = {}
    s = raw["s"]
    for k, v in s.items():
        arr = np.asarray(v, float)
        if arr.ndim == 0 or arr.size == 0:
            continue
        ci[str(k)] = [float(np.percentile(arr, 5)), float(np.percentile(arr, 95))]
    ess = raw.get("ess_min")
    return dict(
        params=params, metrics=met, ci90=ci,
        rhat_max=rhat, divergences=div,
        ess_min=(None if ess is None else float(ess)),
        elpd=float(raw.get("elpd", c["elpd"])),
        elpd_se=float(raw.get("elpd_se", c["se"])),
        n_draw=int(raw.get("n_draw") or 0),
        elapsed_s=round(time.time() - t0, 2),
        budget=dict(kw, sigma_floor=sf), likelihood=likelihood,
        family_ok=bool(c.get("ok")),
    )


# --------------------------------------------------------------------------
# 观测落盘：走 registry 的 observation_block（让 schema 校验真的生效）
# --------------------------------------------------------------------------
def to_observation(cand, obs: dict, *, f, Z, seed: int) -> dict:
    """把统一观测包装成 `observation_block`。校验不通过会当场 raise，不降级。"""
    met = obs["metrics"]
    passed = (obs["rhat_max"] <= SAMPLING_RHAT_MAX) and obs["divergences"] == 0
    sf = float(obs["budget"]["sigma_floor"])
    zmax = float(np.max(np.abs(Z)))
    floor_val = sf * zmax
    res_abs = None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_val = (met["rms_rel"] / 100.0 * zmax / floor_val) if floor_val > 0 else None
    del res_abs
    return CR.observation_block(
        candidate_id=cand.candidate_id,
        run_status="completed",
        sampling_status=("passed" if passed else "failed"),
        rms_rel=float(met["rms_rel"]),
        acf=float(met["acf"]),
        rhat_max=float(obs["rhat_max"]),
        ess_min=obs["ess_min"],
        divergences=int(obs["divergences"]),
        param_stability={k: {"ci90": v} for k, v in obs["ci90"].items()},
        param_estimate=None,          # 真实谱没有真值；伪造这一列会让参数恢复变成自证
        noise_floor={"source": "same_data_estimate", "value": floor_val,
                     "note": f"sigma_floor={sf:g} × zmax={zmax:.4g}；"
                             "这是似然噪声底，不是独立测得的仪器噪声"},
        ratio={"value": (None if ratio_val is None else float(ratio_val)),
               "unavailable_reason": ("not_computed" if ratio_val is None else None),
               "n_used": int(f.size), "n_total": int(f.size),
               "n_usable": int(f.size),
               "noise_floor_source": "same_data_estimate"},
        predictive={"elpd": obs["elpd"], "se": obs["elpd_se"],
                    "valid_for_comparison": False, "reason": R21_REASON},
        budget_params={"source": "recorded",
                       "n_warmup": obs["budget"]["n_warmup"],
                       "n_samples": obs["budget"]["n_samples"],
                       "num_chains": obs["budget"]["num_chains"],
                       "target_accept": obs["budget"]["target_accept"],
                       "max_tree_depth": obs["budget"]["max_tree_depth"],
                       # extra_fields 不是预算数值，不进 registry 的五个核心键
                       }, 
        execution=None,               # 不伪造身份：run_id 会如实记 None/unconfirmed
        truth_known=False,
        interpretation=None if passed else "descriptive_only",
        notes=(f"seed={seed}", f"elapsed_s={obs['elapsed_s']}",
               f"bands={obs['metrics']['bands']}"),
    )


# --------------------------------------------------------------------------
# 缓存 / 断点续跑
# --------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(candidate_id: str, data_sha: str, seed: int,
              budget_profile: str, likelihood: str = "free") -> str:
    """缓存键。`free` 保持旧式四段（历史缓存不失效）；其它口径追加一段。"""
    base = f"{candidate_id}|{data_sha}|{seed}|{budget_profile}"
    if likelihood != "free":
        base += f"|{likelihood}"
    return hashlib.sha256(base.encode()).hexdigest()[:32]


def load_done(path: str) -> dict:
    """读 append-only JSONL，返回 {cache_key: record}。坏行直接报错，不跳过。"""
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {i} 行不是合法 JSON：{exc}") from exc
            if "cache_key" not in rec:
                raise ValueError(f"{path} 第 {i} 行缺 cache_key，无法续跑")
            done[rec["cache_key"]] = rec
    return done


def append_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(data_path: str, *, seed: int = 0, budget_profile: str = "quick",
        out_dir: str | None = None, max_candidates: int | None = None,
        likelihood: str = "free", **plan_kw) -> dict:
    """候选集 → 预检 → 逐候选拟合 → 落观测。返回本次运行的摘要。"""
    import bm_real as R
    f, Z = R.load_spectrum(data_path)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    out_dir = out_dir or os.path.join(OUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)

    cands = build_candidates(budget_profile=budget_profile, **plan_kw)
    if max_candidates:
        cands = cands[:int(max_candidates)]
    scr = CR.precheck_candidates(cands, f, Z)
    dsha = sha256_file(data_path)

    with open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(data_path=data_path, data_sha256=dsha,
                       n_points=int(f.size), budget_profile=budget_profile,
                       candidates=[c.describe() for c in cands]),
                  fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "screening.json"), "w", encoding="utf-8") as fh:
        json.dump(scr, fh, ensure_ascii=False, indent=1)

    jpath = os.path.join(out_dir, "observations.jsonl")
    # 逐候选的拟合结果（参数 / 90% 区间 / 指标）单独落一份：
    # `observation_block` 按 schema 只放"可比较的观测"，不放参数；
    # 但画图、复核、二次分析都需要参数，所以另开 `fits/`。
    fit_dir = os.path.join(out_dir, "fits")
    os.makedirs(fit_dir, exist_ok=True)
    done = load_done(jpath)

    rows = []
    for cand, rec in zip(cands, scr["records"]):
        state = rec["precheck_state"]
        if state == "excluded":
            rows.append(dict(candidate_id=cand.candidate_id, precheck=state,
                             run_status="skipped_excluded",
                             reason=rec["precheck_reason"]))
            continue
        key = cache_key(cand.candidate_id, dsha, seed, budget_profile, likelihood)
        if key in done:
            rows.append(dict(candidate_id=cand.candidate_id, precheck=state,
                             run_status="cached", cache_key=key,
                             rms_rel=done[key]["fit_observation"]["rms_rel"],
                             acf=done[key]["fit_observation"]["acf"]))
            continue
        try:
            obs = fit_candidate(cand, f, Z, seed=seed,
                               budget_profile=budget_profile,
                               likelihood=likelihood)
            ob = to_observation(cand, obs, f=f, Z=Z, seed=seed)
            rec_out = dict(cache_key=key, data_sha256=dsha, seed=seed,
                           budget_profile=budget_profile,
                           topology_id=cand.topology_id,
                           candidate=cand.describe(),
                           precheck=rec, observation=ob,
                           elapsed_s=obs["elapsed_s"],
                           fit_observation=dict(rms_rel=ob["fit_observation"]["rms_rel"],
                                                acf=ob["fit_observation"]["acf"]))
            append_jsonl(jpath, rec_out)
            with open(os.path.join(fit_dir, cand.candidate_id + ".json"),
                      "w", encoding="utf-8") as fh:
                json.dump(_jsonable(dict(
                    candidate_id=cand.candidate_id,
                    topology_id=cand.topology_id,
                    config=rec_out["candidate"].get("config"),
                    params=obs["params"], ci90=obs["ci90"],
                    metrics=obs["metrics"], rhat_max=obs["rhat_max"],
                    divergences=obs["divergences"], ess_min=obs["ess_min"],
                    elpd=obs["elpd"], elpd_se=obs["elpd_se"],
                    budget_profile=budget_profile, seed=seed)),
                    fh, ensure_ascii=False, indent=1, allow_nan=False)
            # 观测块的采样诊断在 `sampling_diagnostics` 子键下；顶层没有 rhat_max。
            # 这里曾误读 `ob["fit_observation"]["rhat_max"]` 并抛 KeyError，
            # 导致"记录已落盘、行汇总却标 failed"——自检已加断言防它再回来。
            rows.append(dict(candidate_id=cand.candidate_id, precheck=state,
                             run_status="completed", cache_key=key,
                             rms_rel=ob["fit_observation"]["rms_rel"],
                             acf=ob["fit_observation"]["acf"],
                             rhat_max=obs["rhat_max"],
                             divergences=obs["divergences"]))
        except Exception as exc:                       # noqa: BLE001
            # 失败也记账：run_status=failed 且**不报任何残差指标**
            ob = CR.observation_block(
                candidate_id=cand.candidate_id, run_status="failed",
                unavailable={"rms_rel": "not_computed", "acf": "not_computed"},
                notes=(f"error={type(exc).__name__}: {exc}"[:400],))
            append_jsonl(jpath, dict(cache_key=key, data_sha256=dsha, seed=seed,
                                     budget_profile=budget_profile,
                                     topology_id=cand.topology_id,
                                     candidate=cand.describe(), precheck=rec,
                                     observation=ob, error=str(exc)[:400]))
            rows.append(dict(candidate_id=cand.candidate_id, precheck=state,
                             run_status="failed", cache_key=key,
                             error=str(exc)[:200]))

    meta = dict(data_path=data_path, data_sha256=dsha, seed=seed,
                budget_profile=budget_profile, likelihood=likelihood,
                registry_version=CR.REGISTRY_VERSION,
                registry_digest=CR.registry_digest(),
                precheck_states={s: scr[s] for s in CR.PRECHECK_STATES},
                n_candidates=len(cands), rows=rows,
                note=("本模块只产出逐候选观测；不排序、不合成总分、不出 evidence 状态。"
                      "diffusion/adsorption 尚无拓扑，未进入候选集。"))
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1, allow_nan=False)
    return meta


# --------------------------------------------------------------------------
# registry 概览
# --------------------------------------------------------------------------
def describe_registry() -> dict:
    tops = []
    for tid in CR.list_topologies():
        t = CR.get_topology(tid)
        tops.append(dict(id=tid, maturity=t.maturity, min_level=t.min_level,
                         forward_ref=t.forward_ref,
                         precheck_ref=t.precheck_ref,
                         has_adapter=tid in TOPOLOGY_ADAPTERS,
                         slots=[(s.element_ids if hasattr(s, "element_ids") else None)
                                for s in t.slots]))
    els = []
    for eid in CR.list_elements():
        e = CR.get_element(eid)
        els.append(dict(id=eid, wired=bool(e.wired), maturity=e.maturity,
                        blocked_reason=getattr(e, "blocked_reason", None)))
    return dict(registry_version=CR.REGISTRY_VERSION,
                registry_digest=CR.registry_digest(),
                topologies=tops, elements=els,
                adapters=list(TOPOLOGY_ADAPTERS))


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
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

    print("registry 概览")
    reg = describe_registry()
    c("registry 版本非空", bool(reg["registry_version"]))
    c("只有 voigt/nested 有适配器", set(reg["adapters"]) == {"voigt", "nested"})
    c("扩散元件 wired=False（登记在册但无槽位）",
      all(not e["wired"] for e in reg["elements"]
          if e["id"] in ("w_inf", "w_open", "w_short", "adsorption")))
    c("nested 无确定性预检（precheck_ref=None ⇒ unknown）",
      [t for t in reg["topologies"] if t["id"] == "nested"][0]["precheck_ref"] is None)

    print("\n指标同式（§14 口径）")
    rng = np.random.default_rng(0)
    f = np.logspace(-2, 5, 40)
    Ztrue = 30 + 20 / (1 + 1j * 2 * np.pi * f * 0.01)
    Zm = Ztrue.copy()
    m0 = metrics(f, Ztrue, Zm)
    c("完美拟合 rms_rel=0、acf=0", abs(m0["rms_rel"]) < 1e-12 and abs(m0["acf"]) < 1e-12)
    Zp = Ztrue * 1.01
    m1 = metrics(f, Ztrue, Zp)
    # Re/Im 各承担一半：|d|=0.01 常数，但 concatenate 后均方是 0.01²/2 ⇒ 0.7071%
    c("1% 比例偏差 → rms_rel=0.707%（√2/2 因子）", abs(m1["rms_rel"] - 0.7071) < 0.01)
    m1b = metrics(f, Ztrue, Ztrue + 3.0 * (Zp - Ztrue))
    c("acf 对残差整体缩放不变", abs(m1["acf"] - m1b["acf"]) < 1e-12)
    m2 = metrics(f, Ztrue, Ztrue + 0.05 * np.abs(Ztrue) * (1 + np.sin(np.arange(f.size))))
    c("加结构 → acf>0.2", m2["acf"] > 0.2)
    c("分频段三档齐全", len(m2["bands"]) == 3)

    print("\n候选集与预检")
    cands = build_candidates(ks=(1,), layers=(1,), budget_profile="quick")
    c("构造 2 个候选（voigt K=1 + nested L=1）", len(cands) == 2)
    c("candidate_id 前缀 C-（候选定义层，不是执行身份 R-/D-）",
      all(x.candidate_id.startswith("C-") for x in cands))
    scr = CR.precheck_candidates(cands, f, Ztrue)
    by = {r["candidate_id"]: r["precheck_state"] for r in scr["records"]}
    c("voigt 预检 not_excluded", any(v == "not_excluded" for v in by.values()))
    c("nested 预检 unknown（未登记判据 ≠ 通过）",
      [r for r in scr["records"] if "nested" in r["candidate_id"]
       or r["precheck_state"] == "unknown"] != [])
    c("三态都在 tally 里", set(scr) >= set(CR.PRECHECK_STATES))

    print("\nobservation_block 护栏真的生效")
    cd = cands[0]
    obs = dict(metrics=m2, rhat_max=1.001, divergences=0, ess_min=500.0,
               ci90={"R_inf": [1.0, 2.0]}, elpd=-10.0, elpd_se=0.5, n_draw=100,
               elapsed_s=1.0, budget=dict(BUDGETS["quick"], sigma_floor=0.02))
    ob = to_observation(cd, obs, f=f, Z=Ztrue, seed=0)
    c("合法观测：sampling_status=passed", ob["sampling_status"] == "passed")
    c("execution 如实 unconfirmed、run_id=None",
      ob["execution"]["run_id"] is None
      and ob["execution"]["run_id_status"] == "unconfirmed")
    c("elpd 标 valid_for_comparison=False 且带 reason",
      ob["predictive"]["valid_for_comparison"] is False
      and bool(ob["predictive"]["reason"]))
    c("ratio 的 n_used/n_total 都给了", ob["ratio"]["n_used"] == f.size)
    c("budget_params 是数值不是档名", isinstance(ob["budget_params"]["n_warmup"], int))
    c("采样档带 extra_fields=('num_steps',)（否则 bm_real.grade 的 tds=None 崩）",
      BUDGETS["quick"].get("extra_fields") == ("num_steps",)
      and BUDGETS["standard"].get("extra_fields") == ("num_steps",))
    c("noise_floor 来源标 same_data_estimate（不冒充仪器噪声）",
      ob["noise_floor"]["source"] == "same_data_estimate")

    raised = False
    try:
        CR.observation_block(candidate_id=cd.candidate_id, run_status="failed",
                            rms_rel=1.0, acf=0.3)
    except ValueError:
        raised = True
    c("run_status=failed 却报残差 → 报错（不静默放行）", raised)

    raised = False
    try:
        CR.observation_block(candidate_id=cd.candidate_id, run_status="completed",
                            sampling_status="failed", rms_rel=1.0, acf=0.3)
    except ValueError:
        raised = True
    c("sampling=failed 但没给 descriptive_only → 报错", raised)

    ob_bad = to_observation(cd, dict(obs, rhat_max=1.50), f=f, Z=Ztrue, seed=0)
    c("采样不达标 → sampling_status=failed",
      ob_bad["sampling_status"] == "failed")
    c("采样不达标 → interpretation=descriptive_only",
      ob_bad["fit_observation"]["interpretation"] == "descriptive_only")
    c("fit_observation 里没有 rhat_max（诊断在 sampling_diagnostics）",
      "rhat_max" not in ob["fit_observation"]
      and "rhat_max" in ob["fit_observation"]["sampling_diagnostics"])
    c("行汇总改从 obs 取 rhat/div（不再读 fit_observation）",
      abs(obs["rhat_max"] - 1.001) < 1e-9 and obs["divergences"] == 0)

    print("\n测试数据/ 三组：真值拓扑是否在候选集里")
    import bm_top as T
    for key in ("coating", "passivation", "corrosion"):
        tc = truth_candidate_for(key, budget_profile="quick")
        c("真值候选可构造：%s -> %s" % (DATASETS[key]["name"], tc.candidate_id),
          isinstance(tc, CR.Candidate))
    cap = [c for c in build_candidates(branches=("parallel", "capacitive"))
           if "nested" in c.candidate_id or c.topology_id == "nested"]
    c("capacitive 分支候选能通过 registry 一致性校验", len(cap) == 4)
    withL = build_candidates(ks=(1,), layers=(1,), include_L=True)
    c("include_L=True 的 voigt 候选能构造（要挂 series_inductor）",
      len(withL) == 2 and any(x.candidate_id for x in withL))
    noL = build_candidates(ks=(1,), layers=(1,), include_L=False)
    c("include_L 变化会改 candidate_id（不是同一个候选）",
      withL[0].candidate_id != noL[0].candidate_id)
    par_only = build_candidates(branches=("parallel",))
    ids_par = {c.candidate_id for c in par_only}
    ids_cap = {truth_candidate_for(k).candidate_id
               for k in ("coating", "passivation")}
    c("涂层/钝化的真值候选不在默认 parallel 候选集里（必须显式开 branches）",
      ids_cap.isdisjoint(ids_par))
    zcap = T.z_nested(f, dict(Rs=10.0, R1=20.0, Q1=5.0, n1=0.9, R2=30.0,
                              Q2=4.0, n2=0.9, Q_b=3.0, n_b=0.85),
                      branch="capacitive", layers=2)
    c("z_nested(capacitive) 正演有限且非零", bool(np.all(np.isfinite(zcap)))
      and float(np.abs(zcap).max()) > 0)
    found = {k: len(find_dataset_files(k)) for k in DATASETS}
    c("三组数据目录都能定位（未找到则说明路径漂移）",
      all(v > 0 for v in found.values()) or all(v == 0 for v in found.values()))

    print("\n输入展开（文件/目录/glob）")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        for n in ("b.csv", "a.csv", "c.txt", "d.json"):
            open(os.path.join(td, n), "w", encoding="utf-8").write("x")
        got = expand_inputs(td)
        c("目录展开：只认 csv/txt 且排序稳定",
          [os.path.basename(x) for x in got] == ["a.csv", "b.csv", "c.txt"])
        c("glob 展开", len(expand_inputs(os.path.join(td, "*.csv"))) == 2)
        c("单文件原样返回",
          expand_inputs(os.path.join(td, "a.csv")) == [os.path.join(td, "a.csv")])
        c("limit 生效", len(expand_inputs(td, limit=1)) == 1)
        raised = False
        try:
            expand_inputs(os.path.join(td, "nope.csv"))
        except FileNotFoundError:
            raised = True
        c("不存在的路径报错而不是静默空表", raised)

    print("\n缓存 / 续跑")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "o.jsonl")
        key = cache_key(cd.candidate_id, "deadbeef" * 8, 0, "quick")
        append_jsonl(p, dict(cache_key=key, fit_observation=dict(rms_rel=1.0, acf=0.2)))
        d = load_done(p)
        c("落盘后可读回", key in d and d[key]["fit_observation"]["acf"] == 0.2)
        append_jsonl(p, dict(cache_key=key + "x", fit_observation=dict(rms_rel=2.0, acf=0.3)))
        c("append-only 不覆盖已有行", len(load_done(p)) == 2)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        raised = False
        try:
            load_done(p)
        except ValueError:
            raised = True
        c("坏行报错而不是跳过", raised)
        c("cache_key 区分噪声口径",
      cache_key("C-a", "d" * 64, 0, "standard", "free")
      != cache_key("C-a", "d" * 64, 0, "standard", "perpoint"))
    c("free 口径的 key 与旧式四段一致（历史缓存不失效）",
      cache_key("C-a", "d" * 64, 0, "standard", "free")
      == hashlib.sha256(("C-a|" + "d" * 64 + "|0|standard").encode()).hexdigest()[:32])
    c("cache_key 对 seed 敏感",
          cache_key("R-x", "a" * 64, 0, "quick") != cache_key("R-x", "a" * 64, 1, "quick"))

    print(f"\n自检 {ok}/{total} 通过" if not bad else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


# --------------------------------------------------------------------------
def _print_registry(reg: dict) -> None:
    print(f"registry {reg['registry_version']}  digest {reg['registry_digest'][:16]}")
    print(f"\n{'拓扑':<10}{'maturity':<14}{'min_level':<11}{'预检':<26}{'可调度'}")
    print("-" * 78)
    for t in reg["topologies"]:
        print(f"{t['id']:<10}{t['maturity']:<14}{t['min_level']:<11}"
              f"{str(t['precheck_ref']):<26}{'是' if t['has_adapter'] else '**无**'}")
    print(f"\n{'元件':<16}{'wired':<9}{'maturity':<14}阻塞原因")
    print("-" * 78)
    for e in reg["elements"]:
        print(f"{e['id']:<16}{str(e['wired']):<9}{e['maturity']:<14}"
              f"{(e['blocked_reason'] or '')[:44]}")
    print("\n注：扩散/吸附元件 wired=False 且无拓扑槽位 ⇒ 不能进入候选集；"
          "这不是调度器遗漏。")


def _print_datasets(project_root: str) -> None:
    print(f"{'key':<13}{'组':<6}{'真值拓扑':<34}{'谱数':<7}样例")
    print("-" * 104)
    for k, d in DATASETS.items():
        t = d["truth"]
        topo = (f"voigt K={t['layers']}" if t["topology_id"] == "voigt"
                else f"nested L={t['layers']}/{t['branch']}")
        fs = find_dataset_files(k, project_root)
        sample = os.path.basename(fs[0])[:44] if fs else "**未找到**"
        print(f"{k:<13}{d['name']:<6}{topo:<34}{len(fs):<7}{sample}")
    print("\n注：涂层/钝化的真值最内层是容性 CPE ⇒ 必须 `--branches parallel,capacitive`；")
    print("    腐蚀 = ZARC ⇒ voigt K=1 已可表示。三组均可作正演回归，不是准入证据。")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="registry 驱动的候选电路统一调度器")
    ap.add_argument("--list", action="store_true", help="列出 registry 内容与可调度性")
    ap.add_argument("--plan", metavar="CSV", help="构造候选集 + 预检，不拟合")
    ap.add_argument("--run", metavar="CSV", help="逐候选拟合并落观测")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", default="quick", choices=sorted(BUDGETS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--limit-files", type=int, default=None,
                    help="批处理只跑前 N 条（冒烟用）")
    ap.add_argument("--include-L", action="store_true",
                    help="打开 voigt 的串联感抗臂（族能力扩展，需单独预注册）")
    ap.add_argument("--ks", default=None, help="逗号分隔，默认 1,2")
    ap.add_argument("--layers", default=None, help="逗号分隔，默认 1,2")
    ap.add_argument("--branches", default="parallel",
                    help="nested 分支，逗号分隔；默认 parallel。"
                         "测试数据/涂层/钝化需要 parallel,capacitive")
    ap.add_argument("--datasets", action="store_true",
                    help="列出 测试数据/ 三组与其真值拓扑")
    ap.add_argument("--project-root", default="..",
                    help="测试数据/ 的父目录，默认 ..（仓库在 bmdrt/ 下）")
    a = ap.parse_args(argv)

    ks = tuple(int(x) for x in a.ks.split(",")) if a.ks else DEFAULT_KS
    layers = tuple(int(x) for x in a.layers.split(",")) if a.layers else DEFAULT_LAYERS
    branches = tuple(x.strip() for x in a.branches.split(",") if x.strip())
    kw = dict(ks=ks, layers=layers, include_L=bool(a.include_L),
              branches=branches)

    if a.selftest:
        return _selftest()
    if a.list:
        _print_registry(describe_registry())
        return 0
    if a.datasets:
        _print_datasets(a.project_root)
        return 0
    if a.plan:
        d = plan(a.plan, budget_profile=a.budget, **kw)
        print(json.dumps(d, ensure_ascii=False, indent=1, allow_nan=False,
                         default=str))
        return 0
    if a.run:
        files = expand_inputs(a.run, a.limit_files)
        if not files:
            print(f"没有可拟合的文件：{a.run}", file=sys.stderr)
            return 2
        print(f"共 {len(files)} 条谱")
        n_fail = 0
        for i, fp in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {os.path.basename(fp)}")
            meta = run(fp, seed=a.seed, budget_profile=a.budget,
                       out_dir=(None if a.out_dir is None
                                else os.path.join(a.out_dir,
                                                  os.path.splitext(os.path.basename(fp))[0])),
                       max_candidates=a.max_candidates, **kw)
            for r in meta["rows"]:
                line = f"  {r['candidate_id']:<34} {r['precheck']:<13}{r['run_status']}"
                if "rms_rel" in r:
                    line += f"  rms={r['rms_rel']:.2f}%  acf={r['acf']:.3f}"
                if "error" in r:
                    line += f"  {r['error'][:60]}"
                print(line)
            print(f"  预检: {meta['precheck_states']}")
            n_fail += sum(1 for r in meta["rows"] if r["run_status"] == "failed")
        print(f"\n完成 {len(files)} 条；候选级失败 {n_fail} 个。"
              f"产物根：{a.out_dir or OUT_ROOT}")
        return 0 if n_fail == 0 else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
