# -*- coding: utf-8 -*-
"""map_stage0_records.py —— 把阶段 0 的历史产物映射成机器可读的观测记录（v4）。

目的**不是**让历史数据好看，而是检验 `circuit_registry.observation_block` 到底
能不能承载它们。凡是映射不了的字段，一律进 `unmapped` 并写明原因，不丢、不改、
不补猜。映射过程**不跑任何拟合**，只读 `evidence/` 与原始谱。

v4 相对 v3 改的三件事（第四轮审计，全部成立）
-------------------------------------------
1. **确认重复的门槛提到 `recorded`**。v3 的 `duplicate_verdict` 只查 seed 与预算两项，
   于是本批 239 条全是 `D-`（没有产物记代码版本）的情况下，仍把两对判成
   `confirmed_duplicate`——用自己定义的门槛之外的两项去确认，等于把 `R-` 的定义降成摆设。
   v4 要求两侧执行身份**整体**是 `recorded`（含代码版本与数据内容哈希）⇒ 本批 0 条满足，
   两对降为 `possible_duplicate`，`n_independent` 由 237 变成 **239**（严格口径 237）。
   **两个数必须一起报**：只给 239 会把疑似重跑算成新信息，只给 237 是把猜当结论。
2. **数据身份改用内容哈希**。v3 的 `data.ref` 是文件名，两条不同实验的谱都叫
   `spectrum.csv` 就会撞 `run_id`——这与"60 条谱不能共享一个号"是同一个错。
   v4 从原始谱算 sha256（63 份），拿不到时留 `ref_only` 并**禁止**参与确认重复。
3. **`--check` 校验整个快照**。v3 只比重算的两份数据文件，没碰 MANIFEST 里的
   `registry_version`/`registry_digest`/`source_sha256`；注册表改了而记录序列化没变时会
   "通过"。v4 用 `verify_frozen()` 分四层查，并支持 `--frozen <dir>` 供测试指向篡改副本。

v3 相对 v2 改的两件事（都是审计指出的，不是措辞）
-------------------------------------------------
1. **去重不再用"摘要哈希相同"当证据**。v2 的 `_fingerprint()` 只哈希
   candidate + fit_observation + param_estimate，因此"摘要相同"被直接写成了
   `independent=false`——而摘要相同并不能证明两条记录来自同一次（甚至同一个）运行。
   v3 把判定交给 `circuit_registry.duplicate_verdict()`：输入是**来源**
   （产物 path/sha、seed 及其证据等级、预算数值及其来源、run_id），
   `summary_equal` 只是其中一个输入事实。三态：`distinct` /
   `possible_duplicate`（标记但**不**从独立样本数里扣）/ `confirmed_duplicate`。
2. **新增执行身份 `run_id`**（`circuit_registry.execution_identity`）。
   `candidate_id` 只哈希档名（`"standard"`），不含 `target_accept=0.98` 这类实际
   数值；同名的两个 `standard` 因此可以拿到同一个 `candidate_id`。执行层单独记
   候选 + 数据 + seed + 实际预算 + 软件，`candidate_id` 保持不动。

v3 顺带修的两处**身份错误**（v2 遗留，与第 1 轮审计指出的是同一类）
------------------------------------------------------------------
3. **pilot 的 `sensitivity` 段不是 nested**。代码是
   `voigt_fit(f, Z, None, seed=0, init=voigt_init(f, Z))`——voigt K=2、自由 σ、
   数据驱动初值，只对 `verdict=unusable_sampling` 的样本重跑。v2 把它映射成
   nested/capacitive/L2，于是这些行与真正的 `nested` 段**共享 candidate_id**，
   残差被记到了另一个模型名下。
4. **pilot 的 `voigt_prod` / `voigt_fix2` 用的是默认初始化**（`init=None`），
   不是 `voigt_init`。`bm_core.fit:603` 只有 `if init_values:` 才设
   `init_to_value`，None 走 numpyro `init_to_uniform`。v2 因为读到
   `voigt_init` 的 docstring（"数据驱动初值"）就把 120 行标成 `data_driven`，
   方向反了——**函数文档描述的是它自己，不是谁在调用它**。

四条口径必须先说清
------------------
- `run_status` 与 `sampling_status` 分开判：`nested.json` 的 `status` 是**成熟度标签**
  （恒 `"exploratory"`），不是执行状态，不可直连；R̂/ESS 在 `diagnostics` 里。
- 初始化**不来自**目录名、也不来自函数 docstring，来自**调用点**：
  `nested_fit.py:339` 调 `T.fit_top(...)` 不传 `init_values`，`bm_top.fit_top:1006`
  把它当 `init_params=None` ⇒ numpyro 默认 `init_to_uniform`。
- 每个 `nested.json` 的 `files.json` 记录它**自己写到哪**，`input` 记录 `f_min/f_max`
  与含真值参数的文件名 ⇒ 原始目录与真值都能从产物自身复原，不必靠 MANIFEST 猜。
- `1113` 那两份 `elapsed` 不同、指标逐位相同：同候选、同 seed（`config.seed=0`，
  **产物自己写的**）、同预算数值 ⇒ 两次执行、一次信息，计数时合并。

用法：
    python docs/round_2026-11-13/map_stage0_records.py            # 写到 out/
    python docs/round_2026-11-13/map_stage0_records.py --freeze   # 另存冻结副本 + MANIFEST
    python docs/round_2026-11-13/map_stage0_records.py --check    # 与冻结副本逐字节比对
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import circuit_registry as R  # noqa: E402

EVID = _ROOT / "evidence" / "stage0_three_spectra"
FROZEN = _ROOT / "evidence" / "stage0_records"
#: pilot 的原始谱目录（与 `coating_voigt_pilot.py:21` 同一处）。用它给 60 条记录
#: 补内容哈希；目录不在时退成 `ref_only`，不假装知道数据内容。
_PILOT_SUB = Path("D:/eis训练数据/测试数据/涂层/"
                  "Coating_Generated EIS data and Nyquist plots/"
                  "Generated EIS data and Nyquist plots")


def _resolve(p, *, bases=(_ROOT, _PILOT_SUB)):
    """产物里记的路径 → 磁盘路径。绝对路径直接用，相对路径依次试 `bases`。"""
    if p is None or not str(p).strip():
        return None
    q = Path(str(p))
    if q.is_absolute():
        return q
    for b in bases:
        cand = Path(b) / q
        if cand.is_file():
            return cand
    return q


# --------------------------------------------------------------------------
# 生成脚本 → 初始化档。判定依据写在这里，不写"档名"了事。
# 规则必须是**调用点**，不是函数 docstring：`voigt_init` 的 docstring 说自己是
# 数据驱动的，但 pilot 的 `voigt_prod`/`voigt_fix2` 传的是 `init=None`。
# --------------------------------------------------------------------------
_INIT_RULES = {
    "nested_fit.py": dict(
        init_profile="uniform",
        init_source="producer_code:nested_fit.py",
        init_note="nested_fit.py:339 调 T.fit_top(...) 未传 init_values；"
                  "bm_top.fit_top:1006 把它当 init_params=None ⇒ numpyro 默认 "
                  "init_to_uniform。产物 config 里确实没有 init 字段，"
                  "所以这是**由生成代码判定**，不是产物记录。",
    ),
    "bm_real.fit_one": dict(
        init_profile="uniform",
        init_source="producer_code:bm_real.fit_one",
        init_note="bm_real.fit_one 未传 init_values ⇒ init_to_uniform。"
                  "调用链 fit_family_compare→bm_select.fit_voigt→bm_real.fit_one。",
    ),
    "coating_voigt_pilot.py:voigt_default": dict(
        init_profile="uniform",
        init_source="producer_code:coating_voigt_pilot.py:voigt_fit(init=None)",
        init_note="voigt_prod / voigt_fix2 调 voigt_fit(f,Z,...) 不传 init ⇒ "
                  "voigt_fit:160 把 init=None 传给 C.fit(init_values=None) ⇒ "
                  "bm_core.fit:603 只在 `if init_values:` 时设 init_to_value，"
                  "None 走 numpyro init_to_uniform。"
                  "**注意**：voigt_init 的 docstring 描述的是它自己，"
                  "不是这两个调用点（v2 在这里读错了）。",
    ),
    "coating_voigt_pilot.py:voigt_data_init": dict(
        init_profile="data_driven",
        init_source="producer_code:coating_voigt_pilot.py:voigt_fit(init=voigt_init)",
        init_note="只有 sensitivity 段传了 init=voigt_init(f,Z)（:268）："
                  "R_inf=min|Z|、W=(zmax−zmin)/2、τ 窗内均铺、φ=0.8、v=1/K，"
                  "只看 f 与 |Z|，不读真值。",
    ),
    "coating_voigt_pilot.py:nested": dict(
        init_profile="data_driven",
        init_source="producer_code:coating_voigt_pilot.py:data_init",
        init_note="nested 段（:219）用 data_init 经 physical_to_unconstrained 传入；"
                  "由 f/|Z| 的 min/zmax/中心频点算出，不读真值。",
    ),
    "diag_passivation_init.py": dict(
        init_profile="value",
        init_source="producer_code:diag_passivation_init.py",
        init_note="初值直接取**文件名里的真值参数**（physical_to_unconstrained）。"
                  "这是 oracle，生产环境拿不到——只能证明拓扑能表达，"
                  "不证明该候选在生产可达的初值下能拟合。",
    ),
}

#: 文件名里的真值参数 → 注册表参数名。Q 需按 `Q_ours=Q_data/ω_ref^n` 换算。
_TRUTH_KEYS = {
    "coating": ["Rs", "Qa", "na", "R1", "Qb", "nb", "R2", "Qc", "nc"],
    "passivation": ["Qa", "na", "Qb", "nb", "R1", "Rs"],
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: 本次映射实际哈希过的**数据文件**：路径 → sha256。冻结清单会把它落盘，
#: `--check` 据此逐份复核原始谱有没有被改（审计 D5）。每次构建前清空。
_HASHED_FILES: dict = {}


def _sha256_file(p) -> str | None:
    """数据文件的**内容**哈希。文件不在（换了机器、原始目录被清）就返回 None。"""
    if p is None:
        return None
    p = Path(p)
    try:
        if not p.is_file():
            return None
        h = _sha256(p)
    except OSError:
        return None
    _HASHED_FILES[str(p)] = h
    return h


def _data_id(ref: str, path=None, *, f_min=None, f_max=None) -> dict:
    """谱标识 = `ref`（给人看）+ `sha256`（给机器认）+ 频率窗口。

    v3 只填 `ref`（文件名）。文件名不是身份：两条不同实验的谱都可能叫
    `spectrum.csv`、窗口也一样，于是 `run_id` 撞车、判重时被读成"同条件不同结果"。
    v4 起优先用**内容哈希**；拿不到原始文件时退到 `ref_only`，并在
    `execution.data_id_kind` 留痕——那种记录最高只能到 `derived`，
    不能参与"确认重复"（见 `circuit_registry.DATA_ID_KINDS`）。
    """
    return {"ref": str(ref), "sha256": _sha256_file(path),
            "f_min": _num(f_min), "f_max": _num(f_max)}


def _num(v):
    """JSON 安全：NaN/Inf → None。不替原始产物做判断，只是不能写进 JSON。"""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and (math.isnan(float(v)) or math.isinf(float(v))):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _sampling(rhat, ess):
    if rhat is None:
        return "not_evaluated"
    return R.sampling_status_from(run_status="completed", rhat_max=float(rhat),
                                  ess_min=None if ess is None else float(ess))


def _fam_of(path_text: str, n_vals: int) -> str:
    if "涂层" in path_text or n_vals == 9:
        return "coating"
    if "钝化" in path_text or n_vals == 6:
        return "passivation"
    raise ValueError(f"无法判断族：{path_text[:80]} (n_vals={n_vals})")


def _truth_from_input(inp: dict) -> tuple:
    """从 `input.path` 的文件名 + `f_min/f_max` 复原真值（我们的单位）。"""
    path = str(inp.get("path") or "")
    lo, hi = path.rfind("["), path.rfind("]")
    if lo < 0 or hi < 0:
        return None, None
    vals = [float(x) for x in re.findall(r"[-+0-9.eE]+", path[lo + 1:hi])]
    fam = _fam_of(path, len(vals))
    keys = _TRUTH_KEYS[fam]
    if len(vals) != len(keys):
        return None, None
    t = dict(zip(keys, vals))
    w = 2 * np.pi * math.sqrt(float(inp["f_min"]) * float(inp["f_max"]))
    if fam == "coating":
        truth = dict(Rs=t["Rs"], R1=t["R1"], Q1=t["Qa"] / w ** t["na"], n1=t["na"],
                     R2=t["R2"], Q2=t["Qb"] / w ** t["nb"], n2=t["nb"],
                     Q_b=t["Qc"] / w ** t["nc"], n_b=t["nc"])
    else:
        truth = dict(Rs=t["Rs"], R1=t["R1"], Q1=t["Qa"] / w ** t["na"], n1=t["na"],
                     Q_b=t["Qb"] / w ** t["nb"], n_b=t["nb"])
    return truth, fam


def _param_estimate(median: dict, truth: dict) -> dict | None:
    if not median or not truth:
        return None
    out = {}
    for k, m in median.items():
        if m is None or truth.get(k) in (None, 0):
            continue
        tv = float(truth[k])
        out[k] = {"estimate": float(m), "truth": tv,
                  "rel_err": abs(float(m) / tv - 1.0)}
    return out or None


# --------------------------------------------------------------------------
# 1. nested_fit 产物（*.nested.json）
# --------------------------------------------------------------------------
def map_nested_json(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = d.get("config", {})
    met = d.get("metrics", {})
    dg = d.get("diagnostics", {})
    inp = d.get("input") or {}

    rule = _INIT_RULES["nested_fit.py"]
    layers = int(cfg.get("layers", 2))
    branch = cfg.get("branch", "parallel")
    els = [("outer_cpe", "cpe")]
    if layers == 2:
        els.append(("inner_cpe", "cpe"))
    els.append(("branch", {"parallel": "rl_parallel", "series": "rl_series",
                           "capacitive": "cpe"}[branch]))
    lik = ("perpoint-%gpct" % cfg.get("sigma_pct", 2.0)
           if cfg.get("sigma_mode") == "perpoint" else "free-sigma")
    cand = R.make_candidate(
        "nested", els, {"branch": branch, "layers": layers},
        init_profile=rule["init_profile"],
        budget_profile=str(cfg.get("budget", "standard")),
        likelihood_profile=lik,
        identity_status="derived",
        init_source=rule["init_source"])

    # `status` 在 nested.json 里是**成熟度标签**（恒 "exploratory"），不是执行状态。
    has_metrics = bool(met) and met.get("rms_rel") is not None
    run_status = "completed" if has_metrics else "failed"
    rhat, ess = _num(dg.get("rhat_max")), _num(dg.get("ess_min"))
    samp = "not_evaluated" if run_status != "completed" else _sampling(rhat, ess)

    med = {k: _num(v.get("median")) for k, v in (d.get("params") or {}).items()
           if isinstance(v, dict)}
    truth, fam = _truth_from_input(inp) if inp else (None, None)

    bk = cfg.get("budget_kw") or {}
    bp = {"source": "recorded", **{k: _num(bk.get(k)) for k in R.BUDGET_PARAM_KEYS}}
    seed = None if cfg.get("seed") is None else int(cfg["seed"])
    exe = R.execution_identity(
        candidate_id=cand.candidate_id,
        # `data.ref` 取**文件名**：完整路径是机器相关的，进 run_id 会让同一个实验
        # 换台机器就换号。原样路径留在 `unmapped["input.path"]` 里可查。
        # v4：身份改由**内容哈希**承担（`sha256`），`ref` 只负责可读。
        data=_data_id(Path(str(inp.get("path") or p.name)).name,
                      _resolve(inp.get("path")),
                      f_min=inp.get("f_min"), f_max=inp.get("f_max")),
        seed=seed,
        # seed 在 `config.seed` 里，是产物自己写的 —— 这是本批唯一够到
        # `artifact` 级别的执行要素；预算也是 recorded。但代码版本没记，
        # 所以 run_id_status 只能到 derived。
        seed_source="artifact:config.seed" if seed is not None else "absent",
        budget_params=bp,
        software={"producer": "nested_fit.py"})
    obs = R.observation_block(
        candidate_id=cand.candidate_id,
        run_status=run_status,
        sampling_status=samp,
        rms_rel=_num(met.get("rms_rel")),
        acf=_num(met.get("acf")),
        rhat_max=rhat, ess_min=ess,
        divergences=None if dg.get("divergences") is None else int(dg["divergences"]),
        param_estimate=_param_estimate(med, truth),
        truth_known=truth is not None,
        param_stability={k: {"ci90": [_num(x) for x in v["ci90"]]}
                         for k, v in (d.get("params") or {}).items()
                         if isinstance(v, dict) and v.get("ci90")} or None,
        noise_floor={"value": None, "source": "not_available",
                     "note": "该产物未记录噪声底；同两条谱的底只出现在 "
                             "family_fit_compare 的 family_fit_results.json 里"},
        ratio={"value": None, "unavailable_reason": "not_computed"},
        predictive={"elpd": _num(dg.get("elpd")), "se": _num(dg.get("elpd_se")),
                    "valid_for_comparison": bool(
                        dg.get("elpd_valid_for_predictive_comparison")),
                    "reason": str(dg.get("elpd_note")
                                  or ("逐点 σ 来自同一批观测，elpd_loo 不是留一预测"
                                      if not dg.get("elpd_valid_for_predictive_comparison")
                                      else "σ 非逐点估计"))},
        budget_params={"source": "recorded", **{k: _num(bk.get(k))
                                                for k in R.BUDGET_PARAM_KEYS}},
        execution=exe,
        interpretation="descriptive_only" if samp != "passed" else None,
        unavailable=({"acf": "not_computed"} if met.get("acf") is None else None),
        notes=(f"来源 {p.name}", f"原始输出目录 {_orig_dir(d)}"),
    )

    unmapped = {
        "status": {"value": d.get("status"),
                   "reason": "产物用 status 表示成熟度（恒 'exploratory'），"
                             "与 registry 的 run_status（执行状态）撞名"},
        "rel_median": {"value": _num(met.get("rel_median")),
                       "reason": "schema 无该字段；与 rms_rel 不同口径，不可互换"},
        "rel_max": {"value": _num(met.get("rel_max")), "reason": "schema 无该字段"},
        "target_accept_override": {
            "value": cfg.get("target_accept_override"),
            "reason": "None 只表示无额外覆盖；有效值在 budget_kw.target_accept"
                      f"={bk.get('target_accept')}（=0.98，不是生产默认 0.9）"},
        "derived.L_ad_H": {"value": (d.get("derived") or {}).get("L_ad_H"),
                           "reason": "capacitive 分支下 null=不适用，非缺失"},
        "voigt_reference": {"value": bool(d.get("voigt_reference")),
                            "reason": "该轮未开 --with-voigt，无跨族对照"},
        "elapsed_s": {"value": _num(d.get("elapsed_s")), "reason": "schema 无成本字段"},
        "input.path": {"value": str(inp.get("path") or ""),
                       "reason": "原样路径机器相关；进执行身份的是文件名"},
    }
    return {
        "record_id": f"nested_json::{p.name}",
        "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                   "sha256": _sha256(p), "producer": "nested_fit.py",
                   "original_dir": _orig_dir(d), "family": fam},
        "candidate": {**cand.describe(),
                      "init_recorded_in_artifact": False,
                      "init_note": rule["init_note"]},
        "observation": obs,
        "provenance": {"production_reachable": True},
        "unmapped": unmapped,
    }


def _orig_dir(d: dict) -> str:
    j = str(((d.get("files") or {}).get("json")) or "")
    return j.replace("\\", "/").rsplit("/", 1)[0] if "/" in j else ""


# --------------------------------------------------------------------------
# 2. oracle 初始化诊断
# --------------------------------------------------------------------------
def _oracle_data(truth: dict) -> dict | None:
    """oracle 产物**没有记录**它跑的是哪条谱（只有 `truth_init`）。

    `diag_passivation_init.py:14-17` 里那条 PATH 与钝化 `nested.json` 的 `input.path`
    是同一个文件。这里不直接断言"同一个"，而是**用产物里的真值参数反查**：
    两者的 Q 在各自 w_ref 下换算后必须逐项相等（容差 1e-6），否则不给 data，
    该记录的执行身份就停在 `unconfirmed`。
    """
    p = EVID / "passivation__nested_capacitive_standard_default_init.nested.json"
    if not p.exists():
        return None
    inp = json.loads(p.read_text(encoding="utf-8")).get("input") or {}
    t, fam = _truth_from_input(inp)
    if fam != "passivation" or not t:
        return None
    common = [k for k in t if k in truth]
    if len(common) < 4:
        return None
    if any(abs(t[k] / float(truth[k]) - 1.0) > 1e-6 for k in common):
        return None
    return _data_id(Path(str(inp.get("path"))).name, _resolve(inp.get("path")),
                    f_min=inp.get("f_min"), f_max=inp.get("f_max"))


def map_oracle(p: Path, budgets: dict) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    t = d["truth_init"]
    rule = _INIT_RULES["diag_passivation_init.py"]
    cand = R.make_candidate(
        "nested", [("outer_cpe", "cpe"), ("branch", "cpe")],
        {"branch": "capacitive", "layers": 1},
        init_profile="value", budget_profile="oracle_diag",
        likelihood_profile="perpoint-2pct",
        identity_status="derived", init_source=rule["init_source"])
    # 预算：产物没记，但 `diag_passivation_init.py:57-60` 写的是
    # BUDGETS["standard"] 的四键 + target_accept=0.98 覆盖 ⇒ 属"由代码推出"。
    std = budgets.get("standard") or {}
    bp = {"source": "derived_from_tier", "tier": "standard",
          **{k: std[k] for k in R.BUDGET_PARAM_KEYS if k in std},
          "target_accept": 0.98}
    exe = R.execution_identity(
        candidate_id=cand.candidate_id, data=_oracle_data(t.get("truth") or {}),
        seed=0, seed_source="producer_code:diag_passivation_init.py:59 seed=0",
        budget_params=bp, software={"producer": "diag_passivation_init.py"})
    return {
        "record_id": f"oracle_init::{p.name}",
        "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                   "sha256": _sha256(p), "producer": "diag_passivation_init.py",
                   "family": "passivation"},
        "candidate": {**cand.describe(),
                      "init_recorded_in_artifact": True,
                      "init_note": rule["init_note"]},
        "observation": R.observation_block(
            candidate_id=cand.candidate_id,
            run_status="completed", sampling_status="passed",
            rms_rel=float(t["rms_rel"]), acf=float(t["acf"]),
            rhat_max=float(t["rhat_max"]), ess_min=float(t["ess_min"]),
            divergences=int(t["divergences"]),
            param_estimate=_param_estimate(
                {k: float(v) for k, v in t["median"].items()},
                {k: float(v) for k, v in t["truth"].items()}),
            truth_known=True,
            noise_floor={"value": None, "source": "not_available",
                         "note": "诊断脚本未算噪声底"},
            ratio={"value": None, "unavailable_reason": "not_computed"},
            predictive={"elpd": None, "se": None, "valid_for_comparison": False,
                        "reason": "该诊断只报残差与参数，未算 elpd"},
            budget_params=bp,
            execution=exe,
            interpretation="reliable",
            notes=("预算数值由 diag_passivation_init.py:57-60 推出"
                   "（BUDGETS['standard'] + target_accept=0.98），不是产物记录",)),
        "provenance": {"production_reachable": False,
                       "note": "oracle 初值：用文件名真值构造，生产环境拿不到"},
        "unmapped": {"elapsed_s": {"value": _num(t.get("seconds")),
                                   "reason": "schema 无成本字段"}},
    }


# --------------------------------------------------------------------------
# 3. 60 条 pilot
# --------------------------------------------------------------------------
_PILOT_SECS = {
    "voigt_prod": dict(topology="voigt", K=2, rule="coating_voigt_pilot.py:voigt_default",
                       likelihood="free-sigma"),
    "voigt_fix2": dict(topology="voigt", K=2, rule="coating_voigt_pilot.py:voigt_default",
                       likelihood="perpoint-2pct"),
    "nested": dict(topology="nested", K=None, rule="coating_voigt_pilot.py:nested",
                   likelihood="perpoint-2pct"),
    # v3 更正：sensitivity 是 **voigt + 自由 σ + 数据驱动初值**，不是 nested。
    "sensitivity": dict(topology="voigt", K=2,
                        rule="coating_voigt_pilot.py:voigt_data_init",
                        likelihood="free-sigma"),
}
#: 比值字段的位置：`("row", key)` 在记录顶层，`("block", key)` 在段内。
_RATIO_KEY = {"voigt_prod": ("row", "ratio"), "voigt_fix2": ("row", "fix2_ratio"),
              "nested": ("row", "nested_ratio"), "sensitivity": ("block", "ratio")}


def _ratio_value(row: dict, sec: str, blk: dict):
    where = _RATIO_KEY.get(sec)
    if where is None:
        return None
    src = row if where[0] == "row" else blk
    return _num(src.get(where[1]))


def _pilot_candidate(sec: str):
    spec = _PILOT_SECS[sec]
    rule = _INIT_RULES[spec["rule"]]
    if spec["topology"] == "voigt":
        return R.make_candidate(
            "voigt", [("relaxation_unit", "hn")] * spec["K"],
            {"K": spec["K"], "include_L": False},
            init_profile=rule["init_profile"], budget_profile="pilot_kw",
            likelihood_profile=spec["likelihood"],
            identity_status="derived", init_source=rule["init_source"]), rule
    return R.make_candidate(
        "nested", [("outer_cpe", "cpe"), ("inner_cpe", "cpe"), ("branch", "cpe")],
        {"branch": "capacitive", "layers": 2},
        init_profile=rule["init_profile"], budget_profile="pilot_kw",
        likelihood_profile=spec["likelihood"],
        identity_status="derived", init_source=rule["init_source"]), rule


def map_pilot(p: Path) -> list:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_total = len(rows)
    # 分母口径 = **比值实际定义在哪几行**。不能按"有没有 rms"算：
    # voigt 自由 σ 是 60/60 都记了 rms，但只有收敛的 14 条算出比值，
    # 按 rms 计会把 14/60 的收敛子集中位数冒充成 60 条的中位数。
    n_used = {s: sum(1 for r in rows
                     if _ratio_value(r, s, r.get(s) or {}) is not None)
              for s in _PILOT_SECS}
    # 其中后验诊断通过的条数（文档里的中位数就是在这部分上算的）
    n_usable = {s: sum(1 for r in rows
                       if (r.get(s) or {}).get("rms") is not None
                       and _sampling(_num(r[s].get("rhat")), _num(r[s].get("ess")))
                       == "passed")
                for s in _PILOT_SECS}
    out = []
    for i, r in enumerate(rows):
        for sec in _PILOT_SECS:
            blk = r.get(sec)
            if not blk:
                continue
            cand, rule = _pilot_candidate(sec)
            rhat, ess = _num(blk.get("rhat")), _num(blk.get("ess"))
            samp = _sampling(rhat, ess)
            rms, acf = _num(blk.get("rms")), _num(blk.get("acf"))
            unavail = {}
            if rms is None:
                unavail["rms_rel"] = "not_computed"
            if acf is None:
                unavail["acf"] = "not_computed"

            err = blk.get("param_err") or {}
            peak = {}
            for k, v in err.items():
                if _num(v) is None:
                    continue
                peak[k] = {"rel_err": float(v)}
            floor = _num(r.get("floor"))
            rval = _ratio_value(r, sec, blk)
            bp = {"source": "derived_from_tier", "tier": "pilot_kw",
                  "n_warmup": 2000, "n_samples": 1000, "num_chains": 2,
                  "target_accept": 0.98, "max_tree_depth": 10}
            fl = [_num(x) for x in (r.get("f") or []) if _num(x) is not None]
            exe = R.execution_identity(
                candidate_id=cand.candidate_id,
                data=_data_id(str(r["name"]), _PILOT_SUB / str(r["name"]),
                              f_min=min(fl) if fl else None,
                              f_max=max(fl) if fl else None),
                seed=0,
                # 四个段的调用点都写死 seed=0（voigt_fit:152 默认参数、
                # :223 与 :268 显式传），但 results.jsonl 里没有 seed 字段
                # ⇒ 属"由代码推出"，不是产物记录。
                seed_source="producer_code:coating_voigt_pilot.py",
                budget_params=bp,
                software={"producer": "coating_voigt_pilot.py"})

            obs = R.observation_block(
                candidate_id=cand.candidate_id,
                run_status="completed", sampling_status=samp,
                rms_rel=rms, acf=acf, rhat_max=rhat, ess_min=ess,
                divergences=None if blk.get("div") is None else int(blk["div"]),
                param_estimate=peak or None,
                truth_known=True,
                noise_floor=({"value": floor, "source": "truth_forward",
                              "note": "数据 vs 真值拓扑正演"} if floor is not None
                             else {"value": None, "source": "not_available"}),
                ratio=({"value": rval, "n_used": n_used[sec], "n_total": n_total,
                        "n_usable": n_usable[sec],
                        "usable_rule": ("只在 verdict=unusable_sampling 的行上重跑，"
                                        "n_total=60 是全表行数、n_used=46 是重跑子集"
                                        if sec == "sensitivity"
                                        else "R̂≤1.05 且 ESS≥100（registry 的 sampling_status）"),
                        "noise_floor_source": "truth_forward"} if rval is not None
                       else {"value": None, "unavailable_reason": "not_computed"}),
                predictive={"elpd": None, "se": None, "valid_for_comparison": False,
                            "reason": "pilot 未记录 elpd（只记残差与 R̂/ESS）"},
                budget_params={"source": "derived_from_tier", "tier": "pilot_kw",
                               "n_warmup": 2000, "n_samples": 1000, "num_chains": 2,
                               "target_accept": 0.98, "max_tree_depth": 10},
                execution=exe,
                interpretation=None if rms is None else "descriptive_only",
                unavailable=unavail or None,
                notes=(r["name"], f"pilot #{i}", f"verdict={r.get('verdict')}"),
            )
            out.append({
                "record_id": f"pilot::{sec}::{i:03d}",
                "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                           "sha256": _sha256(p), "producer": "coating_voigt_pilot.py",
                           "row": i, "section": sec, "family": "coating"},
                "candidate": {**cand.describe(),
                              "init_recorded_in_artifact": False,
                              "init_note": rule["init_note"]},
                "observation": obs,
                "provenance": {"production_reachable": True},
                "unmapped": {
                    "seed": {"value": None,
                             "reason": "results.jsonl 无 seed 字段；四个段在调用点都写死 "
                                       "seed=0（voigt_fit:152 默认参数、:223/:268 显式传），"
                                       "故 seed 记为 producer_code 而非 artifact"},
                    "param_max_err": {"value": _num(blk.get("param_max_err")),
                                      "reason": "全域最大值，是派生标量；"
                                                "逐参数 rel_err 已进 param_estimate"},
                    "verdict": {"value": r.get("verdict"),
                                "reason": "预注册判定属证据层，不进观测记录"},
                    "sec": {"value": _num(blk.get("sec")), "reason": "schema 无成本字段"},
                    "RsR1": {"value": {"fit": _num(blk.get("RsR1")),
                                       "truth": _num(blk.get("RsR1_truth"))},
                             "reason": "派生组合量；只在报告里用"},
                },
            })
    return out


# --------------------------------------------------------------------------
# 4. AZ31（真实谱，无真值）
# --------------------------------------------------------------------------
def map_az31(budgets: dict) -> list:
    out = []
    p = EVID / "az31__m4_real_nested_az31.json"
    if not p.exists():
        return out
    d = json.loads(p.read_text(encoding="utf-8"))
    tier = str(d.get("budget") or "quick")
    bw = budgets.get(tier)
    for idx, r in enumerate(d.get("rows", [])):
        label = str(r.get("label") or "")
        is_voigt = label.startswith("voigt")
        if is_voigt:
            K = int(r.get("layers") or 1)
            cand = R.make_candidate(
                "voigt", [("relaxation_unit", "hn")] * K,
                {"K": K, "include_L": False}, init_profile="uniform",
                budget_profile=tier, likelihood_profile="free-sigma",
                identity_status="derived",
                init_source="producer_code:bm_select.fit_voigt")
            branch, layers = None, K
        else:
            branch = str(r.get("branch") or "parallel")
            layers = int(r.get("layers") or 1)
            els = [("outer_cpe", "cpe")]
            if layers == 2:
                els.append(("inner_cpe", "cpe"))
            els.append(("branch", {"parallel": "rl_parallel",
                                   "series": "rl_series"}[branch]))
            cand = R.make_candidate(
                "nested", els, {"branch": branch, "layers": layers},
                init_profile="uniform", budget_profile=tier,
                likelihood_profile="free-sigma", identity_status="derived",
                init_source="producer_code:nested_fit.py")

        rhat = _num(r.get("rhat"))
        bp = ({"source": "derived_from_tier", "tier": tier, **bw}
              if bw else {"source": "unavailable", "reason": "not_computed"})
        exe = R.execution_identity(
            candidate_id=cand.candidate_id,
            data=_data_id(Path(str(d.get("spec") or p.name)).name,
                          _resolve(d.get("spec")),
                          f_min=d.get("f_min"), f_max=d.get("f_max")),
            seed=0,
            # m4_real_nested_fit.py:105/161 调 fit_voigt / fit_nested 都不传 seed，
            # 两者的默认参数是 seed=0（bm_select:331 / :385）⇒ 由代码推出。
            seed_source=("producer_code:bm_select.fit_voigt(seed=0)" if is_voigt
                         else "producer_code:bm_select.fit_nested(seed=0)"),
            budget_params=bp,
            software={"producer": "evidence/nested_topology/m4_real_nested_fit.py"})
        obs = R.observation_block(
            candidate_id=cand.candidate_id, run_status="completed",
            sampling_status=_sampling(rhat, _num(r.get("ess_min"))),
            rms_rel=None, acf=_num(r.get("acf")), rhat_max=rhat,
            ess_min=_num(r.get("ess_min")),
            divergences=None if r.get("div") is None else int(r["div"]),
            param_estimate=None, truth_known=False,
            noise_floor={"value": None, "source": "not_available",
                         "note": "真实谱无真值，无法正演算底"},
            ratio={"value": None, "unavailable_reason": "not_computed"},
            predictive={"elpd": _num(r.get("elpd")), "se": _num(r.get("se")),
                        "valid_for_comparison": True,
                        "reason": "自由 σ（sigma=None），非 sigma_from_data；"
                                  "同批 6 行同口径"},
            budget_params=({"source": "derived_from_tier", "tier": tier, **bw}
                           if bw else {"source": "unavailable",
                                       "reason": "not_computed"}),
            execution=exe,
            interpretation="descriptive_only",
            unavailable={"rms_rel": "not_computed"},
            notes=("AZ31 真实谱，无真值；rms 记的是绝对 Ω，不是相对残差",),
        )
        out.append({
            "record_id": f"az31::{idx:02d}::{label}",
            "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                       "sha256": _sha256(p),
                       "producer": "evidence/nested_topology/m4_real_nested_fit.py",
                       "row": idx, "family": "az31"},
            "candidate": {**cand.describe(),
                          "init_recorded_in_artifact": False,
                          "init_note": "AZ31 脚本未记录 init；两条调用链"
                                       "（fit_nested / fit_voigt）都不传 init_values"},
            "observation": obs,
            "provenance": {"production_reachable": True},
            "unmapped": {
                "seed": {"value": None,
                         "reason": "AZ31 产物无 seed 字段；调用链 fit_nested(:161) / "
                                   "fit_voigt(:105) 都不传 seed，默认 0"},
                "spec": {"value": str(d.get("spec") or ""),
                         "reason": "原样路径机器相关；进执行身份的是文件名"},
                "rms_re": {"value": _num(r.get("rms_re")),
                           "reason": "绝对 Ω，与 rms_rel 不同口径，不可互换"},
                "rms_im": {"value": _num(r.get("rms_im")), "reason": "同上"},
                "rel_med": {"value": _num(r.get("rel_med")),
                            "reason": "中位相对残差，schema 无对应字段"},
                "rel_max": {"value": _num(r.get("rel_max")), "reason": "schema 无该字段"},
                "acf_base": {"value": _num(d.get("acf_base")),
                             "reason": "voigt 基线 acf，属比较基准不是本记录指标"},
                "sigma_re": {"value": _num(r.get("sigma_re")),
                             "reason": "voigt 行未记录 σ（None）"},
                "sigma_im": {"value": _num(r.get("sigma_im")), "reason": "同上"},
            },
        })
    return out


# --------------------------------------------------------------------------
# 5. 去重与汇总
# --------------------------------------------------------------------------
def _summary_fingerprint(rec: dict) -> str:
    """**结果摘要**指纹。只作 `duplicate_verdict(summary_equal=)` 的一个输入事实，
    单独出现时**不构成**任何重复结论——它证明不了两条记录来自同一次运行。"""
    o = rec["observation"]
    blob = json.dumps({"f": o["fit_observation"], "p": o.get("param_estimate"),
                       "b": o.get("budget_params")},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _identity_view(rec: dict) -> dict:
    """`duplicate_verdict` 要的最小视图：候选 + 执行身份 + 产物出处。"""
    return {"candidate_id": rec["candidate"]["candidate_id"],
            "execution": rec["observation"]["execution"],
            "source": rec["source"]}


def mark_duplicates(recs: list) -> None:
    """按**来源**判重复。判定规则在 `circuit_registry.duplicate_verdict()` 里，
    这里只负责遍历候选组并落盘结论。

    只比同 `candidate_id` 的记录：跨候选的摘要相同不构成重复（另有可能只是巧合）。
    三态落进 `provenance`：
        duplicate_status / duplicate_kind / independent / duplicate_of /
        duplicate_reason / duplicate_evidence
    `independent` 是 `duplicate_status != "confirmed_duplicate"` 的派生值——
    证据不足时**不扣**，只标 `possible_duplicate`。
    """
    for r in recs:
        r["provenance"].update(duplicate_status="distinct", duplicate_kind=None,
                               independent=True, duplicate_of=None,
                               duplicate_reason=None, duplicate_evidence=None)
    fps = {id(r): _summary_fingerprint(r) for r in recs}
    by_cand = defaultdict(list)
    for r in recs:
        by_cand[r["candidate"]["candidate_id"]].append(r)

    for group in by_cand.values():
        for i, a in enumerate(group):
            for b in group[:i]:
                v = R.duplicate_verdict(_identity_view(a), _identity_view(b),
                                        summary_equal=fps[id(a)] == fps[id(b)])
                if v["duplicate_status"] == "confirmed_duplicate":
                    root = b["provenance"].get("duplicate_of") or b["record_id"]
                    a["provenance"].update(
                        duplicate_status="confirmed_duplicate",
                        duplicate_kind=v["duplicate_kind"], independent=False,
                        duplicate_of=root, duplicate_reason=v["reason"],
                        duplicate_evidence=v["evidence"])
                    break
                if (v["duplicate_status"] == "possible_duplicate"
                        and a["provenance"]["duplicate_status"] == "distinct"):
                    a["provenance"].update(
                        duplicate_status="possible_duplicate",
                        duplicate_of=b["record_id"],
                        duplicate_reason=v["reason"],
                        duplicate_evidence=v["evidence"])


def coverage(recs: list) -> dict:
    n = len(recs)
    dup = Counter(r["provenance"]["duplicate_status"] for r in recs)
    n_conf, n_poss = dup.get("confirmed_duplicate", 0), dup.get("possible_duplicate", 0)
    ident = Counter(r["candidate"]["identity_status"] for r in recs)
    inits = Counter(r["candidate"]["init_profile"] for r in recs)
    init_src = Counter(r["candidate"]["init_source"].split(":")[0] for r in recs)
    samp = Counter(r["observation"]["sampling_status"] for r in recs)
    run = Counter(r["observation"]["run_status"] for r in recs)
    interp = Counter(str(r["observation"]["fit_observation"]["interpretation"])
                     for r in recs)
    bud = Counter((r["observation"]["budget_params"] or {}).get("source") for r in recs)
    nf = Counter((r["observation"]["noise_floor"] or {}).get("source") for r in recs)
    ratio_scope = Counter((r["observation"]["ratio"] or {}).get("denominator_scope")
                          for r in recs)
    pred = Counter(None if r["observation"]["predictive"] is None
                   else r["observation"]["predictive"]["valid_for_comparison"]
                   for r in recs)
    blocked = sum(1 for r in recs
                  if r["observation"]["sampling_status"] != "passed"
                  and r["observation"]["fit_observation"]["rms_rel"] is not None)
    unmapped = Counter(k for r in recs for k in r["unmapped"])
    exe = [r["observation"]["execution"] for r in recs]
    run_ids = [e["run_id"] for e in exe if e.get("run_id")]
    return {
        "n_records": n,
        # 主口径：只扣**已确认**的重复；possible 只标记。
        "n_independent": n - n_conf,
        "n_independent_strict": n - n_conf - n_poss,
        "duplicate_status": dict(dup),
        "duplicate_kind": dict(Counter(r["provenance"]["duplicate_kind"] for r in recs
                                       if r["provenance"]["duplicate_kind"])),
        "confirmed_duplicate_records": [
            {"record_id": r["record_id"], "duplicate_of": r["provenance"]["duplicate_of"],
             "kind": r["provenance"]["duplicate_kind"]}
            for r in recs if r["provenance"]["duplicate_status"] == "confirmed_duplicate"],
        "possible_duplicate_records": [
            {"record_id": r["record_id"], "duplicate_of": r["provenance"]["duplicate_of"]}
            for r in recs if r["provenance"]["duplicate_status"] == "possible_duplicate"],
        "dedup_rule": "重复判定只看来源（产物 path/sha、seed 及证据等级、预算数值及来源、"
                      "代码版本、数据内容哈希、run_id），摘要相同只是输入之一；"
                      "确认重复要求两侧执行身份都是 recorded；证据不足只标 "
                      "possible_duplicate、不从独立数里扣（n_independent_strict 给出扣掉后的对照）",
        "run_status": dict(run),
        "sampling_status": dict(samp),
        "interpretation": dict(interp),
        "candidate_identity": dict(ident),
        "init_profile": dict(inits),
        "init_source_kind": dict(init_src),
        "execution": {
            "n_unique_run_id": len(set(run_ids)),
            "run_id_status": dict(Counter(e["run_id_status"] for e in exe)),
            "run_id_prefix": dict(Counter((e["run_id"] or "None")[:2] for e in exe)),
            "seed_source_kind": dict(Counter(str(e["seed_source"]).split(":")[0]
                                             for e in exe)),
            "budget_source_in_identity": dict(Counter(e.get("budget_source") for e in exe)),
            "data_id_kind": dict(Counter(e.get("data_id_kind") for e in exe)),
            "unavailable_reason": dict(Counter(e.get("unavailable_reason") for e in exe)),
            "software_code_version_known": sum(
                1 for e in exe if (e.get("software") or {}).get("code_version")),
            "note": "run_id 是**执行条件**指纹（候选+数据+seed+实际预算+软件），不是墙钟实例号；"
                    "同条件重复执行共享同一个 run_id，实例由 source.path 区分",
        },
        "budget_params_source": dict(bud),
        "noise_floor_source": dict(nf),
        "ratio_denominator_scope": dict(ratio_scope),
        "predictive_valid_for_comparison": {str(k): v for k, v in pred.items()},
        "production_unreachable_records": sum(
            1 for r in recs if r["provenance"].get("production_reachable") is False),
        "blocked_by_old_schema": blocked,
        "blocked_note": "「跑完了、残差算了、后验没过」；0.1.0 的 converged=False "
                        "禁报 rms/acf 会让它们全部无法入库",
        "unmapped_field_counts": dict(unmapped),
    }


def _serialize(recs: list) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                   for r in recs)


def build(budgets: dict) -> list:
    recs = []
    for p in sorted(EVID.glob("*.nested.json")):
        recs.append(map_nested_json(p))
    for p in sorted(EVID.glob("*diag_truth_init.json")):
        recs.append(map_oracle(p, budgets))
    pilot = EVID.parent / "coating_voigt_pilot" / "results.jsonl"
    if pilot.exists():
        recs.extend(map_pilot(pilot))
    recs.extend(map_az31(budgets))

    mark_duplicates(recs)

    # --- 身份完整性断言：这几条正是前几轮出错的地方 ---
    ids = [r["record_id"] for r in recs]
    assert len(set(ids)) == len(ids), \
        f"record_id 重复：{[k for k, v in Counter(ids).items() if v > 1]}"
    seen = {}
    for r in recs:
        cid = r["candidate"]["candidate_id"]
        key = json.dumps({k: r["candidate"][k] for k in
                          ("topology", "elements", "config", "init_profile",
                           "budget_profile", "likelihood_profile")}, sort_keys=True)
        if cid in seen:
            assert seen[cid] == key, f"candidate_id 撞车：{cid} 对应两种不同配置"
        seen[cid] = key
        assert r["candidate"]["identity_status"] in R.IDENTITY_STATES
        if r["candidate"]["init_profile"] == R.INIT_UNKNOWN:
            assert cid.startswith("U-") and not r["candidate"]["comparable"]
        prov = r["provenance"]
        assert prov["duplicate_status"] in R.DUPLICATE_STATES, prov
        assert prov["independent"] == (prov["duplicate_status"]
                                       != "confirmed_duplicate"), prov
        assert r["observation"]["candidate_id"] == cid
        exe = r["observation"]["execution"]
        assert exe["run_id_status"] in R.IDENTITY_STATES
        if exe["run_id_status"] == "unconfirmed":
            assert exe["run_id"] is None and exe["unavailable_reason"] in R.RUN_ID_MISSING
        else:
            assert exe["run_id"], exe
    return recs


def _manifest(texts: dict, cov: dict, recs: list) -> dict:
    """冻结清单。`files` 段哈希的是**即将写入的字节**，不是磁盘上现有的内容。

    这一点不是形式问题：如果哈希从磁盘读，那么"文件被改过"和"清单被改过"会互相
    掩盖——先改文件再改清单哈希，两边自洽，`--check` 就查不出来。
    """
    files = {}
    for name in sorted(texts):
        b = texts[name].encode("utf-8")
        files[name] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    data_ids = sorted({(r["observation"]["execution"].get("data") or {}).get("sha256")
                       for r in recs} - {None})
    return {
        "note": "本目录的冻结副本清单。`records.jsonl` 由 map_stage0_records.py "
                "从 evidence/ 的原始产物映射而来，不跑拟合。",
        "generated_by": "docs/round_2026-11-13/map_stage0_records.py",
        "registry_version": R.REGISTRY_VERSION,
        "registry_digest": R.registry_digest(),
        "n_records": cov["n_records"],
        "n_independent": cov["n_independent"],
        "n_independent_strict": cov["n_independent_strict"],
        "duplicate_status": cov["duplicate_status"],
        "run_id_status": cov["execution"]["run_id_status"],
        "n_unique_run_id": cov["execution"]["n_unique_run_id"],
        "data_id_kind": cov["execution"]["data_id_kind"],
        "source_sha256": {r["source"]["path"]: r["source"]["sha256"] for r in recs},
        "data_sha256": data_ids,
        "data_sha256_files": dict(sorted(_HASHED_FILES.items())),
        "files": files,
    }


def verify_frozen(frozen: Path, texts: dict, manifest: dict,
                  source_paths: dict) -> list:
    """核对**整个**证据快照，返回问题清单（空 = 通过）。

    四个层次，缺一层就会留下"改了但没报"的口子：

    1. **重新生成的三个文件** vs 冻结副本磁盘字节——映射代码/原始产物变了会被抓；
    2. **冻结清单里的 `files`** vs 冻结副本磁盘字节——副本被单独改过会被抓
       （即使清单没动）；
    3. **`registry_version` / `registry_digest`** vs 当前注册表——注册表改过、
       但记录序列化没变时，这一层是唯一能发现的（审计 D4）；
    4. **`source_sha256` / `data_sha256`** vs 磁盘上的证据文件与原始谱——原始证据
       被改会被抓（审计 D5）。

    第 1 层比对"重算文本"，第 2 层比对"磁盘字节"：两者都过才算这份快照没漂移。
    """
    problems = []
    # --- 1. 重新生成 vs 冻结副本 ---
    for name in sorted(texts):
        fp = frozen / name
        if not fp.is_file():
            problems.append(f"{name}: 冻结副本缺失")
            continue
        got = hashlib.sha256(texts[name].encode("utf-8")).hexdigest()
        disk = hashlib.sha256(fp.read_bytes()).hexdigest()
        if got != disk:
            problems.append(f"{name}: 重算 {got[:16]} ≠ 冻结副本 {disk[:16]}")
    # --- 2. 清单自洽 ---
    for name, meta in sorted((manifest.get("files") or {}).items()):
        fp = frozen / name
        if not fp.is_file():
            problems.append(f"MANIFEST.files[{name}]: 文件不存在")
            continue
        disk = hashlib.sha256(fp.read_bytes()).hexdigest()
        if disk != meta.get("sha256"):
            problems.append(f"MANIFEST.files[{name}]: 磁盘 {disk[:16]} ≠ 清单 "
                            f"{str(meta.get('sha256'))[:16]}")
    # --- 3. 注册表身份 ---
    if manifest.get("registry_version") != R.REGISTRY_VERSION:
        problems.append(f"MANIFEST.registry_version={manifest.get('registry_version')!r} "
                        f"≠ 当前 {R.REGISTRY_VERSION!r}")
    if manifest.get("registry_digest") != R.registry_digest():
        problems.append(f"MANIFEST.registry_digest={manifest.get('registry_digest')!r} "
                        f"≠ 当前 {R.registry_digest()!r}")
    # --- 4. 原始证据与数据内容 ---
    for rel, want in sorted((manifest.get("source_sha256") or {}).items()):
        p = source_paths.get(rel)
        if p is None or not Path(p).is_file():
            problems.append(f"source_sha256[{rel}]: 原始证据文件不在了（{p}）")
            continue
        got = _sha256(Path(p))
        if got != want:
            problems.append(f"source_sha256[{rel}]: 磁盘 {got[:16]} ≠ 清单 {str(want)[:16]}")
    for rel, want in sorted((manifest.get("data_sha256_files") or {}).items()):
        p = Path(rel)
        if not p.is_file():
            problems.append(f"data_sha256[{rel}]: 原始谱文件不在了")
            continue
        got = _sha256(p)
        if got != want:
            problems.append(f"data_sha256[{rel}]: 磁盘 {got[:16]} ≠ 清单 {str(want)[:16]}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/stage0_records")
    ap.add_argument("--frozen", default=None,
                    help=f"冻结副本目录（默认 {FROZEN.relative_to(_ROOT)}）")
    ap.add_argument("--freeze", action="store_true",
                    help=f"另存字节级副本到 {FROZEN.relative_to(_ROOT)} 并写 MANIFEST.json")
    ap.add_argument("--check", action="store_true",
                    help="重算整个快照（三份文件 + 注册表指纹 + 原始证据 sha256）并逐项比对，"
                         "不写任何文件")
    args = ap.parse_args(argv)
    outdir = (_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    frozen = FROZEN if args.frozen is None else Path(args.frozen)

    import bm_select  # 只为 BUDGETS 的实际数值（AZ31 / oracle 只记了档名）
    budgets = {k: {kk: int(vv) if isinstance(vv, int) else float(vv)
                   for kk, vv in v.items() if kk in R.BUDGET_PARAM_KEYS}
               for k, v in bm_select.BUDGETS.items()}

    _HASHED_FILES.clear()
    recs = build(budgets)
    cov = coverage(recs)
    texts = {"records.jsonl": _serialize(recs),
             "coverage.json": json.dumps(cov, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n"}
    man = _manifest(texts, cov, recs)
    texts["MANIFEST.json"] = json.dumps(man, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n"
    source_paths = {r["source"]["path"]: _ROOT / r["source"]["path"] for r in recs}

    if args.check:
        if not (frozen / "MANIFEST.json").is_file():
            print(f"{frozen}/MANIFEST.json 不存在，无法核对快照")
            return 2
        frozen_man = json.loads((frozen / "MANIFEST.json").read_text(encoding="utf-8"))
        # MANIFEST 不哈希自己（自指），所以单独比"重新生成的字节 vs 冻结字节"。
        for name in sorted(texts):
            got = hashlib.sha256(texts[name].encode("utf-8")).hexdigest()
            fp = frozen / name
            disk = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else None
            tag = "清单" if name != "MANIFEST.json" else "磁盘"
            print(f"{name}: 重算 {got[:16]} / 冻结{tag} {str(disk)[:16]} → "
                  f"{'一致' if got == disk else '不一致'}")
        print(f"registry: {R.REGISTRY_VERSION} {R.registry_digest()} / "
              f"冻结 {frozen_man.get('registry_version')} "
              f"{frozen_man.get('registry_digest')}")
        print(f"原始证据 {len(man['source_sha256'])} 份、数据文件 "
              f"{len(man['data_sha256_files'])} 份逐份复核")
        problems = verify_frozen(frozen, texts, frozen_man, source_paths)
        # `verify_frozen` 比的是**冻结清单**；这里再比"重新生成的清单"是否与冻结的一致，
        # 否则字段被删（而非被改）时两侧都不报警。
        for k in ("n_records", "n_independent", "n_independent_strict",
                  "duplicate_status", "run_id_status", "n_unique_run_id", "data_id_kind"):
            if man.get(k) != frozen_man.get(k):
                problems.append(f"MANIFEST.{k}: 重算 {man.get(k)!r} ≠ 冻结 "
                                f"{frozen_man.get(k)!r}")
        if problems:
            print(f"发现 {len(problems)} 处不一致：")
            for p in problems:
                print("  -", p)
            return 2
        print(f"通过：{cov['n_records']} 条，{cov['n_independent']} 独立"
              f"（严格口径 {cov['n_independent_strict']}），"
              f"{cov['execution']['n_unique_run_id']} 个唯一 run_id")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "records.jsonl").write_text(texts["records.jsonl"],
                                          encoding="utf-8", newline="\n")
    (outdir / "coverage.json").write_text(texts["coverage.json"],
                                          encoding="utf-8", newline="\n")


    if args.freeze:
        frozen.mkdir(parents=True, exist_ok=True)
        for name in ("records.jsonl", "coverage.json", "MANIFEST.json"):
            dst = frozen / name
            dst.write_text(texts[name], encoding="utf-8", newline="\n")
            assert dst.read_bytes() == texts[name].encode("utf-8"), \
                f"{name} 副本与生成内容不一致"
        print(f"冻结副本 + MANIFEST → {frozen}")

    print(f"记录 {cov['n_records']} 条（独立 {cov['n_independent']}；"
          f"严格口径 {cov['n_independent_strict']}）→ {outdir}")
    print("重复:", cov["duplicate_status"], cov["duplicate_kind"])
    print("run_status:", cov["run_status"])
    print("sampling_status:", cov["sampling_status"])
    print("identity:", cov["candidate_identity"])
    print("init_profile:", cov["init_profile"])
    print("init 判定依据:", cov["init_source_kind"])
    print("执行身份:", cov["execution"])
    print("budget_params 来源:", cov["budget_params_source"])
    print("noise_floor 来源:", cov["noise_floor_source"])
    print("ratio 分母口径:", cov["ratio_denominator_scope"])
    print("生产不可达记录:", cov["production_unreachable_records"])
    print("旧 schema 装不下的记录:", cov["blocked_by_old_schema"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
