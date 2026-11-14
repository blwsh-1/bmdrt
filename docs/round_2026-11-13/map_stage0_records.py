# -*- coding: utf-8 -*-
"""map_stage0_records.py —— 把阶段 0 的历史产物映射成机器可读的观测记录（v2）。

目的**不是**让历史数据好看，而是检验 `circuit_registry.observation_block` 到底
能不能承载它们。凡是映射不了的字段，一律进 `unmapped` 并写明原因，不丢、不改、
不补猜。映射过程**不跑任何拟合**，只读 evidence/ 里的副本。

v2 相对 v1 修的三个身份缺陷（v1 全部是错的，不是措辞问题）
------------------------------------------------------------
1. **目录名推断从未命中**。v1 用 `p.parent.name` 查 `_INIT_HINT`，但归档文件是
   平铺的 `evidence/stage0_three_spectra/`，父目录恒为该目录名，一个都不匹配，
   于是 `init_profile` 静默回落成 `"uniform"`、标记 `default_assumed`。
   更糟的是回落值是一个**真实档名**，看起来合法 ⇒ 13 条记录拿到无依据的身份键。
   v2 改成：初始化由**生成脚本的代码路径**判定；查不到就 `INIT_UNKNOWN`，
   id 前缀变 `U-`，且不许进 manifest。
2. **AZ31 的 2 条 voigt 行被算成了 nested**。v1 写 `branch = r.get("branch") or "parallel"`，
   而 voigt 行的 `branch` 是 `None` ⇒ 落成 nested/parallel，与真实 parallel 行
   `candidate_id` 完全撞车（6 行只剩 4 个 id，`record_id` 也重复）——
   voigt 的残差被记到了 nested 名下。v2 按 `label` 分流并断言 id 唯一。
3. **pilot 的 voigt 初始化标错**。`voigt_init` 的文档写明"数据驱动初值（只看 f、|Z|）"，
   v1 标成 `uniform`。v2 标 `data_driven`，并把 voigt 的该档补进注册表。

四条口径必须先说清
------------------
- `run_status` 与 `sampling_status` 分开判：`nested.json` 的 `status` 是**成熟度标签**
  （恒 `"exploratory"`），不是执行状态，不可直连；R̂/ESS 在 `diagnostics` 里。
- 初始化**不来自**目录名，来自生成脚本：
  `nested_fit.py:339` 调 `T.fit_top(...)` 不传 `init_values`，`bm_top.fit_top:1006`
  把它当 `init_params=None` ⇒ numpyro 默认 `init_to_uniform`。
  `fit_family_compare.py` 只写 `family_fit_results.json`，**不写** `.nested.json`，
  所以 `out/_user_fit_nested/*.nested.json` 不是它的产物（这条曾被搞混）。
- 每个 `nested.json` 的 `files.json` 记录它**自己写到哪**，`input` 记录 `f_min/f_max`
  与含真值参数的文件名 ⇒ 原始目录与真值都能从产物自身复原，不必靠 MANIFEST 猜。
  MANIFEST 只用于交叉核对。
- 同配置同 seed 的重跑会产生**逐位相同**的指标（`1113` 那两份 elapsed 不同、指标相同）
  ⇒ 那是可复现性证据，但**不是独立观测**，计数时必须标 `independent=false`。

用法：
    python docs/round_2026-11-13/map_stage0_records.py
    python docs/round_2026-11-13/map_stage0_records.py --out out/stage0_records
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import circuit_registry as R  # noqa: E402

EVID = _ROOT / "evidence" / "stage0_three_spectra"


# --------------------------------------------------------------------------
# 生成脚本 → 初始化档。判定依据写在这里，不写"档名"了事。
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
    "coating_voigt_pilot.py:voigt": dict(
        init_profile="data_driven",
        init_source="producer_code:coating_voigt_pilot.py:voigt_init",
        init_note="voigt_init 的 docstring 明写「数据驱动初值（只看 f、|Z|，不读真值）」："
                  "R_inf=min|Z|、W=(zmax−zmin)/2、τ 窗内均铺、φ=0.8、v=1/K。",
    ),
    "coating_voigt_pilot.py:nested": dict(
        init_profile="data_driven",
        init_source="producer_code:coating_voigt_pilot.py:data_init",
        init_note="data_init 由 f/|Z| 的 min/zmax/中心频点算出，不读真值。",
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
        "provenance": {"production_reachable": True, "independent": True},
        "unmapped": unmapped,
    }


def _orig_dir(d: dict) -> str:
    j = str(((d.get("files") or {}).get("json")) or "")
    return j.replace("\\", "/").rsplit("/", 1)[0] if "/" in j else ""


# --------------------------------------------------------------------------
# 2. oracle 初始化诊断
# --------------------------------------------------------------------------
def map_oracle(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    t = d["truth_init"]
    rule = _INIT_RULES["diag_passivation_init.py"]
    cand = R.make_candidate(
        "nested", [("outer_cpe", "cpe"), ("branch", "cpe")],
        {"branch": "capacitive", "layers": 1},
        init_profile="value", budget_profile="oracle_diag",
        likelihood_profile="perpoint-2pct",
        identity_status="derived", init_source=rule["init_source"])
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
            budget_params={"source": "unavailable", "reason": "not_computed"},
            interpretation="reliable"),
        "provenance": {"production_reachable": False,
                       "independent": True,
                       "note": "oracle 初值：用文件名真值构造，生产环境拿不到"},
        "unmapped": {"elapsed_s": {"value": _num(t.get("seconds")),
                                   "reason": "schema 无成本字段"}},
    }


# --------------------------------------------------------------------------
# 3. 60 条 pilot
# --------------------------------------------------------------------------
_PILOT_SECS = {
    "voigt_prod": dict(topology="voigt", K=2, rule="coating_voigt_pilot.py:voigt",
                       likelihood="free-sigma"),
    "voigt_fix2": dict(topology="voigt", K=2, rule="coating_voigt_pilot.py:voigt",
                       likelihood="perpoint-2pct"),
    "nested": dict(topology="nested", K=None, rule="coating_voigt_pilot.py:nested",
                   likelihood="perpoint-2pct"),
    "sensitivity": dict(topology="nested", K=None, rule="coating_voigt_pilot.py:nested",
                        likelihood="perpoint-2pct"),
}
_RATIO_KEY = {"voigt_prod": "ratio", "voigt_fix2": "fix2_ratio", "nested": "nested_ratio"}


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
    n_used = {s: (sum(1 for r in rows if _num(r.get(_RATIO_KEY[s])) is not None)
                  if s in _RATIO_KEY else 0)
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
            rkey = _RATIO_KEY.get(sec)
            rval = _num(r.get(rkey)) if rkey else None

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
                        "usable_rule": "R̂≤1.05 且 ESS≥100（registry 的 sampling_status）",
                        "noise_floor_source": "truth_forward"} if rval is not None
                       else {"value": None, "unavailable_reason": "not_computed"}),
                predictive={"elpd": None, "se": None, "valid_for_comparison": False,
                            "reason": "pilot 未记录 elpd（只记残差与 R̂/ESS）"},
                budget_params={"source": "derived_from_tier", "tier": "pilot_kw",
                               "n_warmup": 2000, "n_samples": 1000, "num_chains": 2,
                               "target_accept": 0.98, "max_tree_depth": 10},
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
                "provenance": {"production_reachable": True, "independent": True},
                "unmapped": {
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
            "provenance": {"production_reachable": True, "independent": True},
            "unmapped": {
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
def _fingerprint(rec: dict) -> str:
    o = rec["observation"]
    blob = json.dumps({"c": rec["candidate"]["candidate_id"],
                       "f": o["fit_observation"], "p": o.get("param_estimate")},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def mark_duplicates(recs: list) -> None:
    """同配置、同 seed、指标逐位相同 ⇒ 同一次结果的两个导出，不是独立观测。"""
    seen = {}
    for r in recs:
        fp = _fingerprint(r)
        if fp in seen:
            first = seen[fp]
            r["provenance"]["independent"] = False
            r["provenance"]["duplicate_of"] = first["record_id"]
            r["provenance"]["duplicate_reason"] = (
                "指标与参数逐位相同（同配置同 seed）；elapsed 不同 ⇒ 是重跑而非拷贝，"
                "属可复现性证据，但计数时不能重复算")
        else:
            seen[fp] = r


def coverage(recs: list) -> dict:
    n = len(recs)
    ind = [r for r in recs if r["provenance"].get("independent")]
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
    return {
        "n_records": n,
        "n_independent": len(ind),
        "n_duplicate": n - len(ind),
        "run_status": dict(run),
        "sampling_status": dict(samp),
        "interpretation": dict(interp),
        "candidate_identity": dict(ident),
        "init_profile": dict(inits),
        "init_source_kind": dict(init_src),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/stage0_records")
    args = ap.parse_args(argv)
    outdir = (_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    import bm_select  # 只为 BUDGETS 的实际数值（AZ31 只记了档名）
    budgets = {k: {kk: int(vv) if isinstance(vv, int) else float(vv)
                   for kk, vv in v.items() if kk in R.BUDGET_PARAM_KEYS}
               for k, v in bm_select.BUDGETS.items()}

    recs = []
    for p in sorted(EVID.glob("*.nested.json")):
        recs.append(map_nested_json(p))
    for p in sorted(EVID.glob("*diag_truth_init.json")):
        recs.append(map_oracle(p))
    pilot = EVID.parent / "coating_voigt_pilot" / "results.jsonl"
    if pilot.exists():
        recs.extend(map_pilot(pilot))
    recs.extend(map_az31(budgets))

    mark_duplicates(recs)

    # --- 身份完整性断言：这三条正是 v1 出错的地方 ---
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

    with (outdir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    cov = coverage(recs)
    (outdir / "coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")

    print(f"记录 {cov['n_records']} 条（独立 {cov['n_independent']}，"
          f"重复 {cov['n_duplicate']}）→ {outdir}")
    print("run_status:", cov["run_status"])
    print("sampling_status:", cov["sampling_status"])
    print("identity:", cov["candidate_identity"])
    print("init_profile:", cov["init_profile"])
    print("init 判定依据:", cov["init_source_kind"])
    print("budget_params 来源:", cov["budget_params_source"])
    print("noise_floor 来源:", cov["noise_floor_source"])
    print("ratio 分母口径:", cov["ratio_denominator_scope"])
    print("生产不可达记录:", cov["production_unreachable_records"])
    print("旧 schema 装不下的记录:", cov["blocked_by_old_schema"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
