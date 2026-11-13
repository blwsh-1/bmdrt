# -*- coding: utf-8 -*-
"""map_stage0_records.py —— 把阶段 0 的历史产物映射成机器可读的观测记录。

目的**不是**让历史数据好看，而是检验 `circuit_registry.observation_block` 到底
能不能承载它们。凡是映射不了的字段，一律进 `unmapped` 并写明原因，不丢、不改、
不补猜。映射过程**不跑任何拟合**，只读 evidence/ 里的副本。

两条口径必须先说清
------------------
1. `run_status` 与 `sampling_status` 分开判：
   - `nested.json` 的 `status` 字段是产物自己的状态；`R̂/ESS` 在 `diagnostics` 里；
   - pilot 的 `rms/acf` 与 `rhat/ess` 是**同时**记的（60/60 都有 rms，即使 R̂=16.2），
     所以它们对应"跑完 + 采样未过"，不是"没跑"。
2. `init_profile` 在历史产物里**没有记录**。`nested.json` 的 `config` 只有
   branch/layers/budget/sigma/seed。目录名与生成脚本能给出线索，但那不是产物内字段，
   所以这里把它标成 `inferred`，并同时留 `recorded: false`。
   受此影响，`candidate_id` 只能算"基于推断的候选标识"，不能当作可复现身份。

用法：
    python docs/round_2026-11-13/map_stage0_records.py
    python docs/round_2026-11-13/map_stage0_records.py --out out/stage0_records
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import circuit_registry as R  # noqa: E402

EVID = _ROOT / "evidence" / "stage0_three_spectra"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _num(v):
    """JSON 安全：NaN/Inf → None。这里不替原始产物做判断，只是不能写进 JSON。"""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _sampling_from(rhat, ess):
    if rhat is None:
        return "not_evaluated"
    return R.sampling_status_from(run_status="completed", rhat_max=float(rhat),
                                  ess_min=None if ess is None else float(ess))


# --------------------------------------------------------------------------
# 1. nested_fit 产物（*.nested.json）
# --------------------------------------------------------------------------
#: 目录名 → init_profile 的**推断**依据（不是产物字段）。
_INIT_HINT = {
    "nested_fit_cap_control": "uniform",
    "nested_fit_cap_standard": "data_driven",
    "nested_fit_cap_smoke": "uniform",
    "nested_fit_cap_pass": "uniform",
    "_user_fit_nested": "data_driven",
}


def map_nested_json(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    cfg = d.get("config", {})
    met = d.get("metrics", {})
    dg = d.get("diagnostics", {})

    hint = _INIT_HINT.get(p.parent.name)
    init_profile = hint or "uniform"
    init_source = "inferred_from_dirname" if hint else "default_assumed"

    layers = int(cfg.get("layers", 2))
    branch = cfg.get("branch", "parallel")
    els = [("outer_cpe", "cpe")]
    if layers == 2:
        els.append(("inner_cpe", "cpe"))
    els.append(("branch", {"parallel": "rl_parallel", "series": "rl_series",
                           "capacitive": "cpe"}[branch]))
    cand = R.make_candidate(
        "nested", els,
        {"branch": branch, "layers": layers},
        init_profile=init_profile,
        budget_profile=str(cfg.get("budget", "standard")),
        likelihood_profile=("perpoint-%gpct" % cfg.get("sigma_pct", 2.0)
                            if cfg.get("sigma_mode") == "perpoint" else "free-sigma"))

    # 注意：`status` 在 nested.json 里是**成熟度标签**（值恒为 "exploratory"），
    # 不是执行状态。产物用同一个词 `status` 表达"研究性接口"，而 registry 用
    # `run_status` 表达"跑没跑完"——两个语义撞名，映射时不能直连。
    # 执行状态只能由"有没有 metrics + diagnostics"判断。
    has_metrics = bool(met) and met.get("rms_rel") is not None
    run_status = "completed" if has_metrics else "failed"
    rhat, ess = _num(dg.get("rhat_max")), _num(dg.get("ess_min"))
    samp = "not_evaluated" if run_status != "completed" else _sampling_from(rhat, ess)

    stab = {}
    for k, v in (d.get("params") or {}).items():
        if not isinstance(v, dict) or "median" not in v:
            continue
        blk = {"median": _num(v.get("median"))}
        if v.get("ci90") is not None:
            blk["ci90"] = [_num(x) for x in v["ci90"]]
        if v.get("rhat") is not None:
            blk["rhat"] = _num(v["rhat"])
        if blk.get("median") is None and "ci90" not in blk:
            blk["flag"] = "median_unavailable"
        stab[k] = blk

    obs = R.observation_block(
        candidate_id=cand.candidate_id,
        run_status=run_status,
        sampling_status=samp,
        rms_rel=_num(met.get("rms_rel")),
        acf=_num(met.get("acf")),
        rhat_max=rhat, ess_min=ess,
        divergences=None if dg.get("divergences") is None else int(dg["divergences"]),
        param_stability=stab or None,
        interpretation=("descriptive_only" if samp != "passed" and
                        met.get("rms_rel") is not None else None),
        unavailable=({"acf": "not_computed"} if met.get("acf") is None else None),
        notes=(f"来源 {p.name}",),
    )

    derived = d.get("derived") or {}
    unmapped = {
        "status": {"value": d.get("status"),
                   "reason": "产物用 status 表示成熟度（恒为 'exploratory'），"
                             "与 registry 的 run_status（执行状态）撞名；不可直连"},
        "warning": {"value": d.get("warning"), "reason": "schema 无警示文本字段"},
        "rel_median": {"value": _num(met.get("rel_median")),
                       "reason": "schema 无该字段；它只作为 rms_rel 的补充口径出现在文档里"},
        "rel_max": {"value": _num(met.get("rel_max")),
                    "reason": "schema 无该字段"},
        "elpd": {"value": _num(dg.get("elpd")),
                 "reason": "schema 无证据用量字段（虽有 elpd_valid 标记）"},
        "elpd_valid_for_predictive_comparison": {
            "value": dg.get("elpd_valid_for_predictive_comparison"),
            "reason": "该可用性标记未进 schema，是 §6 待补项"},
        "sigma_mode": {"value": cfg.get("sigma_mode"),
                       "reason": "已折进 likelihood_profile 标签，原始字段未单列"},
        "budget_kw": {"value": cfg.get("budget_kw"),
                      "reason": "budget_profile 只是标签，未结构化保存预算参数"},
        "target_accept_override": {
            "value": cfg.get("target_accept_override"),
            "reason": "None 只表示无额外覆盖；有效值在 budget_kw.target_accept"},
        "derived.L_ad_H": {"value": derived.get("L_ad_H"),
                           "reason": "capacity 分支下为 null=不适用，非缺失"},
        "elapsed_s": {"value": _num(d.get("elapsed_s")), "reason": "schema 无成本字段"},
    }
    return {
        "record_id": f"nested_json::{p.name}",
        "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                   "sha256": _sha256(p), "producer": "nested_fit.py"},
        "candidate": {
            **cand.describe(),
            "identity_status": "inferred",
            "init_profile_recorded": False,
            "init_profile_source": init_source,
            "note": "config 里没有 init 字段；init_profile 由目录名推断，"
                    "candidate_id 因此不是可复现身份",
        },
        "observation": obs,
        "unmapped": unmapped,
    }


# --------------------------------------------------------------------------
# 2. 60 条 pilot（results.jsonl）
# --------------------------------------------------------------------------
_PILOT_CANDS = {
    "voigt_prod": dict(topology="voigt", K=2, init="uniform",
                       likelihood="free-sigma", budget="pilot_kw"),
    "voigt_fix2": dict(topology="voigt", K=2, init="uniform",
                       likelihood="perpoint-2pct", budget="pilot_kw"),
    "nested": dict(topology="nested", K=None, init="data_driven",
                   likelihood="perpoint-2pct", budget="pilot_kw"),
    "sensitivity": dict(topology="nested", K=None, init="data_driven",
                        likelihood="perpoint-2pct", budget="pilot_kw_warmup4000"),
}


def _pilot_candidate(sec: str):
    spec = _PILOT_CANDS[sec]
    if spec["topology"] == "voigt":
        return R.make_candidate(
            "voigt", [("relaxation_unit", "hn")] * spec["K"],
            {"K": spec["K"], "include_L": False}, init_profile=spec["init"],
            budget_profile=spec["budget"], likelihood_profile=spec["likelihood"])
    return R.make_candidate(
        "nested", [("outer_cpe", "cpe"), ("inner_cpe", "cpe"), ("branch", "cpe")],
        {"branch": "capacitive", "layers": 2}, init_profile=spec["init"],
        budget_profile=spec["budget"], likelihood_profile=spec["likelihood"])


def map_pilot(p: Path) -> list:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for i, r in enumerate(rows):
        for sec in ("voigt_prod", "voigt_fix2", "nested", "sensitivity"):
            blk = r.get(sec)
            if not blk:
                continue
            cand = _pilot_candidate(sec)
            rhat, ess = _num(blk.get("rhat")), _num(blk.get("ess"))
            samp = _sampling_from(rhat, ess)
            rms = _num(blk.get("rms"))
            acf = _num(blk.get("acf"))
            unavail = {}
            if rms is None:
                unavail["rms_rel"] = "not_computed"
            if acf is None:
                unavail["acf"] = "not_computed"
            obs = R.observation_block(
                candidate_id=cand.candidate_id,
                run_status="completed",
                sampling_status=samp,
                rms_rel=rms, acf=acf, rhat_max=rhat, ess_min=ess,
                divergences=None if blk.get("div") is None else int(blk["div"]),
                interpretation=None if rms is None else "descriptive_only",
                unavailable=unavail or None,
                notes=(f"{r['name']}", f"pilot #{i}", f"verdict={r.get('verdict')}"),
            )
            unmapped = {
                "param_err": {"value": _num_any(blk.get("param_err")),
                              "reason": "点估计相对误差，不是后验区间；"
                                        "param_stability 只接受 ci90/flag 描述块"},
                "param_max_err": {"value": _num(blk.get("param_max_err")),
                                  "reason": "同上，属派生标量分数"},
                "ratio": {"value": _num(r.get("ratio" if sec == "voigt_prod" else
                                              {"voigt_fix2": "fix2_ratio",
                                               "nested": "nested_ratio"}.get(sec, "ratio"))),
                          "reason": "schema 无 ratio 字段（更缺 n_used/n_total）"},
                "floor": {"value": _num(r.get("floor")),
                          "reason": "噪声底来源未进 schema"},
                "verdict": {"value": r.get("verdict"),
                            "reason": "预注册判定未进观测记录（属证据层）"},
                "sec": {"value": _num(blk.get("sec")), "reason": "schema 无成本字段"},
            }
            out.append({
                "record_id": f"pilot::{sec}::{i:03d}",
                "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                           "sha256": _sha256(p), "producer": "coating_voigt_pilot.py",
                           "row": i, "section": sec},
                "candidate": {**cand.describe(), "identity_status": "complete",
                              "init_profile_recorded": True,
                              "init_profile_source": "producer_code"},
                "observation": obs,
                "unmapped": unmapped,
            })
    return out


def _num_any(v):
    if isinstance(v, dict):
        return {str(k): _num(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_num(x) for x in v]
    return _num(v)


# --------------------------------------------------------------------------
# 3. AZ31（真实谱，无真值）
# --------------------------------------------------------------------------
def map_az31() -> list:
    out = []
    p = EVID / "az31__m4_real_nested_az31.json"
    if not p.exists():
        return out
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d.get("rows", []):
        branch = r.get("branch") or "parallel"
        layers = int(r.get("layers") or 1)
        if branch == "capacitive":
            els = [("outer_cpe", "cpe"), ("branch", "cpe")]
        else:
            els = [("outer_cpe", "cpe"), ("branch",
                                          {"parallel": "rl_parallel",
                                           "series": "rl_series"}[branch])]
        if layers == 2:
            els.insert(1, ("inner_cpe", "cpe"))
        cfg = {"branch": branch, "layers": layers}
        if r.get("family") == "voigt" or r.get("topology") == "voigt":
            cand = R.make_candidate("voigt", [("relaxation_unit", "hn")] * 2,
                                    {"K": 2, "include_L": False},
                                    init_profile="uniform", budget_profile="quick",
                                    likelihood_profile="free-sigma")
        else:
            cand = R.make_candidate("nested", els, cfg, init_profile="uniform",
                                    budget_profile="quick",
                                    likelihood_profile="free-sigma")
        rhat = _num(r.get("rhat_max", r.get("rhat")))
        obs = R.observation_block(
            candidate_id=cand.candidate_id, run_status="completed",
            sampling_status=_sampling_from(rhat, _num(r.get("ess_min"))),
            rms_rel=None, acf=_num(r.get("acf")), rhat_max=rhat,
            ess_min=_num(r.get("ess_min")),
            divergences=None if r.get("divergences") is None else int(r["divergences"]),
            interpretation="descriptive_only",
            unavailable={"rms_rel": "not_computed"},
            notes=("AZ31 真实谱，无真值；elpd 存在但不可用于预测比较",),
        )
        out.append({
            "record_id": f"az31::{cand.candidate_id}",
            "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                       "sha256": _sha256(p), "producer": "bm_select 跨族扫描"},
            "candidate": {**cand.describe(), "identity_status": "inferred",
                          "init_profile_recorded": False,
                          "init_profile_source": "assumed_default"},
            "observation": obs,
            "unmapped": {
                "rel_med": {"value": _num(r.get("rel_med")),
                            "reason": "与 rms_rel 不同口径，不能互相替代"},
                "elpd": {"value": _num(r.get("elpd")),
                         "reason": "schema 无证据用量字段"},
            },
        })
    return out


# --------------------------------------------------------------------------
# 4. 汇总
# --------------------------------------------------------------------------
def coverage(records: list) -> dict:
    n = len(records)
    ident = Counter(r["candidate"]["identity_status"] for r in records)
    samp = Counter(r["observation"]["sampling_status"] for r in records)
    run = Counter(r["observation"]["run_status"] for r in records)
    interp = Counter(str(r["observation"]["fit_observation"]["interpretation"])
                     for r in records)
    # 旧 0.1.0 schema 装不下的记录：跑完但采样未过、却带有残差
    blocked_by_old_schema = sum(
        1 for r in records
        if r["observation"]["sampling_status"] != "passed"
        and r["observation"]["fit_observation"]["rms_rel"] is not None)
    unmapped_keys = Counter(k for r in records for k in r["unmapped"])
    return {
        "n_records": n,
        "run_status": dict(run),
        "sampling_status": dict(samp),
        "interpretation": dict(interp),
        "candidate_identity": dict(ident),
        "init_profile_recorded_true": sum(
            1 for r in records if r["candidate"]["init_profile_recorded"]),
        "blocked_by_old_schema": blocked_by_old_schema,
        "blocked_note": "这些记录是「跑完了、残差算了、后验没过」；0.1.0 的 "
                        "converged=False 禁报 rms/acf 会让它们全部无法入库",
        "unmapped_field_counts": dict(unmapped_keys),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/stage0_records")
    args = ap.parse_args(argv)
    outdir = (_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    recs = []
    for p in sorted(EVID.glob("*.nested.json")):
        recs.append(map_nested_json(p))
    for p in sorted(EVID.glob("*diag_truth_init.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        recs.append({
            "record_id": f"oracle_init::{p.name}",
            "source": {"path": str(p.relative_to(_ROOT)).replace("\\", "/"),
                       "sha256": _sha256(p), "producer": "diag_passivation_init.py"},
            "candidate": {
                "topology": "nested", "config": {"branch": "capacitive", "layers": 1},
                "init_profile": "value", "identity_status": "inferred",
                "init_profile_recorded": True,
                "init_profile_source": "script_hardcodes_truth_params",
                "note": "truth_init 是 oracle：初值直接取文件名里的真值参数，"
                        "生产环境拿不到，不能作为数据驱动初始化成功的证据",
            },
            "observation": R.observation_block(
                candidate_id="C-oracle-passivation",
                run_status="completed", sampling_status="passed",
                rms_rel=float(d["truth_init"]["rms_rel"]),
                acf=float(d["truth_init"]["acf"]),
                rhat_max=float(d["truth_init"]["rhat_max"]),
                ess_min=float(d["truth_init"]["ess_min"]),
                divergences=int(d["truth_init"]["divergences"]),
                interpretation="reliable"),
            "unmapped": {"elapsed_s": {"value": d["truth_init"].get("seconds"),
                                       "reason": "schema 无成本字段"},
                         "median_vs_truth": {
                             "value": {"median": d["truth_init"]["median"],
                                       "truth": d["truth_init"]["truth"]},
                             "reason": "参数恢复对照，schema 无该结构"}},
        })
    pilot = EVID.parent / "coating_voigt_pilot" / "results.jsonl"
    if pilot.exists():
        recs.extend(map_pilot(pilot))
    recs.extend(map_az31())

    with (outdir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    cov = coverage(recs)
    (outdir / "coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")

    print(f"记录 {cov['n_records']} 条 → {outdir}")
    print("run_status:", cov["run_status"])
    print("sampling_status:", cov["sampling_status"])
    print("identity:", cov["candidate_identity"])
    print("init_profile 有记录:", cov["init_profile_recorded_true"], "/", cov["n_records"])
    print("旧 schema 装不下的记录:", cov["blocked_by_old_schema"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
