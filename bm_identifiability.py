# -*- coding: utf-8 -*-
"""统一参数可辨识性报告（P0 ①）：把 voigt / nested 的零散诊断归一到同一 schema。

预注册：`docs/IDENTIFIABILITY_REPORT_PREREG.md`。**先读那份再改本文件。**

设计边界
--------
1. **只记账不裁决**：报告里没有 ``grade`` / ``score`` / ``verdict`` / ``pass`` /
   ``accept``，不调用 ``bm_grade``，不改任何 ``status`` / ``sample_grade`` / 门禁。
   它回答「这个参数被数据约束住了吗」，不回答「这个模型能不能用」。
2. **不碰生产**：本模块是纯函数，只读已落盘的 ``dict``；不 import ``bm_core`` /
   ``bm_top`` / ``bm_result`` / ``bm_corr``，也不写文件。批处理器另放
   ``docs/round_2026-11-12/build_identifiability_report.py``。
3. **三态诚实**：``available`` / ``partial`` / ``unavailable``。缺数据必须写
   ``unavailable`` 且数值全 ``null``，**不许**用 0、空列表或「通过」糊过去。
4. **原尺度纪律**：``param_corr.summary`` 的 q05/q50/q95 **已经是原尺度**
   （对数参数已 exp 回），这里原样搬运，绝不再次 exp。
5. **跨族同口径 = 同 schema，不是数值可比**：族特有量各留原名，
   ``comparable=False``。``nested_fit`` 的 ``rms_rel = Rwp/√2``，别当 Rwp 比。

已知限制（预注册 §4，报告要如实暴露，不是 bug）
--------------------------------------------
``nested_fit.py`` 不落盘后验样本 → nested 侧 ``correlation`` / ``boundary`` /
``combinations`` 只能是 ``unavailable``；两族 PPC 均未落盘 → ``unavailable``。

自检：``python bm_identifiability.py``
"""

from __future__ import annotations

import json
import math
from typing import Any

SCHEMA_VERSION = "1.0"
DIAG_KEYS = ("posterior", "correlation", "boundary",
             "combinations", "sampling", "prediction")
FORBIDDEN_KEYS = ("grade", "score", "verdict", "pass", "accept", "passed")

DISCLAIMER = (
    "只记账不裁决：本报告不判通过/拒绝、不给分数、不改原产物 status 与门禁；"
    "可辨识性结论只是解释边界的依据，模型取舍仍由预注册判据决定。"
)
GENERAL_NOTES = (
    "两族拟合优度字段各留原名：nested_fit 的 rms_rel = Rwp/√2，不得直接当 Rwp 比较。",
    "PPC（后验预测检查）未落盘，一律 unavailable；不得用残差 acf 冒充 PPC。",
    "nested_fit 不落盘后验样本：相关矩阵 / 逐参数贴边率 / 参数组合无法计算，标 unavailable。",
    "剖面平坦度与窗口截断见 evidence/nested_topology/identifiability_check.py（独立脚本，非逐谱产物）。",
)


# ---------------------------------------------------------------- 基础工具

def _finite(v: Any) -> Any:
    """非有限的 float 一律写 None（严格 JSON，不写 nan/inf，也不写 0）。"""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, int):          # 计数 / 样本数保持整数，别显示 2.0
            return v
        return float(v) if math.isfinite(float(v)) else None
    return v


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return _finite(v)
    if isinstance(v, (str, bool)) or v is None:
        return v
    return str(v)


def _blank(status: str, source: str, comparable: bool = False,
           note: str = "") -> dict:
    return dict(status=status, source=source, comparable=comparable, note=note)


def _rel_span(q05, q50, q95):
    """相对跨度 (q95-q05)/|q50|；q50≈0 时返回 None，不写 0。"""
    a, b, c = _finite(q05), _finite(q50), _finite(q95)
    if a is None or b is None or c is None or abs(b) < 1e-30:
        return None
    return _finite((c - a) / abs(b))


# ---------------------------------------------------------------- 六类诊断

def _posterior_voigt(raw: dict) -> dict:
    pc = raw.get("param_corr") or {}
    rows = pc.get("summary") or []
    if not rows:
        return dict(_blank("unavailable", "real_*.json:param_corr.summary（缺失）",
                           True, "无参数后验分位；不得写成「参数可辨识」"),
                    n_params=None, params=None)
    params = [dict(name=r.get("name"), q05=_finite(r.get("q05")),
                   q50=_finite(r.get("q50")), q95=_finite(r.get("q95")),
                   rel_span=_rel_span(r.get("q05"), r.get("q50"), r.get("q95")))
              for r in rows]
    return dict(_blank("available", "real_*.json:param_corr.summary（原尺度 q05/q50/q95）",
                       True, "分位已是原尺度，实现不得再取 exp()"),
                n_params=len(params), params=params)


def _posterior_nested(raw: dict) -> dict:
    p = raw.get("params") or {}
    if not p:
        return dict(_blank("unavailable", "*.nested.json:params（缺失）",
                           True, "无参数后验分位"),
                    n_params=None, params=None)
    params = []
    for name, v in p.items():
        ci = v.get("ci90") or [None, None]
        params.append(dict(name=name, q05=_finite(ci[0]),
                           q50=_finite(v.get("median")), q95=_finite(ci[1]),
                           rel_span=_rel_span(ci[0], v.get("median"), ci[1])))
    return dict(_blank("available", "*.nested.json:params[].median/ci90", True,
                       "ci90 为 [q05,q95]，与 voigt 的 q05/q95 同口径"),
                n_params=len(params), params=params)


def _correlation_voigt(raw: dict) -> dict:
    pc = raw.get("param_corr") or {}
    if not pc.get("names"):
        return dict(_blank("unavailable", "real_*.json:param_corr（缺失）", False,
                           "没有后验样本或相关矩阵"),
                    n_draw=None, strong_tol=None, medium_tol=None,
                    strong_pairs=None, medium_pairs=None, matrix_names=None)
    return dict(_blank("available", "real_*.json:param_corr.strong/medium/corr", False,
                       "强相关对只能说明存在补偿关系，不等于模型该被拒绝"),
                n_draw=_finite(pc.get("n_draw")),
                strong_tol=_finite(pc.get("strong_tol")),
                medium_tol=_finite(pc.get("medium_tol")),
                strong_pairs=_json_safe(pc.get("strong") or []),
                medium_pairs=_json_safe(pc.get("medium") or []),
                matrix_names=list(pc.get("names") or []))


def _boundary_voigt(raw: dict) -> dict:
    pc = raw.get("param_corr") or {}
    cr = ((raw.get("component_review") or {}).get("parameter_identifiability") or {})
    fixed = [f.get("name") for f in (pc.get("fixed") or [])]
    # max_at_bound 来自 component_review / degeneracy，是「峰/参数贴边」的线索，
    # 不是「逐参数贴先验边界的比例」。口径不同 → fraction 一律 None。
    signal = _finite(cr.get("max_at_bound"))
    at_bound = []
    for d in (raw.get("degeneracy") or []):
        if isinstance(d, dict) and d.get("at_bound") is not None:
            at_bound.append(dict(phi=d.get("phi"), f_peak_hz=_finite(d.get("f_peak")),
                                 value=_finite(d.get("at_bound"))))
    return dict(_blank("partial",
                       "param_corr.fixed + component_review.parameter_identifiability "
                       "+ degeneracy[].at_bound",
                       False,
                       "只有贴边线索，没有逐参数先验边界表 + 后验样本，算不出"
                       "「贴先验上下界的比例」；at_bound_fraction 保持 null"),
                n_params=len(pc.get("names") or []) or None,
                fixed_params=fixed, at_bound_params=at_bound,
                at_bound_fraction=None, max_at_bound_signal=signal,
                level=cr.get("level"))


def _combinations_voigt(raw: dict) -> dict:
    pc = raw.get("param_corr") or {}
    if not pc.get("names"):
        return dict(_blank("unavailable", "real_*.json:param_corr.degenerate（缺失）",
                           False, "无相关矩阵 → 无法区分「组合可辨识」与「整支路不可辨识」"),
                    known_combinations=None, single_param_limitation=None)
    deg = pc.get("degenerate") or []
    if isinstance(deg, dict):
        deg = [deg]
    cr = ((raw.get("component_review") or {}).get("parameter_identifiability") or {})
    return dict(_blank("available", "param_corr.degenerate（原样转述，不新增判断）",
                       False,
                       "known_combinations 是已有结论文本，不是新裁决；"
                       "单参数不可辨识但组合可辨识时应限制物理解释，不必然拒绝模型"),
                known_combinations=_json_safe(deg),
                single_param_limitation=_json_safe(cr.get("strong_pairs") or []))


def _sampling_voigt(raw: dict) -> dict:
    diag = raw.get("diag") or {}
    if not diag:
        return dict(_blank("unavailable", "real_*.json:diag（缺失）", False),
                    rhat_max=None, ess_bulk_min=None, ess_tail_min=None,
                    divergences=None, per_param=None)
    per = {}
    rh, eb, et = [], [], []
    for k, v in diag.items():
        if not isinstance(v, dict):
            continue
        per[k] = dict(rhat=_finite(v.get("rhat")),
                      ess_bulk=_finite(v.get("bulk")), ess_tail=_finite(v.get("tail")))
        for val, acc in ((v.get("rhat"), rh), (v.get("bulk"), eb), (v.get("tail"), et)):
            x = _finite(val)
            if x is not None:
                acc.append(x)
    return dict(_blank("available", "real_*.json:diag[].rhat/bulk/tail + divergences",
                       False, "ESS 分 bulk/tail；nested 侧只有单一 ess，不能逐值比"),
                rhat_max=max(rh) if rh else None,
                ess_bulk_min=min(eb) if eb else None,
                ess_tail_min=min(et) if et else None,
                divergences=_finite(raw.get("divergences")),
                per_param=per)


def _sampling_nested(raw: dict) -> dict:
    d = raw.get("diagnostics") or {}
    if not d:
        return dict(_blank("unavailable", "*.nested.json:diagnostics（缺失）", False),
                    rhat_max=None, ess_bulk_min=None, ess_tail_min=None,
                    divergences=None, per_param=None)
    per = {}
    for k, v in (raw.get("params") or {}).items():
        per[k] = dict(rhat=_finite(v.get("rhat")), ess=_finite(v.get("ess")),
                      ess_bulk=None, ess_tail=None)
    return dict(_blank("partial", "*.nested.json:diagnostics + params[].rhat/ess",
                       False,
                       "只有单一 ess，没有 bulk/tail 拆分；两个字段保持 null，"
                       "不许拿 ess 顶替 ess_bulk"),
                rhat_max=_finite(d.get("rhat_max")),
                ess_bulk_min=None, ess_tail_min=None,
                divergences=_finite(d.get("divergences")), per_param=per)


def _prediction_voigt(raw: dict) -> dict:
    metrics = {}
    for k in ("rms_re", "rms_im", "rms_rel", "rwp_pct"):
        if raw.get(k) is not None:
            metrics[k] = _finite(raw[k])
    bands = _json_safe(raw.get("resid_bins") or [])
    return dict(_blank("partial" if metrics else "unavailable",
                       "real_*.json:acf / rms_* / resid_bins / elpd*",
                       False,
                       "拟合优度各留原名；resid_bins 是 σ 归一化残差三段，"
                       "与 nested 的 bands_pct 不是同一口径"),
                acf=_finite(raw.get("acf")),
                fit_metrics=metrics or None,
                residual_bands=dict(kind="resid_bins_sigma_normalized",
                                    values=bands) if bands else None,
                elpd=_finite(raw.get("elpd_loo")),
                elpd_valid=_finite(raw.get("dse")) is not None,
                ppc=dict(status="unavailable",
                         note="产物未落盘后验预测检查；不得用残差 acf 冒充 PPC"))


def _prediction_nested(raw: dict) -> dict:
    m = raw.get("metrics") or {}
    d = raw.get("diagnostics") or {}
    metrics = {k: _finite(m[k]) for k in ("rms_rel", "rms_re", "rms_im")
                if m.get(k) is not None}
    bands = _json_safe(m.get("bands_pct") or [])
    return dict(_blank("partial" if metrics else "unavailable",
                       "*.nested.json:metrics.acf/rms_rel/bands_pct + diagnostics.elpd",
                       False,
                       "rms_rel = Rwp/√2，不是 Rwp；逐点 σ 由同一批观测算出时"
                       " elpd 不作预测诊断（见 diagnostics.elpd_note）"),
                acf=_finite(m.get("acf")),
                fit_metrics=metrics or None,
                residual_bands=dict(kind="bands_pct", values=bands) if bands else None,
                elpd=_finite(d.get("elpd")),
                elpd_valid=bool(d.get("elpd_valid_for_predictive_comparison")),
                ppc=dict(status="unavailable",
                         note="产物未落盘后验预测检查；不得用残差 acf 冒充 PPC"))


# ---------------------------------------------------------------- 主入口

def build_report(raw: dict, *, family: str, model_id: str,
                 source_files: list | tuple = ()) -> dict:
    """把一条已落盘的产物归一成统一 schema 报告。纯函数，不读写文件。

    ``family`` 只允许 ``voigt`` / ``nested``；其它值直接 raise（不许悄悄降级）。
    """
    if family not in ("voigt", "nested"):
        raise ValueError(f"family 只能是 voigt/nested，收到 {family!r}")
    if not isinstance(raw, dict):
        raise TypeError("raw 必须是产物字典")

    if family == "voigt":
        dg = dict(
            posterior=_posterior_voigt(raw),
            correlation=_correlation_voigt(raw),
            boundary=_boundary_voigt(raw),
            combinations=_combinations_voigt(raw),
            sampling=_sampling_voigt(raw),
            prediction=_prediction_voigt(raw),
        )
    else:
        missing = ("*.nested.json 未落盘后验样本：无相关矩阵、无逐参数贴边率、"
                   "无法区分单参数与组合可辨识")
        dg = dict(
            posterior=_posterior_nested(raw),
            correlation=dict(_blank("unavailable", "*.nested.json（无后验样本）",
                                   False, missing),
                            n_draw=None, strong_tol=None, medium_tol=None,
                            strong_pairs=None, medium_pairs=None, matrix_names=None),
            boundary=dict(_blank("unavailable", "*.nested.json（无后验样本 + 无先验边界表）",
                                 False, missing),
                          n_params=None, fixed_params=None, at_bound_params=None,
                          at_bound_fraction=None, max_at_bound_signal=None, level=None),
            combinations=dict(_blank("unavailable", "*.nested.json（无相关矩阵）",
                                     False, missing),
                              known_combinations=None, single_param_limitation=None),
            sampling=_sampling_nested(raw),
            prediction=_prediction_nested(raw),
        )

    counts = {s: sum(1 for v in dg.values() if v["status"] == s)
              for s in ("available", "partial", "unavailable")}
    out = dict(
        schema_version=SCHEMA_VERSION,
        family=family,
        model_id=str(model_id),
        source_files=[str(x) for x in source_files],
        status=raw.get("status"),          # 原样抄，不改（voigt 无此键 → None）
        disclaimer=DISCLAIMER,
        diagnostics=dg,
        coverage=counts,
        notes=list(GENERAL_NOTES),
    )
    out = _json_safe(out)
    json.dumps(out, ensure_ascii=False, allow_nan=False)   # 严格 JSON 自检
    for k in FORBIDDEN_KEYS:                              # 只记账不裁决
        if k in out:
            raise AssertionError(f"报告不得含裁决字段 {k!r}")
    return out


def assert_same_schema(*reports: dict) -> None:
    """跨族同 schema 校验：键集合必须完全一致。"""
    if len(reports) < 2:
        return
    ref = set(reports[0]["diagnostics"])
    for i, r in enumerate(reports[1:], 1):
        got = set(r["diagnostics"])
        if got != ref:
            raise AssertionError(f"第 {i} 份报告 schema 不一致：缺 {ref - got}、多 {got - ref}")
        for key in ref:
            a, b = set(reports[0]["diagnostics"][key]), set(r["diagnostics"][key])
            # 专属字段允许族间不同名（rms_re vs rms_rel），但诊断头四键必须齐
            for must in ("status", "source", "comparable", "note"):
                if must not in a or must not in b:
                    raise AssertionError(f"{key} 缺公共字段 {must}")


def format_markdown(rep: dict) -> str:
    """把一份报告渲染成几行 Markdown（批处理器汇总用）。"""
    L = [f"### {rep['model_id']} · `{rep['family']}`",
         "", f"> {rep['disclaimer']}", "",
         "| 诊断 | 状态 | 跨族可比 | 要点 |", "|---|---|---|---|"]
    label = dict(posterior="参数后验", correlation="参数相关", boundary="边界诊断",
                  combinations="参数组合", sampling="采样诊断", prediction="预测诊断")
    for k, v in rep["diagnostics"].items():
        if k == "posterior":
            brief = f"{v['n_params'] or '—'} 个参数，q05/q50/q95 原尺度"
        elif k == "correlation":
            brief = (f"强相关 {len(v['strong_pairs'] or [])} 对"
                     if v["status"] == "available" else v["note"])
        elif k == "boundary":
            brief = (f"钉死 {len(v['fixed_params'] or [])} 个，贴边比例 null"
                     if v["status"] != "unavailable" else v["note"])
        elif k == "combinations":
            brief = (f"已知组合 {len(v['known_combinations'] or [])} 条"
                     if v["status"] == "available" else v["note"])
        elif k == "sampling":
            brief = (f"R̂max {v['rhat_max']:.3f}" if isinstance(v.get("rhat_max"), float)
                     else f"R̂max {v.get('rhat_max')}") + \
                    f"，div {v.get('divergences')}"
        else:
            acf = v.get('acf')
            brief = (f"acf {acf:.3f}" if isinstance(acf, float) else f"acf {acf}") \
                    + f"，PPC {v['ppc']['status']}"
        L.append(f"| {label[k]} | `{v['status']}` | "
                 f"{'是' if v['comparable'] else '否'} | {brief} |")
    c = rep["coverage"]
    L += ["", f"覆盖：available {c['available']} / partial {c['partial']} / "
              f"unavailable {c['unavailable']}", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- 自检

def _selftest() -> None:
    voigt = dict(
        param_corr=dict(
            names=["R_inf", "W"], corr=[[1.0, -0.94], [-0.94, 1.0]],
            summary=[dict(name="R_inf", q05=0.006, q50=0.0062, q95=0.0063),
                     dict(name="W", q05=40.0, q50=55.0, q95=70.0)],
            fixed=[dict(name="zeta", value=1.0)],
            strong=[dict(a="R_inf", b="W", rho=-0.94, why="x")],
            medium=[], n_draw=1000, strong_tol=0.9, medium_tol=0.7,
            degenerate=[dict(kind="Rinf-Wtot", comp=0, rho=-0.94, text="只报和")]),
        diag=dict(R_inf=dict(rhat=1.001, bulk=1500.0, tail=1800.0)),
        component_review=dict(parameter_identifiability=dict(
            max_at_bound=0.899, level="degenerate",
            strong_pairs=[dict(a="R_inf", b="W", rho=-0.94)])),
        degeneracy=[dict(phi=0.56, f_peak=0.086, at_bound=0.899)],
        resid_bins=[dict(n=19, mean_re=0.0, rms_re=0.001)],
        acf=0.42, rms_re=0.001, rms_im=0.002, elpd_loo=-100.0, dse=2.0,
        divergences=0, sample_grade=2,
    )
    nested = dict(
        status="exploratory",
        params=dict(Rs=dict(median=10.0, ci90=[9.0, 11.0], rhat=1.002, ess=800.0)),
        diagnostics=dict(rhat_max=1.002, ess_min=800.0, divergences=0,
                          elpd=-90.0, elpd_valid_for_predictive_comparison=False),
        metrics=dict(acf=0.55, rms_rel=6.7, bands_pct=[1.0, 2.0, 3.0]),
    )
    a = build_report(voigt, family="voigt", model_id="selftest/voigt")
    b = build_report(nested, family="nested", model_id="selftest/nested")
    assert_same_schema(a, b)

    # 原尺度不 exp：q50 必须逐值相等
    assert a["diagnostics"]["posterior"]["params"][1]["q50"] == 55.0
    assert a["diagnostics"]["posterior"]["params"][1]["rel_span"] == (70.0 - 40.0) / 55.0
    assert a["diagnostics"]["boundary"]["at_bound_fraction"] is None
    assert b["diagnostics"]["correlation"]["status"] == "unavailable"
    assert b["diagnostics"]["correlation"]["strong_pairs"] is None
    assert b["diagnostics"]["sampling"]["ess_bulk_min"] is None
    assert b["status"] == "exploratory" and a["status"] is None
    assert a["diagnostics"]["posterior"]["comparable"] is True
    assert a["diagnostics"]["prediction"]["comparable"] is False
    assert "rms_rel" in b["diagnostics"]["prediction"]["fit_metrics"]
    assert "rms_rel" not in a["diagnostics"]["prediction"]["fit_metrics"]
    for r in (a, b):
        assert not (set(r) & set(FORBIDDEN_KEYS))
        assert r["coverage"]["available"] + r["coverage"]["partial"] + \
               r["coverage"]["unavailable"] == len(DIAG_KEYS)
        json.dumps(r, ensure_ascii=False, allow_nan=False)
    # 非有限值必须变 null，不许写 0
    v2 = dict(voigt); v2["acf"] = float("nan")
    assert build_report(v2, family="voigt", model_id="nan")[
        "diagnostics"]["prediction"]["acf"] is None
    try:
        build_report({}, family="bogus", model_id="x")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 family 必须 raise")
    print("bm_identifiability 自检通过：同 schema / 原尺度 / 三态 / 不裁决 / 严格 JSON")
    print(format_markdown(a))
    print(format_markdown(b))


if __name__ == "__main__":
    _selftest()
