"""离线回放 v2（第 3 轮审查修订版）：四层状态，只读，不改算法代码。

本轮修订：
- k_reference 改为完整模型标识 {family, K, elpd}，取全局 LOO 最优，且只作预测参照。
- 严格门控逐 control 检查 kk.status / grade / div / tree_cap / R-hat / ESS；
  effective 统一为完整严格有效性（含 KK），不再只用 G_K2_none 代表全部。
- 机制判定先检查 delta/dse 完整性：缺失或非有限 -> comparison_incomplete，
  不得落入 evidence_for_model_preference。
- K 选择与机制比较的阈值规则分开（pairwise_dse.csv 的 comparison_type /
  threshold_rule：K 用 max(2,2*dse)，机制用 1*dse）。
- 第四层取消 asymptotic_only 语义，状态只用
  reference_frequency_stable / reference_frequency_dependent / unidentifiable / not_applicable。
- 增加按模型区分的 component_out_of_window。
- 机制 ambiguous 附高 Pareto-k 敏感性复核提示。

运行：
  D:/autoeis_env/Scripts/python.exe replay_identifiability.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TH = dict(p_out=0.50, elpd_abs=2.0, elpd_k=2.0, rho_degen=0.90,
          cd_rel_width=0.50)


def read_json(name):
    p = glob.glob(str(ROOT / "results" / name / "real_*.json"))[0]
    return json.loads(Path(p).read_text(encoding="utf-8"))


def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))


def corr_rho(js, a, b):
    pc = js.get("param_corr") or {}
    names, mat = pc.get("names") or [], pc.get("corr") or []
    if a not in names or b not in names:
        return None
    i, j = names.index(a), names.index(b)
    try:
        return float(mat[i][j])
    except Exception:
        return None


def cd_relative_width(name):
    g = glob.glob(str(ROOT / "results" / name / "*_samples.npz"))
    if not g:
        return None                      # 原始数据未随包发布时，不伪造通过
    z = np.load(g[0], allow_pickle=True)
    if "tauD" not in z or "Rwo" not in z:
        return None
    cd = np.asarray(z["tauD"], float) / np.maximum(np.asarray(z["Rwo"], float), 1e-300)
    return float((q(cd, 95) - q(cd, 5)) / q(cd, 50))


def load_diagnostics():
    """优先读随包发布的 control_diagnostics.csv；独立环境无需仓库 run.log。"""
    p = ROOT / "control_diagnostics.csv"
    if not p.exists():
        return {}
    return {r["run_id"]: r.to_dict()
            for _, r in pd.read_csv(p).iterrows()}


def gate_of(js, row):
    """fail-closed：任一必要诊断缺失/非有限 -> effective=False。

    不再把 None 跳过，也不再把缺失的 tree_cap 当成 0（通过）。
    """
    missing = []

    def need(name, v):
        if v is None:
            missing.append(name); return None
        if isinstance(v, (int, float, np.floating)) and not np.isfinite(float(v)):
            missing.append(name); return None
        return v

    kk = (js.get("kk") or {}).get("status")
    grade = need("sample_grade", js.get("sample_grade"))
    div = need("divergences", js.get("divergences"))
    cap = need("tree_cap", js.get("tree_cap"))
    rhat = need("rhat_max", row.get("rhat_max") if row else None)
    bulk = need("ess_bulk", row.get("ess_bulk") if row else None)
    tail = need("ess_tail", row.get("ess_tail") if row else None)
    if kk is None:
        missing.append("kk.status")
    checks = dict(
        kk_ok=(None if kk is None else kk == "ok"),
        grade_ok=(None if grade is None else int(grade) >= 2),
        div_ok=(None if div is None else int(div) == 0),
        cap_ok=(None if cap is None else float(cap) < 0.01),
        rhat_ok=(None if rhat is None else float(rhat) <= 1.01),
        ess_ok=(None if (bulk is None or tail is None)
                else min(float(bulk), float(tail)) >= 400),
        rhat_value=rhat, ess_bulk=bulk, ess_tail=tail,
    )
    effective = (not missing) and all(v is True for k, v in checks.items()
                                      if k in ("kk_ok", "grade_ok", "div_ok",
                                               "cap_ok", "rhat_ok", "ess_ok"))
    return checks, effective, missing


def main():
    g = pd.read_csv(ROOT / "grid_comparison.csv")
    g["K"] = g["model"].str.extract(r"K=(\d)").astype(int)
    g["diff"] = g["model"].str.split("/").str[1].str.strip()
    pw = pd.read_csv(ROOT / "pairwise_dse.csv").set_index("pair")
    rec = pd.read_csv(ROOT / "reference_frequency_contribution.csv")
    base = json.loads((ROOT / "baseline.json").read_text(encoding="utf-8"))
    data_sha = base["inputs"]["original_50pt.txt"]["sha256"]
    names = {"none": ("G_K1_none", "G_K2_none"),
             "Ws": ("G_K1_Ws", "G_K2_Ws"),
             "Wo": ("G_K1_Wo", "G_K2_Wo")}

    # ---- 1. 逐 control 严格门控 ----
    diag = load_diagnostics()
    controls_done, controls_missing, gate_detail = [], [], {}
    for d, pair in names.items():
        for K, run_id in zip((1, 2), pair):
            js = read_json(run_id)
            checks, effective, missing = gate_of(js, diag.get(run_id))
            gate_detail[f"{d}/K={K}"] = dict(checks,
                                             missing_diagnostics=missing)
            controls_done.append(dict(
                family=d, K=K, data_sha256=data_sha,
                status=js.get("k_selection_status"), run_id=run_id,
                effective=effective, checks=checks,
                missing_diagnostics=missing,
                p_peak_out_max=js.get("p_peak_out_max")))
    kk_all_ok = all(c["checks"]["kk_ok"] is True for c in controls_done)
    grade_all_ok = all(c["checks"]["grade_ok"] is True for c in controls_done)
    n_effective = sum(bool(c["effective"]) for c in controls_done)
    controls_complete = ((not controls_missing)
                         and n_effective == len(controls_done) and kk_all_ok)

    # ---- 2. K 选择（comparison_type = K_selection）----
    k_rows, k_stable = [], True
    for d in ("none", "Ws", "Wo"):
        row = pw.loc[f"K2-K1 / {d}"]
        assert row.comparison_type == "K_selection"
        delta, dse, thr = float(row.delta), float(row.dse), float(row.thr)
        sig = np.isfinite(delta) and delta > thr
        k_rows.append(dict(diffusion=d, delta=delta, dse=dse, thr=thr,
                           threshold_rule=str(row.threshold_rule),
                           n_observations=int(row.n_observations),
                           K2_better=bool(sig)))
        k_stable &= bool(sig)
    if not kk_all_ok or not grade_all_ok or not controls_complete:
        k_status = "exploratory_only"
    elif k_stable:
        k_status = "converged"
    else:
        k_status = "K_SELECTION_AMBIGUOUS"

    # ---- 3. k_reference：完整模型标识（全局 LOO 最优，仅预测参照）----
    best = g.loc[g["elpd_loo_recalc"].idxmax()]
    k_reference = dict(family=str(best["diff"]), K=int(best["K"]),
                       elpd=float(best["elpd_loo_recalc"]),
                       role="prediction_reference_only")

    # ---- 4. 机制：先查数据完整性 ----
    mech_rows = []
    for label in ("Ws-Wo / K=2", "Ws-none / K=2"):
        row = pw.loc[label]
        assert row.comparison_type == "mechanism"
        mech_rows.append(dict(pair=label, delta=float(row.delta),
                              dse=float(row.dse), thr=float(row.thr),
                              threshold_rule=str(row.threshold_rule),
                              direction=str(row.direction)))
    finite = all(np.isfinite(m["delta"]) and np.isfinite(m["dse"])
                 and m["dse"] > 0 for m in mech_rows)
    if not finite:
        mech = "comparison_incomplete"
    elif not controls_complete:
        mech = "not_tested"
    else:
        insignificant = [abs(m["delta"]) < m["dse"] for m in mech_rows]
        sig_dirs = {m["direction"] for m in mech_rows
                    if abs(m["delta"]) >= m["dse"]}
        if all(insignificant):
            mech = "ambiguous"
        elif len(sig_dirs) > 1:
            mech = "preference_conflict"
        else:
            mech = "evidence_for_model_preference"

    # ---- 5. 扩散参数（rho 或 C_D 区间，逻辑或）----
    j_wo2 = read_json("G_K2_Wo")
    d2 = j_wo2.get("diffusion_resolution") or {}
    rho_rw = corr_rho(j_wo2, "ln Rwo", "ln tauD")
    cd_w = cd_relative_width("G_K2_Wo")
    by_rho = rho_rw is not None and abs(rho_rw) > TH["rho_degen"]
    by_cd = cd_w is not None and cd_w > TH["cd_rel_width"]
    diff_id = ("shape_resolvable_params_degenerate"
               if d2.get("level") == "resolvable" and (by_rho or by_cd)
               else (d2.get("level") or "unknown"))

    # ---- 6. 第四层：精确 HN 幅值，取消 asymptotic_only 语义 ----
    verdicts = rec["verdict"].tolist()
    if all(v == "reference_frequency_stable" for v in verdicts):
        comp_id = "reference_frequency_stable"
    elif all(v == "unidentifiable" for v in verdicts):
        comp_id = "unidentifiable"
    else:
        comp_id = "reference_frequency_dependent"
    comp_rows = [dict(family=r.family,
                      omega_tau_fmin=r.omega_tau_fmin,
                      asymptotic_valid_fmin=bool(r.asymptotic_valid_fmin),
                      asymptotic_valid_geom=bool(r.asymptotic_valid_geom),
                      CV_Aexact_fmin=r.CV_Aexact_fmin,
                      CV_Aexact_geom=r.CV_Aexact_geom,
                      verdict=r.verdict)
                 for _, r in rec.iterrows()]

    # ---- 7. 按模型区分的出窗标记 ----
    out_of_window = {r["model"]: float(r["p_peak_out_max"])
                     for _, r in g.iterrows()}

    v1 = pd.read_csv(ROOT / "replay_ambiguous.csv")
    v1_map = dict(zip(v1.case, v1.new_status))
    out = pd.DataFrame([dict(
        case="Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        v1_status=v1_map.get("K1_none", "n/a"),
        v2_k_selection_status=k_status,
        v2_k_reference=json.dumps(k_reference, ensure_ascii=False),
        v2_mechanism_identifiability=mech,
        v2_diffusion_identifiability=diff_id,
        v2_reference_frequency_contribution=comp_id,
        v2_component_out_of_window=json.dumps(out_of_window, ensure_ascii=False),
        kk_all_ok=kk_all_ok, grade_all_ok=grade_all_ok,
        controls_complete=controls_complete,
        controls_effective=n_effective, controls_total=len(controls_done),
        mechanism_pareto_k_sensitive=True,
        mechanism_evidence_level="exploratory",
        mechanism_pareto_k_caveat=(
            "完整频段 PSIS-LOO 下 Ws/Wo/none 的差异均未超过配对 dse；"
            "删除高 Pareto-k 频点的敏感性分析中 Ws-Wo 判定翻转，"
            "说明该结论对单个频点敏感。当前 ambiguous 属探索性判断，"
            "待 91.03 Hz 的 moment matching / 精确 LOO 复核"),
        dse_source="pairwise_dse.csv（逐频点 LOO 差值；comparison_type 区分阈值规则）",
        gate_detail=json.dumps(gate_detail, ensure_ascii=False),
        evidence_k="; ".join(f"{r['diffusion']}: Δ={r['delta']:.1f} > {r['thr']:.1f} "
                             f"({r['threshold_rule']})" for r in k_rows),
        evidence_mech="; ".join(f"{m['pair']}: |Δ|={abs(m['delta']):.3f} vs "
                                f"dse {m['dse']:.3f} ({m['threshold_rule']})"
                                for m in mech_rows),
        evidence_diff=f"rho={rho_rw}, C_D rel.width={cd_w}, by_rho={by_rho}, by_cd={by_cd}",
        evidence_comp=json.dumps(comp_rows, ensure_ascii=False),
    )])
    out.to_csv(ROOT / "replay_identifiability.csv", index=False,
               float_format="%.6g")

    print("=== 逐 control 门控（含 KK）===")
    for k, v in gate_detail.items():
        print(f"  {k:10s} {v}")
    print("  kk_all_ok =", kk_all_ok, "| grade_all_ok =", grade_all_ok,
          "| effective =", n_effective, "/", len(controls_done))
    print("\n=== 四层汇总（第 3 轮修订）===")
    print(out[["case", "v2_k_selection_status", "v2_k_reference",
               "v2_mechanism_identifiability", "v2_diffusion_identifiability",
               "v2_reference_frequency_contribution", "kk_all_ok"]]
          .to_string(index=False))
    print("\nDONE ->", ROOT / "replay_identifiability.csv")


if __name__ == "__main__":
    main()
