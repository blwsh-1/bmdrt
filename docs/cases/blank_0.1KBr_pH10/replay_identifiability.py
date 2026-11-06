"""离线回放 v2（审查修订版）：四层状态，只读，不改算法代码。

修订点：
- 第四层改用固定参考频率 A_*(w_*)（单位 Ω），指标来自 asymptotic_recheck.csv；
  asymptotic_only 需三个条件 + 两个参考频率结论一致，否则 reference_frequency_dependent。
- 第三层扩散参数简并 = |rho(ln R_W,ln tau_D)|>0.9 **或** C_D 相对区间宽度 >50%。
- 严格门控检查全部 6 个 control 的采样等级，不只 K=2。
- controls 区分「已运行」与「有效完成」（grade>=2 且 div=0）。

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
REPO = next((q for q in [ROOT, *ROOT.parents] if (q / "bm_core.py").exists()),
            Path(r"D:/eis训练数据/bmdrt"))
sys.path.insert(0, str(REPO))
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
    """C_D = tauD/Rwo 的 (q95-q05)/q50；无 tauD 时返回 None。"""
    p = glob.glob(str(ROOT / "results" / name / "*_samples.npz"))[0]
    z = np.load(p, allow_pickle=True)
    if "tauD" not in z or "Rwo" not in z:
        return None
    cd = np.asarray(z["tauD"], float) / np.maximum(np.asarray(z["Rwo"], float), 1e-300)
    return float((q(cd, 95) - q(cd, 5)) / q(cd, 50))


def main():
    g = pd.read_csv(ROOT / "grid_comparison.csv")
    g["K"] = g["model"].str.extract(r"K=(\d)").astype(int)
    g["diff"] = g["model"].str.split("/").str[1].str.strip()
    piv = {(r.K, r["diff"]): r for _, r in g.iterrows()}
    rec = pd.read_csv(ROOT / "reference_frequency_contribution.csv")
    pw = pd.read_csv(ROOT / "pairwise_dse.csv").set_index("pair")

    # ---- 严格门控：检查全部 6 个 control ----
    all_grade = {r["model"]: int(r["grade"]) for _, r in g.iterrows()}
    grade_ok = all(v >= 2 for v in all_grade.values())

    # ---- controls：已运行 vs 有效完成 ----
    base = json.loads((ROOT / "baseline.json").read_text(encoding="utf-8"))
    data_sha = base["inputs"]["original_50pt.txt"]["sha256"]
    names = {"none": ("G_K1_none", "G_K2_none"),
             "Ws": ("G_K1_Ws", "G_K2_Ws"),
             "Wo": ("G_K1_Wo", "G_K2_Wo")}
    controls_done, controls_missing = [], []
    for d, pair in names.items():
        for K, run_id in zip((1, 2), pair):
            if (K, d) not in piv:
                controls_missing.append(dict(family=d, K=K, reason="未运行"))
                continue
            js = read_json(run_id)
            effective = (int(js.get("sample_grade") or 0) >= 2
                         and int(js.get("divergences") or 0) == 0
                         and float(js.get("tree_cap") or 0) < 0.01)
            controls_done.append(dict(
                family=d, K=K, data_sha256=data_sha,
                status=js.get("k_selection_status"), run_id=run_id,
                elpd_loo=float(piv[(K, d)].elpd_loo_recalc),
                effective=effective))
    n_effective = sum(bool(c["effective"]) for c in controls_done)
    controls_complete = (not controls_missing) and n_effective == len(controls_done)

    # ---- 第一层：K 选择（家族内配对 dse，由 analyze_grid 重算）----
    k_rows, k_stable = [], True
    for d in ("none", "Ws", "Wo"):
        # dse / Delta / thr 唯一来源：逐频点配对差值（pairwise_dse.csv）
        row = pw.loc[f"K2-K1 / {d}"]
        delta, dse, thr = float(row.delta), float(row.dse), float(row.thr)
        n_obs = int(row.n_observations)
        sig = delta > thr
        k_rows.append(dict(diffusion=d, delta=delta, dse=dse, thr=thr,
                           n_observations=n_obs, K2_better=sig))
        k_stable &= sig
    kk_status = read_json("G_K2_none").get("kk", {}).get("status")
    if kk_status != "ok" or not grade_ok:
        k_status = "exploratory_only"
    elif k_stable:
        k_status = "converged"
    else:
        k_status = "K_SELECTION_AMBIGUOUS"

    # ---- 第二层：机制三态 ----
    pairs = []
    for label in ("Ws-Wo / K=2", "Ws-none / K=2"):
        row = pw.loc[label]
        pairs.append((label, float(row.delta), float(row.dse),
                      str(row.direction), int(row.n_observations)))
    insignificant = [abs(dd) < dse for _, dd, dse, _, _ in pairs]
    sig_dirs = {direction for _, dd, dse, direction, _ in pairs
                if abs(dd) >= dse}
    if not controls_complete:
        mech = "not_tested"
    elif all(insignificant):
        mech = "ambiguous"
    elif len(sig_dirs) > 1:
        mech = "preference_conflict"
    else:
        mech = "evidence_for_model_preference"

    # ---- 第三层：扩散参数（rho 或 C_D 区间，逻辑或）----
    j_wo2, j_wo1 = read_json("G_K2_Wo"), read_json("G_K1_Wo")
    d2 = j_wo2.get("diffusion_resolution") or {}
    d1 = j_wo1.get("diffusion_resolution") or {}
    rho_rw = corr_rho(j_wo2, "ln Rwo", "ln tauD")
    cd_w = cd_relative_width("G_K2_Wo")
    by_rho = rho_rw is not None and abs(rho_rw) > TH["rho_degen"]
    by_cd = cd_w is not None and cd_w > TH["cd_rel_width"]
    if d2.get("level") == "resolvable" and (by_rho or by_cd):
        diff_id = "shape_resolvable_params_degenerate"
    else:
        diff_id = d2.get("level") or "unknown"
    diff_note = (f"K=2/Wo level={d2.get('level')}, rho={rho_rw}, "
                 f"C_D rel.width={cd_w}, by_rho={by_rho}, by_cd={by_cd}; "
                 f"K=1/Wo level={d1.get('level')}")

    # ---- 第四层：A_*(omega_*) 与三条件，两个参考频率必须一致 ----
    comp_rows = []
    for _, r in rec.iterrows():
        comp_rows.append(dict(
            family=r.family, omega_tau_fmin=r.omega_tau_fmin,
            asymptotic_valid_fmin=bool(r.asymptotic_valid_fmin),
            CV_Aasym_fmin=r.CV_Aasym_fmin, CV_Aexact_fmin=r.CV_Aexact_fmin,
            CV_Aexact_geom=r.CV_Aexact_geom, CV_w=r.CV_w,
            CV_tau=r.CV_tau, rho=r.rho_lnw_lntau,
            stable_fmin=bool(r.three_exact_fmin),
            stable_geom=bool(r.three_exact_geom)))
    comp_id = ("asymptotic_only"
               if all(c["stable_fmin"] and c["stable_geom"] for c in comp_rows)
               else "reference_frequency_dependent")

    v1 = pd.read_csv(ROOT / "replay_ambiguous.csv")
    v1_map = dict(zip(v1.case, v1.new_status))

    out = pd.DataFrame([dict(
        case="Blank_0.1KBr_pH10_EIS_PtCu_1.2V",
        v1_status=v1_map.get("K1_none", "n/a"),
        v2_k_selection_status=k_status, v2_K_reference=2,
        v2_mechanism_identifiability=mech,
        v2_diffusion_identifiability=diff_id,
        v2_reference_frequency_contribution=comp_id,
        controls_complete=controls_complete, controls_effective=n_effective,
        controls_total=len(controls_done),
        v2_controls_done=json.dumps(controls_done, ensure_ascii=False),
        v2_controls_missing=json.dumps(controls_missing, ensure_ascii=False),
        evidence_k="; ".join(f"{r['diffusion']}: Δ={r['delta']:.1f} > {r['thr']:.1f}"
                             for r in k_rows),
        evidence_mech="; ".join(f"{a}: |Δ|={abs(dd):.3f} vs dse {dse:.3f} ({d})"
                                for a, dd, dse, d, _ in pairs),
        dse_source="pairwise_dse.csv（逐频点 LOO 差值；合成用例为 run.log ③）",
        n_observations=int(k_rows[0]["n_observations"]),
        evidence_diff=diff_note,
        evidence_comp=json.dumps(comp_rows, ensure_ascii=False),
        B2_unit_dependence=json.dumps(
            rec[["family", "CV_B2_seconds", "CV_B2_milliseconds"]]
            .to_dict("records"), ensure_ascii=False),
    )])
    out.to_csv(ROOT / "replay_identifiability.csv", index=False,
               float_format="%.6g")

    print("=== 严格门控（全部 6 个 control）===")
    print(all_grade, "grade_ok =", grade_ok, "kk =", kk_status)
    print("\n=== controls ===")
    for c in controls_done:
        print(f"  {c['family']:4s} K={c['K']} {c['run_id']:9s} "
              f"status={c['status']:14s} effective={c['effective']}")
    print("  effective:", n_effective, "/", len(controls_done),
          "| complete:", controls_complete)
    print("\n=== 四层汇总（审查修订版）===")
    print(out[["case", "v1_status", "v2_k_selection_status",
               "v2_mechanism_identifiability", "v2_diffusion_identifiability",
               "v2_reference_frequency_contribution", "controls_complete"]]
          .to_string(index=False))
    print("\n=== A_* 复核 ===")
    print(rec[["family", "omega_tau_fmin", "asymptotic_valid_fmin",
               "CV_Aexact_fmin", "CV_Aexact_geom", "three_exact_fmin",
               "three_exact_geom", "verdict"]].to_string(index=False))
    print("\nDONE ->", ROOT / "replay_identifiability.csv")


if __name__ == "__main__":
    main()
