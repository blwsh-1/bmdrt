"""v2 状态机：用例 2/2b/4/4b 验证 + 历史回放离线回归（只读，不改算法代码）。

输入全部来自已落盘的 run.log / real_*.json，不重跑 NUTS。
产出 v2_case_tests.csv、v2_history_regression.csv 与控制台摘要。

运行：
  D:/autoeis_env/Scripts/python.exe v2_state_machine.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = Path(r"D:/eis训练数据/bmdrt")
CASES_DIR = REPO / "out/_v2cases"

TH = dict(p_out=0.50, elpd_abs=2.0, elpd_k=2.0, grade_min=2)


# --------------------------------------------------------------------------
# 只读解析
# --------------------------------------------------------------------------
def parse_log(p: Path):
    txt = p.read_text(encoding="utf-8", errors="replace")
    kk = re.search(r"Lin-KK 门禁：(通过|跳过|不通过)[^\n]*", txt)
    kk_line = kk.group(0) if kk else ""
    if "有归因" in kk_line or "warn" in kk_line.lower():
        kk_status = "warn"
    elif "不通过" in kk_line or "拒绝" in kk_line:
        kk_status = "fail"
    else:
        kk_status = "ok"
    cands, cur = {}, None
    for ln in txt.splitlines():
        m = re.search(r"--- K=(\d+) 拟合中", ln)
        if m:
            cur = int(m.group(1)); cands[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"R̂max=([\d.]+)\s+bulk=([\d.]+)\s+tail=([\d.]+)\s+"
                      r"div=(\d+)\s+触顶=([\d.]+)%\s+→ 分级 (\d)", ln)
        if m:
            cands[cur].update(rhat=float(m.group(1)), bulk=float(m.group(2)),
                              tail=float(m.group(3)), div=int(m.group(4)),
                              cap=float(m.group(5))/100, grade=int(m.group(6)))
            continue
        m = re.search(r"elpd=(-?[\d.]+)\s+rms/σ=([\d.]+)/([\d.]+)\s+"
                      r"acf=([\d.]+)\s+P\(峰位出窗\)max=([\d.]+)", ln)
        if m:
            cands[cur].update(elpd=float(m.group(1)), p_out=float(m.group(5)))
    pair = {}
    for ln in txt.splitlines():
        m = re.match(r"\s*K=(\d+)\s+\d+\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s", ln)
        if m:
            pair[int(m.group(1))] = dict(elpd=float(m.group(2)),
                                         elpd_diff=float(m.group(3)),
                                         dse=float(m.group(4)))
    removed = [ln.strip() for ln in txt.splitlines()
               if "**剔除**" in ln]
    return dict(kk_status=kk_status, kk_line=kk_line, cands=cands,
                pair=pair, removed=removed, txt=txt)


# --------------------------------------------------------------------------
# v2 状态机（纯函数）
# --------------------------------------------------------------------------
def decide(kk_status, cands, pair, n_candidates_planned, removed,
           controls_done, controls_missing, mech_diffs, family="none",
           pair_dse=None):
    """四层判定。cands 已过滤为「采样合格」的候选。

    审查修订（2026-11-06）：
    - k_reference 统一取 LOO 最优候选，且返回完整模型标识 {family,K,elpd}；
      无可用 LOO 时返回 None（不再用 max(K) 代替）。
    - 配对 dse 缺失/非有限时禁止严格 converged。
    - Δ 不显著、候选不完整 → K_SELECTION_AMBIGUOUS（不再「取小 K 当 converged」）。
    - 多 K 时与「最强较低 K 竞争者」比较，而不是 min(cands)。
    """
    reasons = []

    def make_ref(cc):
        if not cc or any(not np.isfinite(c.get("elpd", np.nan))
                         for c in cc.values()):
            return None
        K = max(cc, key=lambda k: cc[k]["elpd"])
        return dict(family=family, K=int(K), elpd=float(cc[K]["elpd"]))

    k_ref = make_ref(cands)

    # 严格门控：检查所有参与比较的候选，不只最终胜出的 K
    gated = (kk_status != "ok") or any(
        c.get("grade", 0) < TH["grade_min"] or c.get("div", 0) != 0
        or c.get("cap", 1) >= 0.01 or c.get("rhat", 9) > 1.01
        or min(c.get("bulk", 0), c.get("tail", 0)) < 400
        for c in cands.values())

    # 配对 dse 唯一来源：pairwise_dse.csv 中实际参与比较的那一对
    # （best 与 K_lower*），不再从所有比较里取 max。
    dse = float(pair_dse) if pair_dse is not None else float("nan")
    dse_ok = bool(np.isfinite(dse)) and dse > 0

    # 第一层：K 选择
    if kk_status == "fail":
        k_status = "rejected_by_kk"
    elif gated:
        k_status = "exploratory_only"
    elif not cands:
        k_status = "rejected"
    elif n_candidates_planned == 1:
        k_status = "not_compared"
    else:
        best = max(cands, key=lambda K: cands[K]["elpd"])
        lowers = [K for K in cands if K < best]
        lower_star = (max(lowers, key=lambda K: cands[K]["elpd"])
                      if lowers else None)
        delta = (cands[best]["elpd"] - cands[lower_star]["elpd"]
                 if lower_star is not None else float("inf"))
        thr = max(TH["elpd_abs"], TH["elpd_k"] * (dse if dse_ok else 0))
        incomplete = len(cands) < n_candidates_planned
        if incomplete:
            k_status = "K_SELECTION_AMBIGUOUS"
            reasons.append("候选集合不完整（不能当作完整比较的结论）")
        elif not dse_ok:
            k_status = "K_SELECTION_AMBIGUOUS"
            reasons.append("配对 dse 缺失/非有限，禁止严格 converged")
        elif delta <= thr:
            k_status = "K_SELECTION_AMBIGUOUS"
            reasons.append(f"与最强较低 K 竞争者 Δ={delta:.1f} ≤ thr={thr:.1f}")
        else:
            k_status = "converged"
            reasons.append(f"与最强较低 K 竞争者 Δ={delta:.1f} > thr={thr:.1f}")

    # 第二层：机制三态
    if controls_missing:
        mech = "not_tested"
    elif mech_diffs and all(abs(d) < dse for d, dse in mech_diffs):
        mech = "ambiguous"
    elif mech_diffs:
        mech = "evidence_for_model_preference"
    else:
        mech = "not_tested"

    # 第三/四层：本脚本不重算（由 replay_identifiability.py 负责）
    return dict(k_selection_status=k_status,
                k_reference=json.dumps(k_ref, ensure_ascii=False) if k_ref else None,
                mechanism_identifiability=mech,
                controls_done=len(controls_done),
                controls_missing=len(controls_missing),
                reasons="; ".join(reasons))


# --------------------------------------------------------------------------
# 用例 2 / 2b / 4 / 4b
# --------------------------------------------------------------------------
def build_cases():
    pw = pd.read_csv(ROOT / "pairwise_dse.csv").set_index("pair")
    specs = [
        ("用例 2  K 选择不确定", "U2_noise2", 2, ["none"]),
        ("用例 2b 反向：K=2 优势超阈值", "U2b_K2", 2, ["none"]),
        ("用例 4  KK warn", "E3", 2, ["none"]),
        ("用例 4b 反向：KK warn + K=2 优势超阈值", "U4b_warn_K2", 2, ["none"]),
    ]
    rows = []
    for label, name, nplan, done in specs:
        p = next((CASES_DIR / name).rglob("run.log"))
        info = parse_log(p)
        # 采样不合格的候选被剔除；但保留其 elpd 供 Δ 判断
        ok = {}
        for K, c in info["cands"].items():
            bad = (c.get("grade", 0) < 2 or c.get("div", 0) != 0
                   or c.get("cap", 1) >= 0.01 or c.get("rhat", 9) > 1.01
                   or min(c.get("bulk", 0), c.get("tail", 0)) < 400)
            if not bad:
                ok[K] = c
        r = decide(info["kk_status"], ok, info["pair"], nplan,
                   info["removed"], done, [], [],
                   pair_dse=float(pw.loc[f"{name} / K2-K1", "dse"]))
        allc = info["cands"]
        elpd_txt = ", ".join(f"K={K}: {c.get('elpd', float('nan')):.1f}"
                             for K, c in sorted(allc.items()))
        dse = max((v["dse"] for v in info["pair"].values() if v["dse"] > 0),
                  default=float("nan"))
        rows.append(dict(case=label, run=name, kk=info["kk_status"],
                         planned_K=nplan, acceptable_K=",".join(map(str, sorted(ok))),
                         elpd_by_K=elpd_txt, dse=dse,
                         removed=" | ".join(info["removed"]) or "无",
                         **r))
    return rows


# --------------------------------------------------------------------------
# 第 4 步：历史回放离线回归
# --------------------------------------------------------------------------
def history_regression():
    rows = []
    for p in sorted((REPO / "out").rglob("real_*.json")):
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        st = js.get("k_selection_status")
        if st is None:
            continue
        kk = (js.get("kk") or {}).get("status", "unknown")
        grade = js.get("sample_grade")
        ksel = " ".join(js.get("k_selection") or [])
        delta = js.get("elpd_diff")
        dse = js.get("dse")
        thr = max(TH["elpd_abs"], TH["elpd_k"] * (dse or 0))
        # 简化重判：只用 JSON 里有的字段
        reason = ""
        if kk == "fail":
            new = "rejected_by_kk"; reason = "KK gate"
        elif kk not in ("ok", "unknown"):
            new = "exploratory_only"; reason = "KK gate"
        elif grade is not None and grade < 2:
            new = "exploratory_only"; reason = "sampling gate"
        elif "剔除" in ksel and "峰位出窗" in ksel:
            # 剔除理由里若含采样不合格，说明候选集合不完整 → K 本身不稳，
            # 不能因为 Δ 大就直接改选被剔除的 K。
            sampling_bad = any(t in ksel for t in
                               ["R̂=", "bulk=", "tail=", "div=", "分级"])
            if sampling_bad:
                new = "K_SELECTION_AMBIGUOUS"; reason = "K candidate incomplete"
            elif delta is not None and delta > thr:
                new = "WOULD_CHANGE_K"; reason = "peak-out policy change"
            else:
                new = "K_SELECTION_AMBIGUOUS"; reason = "K candidate incomplete"
        elif "剔除" in ksel and delta is not None and dse and delta <= thr:
            new = "K_SELECTION_AMBIGUOUS"; reason = "K candidate incomplete"
        else:
            new = st
        rows.append(dict(path=str(p.relative_to(REPO)), source=p.parent.name,
                         old_status=st, K=js.get("K"), kk=kk, grade=grade,
                         elpd_diff=delta, dse=dse, thr=thr,
                         ksel=ksel[:120], new_status=new,
                         change_reason=reason,
                         changed=(new != st)))
    return rows


def main():
    cases = build_cases()
    df = pd.DataFrame(cases)
    df.to_csv(ROOT / "v2_case_tests.csv", index=False, float_format="%.4g")

    hist = history_regression()
    hd = pd.DataFrame(hist)
    hd.to_csv(ROOT / "v2_history_regression.csv", index=False, float_format="%.4g")

    print("=== 用例 2 / 2b / 4 / 4b ===")
    print(df[["case", "kk", "acceptable_K", "elpd_by_K", "dse",
              "k_selection_status", "k_reference", "removed"]].to_string(index=False))
    print("\n=== 第 4 步：历史回放离线回归 ===")
    print("总条数:", len(hd))
    print(hd["new_status"].value_counts().to_string())
    ch = hd[hd.changed]
    print("\n会被新状态机改判的条数:", len(ch))
    if len(ch):
        print(ch.groupby(["source", "old_status", "new_status"]).size().to_string())
    print("\nDONE ->", ROOT)


if __name__ == "__main__":
    main()
