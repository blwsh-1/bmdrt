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
def decide_mechanism(mech_diffs, controls_missing=False, controls_complete=True):
    """机制可辨识性纯函数（v5.1 修订）。

    mech_diffs 元素为 (delta, dse, direction)；二元组 (delta, dse) 仍兼容，
    但按 delta 符号自动推导方向（A>B / B>A），不再让 "unknown" 参与显著偏好判断。
    严格顺序：空比较 -> 完整性 -> controls -> 显著性与方向。
    """
    if not mech_diffs:
        return "not_tested"
    norm = []
    for item in mech_diffs:
        if len(item) == 2:
            delta, dse, direction = item[0], item[1], None
        else:
            delta, dse, direction = item[0], item[1], item[2]
        if not (np.isfinite(delta) and np.isfinite(dse)) or dse <= 0:
            return "comparison_incomplete"
        if direction not in ("A>B", "B>A"):
            direction = "A>B" if delta > 0 else ("B>A" if delta < 0 else "unknown")
        norm.append((delta, dse, direction))
    if controls_missing or not controls_complete:
        return "not_tested"
    sig = [direction for delta, dse, direction in norm if abs(delta) >= dse]
    if any(d == "unknown" for d in sig):
        return "comparison_incomplete"
    if not sig:
        return "ambiguous"
    if len(set(sig)) > 1:
        return "preference_conflict"
    return "evidence_for_model_preference"


def normalize_pair(pairwise_comparisons, key):
    """按 (best, lower_star) 取配对记录；若只有反向记录，同步反转 Δ / direction / 端点。

    返回 (rec_or_none, reversed_flag)。dse / thr / n_observations 对方向对称，保持不变。
    """
    pc = pairwise_comparisons or {}
    rec = pc.get(key)
    if rec is not None:
        return dict(rec), False
    rev_key = (key[1], key[0])
    rec = pc.get(rev_key)
    if rec is None:
        return None, False
    out = dict(rec)
    try:
        out["delta"] = -float(out["delta"])
    except (KeyError, TypeError, ValueError):
        pass
    d = out.get("direction")
    if d == "A>B":
        out["direction"] = "B>A"
    elif d == "B>A":
        out["direction"] = "A>B"
    for a, b in (("K_a", "K_b"), ("model_a", "model_b")):
        if a in out and b in out:
            out[a], out[b] = out[b], out[a]
    return out, True


def validate_pair_record(rec, key):
    """配对证据整体校验（v5.1 P0）：任何一项不满足都不能输出严格 converged。"""
    if rec is None:
        return False, "缺少配对记录"
    required = ("delta", "dse", "thr", "comparison_type",
                "n_observations", "K_a", "K_b")
    for k in required:
        if k not in rec:
            return False, f"配对记录缺字段 {k}"
    try:
        delta, dse, thr = (float(rec["delta"]), float(rec["dse"]),
                           float(rec["thr"]))
    except (TypeError, ValueError):
        return False, "配对记录数值不可解析"
    if not all(np.isfinite(x) for x in (delta, dse, thr)):
        return False, "delta/dse/thr 含非有限值（含 thr=NaN）"
    if dse <= 0:
        return False, "dse <= 0"
    if thr < 0:
        return False, "thr < 0"
    if str(rec["comparison_type"]) != "K_selection":
        return False, "comparison_type 不是 K_selection"
    try:
        n = int(rec["n_observations"])
    except (TypeError, ValueError):
        return False, "n_observations 无效"
    if n <= 0:
        return False, "n_observations <= 0"
    try:
        if int(rec["K_a"]) != key[0] or int(rec["K_b"]) != key[1]:
            return False, "配对方向与 model_a/model_b(K_a/K_b) 不一致"
    except (TypeError, ValueError):
        return False, "K_a/K_b 不可解析"
    return True, "ok"


def decide(kk_status, cands, pair, n_candidates_planned, removed,
           controls_done, controls_missing, mech_diffs, family="none",
           pairwise_comparisons=None, controls_complete=True):
    """四层判定。cands 已过滤为「采样合格」的候选。

    第 4 轮审查修订：
    - 机制判定交给纯函数 decide_mechanism()（缺失 dse -> comparison_incomplete，
      显著方向相反 -> preference_conflict）。
    - 配对证据改为 pairwise_comparisons[(best, lower_star)] 映射，直接读取该配对的
      delta / dse / thr / n_observations / comparison_type，不再用单一标量 pair_dse，
      也不再用两个汇总 ELPD 相减代替配对 Δ。
    - k_reference 增加 role 字段，与 replay_identifiability.py 语义一致。
    """
    reasons = []
    pairwise_comparisons = pairwise_comparisons or {}

    def make_ref(cc):
        if not cc or any(not np.isfinite(c.get("elpd", np.nan))
                         for c in cc.values()):
            return None
        K = max(cc, key=lambda k: cc[k]["elpd"])
        return dict(family=family, K=int(K), elpd=float(cc[K]["elpd"]),
                    role="prediction_reference_only")

    k_ref = make_ref(cands)

    # 严格门控：检查所有参与比较的候选，不只最终胜出的 K
    gated = (kk_status != "ok") or any(
        c.get("grade", 0) < TH["grade_min"] or c.get("div", 0) != 0
        or c.get("cap", 1) >= 0.01 or c.get("rhat", 9) > 1.01
        or min(c.get("bulk", 0), c.get("tail", 0)) < 400
        for c in cands.values())

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
        key = (best, lower_star)
        rec, reversed_pair = normalize_pair(pairwise_comparisons, key)
        valid, why = validate_pair_record(rec, key)
        incomplete = len(cands) < n_candidates_planned
        if incomplete:
            k_status = "K_SELECTION_AMBIGUOUS"
            reasons.append("候选集合不完整（不能当作完整比较的结论）")
        elif not valid:
            k_status = "K_SELECTION_AMBIGUOUS"
            reasons.append(f"配对证据无效：{why}，禁止严格 converged")
        else:
            delta, dse, thr = (float(rec["delta"]), float(rec["dse"]),
                               float(rec["thr"]))
            if delta <= thr:
                k_status = "K_SELECTION_AMBIGUOUS"
                reasons.append(f"配对 Δ={delta:.1f} ≤ thr={thr:.1f}")
            else:
                k_status = "converged"
                reasons.append(f"配对 Δ={delta:.1f} > thr={thr:.1f} "
                               f"(n_obs={rec['n_observations']}, "
                               f"{rec['comparison_type']}"
                               + ("，反向配对已规范化" if reversed_pair else "")
                               + ")")

    # 第二层：机制（纯函数）
    mech = decide_mechanism(mech_diffs,
                            controls_missing=bool(controls_missing),
                            controls_complete=controls_complete)

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
        prow = pw.loc[f"{name} / K2-K1"]
        prec = dict(delta=float(prow.delta), dse=float(prow.dse),
                    thr=float(prow.thr),
                    n_observations=int(prow.n_observations),
                    comparison_type=str(prow.comparison_type),
                    source=str(prow.source), K_a=2, K_b=1,
                    model_a=str(prow.model_a), model_b=str(prow.model_b),
                    direction=str(prow.direction))
        r = decide(info["kk_status"], ok, info["pair"], nplan,
                   info["removed"], done, [], [],
                   pairwise_comparisons={(2, 1): prec},
                   controls_complete=True)
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
