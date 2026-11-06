"""状态机纯函数边界测试（第 4 轮审查要求）。

覆盖：
- decide_mechanism 的缺失/非有限 dse、controls、方向冲突、全不显著；
- decide 的多 K 配对映射（按 (best, lower_star) 取配对）、缺失配对证据、
  k_reference 的 role 字段；
- gate_of 的 fail-closed（缺诊断 / tree_cap 缺失不得静默通过）。

只读，不改算法代码。运行：
  D:/autoeis_env/Scripts/python.exe test_state_machine.py
输出：state_machine_boundary_tests.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import replay_identifiability as RI  # noqa: E402
import v2_state_machine as SM  # noqa: E402

NAN = float("nan")


def cand(elpd, grade=2, div=0, cap=0.0, rhat=1.001, bulk=1000, tail=1200):
    return dict(elpd=elpd, grade=grade, div=div, cap=cap,
                rhat=rhat, bulk=bulk, tail=tail)


def full_js():
    return dict(kk={"status": "ok"}, sample_grade=2, divergences=0,
                tree_cap=0.0)


def full_row():
    return dict(rhat_max=1.001, ess_bulk=1000.0, ess_tail=1200.0)


def main():
    rows = []

    def check(name, expected, actual, detail=""):
        rows.append(dict(test=name, expected=str(expected), actual=str(actual),
                         passed=(expected == actual), detail=detail))

    # ---- decide_mechanism ----
    cases = [
        ("空比较 -> not_tested", [], "not_tested"),
        ("缺失 dse(NaN) -> comparison_incomplete",
         [(0.1, NAN, "A>B")], "comparison_incomplete"),
        ("dse=0 -> comparison_incomplete",
         [(0.1, 0.0, "A>B")], "comparison_incomplete"),
        ("delta=NaN -> comparison_incomplete",
         [(NAN, 0.6, "A>B")], "comparison_incomplete"),
        ("二元组缺失 dse 也拦截",
         [(0.1, NAN)], "comparison_incomplete"),
        ("全不显著 -> ambiguous",
         [(0.2, 0.6, "A>B"), (0.3, 1.0, "A>B")], "ambiguous"),
        ("显著同向 -> evidence_for_model_preference",
         [(2.0, 0.6, "A>B"), (3.0, 1.0, "A>B")], "evidence_for_model_preference"),
        ("显著反向 -> preference_conflict",
         [(2.0, 0.6, "A>B"), (-3.0, 1.0, "B>A")], "preference_conflict"),
    ]
    for name, diffs, exp in cases:
        check("mechanism: " + name, exp, SM.decide_mechanism(diffs))
    check("mechanism: controls_missing -> not_tested", "not_tested",
          SM.decide_mechanism([(2.0, 0.6, "A>B")], controls_missing=True))
    check("mechanism: controls_complete=False -> not_tested", "not_tested",
          SM.decide_mechanism([(2.0, 0.6, "A>B")], controls_complete=False))
    check("mechanism: 缺失优先于显著判定", "comparison_incomplete",
          SM.decide_mechanism([(2.0, 0.6, "A>B"), (0.1, NAN, "A>B")],
                              controls_complete=False))

    # ---- decide: 多 K 配对映射 ----
    pc = {(3, 2): dict(delta=30.0, dse=4.0, thr=10.0,
                       n_observations=50, comparison_type="K_selection"),
          (3, 1): dict(delta=300.0, dse=40.0, thr=80.0,
                       n_observations=50, comparison_type="K_selection")}
    cands3 = {1: cand(-100.0), 2: cand(10.0), 3: cand(40.0)}
    r = SM.decide("ok", cands3, {}, 3, [], ["none"], [],
                  [(0.2, 0.6, "A>B")], pairwise_comparisons=pc)
    check("decide: 多 K 用 (3,2) 配对而非 (3,1)", "converged",
          r["k_selection_status"], r["reasons"])
    check("decide: k_reference 取 LOO 最优 K=3", 3,
          json.loads(r["k_reference"])["K"])
    check("decide: k_reference 含 role", "prediction_reference_only",
          json.loads(r["k_reference"]).get("role"))
    r2 = SM.decide("ok", cands3, {}, 3, [], ["none"], [],
                   [(0.2, 0.6, "A>B")], pairwise_comparisons={})
    check("decide: 缺配对证据 -> 不 converged", "K_SELECTION_AMBIGUOUS",
          r2["k_selection_status"], r2["reasons"])
    r3 = SM.decide("ok", cands3, {}, 3, [], ["none"], [],
                   [(0.1, NAN, "A>B")], pairwise_comparisons=pc)
    check("decide: 机制缺失 dse -> comparison_incomplete",
          "comparison_incomplete", r3["mechanism_identifiability"])

    # ---- gate_of: fail-closed ----
    checks, eff, missing = RI.gate_of(full_js(), full_row())
    check("gate: 完整诊断 -> effective", True, eff, str(missing))
    checks, eff, missing = RI.gate_of(full_js(), None)
    check("gate: 缺 R-hat/ESS 日志 -> 无效", False, eff, str(missing))
    check("gate: 缺诊断被记录", True,
          any("rhat" in m or "ess" in m for m in missing), str(missing))
    js = full_js(); js.pop("tree_cap")
    checks, eff, missing = RI.gate_of(js, full_row())
    check("gate: tree_cap 缺失 -> 无效", False, eff, str(missing))
    check("gate: tree_cap 缺失被记录", True,
          "tree_cap" in missing, str(missing))
    js = full_js(); js.pop("sample_grade")
    checks, eff, missing = RI.gate_of(js, full_row())
    check("gate: grade 缺失 -> 无效", False, eff, str(missing))
    js = full_js(); js["kk"] = {"status": "warn"}
    checks, eff, missing = RI.gate_of(js, full_row())
    check("gate: KK warn -> 无效", False, eff, str(checks["kk_ok"]))

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "state_machine_boundary_tests.csv", index=False)
    print(df[["test", "expected", "actual", "passed"]].to_string(index=False))
    print("\nPASS %d / %d" % (int(df.passed.sum()), len(df)))
    print("DONE ->", ROOT / "state_machine_boundary_tests.csv")
    if not df.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
