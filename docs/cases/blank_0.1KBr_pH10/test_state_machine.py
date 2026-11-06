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
                       n_observations=50, comparison_type="K_selection",
                       K_a=3, K_b=2, model_a="X_K3", model_b="X_K2",
                       direction="A>B"),
          (3, 1): dict(delta=300.0, dse=40.0, thr=80.0,
                       n_observations=50, comparison_type="K_selection",
                       K_a=3, K_b=1, model_a="X_K3", model_b="X_K1",
                       direction="A>B")}
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

    # ---- v5.1 反例：配对证据整体校验 / 反向配对 / 机制方向 ----
    def prec(**kw):
        base = dict(delta=10.0, dse=1.0, thr=5.0,
                    comparison_type="K_selection", n_observations=50,
                    K_a=2, K_b=1, model_a="X_K2", model_b="X_K1",
                    direction="A>B")
        base.update(kw)
        return base

    def kstatus(pc):
        return SM.decide("ok", {1: cand(0.0), 2: cand(10.0)}, {}, 2,
                         [], ["none"], [], [],
                         pairwise_comparisons=pc)["k_selection_status"]

    check("v5.1: 合法配对 -> converged", "converged",
          kstatus({(2, 1): prec()}))
    check("v5.1: thr=NaN -> 不得 converged", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(thr=NAN)}))
    _r = prec(); _r.pop("thr")
    check("v5.1: 缺 thr -> 不抛 KeyError 且拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): _r}))
    check("v5.1: comparison_type 错误 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(comparison_type="mechanism")}))
    check("v5.1: dse=0 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(dse=0.0)}))
    check("v5.1: thr<0 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(thr=-1.0)}))
    check("v5.1: n_observations=0 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(n_observations=0)}))
    check("v5.1: n_observations 非整数 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(n_observations="bad")}))
    check("v5.1: K_a/K_b 方向不符 -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(K_a=1, K_b=2)}))
    check("v5.1: delta=NaN -> 拒绝", "K_SELECTION_AMBIGUOUS",
          kstatus({(2, 1): prec(delta=NAN)}))
    check("v5.1: 缺配对记录 -> 拒绝", "K_SELECTION_AMBIGUOUS", kstatus({}))
    # 反向配对 (1,2) Δ=-10，规范化后应为 (2,1) Δ=+10
    rev = prec(delta=-10.0, K_a=1, K_b=2, model_a="X_K1", model_b="X_K2",
               direction="B>A")
    norm, flag = SM.normalize_pair({(1, 2): rev}, (2, 1))
    check("v5.1: 反向配对 Δ 反号", 10.0, float(norm["delta"]), str(flag))
    check("v5.1: 反向配对方向反转", "A>B", norm["direction"])
    check("v5.1: 反向配对 K_a/K_b 交换", (2, 1),
          (int(norm["K_a"]), int(norm["K_b"])))
    check("v5.1: 反向配对可用且判 converged", "converged",
          kstatus({(1, 2): rev}))
    ok, why = SM.validate_pair_record(prec(thr=NAN), (2, 1))
    check("v5.1: validate 直接拒绝 thr=NaN", False, ok, why)
    ok, why = SM.validate_pair_record(prec(comparison_type="mechanism"), (2, 1))
    check("v5.1: validate 直接拒绝错误类型", False, ok, why)

    # 机制二元组方向：按 Δ 符号推导，不让 unknown 参与显著偏好
    check("v5.1: 二元组同向 -> 有方向偏好", "evidence_for_model_preference",
          SM.decide_mechanism([(2, 1), (3, 1)]))
    check("v5.1: 二元组反向 -> preference_conflict", "preference_conflict",
          SM.decide_mechanism([(2, 1), (-3, 1)]))
    check("v5.1: 二元组不显著 -> ambiguous", "ambiguous",
          SM.decide_mechanism([(0.2, 1), (0.3, 1)]))
    check("v5.1: 显式 unknown 方向按 Δ 推导", "preference_conflict",
          SM.decide_mechanism([(2, 1, "unknown"), (-3, 1, "unknown")]))

    # ---- 用例 2 正向（状态机层）：两候选都合格、KK ok、Δ 不显著 ----
    pos = prec(delta=0.4, dse=0.5, thr=2.0)   # Δ=0.4 < thr=max(2,2*dse)
    r_pos = SM.decide("ok", {1: cand(0.0), 2: cand(0.4)}, {}, 2,
                      [], ["none"], [], [], pairwise_comparisons={(2, 1): pos})
    check("用例2正向: 两候选合格 + Δ<2dse -> K_SELECTION_AMBIGUOUS",
          "K_SELECTION_AMBIGUOUS", r_pos["k_selection_status"], r_pos["reasons"])
    check("用例2正向: 未走候选不完整分支", True,
          "候选集合不完整" not in r_pos["reasons"], r_pos["reasons"])
    check("用例2正向: k_reference 仍指向 LOO 最优", 2,
          json.loads(r_pos["k_reference"])["K"])
    # 反向对照：Δ 显著 -> converged
    big = prec(delta=30.0, dse=4.0, thr=10.0)
    check("用例2反向: Δ 显著 -> converged", "converged",
          kstatus({(2, 1): big}))

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "state_machine_boundary_tests.csv", index=False)
    print(df[["test", "expected", "actual", "passed"]].to_string(index=False))
    print("\nPASS %d / %d" % (int(df.passed.sum()), len(df)))
    print("DONE ->", ROOT / "state_machine_boundary_tests.csv")
    if not df.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
