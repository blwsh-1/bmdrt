"""A 臂（基线）vs B 臂（voigt+adsorption）逐项对照，按 §9.4.11 预注册判读。

预注册里写死了 4 条预测（P1–P4）和 3 条否决条件（K1–K3）。本脚本**只做机械核对**，
不重新解释——任何"看起来更好"的结论都必须先过 K1/K2。

用法：
    python evidence/nested_topology/m4_arm_compare.py \
        out/m4_select_selftest_B2.json out/m4_armB_adRL.json
"""

from __future__ import annotations

import json
import sys

#: A 臂的实测降级 seed 数（8/20），预注册 P4 的比较基准。
BASE_DEGRADED = 8
BASE_TOTAL = 20


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fp:
        d = json.load(fp)
    d["_rows"] = {r["case"]: r for r in d.get("rows", [])}
    return d


def fmt(v, w: int = 9, p: int = 1) -> str:
    return "-".rjust(w) if v is None else f"{v:>{w}.{p}f}"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    a, b = load(argv[1]), load(argv[2])
    caps = b.get("voigt_caps")
    print(f"A 臂：{argv[1]}   voigt_caps={a.get('voigt_caps', '未记录（基线=两项全关）')}")
    print(f"B 臂：{argv[2]}   voigt_caps={caps}")
    if caps and caps.get("adsorption") is None:
        print("⚠ B 臂的 adsorption 是 None——这一臂没开能力，下面的对照无意义。")

    print(f"\n{'case':<22}{'A decision':<18}{'B decision':<18}"
          f"{'A Δ':>10}{'B Δ':>10}{'A z':>7}{'B z':>7}{'A wrong':>8}{'B wrong':>8}")
    rows = []
    for case, ra in a["_rows"].items():
        rb = b["_rows"].get(case)
        if rb is None:
            print(f"{case:<22}  B 臂缺失")
            continue
        rows.append((case, ra, rb))
        print(f"{case:<22}{ra['decision']:<18}{rb['decision']:<18}"
              f"{fmt(ra['d_elpd'])}{fmt(rb['d_elpd'])}"
              f"{fmt(ra['z'], 7, 1)}{fmt(rb['z'], 7, 1)}"
              f"{ra['wrong']:>8}{rb['wrong']:>8}")

    deg_a = sum(len(r["degraded_seeds"]) for _, r, _ in rows)
    deg_b = sum(len(r["degraded_seeds"]) for _, _, r in rows)
    wrong_a = sum(r["wrong"] for _, r, _ in rows)
    wrong_b = sum(r["wrong"] for _, _, r in rows)
    inad_b = sum(1 for _, _, r in rows if r["decision"] == "voigt_inadmissible")
    hits_a = sum(r["hits"] for _, r, _ in rows)
    hits_b = sum(r["hits"] for _, _, r in rows)

    print(f"\n汇总：A 错切 {wrong_a} / 命中 {hits_a} / 降级 seed {deg_a}"
          f"    B 错切 {wrong_b} / 命中 {hits_b} / 降级 seed {deg_b}"
          f" / voigt_inadmissible {inad_b}")

    # ------------------------------------------------------------------
    # 逐条核对预注册
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("§9.4.11 预注册核对")
    print("=" * 74)

    fails = []

    # K1 / K2 先查——否决条件一旦触发，后面"改善"都不算数
    print(f"\n[K1] 零错切：A={wrong_a}，B={wrong_b}  → "
          f"{'通过' if wrong_b == 0 else '**否决**'}")
    if wrong_b:
        fails.append("K1")

    vt = b["_rows"].get("voigt_true")
    k2 = bool(vt and vt["decision"] == "nested")
    print(f"[K2] voigt_true 不得判 nested：B={vt['decision'] if vt else '缺失'}  → "
          f"{'**否决**' if k2 else '通过'}")
    if k2:
        fails.append("K2")

    # P1：两条 nested 真值的 |Δ| 应下降
    print("\n[P1] nested 真值的 Δ 应下降（voigt 不再系统性吃亏）")
    p1 = True
    for case in ("nested_true", "arcs_merged"):
        ra, rb = a["_rows"].get(case), b["_rows"].get(case)
        if not ra or not rb or ra["d_elpd"] is None or rb["d_elpd"] is None:
            print(f"  {case:<20} A={ra['d_elpd'] if ra else '-'}  "
                  f"B={rb['d_elpd'] if rb else '-'}  → 无法比较（有 None）")
            p1 = False
            continue
        drop = ra["d_elpd"] - rb["d_elpd"]
        ok = drop > 0
        p1 &= ok
        print(f"  {case:<20} A={ra['d_elpd']:+9.1f}  B={rb['d_elpd']:+9.1f}  "
              f"变化 {drop:+9.1f}  → {'下降 ✓' if ok else '未下降 ✗'}")
    if not p1:
        fails.append("P1")

    # P2：voigt_true 不得出现 voigt_inadmissible
    p2 = bool(vt and vt["decision"] != "voigt_inadmissible")
    print(f"\n[P2] voigt_true 不出现 voigt_inadmissible："
          f"{'通过' if p2 else '**不符**'}")
    if not p2:
        fails.append("P2")

    # P3：loop_out_of_window 变动属预期内，只报不判
    lo = b["_rows"].get("loop_out_of_window")
    la = a["_rows"].get("loop_out_of_window")
    print(f"\n[P3] loop_out_of_window（变动属预期内，不算错）："
          f"A={la['decision'] if la else '-'} → B={lo['decision'] if lo else '-'}")

    # P4：降级数 ≤ 8/20（写成双向，升了也要报）
    print(f"\n[P4] 降级 seed 数：A={deg_a}/{BASE_TOTAL}  B={deg_b}/{BASE_TOTAL}  → "
          f"{'持平或下降' if deg_b <= deg_a else '**上升**（反方向风险，须如实记）'}")

    # K3：|Δ| 不降反升且 z 增大
    k3 = False
    for case in ("nested_true", "arcs_merged"):
        ra, rb = a["_rows"].get(case), b["_rows"].get(case)
        if not (ra and rb) or ra["d_elpd"] is None or rb["d_elpd"] is None:
            continue
        if abs(rb["d_elpd"]) > abs(ra["d_elpd"]) and (rb["z"] or 0) > (ra["z"] or 0):
            k3 = True
    print(f"\n[K3] |Δ| 不降反升且 z 增大：{'**触发**' if k3 else '未触发'}"
          f"  → {'记「缺项不是原因」' if k3 else '见 P1 判读'}")
    if k3:
        fails.append("K3")

    print("\n" + "-" * 74)
    if fails:
        print(f"未通过项：{fails}")
        print("按 §9.4.11.5：K1/K2 触发即**否决该臂**，不得以总体改善为由保留；")
        print("K3 触发则记「缺项不是原因」，且**不得**改用 C 臂反复试（多重比较）。")
    else:
        print("预注册各项均通过。")
    print("提醒：这只是**合成谱上的判定分布**。真实谱分两级：「哪族能表示」记 B-缺项，"
          "「机制是什么」仍 C 档（§9.4.12.7）；且排除 voigt ≠ 证明 nested。")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
