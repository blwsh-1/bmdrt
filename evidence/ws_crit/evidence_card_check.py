# -*- coding: utf-8 -*-
"""证据卡生成器的契约检查（只读、逐条 not、不排序/不打分）。

对应 `THREE_LAYER_SCOPE.md` §4。检查的不是「卡好不好看」，而是五条硬约束：

  1. **只读**：`build_document()` 前后，输入 JSON 的 sha256 指纹逐位不变；
  2. **每条证据一个 `not`**：同一张卡里 `not` 的去重数 = 证据条数（不是一条结论一句）；
  3. **`next_evidence_needed` 必填**：A/B/C 三档的卡都必须说得出还缺什么；
  4. **不打分**：全文档不出现 score / confidence / probability / weight 这类键；
  5. **不排序、不单选**：`multi_model` 默认关闭；打开后 arm 顺序 = 输入顺序，
     且不出现 best / rank / selected / recommend。

用例来源：`grade_check.py` 的 30 个选择器级合成例（覆盖全部分支）+ 3 个真实结果
JSON（Blank / AZ31-4Li_100 / WP2 的 k2_Ws 臂）。真实文件只读，不落任何新结果。

用法：
    D:/autoeis_env/Scripts/python.exe evidence/ws_crit/evidence_card_check.py
    ... -v        # 打印每张卡的 claim_id / grade / 证据条数
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bm_evidence as EV                                  # noqa: E402
import grade_check as GC                                  # noqa: E402

REAL = [
    "out/Blank_0.1KBr_pH10_EIS_PtCu_1.2V/real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V.json",
    "out/AZ31-4Li_100/real_AZ31-4Li_100.json",
    "out/_wp2_blank/k2_Ws/Blank_0.1KBr_pH10_EIS_PtCu_1.2V/real_Blank_0.1KBr_pH10_EIS_PtCu_1.2V.json",
]

FORBIDDEN = ("score", "confidence", "probability", "weight", "rank", "best",
             "selected", "recommend")


def _base_run(**kw):
    r = dict(file=kw.pop("file", "synthetic"), K=2, include_L=True,
             k_selection_status="converged", sample_grade=2, grade_fail=[],
             frac_neg=0.0, cover_re=0.9, cover_im=0.9, acf=0.1,
             component_review=dict(
                 component_identifiability={"component_1": "in_window",
                                            "component_2": "out_of_window"},
                 window_contribution=[
                     dict(component=2, state="out_of_window", p_peak_out=0.99,
                          f0_hz=0.071, w_ohm=10.0, frac_abs_at_fmin=0.086,
                          frac_im_at_fmin=-0.95, frac_abs_at_geom=0.009)],
                 parameter_identifiability=dict(level="degenerate", n_strong=1),
                 residual_review=dict(acf=0.38, trend_re_sigma=0.75,
                                      trend_im_sigma=1.09, systematic=True),
                 loo_robustness=dict(available=True, pareto_k_max=0.75)),
             param_corr=dict(strong=[dict(a="ln τ_2", b="ln w_2", rho=0.94)],
                             n_strong=1),
             degeneracy=[dict(level="no", at_bound=0.0, phi=0.51, a_impl=6.7)])
    r.update(kw)
    return r


def fixtures():
    """→ [(name, run_dict, 期望的卡 grade)]"""
    out = []
    cases = GC.build_cases()
    for name, c in cases.items():
        if name.startswith("mech_"):
            out.append((name, _base_run(file=name, auto_mechanism=c), c.get("grade")))
        elif name.startswith("ad_"):
            out.append((name, _base_run(file=name, ad_selection=c), c.get("grade")))
        elif name.startswith("dres_"):
            out.append((name, _base_run(file=name, diffusion_resolution=c),
                        c.get("grade")))
    return out


def check(verbose=False):
    ok, fails = 0, []

    def chk(cond, msg):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(msg)

    # ---------- 0. 台账：每条证据都必须有 not ----------
    for k, spec in EV.ITEM.items():
        chk(bool(spec.get("not_")), f"ITEM[{k}] 缺 not")
        chk(bool(spec.get("reads")), f"ITEM[{k}] 缺 reads")
    chk(len(EV.ITEM) >= 20, f"证据项只有 {len(EV.ITEM)} 条")

    # ---------- 1. 合成例：契约 ----------
    fx = fixtures()
    chk(len(fx) >= 25, f"合成例只有 {len(fx)} 个")
    for name, run, want in fx:
        fp = EV.fingerprint(run)
        doc = EV.build_document([run], [name])
        chk(EV.fingerprint(run) == fp, f"{name}: 生成过程改了输入（只读失败）")
        chk("multi_model" not in doc, f"{name}: 默认不该输出多模型表")
        cards = doc["runs"][0]["cards"]
        chk(len(cards) >= 2, f"{name}: 卡数异常（{len(cards)}）")
        for c in cards:
            chk(c["next_evidence_needed"], f"{name}/{c['claim_id']}: next 为空")
            chk(c["grade"] in (None, "A", "B", "C"),
                f"{name}/{c['claim_id']}: grade={c['grade']!r} 非法")
            chk(bool(c["not"]), f"{name}/{c['claim_id']}: 缺整卡 not")
            chk(bool(c["evidence"]), f"{name}/{c['claim_id']}: 没有证据条目")
            for e in c["evidence"]:
                chk(e["item"] in EV.ITEM, f"{name}: 未登记证据 {e['item']}")
                chk(bool(e.get("not")), f"{name}/{c['claim_id']}/{e['item']}: 缺 not")
                chk(e["state"] in ("read", "unusable", "missing"),
                    f"{name}/{e['item']}: state={e['state']!r}")
            # 每条证据的 not 必须各不相同（证明是逐条，不是一条结论一句）
            nots = [e.get("not") for e in c["evidence"]]
            chk(len(set(nots)) == len(nots),
                f"{name}/{c['claim_id']}: 有重复 not（说明仍是结论级）")
            chk(len(set(nots)) >= 3,
                f"{name}/{c['claim_id']}: 逐条 not 只有 {len(set(nots))} 种")
        if want is not None:
            got = {c["claim_id"]: c["grade"] for c in cards}
            if name.startswith("mech_"):
                chk(got.get("diffusion") == want,
                    f"{name}: 扩散卡 {got.get('diffusion')} != 选择器 {want}")
            elif name.startswith("ad_"):
                chk(got.get("adsorption") == want,
                    f"{name}: 吸附卡 {got.get('adsorption')} != 选择器 {want}")
        if verbose:
            print(f"  {name:34s} " + " | ".join(
                f"{c['claim_id']}:{c['grade']}:{len(c['evidence'])}" for c in cards))

    # ---------- 2. 真实结果：只读 + 可序列化 ----------
    n_real = 0
    for rel in REAL:
        p = ROOT / rel
        if not p.exists():
            print(f"  [跳过] 不存在 {rel}")
            continue
        run = EV.load_run(p)
        fp = EV.fingerprint(run)
        doc = EV.build_document([run], [p.stem])
        chk(EV.fingerprint(run) == fp, f"{rel}: 生成过程改了输入（只读失败）")
        txt = json.dumps(doc, ensure_ascii=False, allow_nan=False)
        chk("NaN" not in txt, f"{rel}: JSON 里有 NaN（应转 null）")
        n_real += 1
        if verbose:
            print(f"  {p.name:44s} cards={len(doc['runs'][0]['cards'])}")
    chk(n_real >= 2, "真实结果 JSON 少于 2 个（路径可能变了）")

    # ---------- 3. 不打分、不排序 ----------
    sample = EV.build_document([fx[0][1]], ["s"])
    blob = json.dumps(sample, ensure_ascii=False).lower()
    for bad in ("score", "confidence"):
        chk(bad not in blob, f"文档出现 {bad}")
    keys = set()

    def _walk(o):
        if isinstance(o, dict):
            keys.update(o.keys())
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(sample)
    for bad in FORBIDDEN:
        chk(bad not in keys, f"文档出现禁忌键 {bad}")

    # ---------- 4. 多模型：默认关、开了保序、不单选 ----------
    runs = [fx[0][1], fx[1][1], fx[4][1]]
    labels = ["a", "b", "c"]
    off = EV.build_document(runs, labels)
    chk("multi_model" not in off, "默认必须关掉多模型")
    on = EV.build_document(runs, labels, multi_model=True)
    mm = on["multi_model"]
    chk([r["label"] for r in mm["rows"]] == labels, "多模型表没按输入顺序")
    mkeys = set()
    for r in mm["rows"]:
        mkeys.update(r.keys())
    for bad in FORBIDDEN:
        chk(bad not in mkeys, f"多模型表出现禁忌键 {bad}")
    chk(mm.get("note") and "不排序" in mm["note"], "多模型表缺「不排序」说明")

    # ---------- 5. 开关默认值（config.json） ----------
    cfg = EV._load_cfg()
    chk(cfg.get("multi_model", False) is False,
        "config.evidence.multi_model 默认应为 false")

    # ---------- 汇总 ----------
    total = ok + len(fails)
    print(f"\n证据卡契约检查 {ok}/{total} 通过"
          f"（合成 {len(fx)} 例 + 真实 {n_real} 例）")
    for f in fails[:20]:
        print("  ✗ " + f)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    return check(a.verbose)


if __name__ == "__main__":
    sys.exit(main())
