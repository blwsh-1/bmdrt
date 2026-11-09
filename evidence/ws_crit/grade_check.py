# -*- coding: utf-8 -*-
"""证据资格字段拆分的前后对拍（只加键、不改值）。

对应 `THREE_LAYER_SCOPE.md` §3.2 的三处混用：
  ① `select_mechanism` / `select_adsorption` 的 8 个 guard 汇入单一
     `guard_degraded` → 补 `guard_classes` / `grade` / `grade_reason` / `not`
  ② `diffusion_resolution` 的 `unknown` 与 `indistinguishable` 都读作「不可分」
     → 补 `grade`（A / B）
  ③ `component_identifiability = uncertain` 未分 A/B → 补 `component_grade`（C 档）

用法（**先快照后改代码**，这个顺序不能反）：

    D:/autoeis_env/Scripts/python.exe evidence/ws_crit/grade_check.py --snapshot /tmp/g.json
    # ... 改 bm_real.py / bm_diff.py ...
    D:/autoeis_env/Scripts/python.exe evidence/ws_crit/grade_check.py --verify /tmp/g.json

`--verify` 的判据只有一条：**旧字段一个都不能变**（含 nan 相等），
新增键单独列出来给人看。任一旧字段变化即返回码 1。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bm_diff                                   # noqa: E402
import bm_real                                   # noqa: E402


def _pa(delta_per_pt, n=24, spread=0.0, seed=0):
    """构造一对逐点 LOO：pb − pa 的均值 = delta_per_pt。"""
    rng = np.random.default_rng(seed)
    pa = rng.normal(0.0, 1.0, n)
    pb = pa + delta_per_pt + (spread * rng.normal(0.0, 1.0, n) if spread else 0.0)
    return pa.tolist(), pb.tolist()


def _ws_res(**kw):
    d = dict(grade=1, bad_k=0.0, elpd=-10.0, p_diff_small=0.05,
             diffusion_resolution={"resolvable": None})
    d.update(kw)
    return d


def _share(cv=0.05, aw=2.0, sd=0.1):
    return dict(A_W=aw, A_W_sd=sd, cv_aw=cv, identifiable=cv <= 0.15,
                snr_med=0.40 / max(cv, 1e-9), snr_lo=1.0, share_med=0.2,
                share_lo=0.1, share_hi=0.3, f_min=0.1, sig_im=0.1)


def _gate(status="candidate"):
    return dict(status=status, is_candidate=(status == "candidate"))


def build_cases():
    """每条 case 返回一个 dict（调用被测函数的结果）。全部 JSON 可序列化。"""
    C = {}

    # ---------------- select_mechanism ----------------
    pa, pb = _pa(0.0)

    C["mech_fit_missing"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa), None, _gate())

    C["mech_guard_ws_grade0"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(grade=0, pointwise_elpd=pb), _gate())

    C["mech_guard_ws_bad_k"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(bad_k=0.3, pointwise_elpd=pb), _gate())

    C["mech_guard_ws_loo_unavailable"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(elpd=None, pointwise_elpd=pb), _gate())

    C["mech_guard_diff_amp"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(p_diff_small=0.9, pointwise_elpd=pb), _gate())

    C["mech_guard_hn_degenerate"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(pointwise_elpd=pb), _gate("degenerate"))

    C["mech_guard_ws_amp"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(pointwise_elpd=pb), _gate(), share=_share(cv=0.40))

    C["mech_guard_mixed_AB"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa),
        _ws_res(grade=0, p_diff_small=0.9, pointwise_elpd=pb), _gate())

    C["mech_comparison_incomplete"] = lambda: bm_real.select_mechanism(
        dict(), _ws_res(pointwise_elpd=pb), _gate())

    pa2, pb2 = _pa(0.0)
    C["mech_uncertain"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa2), _ws_res(pointwise_elpd=pb2), _gate())

    pa3, pb3 = _pa(-0.2, spread=0.1, seed=3)
    C["mech_ws_worse"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa3), _ws_res(pointwise_elpd=pb3), _gate())

    pa4, pb4 = _pa(0.6, spread=0.05, seed=4)
    C["mech_switched"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa4), _ws_res(pointwise_elpd=pb4), _gate(),
        share=_share(cv=0.05))

    C["mech_share_nan"] = lambda: bm_real.select_mechanism(
        dict(pointwise_elpd=pa4), _ws_res(pointwise_elpd=pb4), _gate(),
        share=dict(A_W=1.0, A_W_sd=float("nan"), cv_aw=float("nan"),
                   identifiable=True, share_med=float("nan"),
                   share_lo=float("nan"), share_hi=float("nan"), f_min=0.1))

    C["mech_skipped_inductive"] = lambda: bm_real.GRADE.annotate(dict(
        choice="none", status="skipped_inductive", reason="感抗尾",
        gate=_gate("inductive_tail"), residual=None, delta=None, dse=None,
        threshold=None, guards={}, note=[]))

    # ---------------- select_adsorption ----------------
    C["ad_gate_not_strong"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa), _ws_res(pointwise_elpd=pb),
        dict(status="monotonic_rise", strong=False, enter=False))

    C["ad_gate_missing"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa), _ws_res(pointwise_elpd=pb), None)

    C["ad_fit_missing"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa), None,
        dict(status="strong", strong=True, enter=True))

    C["ad_guard_grade0"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa),
        dict(grade=0, bad_k=0.0, elpd=-5.0, p_ad_small=0.1,
             f_c_out_prob=0.0, pointwise_elpd=pb),
        dict(status="strong", strong=True, enter=True))

    C["ad_guard_amp"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa),
        dict(grade=1, bad_k=0.0, elpd=-5.0, p_ad_small=0.8,
             f_c_out_prob=0.1, pointwise_elpd=pb),
        dict(status="strong", strong=True, enter=True))

    C["ad_guard_corner"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa),
        dict(grade=1, bad_k=0.0, elpd=-5.0, p_ad_small=0.1,
             f_c_out_prob=0.9, pointwise_elpd=pb),
        dict(status="strong", strong=True, enter=True))

    C["ad_guard_mixed"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa),
        dict(grade=0, bad_k=0.0, elpd=-5.0, p_ad_small=0.8,
             f_c_out_prob=0.1, pointwise_elpd=pb),
        dict(status="strong", strong=True, enter=True))

    C["ad_switched"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa4),
        dict(grade=1, bad_k=0.0, elpd=-5.0, p_ad_small=0.1,
             f_c_out_prob=0.1, pointwise_elpd=pb4),
        dict(status="strong", strong=True, enter=True))

    C["ad_uncertain"] = lambda: bm_real.select_adsorption(
        dict(pointwise_elpd=pa2),
        dict(grade=1, bad_k=0.0, elpd=-5.0, p_ad_small=0.1,
             f_c_out_prob=0.1, pointwise_elpd=pb2),
        dict(status="strong", strong=True, enter=True))

    C["ad_ws_taken"] = lambda: bm_real.GRADE.annotate(dict(
        choice="none", status="ws_taken", reason="机制已判 Ws",
        gate=None, delta=None, dse=None, threshold=None, guards={}, note=[]))

    # ---------------- diffusion_resolution ----------------
    f_win = np.logspace(-1, 4, 50)

    C["dres_no_diffusion"] = lambda: bm_diff.diffusion_resolution(
        f_win, {}, diffusion=None)
    C["dres_unknown"] = lambda: bm_diff.diffusion_resolution(
        f_win, {}, diffusion="Ws")
    C["dres_indistinguishable"] = lambda: bm_diff.diffusion_resolution(
        f_win, {"tauD": np.full(200, 1e5)}, diffusion="Ws")
    C["dres_resolvable"] = lambda: bm_diff.diffusion_resolution(
        f_win, {"tauD": np.full(200, 0.5)}, diffusion="Ws")
    C["dres_edge"] = lambda: bm_diff.diffusion_resolution(
        f_win, {"tauD": np.full(200, 3.0)}, diffusion="Ws")
    C["dres_empty_f"] = lambda: bm_diff.diffusion_resolution(
        np.array([]), {}, diffusion="Ws")

    return {k: v() for k, v in C.items()}


def _walk(old, new, path="", acc=None):
    if acc is None:
        acc = {"added": [], "changed": [], "missing": []}
    if isinstance(old, dict):
        if not isinstance(new, dict):
            acc["changed"].append((path, old, new))
            return acc
        for k, v in old.items():
            if k not in new:
                acc["missing"].append(f"{path}/{k}")
            else:
                _walk(v, new[k], f"{path}/{k}", acc)
        for k in new:
            if k not in old:
                acc["added"].append(f"{path}/{k}")
        return acc
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            acc["changed"].append((path, old, new))
            return acc
        for i, (a, b) in enumerate(zip(old, new)):
            _walk(a, b, f"{path}[{i}]", acc)
        return acc
    if isinstance(old, float) and isinstance(new, float):
        if math.isnan(old) and math.isnan(new):
            return acc
        if old == new:
            return acc
        acc["changed"].append((path, old, new))
        return acc
    if old != new:
        acc["changed"].append((path, old, new))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", metavar="PATH")
    ap.add_argument("--verify", metavar="PATH")
    ap.add_argument("--table", action="store_true",
                    help="打印每例的 status/level → grade 对照表（给审阅人看）")
    a = ap.parse_args()
    res = json.loads(json.dumps(build_cases(), default=str))  # 归一化，便于比

    if a.table:
        print(f"{'case':32s} {'status / level':30s} {'档':4s} "
              f"{'guard_classes':34s} 判据")
        for k in sorted(res):
            d = res[k]
            g = d.get("grade")
            gc = d.get("guard_classes") or {}
            st = d.get("status") or ("level=" + str(d.get("level")))
            src = (f"guards: {'、'.join(gc)}" if gc
                   else (d.get("grade_reason") or "").split(" 判 ")[0][:34].rstrip())
            print(f"{k:32s} {st:30s} {str(g):4s} "
                  f"{(','.join(f'{x}→{y}' for x, y in gc.items()) or '-'):34s} {src}")
        return 0

    if a.snapshot:
        Path(a.snapshot).write_text(json.dumps(res, ensure_ascii=False),
                                    encoding="utf-8")
        print(f"快照已存 {a.snapshot}（{len(res)} 例）")
        for k in sorted(res):
            d = res[k]
            print(f"  {k:32s} status={d.get('status') or '-':22s} "
                  f"level={d.get('level') or '-'}")
        return 0

    if a.verify:
        old = json.loads(Path(a.verify).read_text(encoding="utf-8"))
        acc = _walk(old, res)
        print(f"比对 {len(old)} 例：缺失 {len(acc['missing'])} / "
              f"值变 {len(acc['changed'])} / 新增键 {len(acc['added'])}")
        for p in acc["missing"][:20]:
            print("  **缺失** " + p)
        for p, o, n in acc["changed"][:20]:
            print(f"  **值变** {p}: {o!r} -> {n!r}")
        keys = {}
        for p in acc["added"]:
            leaf = p.rsplit("/", 1)[-1]
            keys[leaf] = keys.get(leaf, 0) + 1
        if keys:
            print("  新增键（按名字）：" + "、".join(
                f"{k}×{v}" for k, v in sorted(keys.items(), key=lambda x: -x[1])))
        ok = not acc["missing"] and not acc["changed"]
        print("\n结论：" + ("旧字段全部一致，只多出新键 ✓" if ok
                          else "**有旧字段被改动，不可接受 ✗**"))
        return 0 if ok else 1

    ap.error("必须给 --snapshot 或 --verify")


if __name__ == "__main__":
    sys.exit(main())
