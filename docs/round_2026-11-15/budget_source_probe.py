# -*- coding: utf-8 -*-
"""③ 第 1 步只读审计取证：预算来源链的静态装配与库默认值。

边界（重要）
------------
· 不构造 MCMC、不 run、不拟合、不写任何产物；
· **不产出 `budget_params`** —— 本脚本只把生产代码里"会走到的那条链"照抄成
  dict 操作并打印，供人工核对；
· 只回答一句话：从入口声明到最终构造实参，中间改了几次、改在哪。

    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/budget_source_probe.py
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bm_core                                              # noqa: E402
import bm_real                                              # noqa: E402
import bm_select                                            # noqa: E402
import bm_top                                               # noqa: E402
import circuit_registry as R                                # noqa: E402
import exec_identity as E                                   # noqa: E402
import nested_fit as N                                      # noqa: E402
from numpyro.infer import HMC, MCMC, NUTS                   # noqa: E402

FIVE = R.BUDGET_PARAM_KEYS
EIGHT = FIVE + ("chain_method", "progress_bar", "extra_fields")


# ---------------------------------------------------------------------------
# 三层默认值（库 / 两个 fit 签名）
# ---------------------------------------------------------------------------

def _defaults(cls, names):
    sig = inspect.signature(cls.__init__)
    out = {}
    for n in names:
        p = sig.parameters.get(n)
        out[n] = "<无此形参>" if p is None else (
            "<必填>" if p.default is inspect.Parameter.empty else p.default)
    return out


def lib_defaults():
    return {
        "MCMC.__init__": _defaults(MCMC, ("num_warmup", "num_samples", "num_chains",
                                          "chain_method", "progress_bar")),
        "NUTS.__init__": _defaults(NUTS, ("target_accept_prob", "max_tree_depth",
                                          "adapt_step_size")),
        "HMC.__init__": _defaults(HMC, ("target_accept_prob", "adapt_step_size")),
    }


def fit_defaults():
    out = {}
    for label, fn, names in (
        ("bm_core.fit", bm_core.fit, ("n_warmup", "n_samples", "num_chains",
                                      "target_accept", "max_tree_depth",
                                      "progress", "extra_fields")),
        ("bm_top.fit_top", bm_top.fit_top, ("n_warmup", "n_samples", "num_chains",
                                            "target_accept", "max_tree_depth",
                                            "progress")),
    ):
        sig = inspect.signature(fn)
        out[label] = {n: sig.parameters[n].default for n in names}
    return out


# ---------------------------------------------------------------------------
# 三条路径的装配（照抄生产代码，只做 dict 操作）
# ---------------------------------------------------------------------------

def a_path(*, quick=False, ta=None):
    """A：`bmdrt.py → bm_real.run → fit_one → bm_core.fit(**kw)`。"""
    kw = dict(bm_real.KW)                                # bm_real.py:2766
    if quick:                                            # :2770 —— 只覆盖三键
        kw.update(n_warmup=800, n_samples=400, num_chains=2)
    if ta is not None:                                   # :2773
        kw["target_accept"] = float(ta)
    return kw                                            # fit_one 里 B.fit(**kw)


def b_path(budget="quick"):
    """B：`bm_select.py --budget → kw → fit_voigt / fit_nested`。"""
    kw = dict(bm_select.BUDGETS[budget])                 # :1432
    voigt = dict(kw)                                     # fit_voigt 整包转发
    nested = {k: kw[k] for k in FIVE}                    # fit_nested 挑五键
    nested["progress"] = False                           # 硬编码 :394
    nested["extra_fields"] = ("diverging",)              # fit_top 硬编码
    return voigt, nested


def c_path(budget="standard", ta=None, progress=False):
    """C：`nested_fit.py --budget → budget_kw → fit_top(**kw)`。"""
    kw = N.budget_kw(budget)                             # :334
    if ta is not None:                                   # :335
        kw["target_accept"] = float(ta)
    d = dict(kw)
    d["progress"] = progress                             # CLI --progress
    d["extra_fields"] = ("diverging",)                   # fit_top 硬编码
    return d


def show(label, d):
    print(f"\n  {label}")
    for k in EIGHT:
        key = {"progress_bar": "progress"}.get(k, k)
        v = d.get(key, "<未传 ⇒ 走函数/库默认>")
        print(f"    {k:<16} {v}")


def main() -> int:
    print("=" * 76)
    print("1 · 三层默认值（库签名 / 两个 fit 签名）")
    print("=" * 76)
    for owner, d in lib_defaults().items():
        print(f"  {owner:<18} {d}")
    for owner, d in fit_defaults().items():
        print(f"  {owner:<18} {d}")

    print("\n" + "=" * 76)
    print("2 · config.json 里有没有预算数值")
    print("=" * 76)
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    fit = cfg["fit"]
    print(f"  fit 段键：{sorted(fit)}")
    print(f"  五个核心预算键命中：{[k for k in FIVE if k in fit] or '一个都没有'}")
    print("  ⇒ config.json 只管开关（quick / ta / seed），不提供任何预算数值")

    print("\n" + "=" * 76)
    print("3 · 三条路径（同名 quick 档）逐项装配")
    print("=" * 76)
    av, an = b_path("quick")
    show("A / bm_real.run --quick（fit_one 收到的 kw）", a_path(quick=True))
    show("B / bm_select --budget quick，voigt 侧（整包转发）", av)
    show("B / bm_select --budget quick，nested 侧（挑五键 + 两处硬编码）", an)
    show("C / nested_fit --budget quick", c_path("quick"))

    print("\n" + "=" * 76)
    print("4 · 同名档不同值（档名不能当预算身份）")
    print("=" * 76)
    a_quick = a_path(quick=True)
    b_quick = b_path("quick")[0]
    print(f"  A quick    max_tree_depth = {a_quick['max_tree_depth']}"
          f"（--quick 不覆盖这一键，仍是 KW 的值）")
    print(f"  B quick    max_tree_depth = {b_quick['max_tree_depth']}")
    print(f"  C quick    max_tree_depth = {c_path('quick')['max_tree_depth']}")
    print(f"  BUDGETS['standard'].target_accept       = "
          f"{bm_select.BUDGETS['standard']['target_accept']}")
    print(f"  VALIDATION_STANDARD.target_accept       = "
          f"{N.VALIDATION_STANDARD['target_accept']}")
    print(f"  nested_fit.budget_kw('standard')        = {N.budget_kw('standard')}")
    print(f"  A 默认 KW  vs B/C quick 的 warmup/samples/chains："
          f"{tuple(bm_real.KW[k] for k in FIVE[:3])} vs "
          f"{tuple(b_quick[k] for k in FIVE[:3])}")

    print("\n" + "=" * 76)
    print("5 · 两层 budget_params 的形状冲突（同一份数值喂给两个校验器）")
    print("=" * 76)
    flat = {"source": "recorded", **{k: a_path()[k] for k in FIVE}}
    print(f"  registry 0.7.0 形状：{flat}")
    checked = R._check_budget_params(flat, candidate_id="audit")
    print(f"  registry _check_budget_params ⇒ OK：{checked}")
    shape = {k: {"value": "…", "source": "…"} for k in FIVE}
    print(f"  v1.4 BUDGET_KEYS 要求：{shape}")
    try:
        E._budget_payload(flat)
        print("  v1.4 _budget_payload ⇒ 意外通过（两层竟然兼容）")
    except ValueError as exc:
        print(f"  v1.4 _budget_payload ⇒ ValueError：{exc}")
    print("  registry run_id 前缀 R-/D-（10 hex） vs v1.4 R1-/D1-（64 hex）："
          "两套号不可直接比对")

    print("\n" + "=" * 76)
    print("6 · 身份边界：哪些键进 run_id / workflow_id")
    print("=" * 76)
    print(f"  RUN_ID_CONFIG_KEYS 里是否含五个预算键："
          f"{[k for k in FIVE if k in E.RUN_ID_CONFIG_KEYS] or '没有（预算单独成块）'}")
    print(f"  progress 的 effect = {E.RUN_ID_CONFIG_KEYS.get('progress')}"
          " ⇒ display_only，不进 run_id")
    print(f"  workflow_payload 含 retry_kw（数值 dict）⇒ 预算数值可能第二次被哈希")
    print(f"  registry RUN_ID_BASIS = {R.RUN_ID_BASIS}")
    print(f"  v1.4 BUDGET_KEYS = {E.BUDGET_KEYS}")

    print("\n审计取证结束：未拟合、未写产物、未产出 budget_params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
