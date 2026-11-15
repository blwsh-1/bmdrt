# -*- coding: utf-8 -*-
"""test_sampler_capture.py —— ③ 阶段 C 第一轮：捕获层的 28 格。

阶段 B 测的是"这条记录该不该被当成证据"；这一轮测的是**采集本身可不可信**：

· 拿到的值是不是**真实**构造实参与**真实**对象属性（不是"我以为传了什么"）；
· 采集失败时会不会**假装通过**，会不会**打断拟合**；
· 捕获层有没有**自己造结论**（所有状态必须出自 `evaluate()`）。

四组：

· **K**（4 格）内核判别：NUTS/HMC、`NUTS` 是 `HMC` 子类这个陷阱、非内核对象
· **A**（5 格）`adapt_step_size`：两内核各取真值、缺属性必须 raise、fallback 的来源标签
· **B**（11 格）构造边界：八项齐全、单链、三链改写、环境覆盖、extra_fields 三态、
  单点采集失败不扩散、上下文不可用、库签名被裸包装时不编默认值
· **G**（8 格）站点与护栏：未知站点、非法字段、站点行号对得上、
  生产构造点数 == 已接 + 未接、生产模块不直连纯函数层、状态字面量只有一处、
  观测点必须是契约点

    D:/autoeis_env/Scripts/python.exe test_sampler_capture.py
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sampler_capture as C            # noqa: E402
import sampler_provenance as S         # noqa: E402

_mcmc = C._mcmc                        # 与模块自检共用一份夹具，避免两处悄悄走样


@contextmanager
def _env(**kv):
    """临时改环境变量。numpyro 的 progress_bar 覆盖是**静默**的，只能这样构造。"""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _ev(*, site="bm_core.fit", num_chains=1, progress=True, run_extra=(),
        passed_extra=None, mcmc=None, **over):
    """按 bm_core.fit 的形状采一次。`over` 覆盖 passed 里的值（用来造不一致）。"""
    m = mcmc if mcmc is not None else _mcmc(num_chains=num_chains, progress=progress)
    passed = dict(n_warmup=8, n_samples=8, num_chains=num_chains,
                  target_accept=0.9, max_tree_depth=10, progress_bar=progress)
    if passed_extra is not None:
        passed["extra_fields"] = passed_extra
    passed.update(over)
    rc = {} if run_extra is None else {"extra_fields": run_extra}
    return C.capture_at_construction(m, site=site, passed=passed, run_call=rc)


# ---------------------------------------------------------------------------
# K：内核判别
# ---------------------------------------------------------------------------

def case_K1_nuts():
    m = _mcmc()
    assert C.kernel_of(m) == "NUTS"
    return "NUTS"


def case_K2_hmc():
    m = _mcmc(kernel="HMC")
    assert C.kernel_of(m) == "HMC"
    return "HMC"


def case_K3_nuts_is_hmc_subclass():
    """`NUTS` 继承 `HMC` ⇒ 先判 HMC 会把 NUTS 全标错。这一格把陷阱钉住。"""
    from numpyro.infer import HMC

    m = _mcmc()
    assert isinstance(m.sampler, HMC), "numpyro 版本变了？NUTS 不再继承 HMC"
    assert C.kernel_of(m) == "NUTS", "先判 HMC 的实现会让这条变 HMC"
    return "isinstance(NUTS, HMC)=True 但判别仍给 NUTS"


def case_K4_unknown_kernel():
    """非 NUTS/HMC 走 raise——**不许**用 `hasattr(sampler,'_max_tree_depth')` 顶替。"""
    class _Fake:
        sampler = type("S", (), {"_max_tree_depth": 10})()

    try:
        C.kernel_of(_Fake())
    except TypeError as exc:
        assert "NUTS" in str(exc) and "hasattr" in str(exc), exc
        return "带 `_max_tree_depth` 的假对象也被拒"
    raise AssertionError("未知内核没有 raise")


# ---------------------------------------------------------------------------
# A：adapt_step_size
# ---------------------------------------------------------------------------

def case_A1_nuts_default_true():
    assert C.adapt_step_size_of(_mcmc()) == (True, "attr")
    return "True / attr"


def case_A2_hmc_false_is_readable():
    """HMC 上拿到 False——**这正是**"类型判别不能代替开关"的反例。"""
    assert C.adapt_step_size_of(_mcmc(kernel="HMC", adapt_step_size=False)) == (False, "attr")
    return "HMC + False 如实读出"


def case_A3_nuts_false():
    assert C.adapt_step_size_of(_mcmc(adapt_step_size=False)) == (False, "attr")
    return "NUTS + False 如实读出"


def case_A4_missing_attr_raises():
    class _NoFlag:
        sampler = type("S", (), {})()

    try:
        C.adapt_step_size_of(_NoFlag())
    except ValueError as exc:
        assert "adapt_step_size" in str(exc), exc
        return "缺属性 + 无 fallback ⇒ raise"
    raise AssertionError("缺属性没有 raise")


def case_A5_fallback_labelled_and_typed():
    class _NoFlag:
        sampler = type("S", (), {})()

    assert C.adapt_step_size_of(_NoFlag(), fallback=False) == (False, "call_arg")
    for bad in (None, 0, "yes"):
        try:
            C.adapt_step_size_of(_NoFlag(), fallback=bad)
        except ValueError:
            continue
        raise AssertionError(f"fallback={bad!r} 没有被拒")
    return "fallback 标 call_arg，非 bool 一律拒"


# ---------------------------------------------------------------------------
# B：构造边界
# ---------------------------------------------------------------------------

def case_B1_all_eight_judged_and_attached():
    ev = _ev(run_extra=("num_steps",), passed_extra=("num_steps",))
    assert ev["judged"] and ev["n_capture_errors"] == 0, ev["n_capture_errors"]
    assert list(ev["fields"]) == list(S.FIELDS)
    assert ev["aggregate"]["n"] == 8
    assert ev["site"] == "bm_core.fit" and ev["scope"] == "construction_boundary"
    assert ev["upstream_origin_captured"] is False
    return f"8/8 judged，聚合 {ev['aggregate']['weakest']}"


def case_B2_single_chain_chain_method_na():
    ev = _ev(num_chains=1)
    r = ev["fields"]["chain_method"]
    assert r["observed_status"] == "not_applicable", r["observed_status"]
    assert r["reason_code"] == "single_chain", r["reason_code"]
    assert r["status"] == "derived"
    assert ev["aggregate"]["weakest"] != "complete", "不适用不许让整体更完整"
    return "not_applicable / single_chain，聚合被压住"


def case_B3_three_chains_sequentially_resolved():
    ev = _ev(num_chains=3)
    r = ev["fields"]["chain_method"]
    assert r["status"] == "mismatch" and r["reason_code"] == "sequentially_resolved", r
    assert ev["aggregate"]["override_fields"] == ["chain_method"]
    assert ev["aggregate"]["n_override"] == 1
    assert r["resolved"]["origin"] == "default", "来源事实被 mismatch 吸收了"
    return "mismatch + 偏移单列，origin 仍记 default"


def case_B4_progress_forced_by_env():
    with _env(CI="1"):
        m = _mcmc(progress=False)                 # 被环境压成 False
        ev = _ev(mcmc=m, progress=True)           # 但请求的是 True
    r = ev["fields"]["progress_bar"]
    assert ev["progress_bar_env_forced"] is True, ev["progress_bar_env_forced"]
    assert r["status"] == "mismatch" and r["reason_code"] == "forced_by_env", r
    return "请求 True / 生效 False + CI ⇒ forced_by_env"


def case_B5_no_env_is_plain_value_differs():
    """同样的不一致，但**没有**环境变量 ⇒ 不许冒充"环境干的"。"""
    with _env(CI=None, PYTEST_XDIST_WORKER=None):
        m = _mcmc(progress=False)
        ev = _ev(mcmc=m, progress=True)
    r = ev["fields"]["progress_bar"]
    assert ev["progress_bar_env_forced"] is False
    assert r["status"] == "mismatch" and r["reason_code"] == "value_differs", r
    assert ev["aggregate"]["override_fields"] == [], ev["aggregate"]["override_fields"]
    return "无环境证据 ⇒ value_differs，不进 override"


def case_B6_extra_fields_declared_and_delivered():
    ev = _ev(run_extra=("num_steps",), passed_extra=("num_steps",))
    r = ev["fields"]["extra_fields"]
    assert r["status"] == "complete", r["status"]
    assert r["observed"]["how"] == "call_arg"
    assert r["observed"]["where"] == "MCMC.run.extra_fields"
    return "声明 = 实参 ⇒ complete（call_arg）"


def case_B7_extra_fields_no_declaration():
    """`fit_top` 没有 `extra_fields` 形参：只有观测、没有声明 ⇒ 必须落 derived。"""
    ev = _ev(site="bm_top.fit_top", run_extra=("diverging",))
    r = ev["fields"]["extra_fields"]
    assert r["status"] == "derived" and r["reason_code"] == "resolved_missing", r
    assert ev["aggregate"]["weakest"] != "complete"
    return "无声明 ⇒ derived / resolved_missing"


def case_B8_run_boundary_not_connected():
    ev = _ev(run_extra=None)
    r = ev["fields"]["extra_fields"]
    assert r["observed_status"] == "hook_not_reached", r["observed_status"]
    assert r["status"] == "incomplete"
    return "run 边界缺失 ⇒ incomplete / hook_not_reached"


def case_B9_single_field_failure_does_not_spread():
    """一个属性改名，只该坏一项；其余七项照旧有结论。"""
    class _Partial:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "num_warmup":
                raise AttributeError("num_warmup")
            return getattr(self._inner, name)

    ev = _ev(mcmc=_Partial(_mcmc()), run_extra=())
    r = ev["fields"]["n_warmup"]
    assert r["observed_status"] == "attribute_missing", r["observed_status"]
    assert r["status"] == "incomplete"
    assert ev["n_capture_errors"] == 0, "属性读失败是判定过的，不是采集异常"
    assert ev["judged"] is True
    assert sum(1 for x in ev["fields"].values() if x["status"] != "incomplete") == 7
    return "一项 attribute_missing，七项仍有结论"


def case_B10_context_unavailable_is_incomplete_not_pass():
    class _Broken:
        sampler = type("S", (), {})()
        num_warmup, num_samples, num_chains = 8, 8, 1

    ev = C.capture_at_construction(_Broken(), site="bm_core.fit",
                                  passed={"n_warmup": 8}, run_call={})
    assert ev["judged"] is False
    assert ev["n_capture_errors"] == 8
    assert ev["aggregate"]["weakest"] == "incomplete"
    assert all(r["judged"] is False for r in ev["fields"].values())
    return "上下文不可用 ⇒ 8/8 incomplete，judged=False"


def case_B11_bare_wrapper_does_not_invent_defaults():
    """库签名被**裸包装**（没有 `functools.wraps`）时，读不到默认值就记 capture_error。

    这是探针第一版踩到的真事：钩子把 `MCMC.__init__` 的签名换成 `(obj, sampler, **kw)`，
    于是"没传的字段"读不出库默认。捕获层**拒绝猜**——记 `capture_error`，不许编一个
    `"parallel"` 出来把链路糊过去。
    """
    from numpyro.infer import MCMC

    orig = MCMC.__init__

    def bare(self, sampler, **kw):
        return orig(self, sampler, **kw)

    MCMC.__init__ = bare
    try:
        ev = _ev(run_extra=(), passed_extra=())
    finally:
        MCMC.__init__ = orig
    r = ev["fields"]["chain_method"]
    assert r["observed_status"] == "capture_error", r["observed_status"]
    assert r["status"] == "incomplete" and r["judged"] is False
    assert ev["judged"] is False and ev["n_capture_errors"] == 1
    assert sum(1 for x in ev["fields"].values() if x["judged"]) == 7
    return "签名读不到 ⇒ capture_error，不编默认值"


# ---------------------------------------------------------------------------
# G：站点与护栏
# ---------------------------------------------------------------------------

def case_G1_unknown_site_raises():
    m = _mcmc()
    try:
        C.capture_at_construction(m, site="bm_whatever", passed={}, run_call={})
    except ValueError as exc:
        assert "未知站点" in str(exc), exc
        return "未知站点 raise"
    raise AssertionError("未知站点没有 raise")


def case_G2_passed_rejects_non_construction_field():
    m = _mcmc()
    try:
        C.capture_at_construction(m, site="bm_core.fit",
                                  passed={"seed": 0}, run_call={})
    except ValueError as exc:
        assert "seed" in str(exc), exc
        return "构造边界不收 seed"
    raise AssertionError("非法构造字段没有 raise")


def case_G3_run_call_rejects_other_fields():
    m = _mcmc()
    try:
        C.capture_at_construction(m, site="bm_core.fit", passed={},
                                  run_call={"init_params": {}})
    except ValueError as exc:
        assert "init_params" in str(exc), exc
        return "run 边界只收 extra_fields"
    raise AssertionError("非法 run 字段没有 raise")


def case_G4_site_line_numbers_match():
    """站点字符串里的行号必须真的指到 `MCMC(`：改了生产代码就该改这里。"""
    for site, where in C.CALL_SITES.items():
        fname, rest = where.split(" ", 1)
        path, line = fname.split(":")
        text = (_ROOT / path).read_text(encoding="utf-8").splitlines()
        assert "MCMC(" in text[int(line) - 1], f"{site}: {path}:{line} 不是构造行"
    return f"{len(C.CALL_SITES)} 个站点行号全部对得上"


def case_G5_construction_points_accounted():
    """生产构造点数 == 已接 + 未接。**不许静默漏掉**第三处。"""
    prod = ("bm_core.py", "bm_top.py", "bm_real.py", "bm_select.py",
            "nested_fit.py", "bmdrt.py")
    hits = []
    for name in prod:
        p = _ROOT / name
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"=\s*MCMC\(", line):
                hits.append(f"{name}:{i}")
    assert len(hits) == len(C.CALL_SITES), f"生产里 {hits}，已接 {sorted(C.CALL_SITES)}"
    m35 = _ROOT / "evidence/nested_topology/m35_cond_n2.py"
    txt = m35.read_text(encoding="utf-8").splitlines()
    assert any(re.search(r"=\s*MCMC\(", ln) for ln in txt)
    key = next(iter(C.NON_PRODUCTION_SITES))
    assert "m35_cond_n2.py" in key
    return f"{len(hits)} 处已接 + {len(C.NON_PRODUCTION_SITES)} 处明账"


def case_G6_production_imports_capture_only():
    """生产模块只 import 捕获层；纯函数层不许被它们直接 import（阶段 B 的 X1）。"""
    importers = []
    for p in _ROOT.glob("*.py"):
        if p.name.startswith("test_") or p.name == "sampler_capture.py":
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*import sampler_capture", txt, re.M):
            importers.append(p.name)
        assert not re.search(r"^\s*(import|from)\s+sampler_provenance", txt, re.M), \
            f"{p.name} 直接 import 了纯函数层"
    assert sorted(importers) == ["bm_core.py", "bm_top.py"], importers
    return f"{importers} 只连捕获层"


def case_G7_no_status_literals_except_incomplete():
    """捕获层**不自己造结论**：能写死的状态只有采集中断那一处（`incomplete`）。

    自检段不算——它只是断言，不改代码路径。所以只看 `_selftest` 之前。
    """
    src = (_ROOT / "sampler_capture.py").read_text(encoding="utf-8")
    src = src[:src.index("def _selftest")]
    assigned = set(re.findall(r'"status":\s*"(\w+)"', src))
    assert assigned <= {"incomplete"}, f"捕获层自己在写 status：{sorted(assigned)}"
    obs = set(re.findall(r'"observed_status":\s*"(\w+)"', src))
    assert obs <= {"capture_error"}, f"捕获层自己在写 observed_status：{sorted(obs)}"
    return f"status 字面量 {sorted(assigned)}、observed_status {sorted(obs)}"


def case_G8_observed_where_is_contract_point():
    """八项观测点必须逐项等于 `OBSERVE_AT`——捕获层不许自己拼位置。"""
    ev = _ev(run_extra=(), passed_extra=())
    for name, rec in ev["fields"].items():
        obs = rec["observed"]
        if obs is None:
            continue
        assert obs["where"] == S.OBSERVE_AT[name], (name, obs["where"])
    assert ev["fields"]["target_accept"]["observed"]["where"] == \
        "mcmc.sampler._target_accept_prob"
    assert ev["fields"]["max_tree_depth"]["observed"]["where"] == \
        "mcmc.sampler._max_tree_depth"
    return "8 项位置全部等于契约点"


CASES = [
    ("K1 NUTS", case_K1_nuts),
    ("K2 HMC", case_K2_hmc),
    ("K3 NUTS 是 HMC 子类", case_K3_nuts_is_hmc_subclass),
    ("K4 未知内核", case_K4_unknown_kernel),

    ("A1 NUTS 默认 True", case_A1_nuts_default_true),
    ("A2 HMC 关掉也读得出", case_A2_hmc_false_is_readable),
    ("A3 NUTS 关掉", case_A3_nuts_false),
    ("A4 缺属性 raise", case_A4_missing_attr_raises),
    ("A5 fallback 标签与类型", case_A5_fallback_labelled_and_typed),

    ("B1 八项齐全并挂载", case_B1_all_eight_judged_and_attached),
    ("B2 单链 chain_method 不适用", case_B2_single_chain_chain_method_na),
    ("B3 三链被改写", case_B3_three_chains_sequentially_resolved),
    ("B4 progress 被环境压掉", case_B4_progress_forced_by_env),
    ("B5 无环境证据是普通改值", case_B5_no_env_is_plain_value_differs),
    ("B6 extra_fields 声明=实参", case_B6_extra_fields_declared_and_delivered),
    ("B7 extra_fields 无声明", case_B7_extra_fields_no_declaration),
    ("B8 run 边界未接", case_B8_run_boundary_not_connected),
    ("B9 单项失败不扩散", case_B9_single_field_failure_does_not_spread),
    ("B10 上下文不可用", case_B10_context_unavailable_is_incomplete_not_pass),

    ("B11 裸包装不编默认值", case_B11_bare_wrapper_does_not_invent_defaults),

    ("G1 未知站点", case_G1_unknown_site_raises),
    ("G2 passed 非法字段", case_G2_passed_rejects_non_construction_field),
    ("G3 run_call 非法字段", case_G3_run_call_rejects_other_fields),
    ("G4 站点行号", case_G4_site_line_numbers_match),
    ("G5 构造点账目", case_G5_construction_points_accounted),
    ("G6 生产只连捕获层", case_G6_production_imports_capture_only),
    ("G7 不自己造结论", case_G7_no_status_literals_except_incomplete),
    ("G8 观测点=契约点", case_G8_observed_where_is_contract_point),
]


def main() -> int:
    ok, bad = 0, []
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except AssertionError as exc:
            bad.append((name, str(exc)))
            print(f"[FAIL] {name}：{exc}")
        except BaseException as exc:                 # noqa: BLE001
            bad.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERR ] {name}：{type(exc).__name__}: {exc}")
    print(f"\n{ok}/{len(CASES)} 通过"
          + ("；未通过：" + ", ".join(n for n, _ in bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
