# -*- coding: utf-8 -*-
"""test_sampler_provenance.py —— ③ 阶段 B 的 15 格反例矩阵 + 接口护栏。

测的是**语义**，不是"能不能跑通"：每一格问"这条记录该不该被当成证据"。
重点在**失败分支**——正常路径只占一格（M1），其余都在造错：改值、采不到、
属性改名、换内核、单链、声明了但没送到。

四组：

· **P**（10 格）纯函数契约：通道白名单、契约观测点、origin 必填、NaN、原因码闭集…
· **M**（15 格）反例矩阵：M1–M13，其中 M5 拆成 a/b/c/d 四种改法（各改一处）
· **G**（5 格）聚合：最弱项、不适用计数、两条事件互不覆盖、缺项、偏移单列
· **X**（5 格）护栏：纯函数层不被生产模块 import、不 import 重量级依赖、
  不改入参、可 JSON 直出、字段顺序稳定

    python test_sampler_provenance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import sampler_provenance as S          # noqa: E402

_slot = S._slot


class Skip(Exception):
    """当前环境做不到的格子：显式跳过，不算通过也不假装通过。"""


# ---------------------------------------------------------------------------
# 夹具：B/quick 档口径的八项，全一致
# ---------------------------------------------------------------------------

def _recs(**over) -> list[dict]:
    """八项默认夹具（quick 档：800/400/2/0.9/10；`progress=False`；
    `extra_fields=("num_steps",)`）。`over` 按字段名覆盖单条记录。

    注意 `chain_method` 的 `resolved.origin` **默认就是 `default`**：
    `grep -rn chain_method --include=*.py .` 全仓零命中，项目从没设过它。
    """
    base = {
        "n_warmup": S.evaluate(
            "n_warmup", resolved=_slot(800, "tier", "BUDGETS['quick'].n_warmup"),
            observation=S.observe("n_warmup", 800)),
        "n_samples": S.evaluate(
            "n_samples", resolved=_slot(400, "tier", "BUDGETS['quick'].n_samples"),
            observation=S.observe("n_samples", 400)),
        "num_chains": S.evaluate(
            "num_chains", resolved=_slot(2, "tier", "BUDGETS['quick'].num_chains"),
            observation=S.observe("num_chains", 2), num_chains=2),
        "target_accept": S.evaluate(
            "target_accept", resolved=_slot(0.9, "tier", "BUDGETS['quick']"),
            observation=S.observe("target_accept", 0.9)),
        "max_tree_depth": S.evaluate(
            "max_tree_depth", resolved=_slot(10, "tier", "BUDGETS['quick']"),
            observation=S.observe("max_tree_depth", 10), kernel="NUTS"),
        "chain_method": S.evaluate(
            "chain_method",
            resolved=_slot("parallel", "default", "numpyro.MCMC.__init__:chain_method"),
            observation=S.observe("chain_method", "parallel"), num_chains=2),
        "progress_bar": S.evaluate(
            "progress_bar",
            resolved=_slot(False, "default", "bm_core.fit:progress"),
            observation=S.observe("progress_bar", False)),
        "extra_fields": S.evaluate(
            "extra_fields", resolved=_slot(("num_steps",), "tier", "BUDGETS['quick']"),
            observation=S.observe("extra_fields", ("num_steps",), how="call_arg")),
    }
    base.update(over)
    return [base[f] for f in S.FIELDS]


def _agg(recs, event_id=None):
    return S.aggregate(recs, event_id=event_id)


def _must_raise(fn, needle: str = "") -> str:
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        assert needle in msg, f"raise 了但理由不含 {needle!r}：{msg}"
        return msg
    raise AssertionError(f"本该 raise（{needle or '任意 ValueError'}），没有 raise")


# ---------------------------------------------------------------------------
# P：纯函数契约
# ---------------------------------------------------------------------------

def case_P1_channel_whitelist():
    """`extra_fields` 没有属性；五项不能用 call_arg 冒充。"""
    _must_raise(lambda: S.observe("extra_fields", ("num_steps",), how="attr"),
                "没有属性")
    _must_raise(lambda: S.observe("n_warmup", 800, how="call_arg"), "观测通道")
    # derived 只对 extra_fields 开放
    _must_raise(lambda: S.observe("n_warmup", 800, how="derived",
                                  where="keys(x)"), "观测通道")
    return "attr/call_arg/derived 各归其位"


def case_P2_observe_at_is_fixed():
    """观测点必须就是契约点：从 `kw` 读不能冒充从 `mc` 读。"""
    _must_raise(lambda: S.observe("n_warmup", 800, where="kw.n_warmup"),
                "契约观测点")
    _must_raise(lambda: S.observe("extra_fields", ("num_steps",), how="derived"),
                "契约点")
    return "换位置就是换语义"


def case_P3_origin_required():
    """`resolved` 必须带出处；`None`/NaN 作为值一律拒绝。"""
    _must_raise(lambda: S.evaluate("n_warmup", resolved={"value": 800},
                                   observation=S.observe("n_warmup", 800)),
                "origin")
    _must_raise(lambda: S.evaluate("n_warmup", resolved=_slot(None),
                                   observation=S.observe("n_warmup", 800)),
                "哨兵值")
    _must_raise(lambda: S.evaluate("n_warmup", resolved=_slot(float("nan")),
                                   observation=S.observe("n_warmup", 800)), "NaN")
    return "origin 必填、None/NaN 不是值"


def case_P4_observed_nan():
    """采到 NaN 是采集问题，不是"值不一致"。"""
    r = S.evaluate("target_accept", resolved=_slot(0.9),
                   observation=S.observe("target_accept", float("nan")))
    assert r["status"] == "incomplete", r["status"]
    assert r["reason_code"] == "observed_nan", r["reason_code"]
    return "NaN 落 incomplete/observed_nan"


def case_P5_reason_codes_closed():
    _must_raise(lambda: S.evaluate("n_warmup", resolved=_slot(800),
                                   observation=S.observe("n_warmup", 800),
                                   reason_code="forced_by_ci"), "原因码")
    return f"{len(S.REASON_CODES)} 个原因码是闭集"


def case_P6_not_applicable_cannot_self_declare():
    """不适用必须由 applicability() 判出来，调用方不能自报。"""
    _must_raise(lambda: S.evaluate("n_warmup", resolved=_slot(800),
                                   observation=S.observe("n_warmup", 800),
                                   observed_status="not_applicable"),
                "是适用的")
    return "自报 not_applicable 被拒"


def case_P7_tuple_order_insensitive():
    """`collect_fields` 是 dict 建的：字段顺序不改变采到的东西。"""
    r = S.evaluate("extra_fields", resolved=_slot(("num_steps", "diverging")),
                   observation=S.observe("extra_fields", ("diverging", "num_steps"),
                                         how="call_arg"))
    assert r["status"] == "complete", r["status"]
    return "元组无序比较"


def case_P8_tree_budget():
    assert S.tree_budget(13) == 8191 and S.tree_budget(10) == 1023
    _must_raise(lambda: S.tree_budget(0), "正整数")
    return "13→8191 / 10→1023"


def case_P9_applicability_needs_context():
    """`max_tree_depth` 没有 kernel 就没有适用性可判——不许给默认值。"""
    _must_raise(lambda: S.applicability("max_tree_depth"), "kernel")
    _must_raise(lambda: S.applicability("max_tree_depth", kernel="HMC2"), "kernel")
    _must_raise(lambda: S.applicability("chain_method"), "num_chains")
    _must_raise(lambda: S.applicability("chain_method", num_chains=0), "正整数")
    return "缺上下文直接 raise"


def case_P10_where_required():
    _must_raise(lambda: S.observe("n_warmup", 800, where=""), "非空字符串")
    _must_raise(lambda: S.evaluate("n_warmup", resolved=_slot(800, where=""),
                                   observation=S.observe("n_warmup", 800)), "where")
    return "证据要能指到位置"


# ---------------------------------------------------------------------------
# M：15 格反例矩阵
# ---------------------------------------------------------------------------

def case_M1_all_explicit():
    """M1 八项全显式且一致 ⇒ 全 `complete`。

    现实里 `chain_method` 没有任何入口（库默认），所以这一格是"假设都显式传入"
    的上界；它的现实版本在 M9/M12 与模块自检里。
    """
    recs = _recs()
    recs[5] = S.evaluate("chain_method", resolved=_slot("parallel", "cli",
                                                        "MCMC.__init__:chain_method"),
                         observation=S.observe("chain_method", "parallel"),
                         num_chains=2)
    recs[6] = S.evaluate("progress_bar",
                         resolved=_slot(False, "tier", "BUDGETS['quick'].progress"),
                         observation=S.observe("progress_bar", False))
    a = _agg(recs)
    assert a["n_by_status"]["complete"] == 8, a["n_by_status"]
    assert a["weakest"] == "complete", a["weakest"]
    assert a["complete_evidence"] is True
    return "8/8 complete"


def case_M2_all_default():
    """M2 八项全走默认 ⇒ 逐项 `defaulted`，**不等于** `derived`。"""
    recs = [
        S.evaluate(f, resolved=_slot(v, "default", f"somewhere.{f}"),
                   observation=S.observe(
                       f, v, how=("call_arg" if f == "extra_fields" else "attr")),
                   num_chains=nc,
                   kernel="NUTS" if f == "max_tree_depth" else None)
        for f, v, nc in (
            ("n_warmup", 600, None), ("n_samples", 1200, None),
            ("num_chains", 1, 1), ("target_accept", 0.9, None),
            ("max_tree_depth", 10, None), ("chain_method", "parallel", 1),
            ("progress_bar", True, None),
            ("extra_fields", (), None),
        )
    ]
    # chain_method 在单链下不适用，会被改判；这里要看"全默认"本身
    recs[5] = S.evaluate("chain_method", resolved=_slot("parallel", "default",
                                                        "numpyro.MCMC.__init__"),
                         observation=S.observe("chain_method", "parallel"),
                         num_chains=2)
    recs[2] = S.evaluate("num_chains", resolved=_slot(1, "default", "bm_core.fit"),
                         observation=S.observe("num_chains", 1), num_chains=2)
    a = _agg(recs)
    assert a["n_by_status"]["defaulted"] == 8, a["n_by_status"]
    assert a["n_by_status"]["derived"] == 0
    assert a["weakest"] == "defaulted", a["weakest"]
    return "8/8 defaulted（不是 derived）"


def case_M3_partial_explicit():
    """M3 部分显式 ⇒ 逐项不同，聚合取最弱。"""
    recs = _recs()
    recs[0] = S.evaluate("n_warmup", resolved=_slot(3000, "cli", "bm_real.KW"),
                         observation=S.observe("n_warmup", 3000))
    recs[6] = S.evaluate("progress_bar", resolved=_slot(False, "default",
                                                        "bm_core.fit:progress"),
                         observation=S.observe("progress_bar", False))
    a = _agg(recs)
    assert a["fields"]["n_warmup"] == "complete", a["fields"]["n_warmup"]
    assert a["fields"]["progress_bar"] == "defaulted", a["fields"]["progress_bar"]
    assert a["weakest"] == "defaulted", a["weakest"]   # 两条 defaulted：chain_method
    return f"逐项 {a['n_by_status']}"


def case_M4_quick_override_recorded():
    """M4 `--quick` 改三个数 ⇒ 记**改后**值，且 `origin` 不是 default。"""
    recs = _recs()
    a = _agg(recs)
    assert recs[0]["resolved"]["value"] == 800, recs[0]["resolved"]
    assert recs[0]["resolved"]["origin"] == "tier"
    assert recs[2]["resolved"]["value"] == 2
    assert a["fields"]["n_warmup"] == "complete"
    # 另造一条：把"改后值"和 KW 的 3000 比 ⇒ 必须 mismatch
    wrong = S.evaluate("n_warmup", resolved=_slot(3000, "cli", "bm_real.KW"),
                       observation=S.observe("n_warmup", 800))
    assert wrong["status"] == "mismatch", wrong["status"]
    return "记 800 不是 3000"


def case_M5a_core_value_tampered():
    """M5a 五参数之一被人为改写 ⇒ 只有那一项 `mismatch`。"""
    recs = _recs()
    recs[0] = S.evaluate("n_warmup", resolved=_slot(800, "tier", "BUDGETS.quick"),
                         observation=S.observe("n_warmup", 400))
    a = _agg(recs)
    assert a["n_by_status"]["mismatch"] == 1, a["n_by_status"]
    assert a["weakest"] == "mismatch"
    assert a["override_fields"] == [], a["override_fields"]  # 不是"条件偏移"
    assert recs[0]["reason_code"] == "value_differs"
    return "1 项 mismatch，其余 7 项不动"


def case_M5b_chain_method():
    """M5b `chain_method`：显式传 vectorized ⇒ complete；被设备数改 ⇒ mismatch。"""
    ok = S.evaluate("chain_method", resolved=_slot("vectorized", "cli", "KW"),
                    observation=S.observe("chain_method", "vectorized"),
                    num_chains=3)
    assert ok["status"] == "complete", ok["status"]
    bad = S.evaluate("chain_method",
                     resolved=_slot("parallel", "default", "numpyro.MCMC.__init__"),
                     observation=S.observe("chain_method", "sequential"),
                     num_chains=3, reason_code="sequentially_resolved")
    assert bad["status"] == "mismatch", bad["status"]
    assert bad["reason_code"] == "sequentially_resolved"
    assert bad["observed_status"] == "captured"      # 采到了，只是值被改了
    return "complete / mismatch+sequentially_resolved"


def case_M5c_progress_bar_env():
    """M5c `CI=1` 把 progress_bar 关掉：请求 True、生效 False ⇒ mismatch。"""
    bad = S.evaluate("progress_bar", resolved=_slot(True, "cli", "bm_real.KW"),
                     observation=S.observe("progress_bar", False),
                     reason_code="forced_by_env")
    assert bad["status"] == "mismatch", bad["status"]
    assert bad["reason_code"] == "forced_by_env"
    # 另一条：请求就是 False，环境也关 ⇒ 一致，不许"因环境覆盖"就放过差异，
    # 但值确实相等时也不许凭空造错
    ok = S.evaluate("progress_bar", resolved=_slot(False, "default", "bm_core.fit"),
                    observation=S.observe("progress_bar", False))
    assert ok["status"] == "defaulted", ok["status"]
    return "True→False 记 mismatch；False→False 不造假错"


def case_M5d_extra_fields():
    """M5d `extra_fields`：声明 `num_steps` 但 call_arg 只收到 `diverging`。"""
    bad = S.evaluate("extra_fields",
                     resolved=_slot(("num_steps",), "tier", "BUDGETS['quick']"),
                     observation=S.observe("extra_fields", ("diverging",),
                                           how="call_arg"))
    assert bad["status"] == "mismatch", bad["status"]
    # 同一条差异，走反推通道也必须报出来，只是不能标 captured
    derived = S.evaluate(
        "extra_fields", resolved=_slot(("num_steps",), "tier", "BUDGETS['quick']"),
        observation=S.observe("extra_fields", ("diverging",), how="derived",
                              where="keys(get_extra_fields)-default_fields"),
        observed_status="hook_not_reached", why="inferred_only")
    assert derived["observed"]["how"] == "derived"
    assert derived["status"] == "incomplete", derived["status"]
    assert derived["observed_status"] != "captured"
    return "两条通道都报差异；反推不许标 captured"


def case_M6_observation_missing():
    """M6 观测缺失 ⇒ `incomplete`，且**不许**用 `resolved` 冒充。"""
    r = S.evaluate("n_warmup", resolved=_slot(800, "tier", "BUDGETS.quick"),
                   observation=None)
    assert r["status"] == "incomplete", r["status"]
    assert r["observed"] is None
    assert r["reason_code"] == "observation_missing"
    # 手上有 resolved 也不能因此变 complete
    assert r["status"] != "complete"
    return "incomplete，resolved 不顶替"


def case_M7_attribute_renamed():
    """M7 属性改名（模拟库升级）⇒ `attribute_missing` ⇒ `incomplete`。"""
    r = S.evaluate("max_tree_depth", resolved=_slot(10, "tier", "BUDGETS"),
                   observation=None, observed_status="attribute_missing",
                   kernel="NUTS")
    assert r["status"] == "incomplete", r["status"]
    assert r["observed_status"] == "attribute_missing"
    assert r["why"] == "attribute_missing"
    return "attribute_missing → incomplete"


def case_M8_two_events_independent():
    """M8 同一工作流两次不同预算 ⇒ 两条独立记录，互不覆盖。"""
    e1 = _recs()
    e2 = _recs()
    e2[0] = S.evaluate("n_warmup", resolved=_slot(2000, "tier", "BUDGETS.standard"),
                       observation=S.observe("n_warmup", 2000))
    e2[1] = S.evaluate("n_samples", resolved=_slot(1000, "tier", "BUDGETS.standard"),
                       observation=S.observe("n_samples", 1000))
    a1 = _agg(e1, event_id="R1-a#0")
    a2 = _agg(e2, event_id="R1-a#1")
    assert a1["event_id"] != a2["event_id"]
    assert e1[0]["resolved"]["value"] == 800, "第二次运行改写了第一次的记录"
    assert e2[0]["resolved"]["value"] == 2000
    assert a1["n"] == a2["n"] == 8
    return "800 与 2000 各自留住"


def case_M9_sequential_override():
    """M9 `num_chains=3` + 1 台设备：五项仍 complete，另附 `sequentially_resolved`。"""
    recs = _recs()
    recs[2] = S.evaluate("num_chains", resolved=_slot(3, "tier", "BUDGETS"),
                         observation=S.observe("num_chains", 3), num_chains=3)
    recs[5] = S.evaluate("chain_method",
                         resolved=_slot("parallel", "default",
                                        "numpyro.MCMC.__init__"),
                         observation=S.observe("chain_method", "sequential"),
                         num_chains=3, reason_code="sequentially_resolved")
    a = _agg(recs)
    assert all(a["fields"][f] == "complete" for f in S.CORE_FIELDS), a["fields"]
    assert a["weakest"] == "mismatch", a["weakest"]
    assert a["override_fields"] == ["chain_method"], a["override_fields"]
    assert a["n_override"] == 1
    return "五项 complete + 1 项条件偏移"


def case_M10_key_missing_tier():
    """M10 档里缺键 ⇒ 落到 `fit_top` 的函数默认，`origin` 要指到**哪个函数**。"""
    r = S.evaluate("num_chains", resolved=_slot(4, "default", "bm_top.fit_top.num_chains"),
                   observation=S.observe("num_chains", 4), num_chains=4)
    assert r["status"] == "defaulted", r["status"]
    assert "fit_top" in r["resolved"]["where"], r["resolved"]["where"]
    assert r["reason_code"] is None            # 有出处 ⇒ 不是 unsourced
    # 没有出处时必须显式标记：default 不等于"源码默认"
    r2 = S.evaluate("num_chains", resolved=_slot(4, "default", None),
                    observation=S.observe("num_chains", 4), num_chains=4)
    assert r2["status"] == "defaulted", r2["status"]
    assert r2["reason_code"] == "origin_default_unsourced", r2["reason_code"]
    return "defaulted + 出处 / 无出处另有原因码"


def case_M11_kernel_hmc():
    """M11 换 HMC：`max_tree_depth` = not_applicable→derived，不是 complete=10、不是 mismatch。"""
    recs = _recs()
    recs[4] = S.evaluate("max_tree_depth", resolved=_slot(7, "cli", "KW"),
                         observation=S.observe("max_tree_depth", 10),
                         kernel="HMC")
    a = _agg(recs)
    assert recs[4]["status"] == "derived", recs[4]["status"]
    assert recs[4]["observed_status"] == "not_applicable"
    assert recs[4]["reason_code"] == "kernel_inapplicable"
    assert a["n_by_status"]["mismatch"] == 0, a["n_by_status"]   # 不许假 mismatch
    assert a["weakest"] == "derived"
    assert a["n_not_applicable"] == 1
    assert a["complete_evidence"] is False        # 不适用不放大完整性
    return "derived，无假 mismatch，整体封顶"


def case_M12_single_chain():
    """M12 `num_chains=1`：`chain_method` 不适用（`run` 根本不读它）。"""
    recs = _recs()
    recs[2] = S.evaluate("num_chains", resolved=_slot(1, "tier", "BUDGETS"),
                         observation=S.observe("num_chains", 1), num_chains=1)
    recs[5] = S.evaluate("chain_method",
                         resolved=_slot("parallel", "default",
                                        "numpyro.MCMC.__init__"),
                         observation=S.observe("chain_method", "parallel"),
                         num_chains=1)
    a = _agg(recs)
    assert recs[5]["observed_status"] == "not_applicable", recs[5]
    assert recs[5]["reason_code"] == "single_chain"
    assert recs[5]["status"] == "derived"
    assert a["n_by_status"]["mismatch"] == 0
    return "single_chain → derived"


def case_M13_b_path_declared_not_delivered():
    """M13 B 路径：预算里声明 `num_steps`，`fit_top` 硬编码 `("diverging",)`。"""
    recs = _recs()
    recs[7] = S.evaluate("extra_fields",
                         resolved=_slot(("num_steps",), "tier",
                                        "nested_fit.budget_kw:extra_fields"),
                         observation=S.observe("extra_fields", ("diverging",),
                                               how="call_arg"),
                         reason_code="declared_not_delivered")
    a = _agg(recs)
    assert recs[7]["status"] == "mismatch", recs[7]["status"]
    assert recs[7]["reason_code"] == "declared_not_delivered"
    assert a["weakest"] == "mismatch"
    # 同一个差异不许被 defaulted/complete 掩掉
    assert a["fields"]["extra_fields"] == "mismatch"
    return "声明未送达 ⇒ mismatch"


# ---------------------------------------------------------------------------
# G：聚合
# ---------------------------------------------------------------------------

def case_G1_weakest_independent_of_order():
    """最弱项与记录顺序无关。"""
    five = [
        S.evaluate("n_warmup", resolved=_slot(800), observation=S.observe("n_warmup", 800)),
        S.evaluate("n_samples", resolved=_slot(400, "default", "x"),
                   observation=S.observe("n_samples", 400)),
        S.evaluate("num_chains", resolved=None, num_chains=2,
                   observation=S.observe("num_chains", 2)),
        S.evaluate("target_accept", resolved=_slot(0.9),
                   observation=S.observe("target_accept", 0.98)),
        S.evaluate("max_tree_depth", resolved=_slot(10), observation=None,
                   kernel="NUTS"),
        S.evaluate("chain_method", resolved=_slot("parallel", "default", "lib"),
                   observation=S.observe("chain_method", "parallel"), num_chains=2),
        S.evaluate("progress_bar", resolved=_slot(False, "default", "lib"),
                   observation=S.observe("progress_bar", False)),
        S.evaluate("extra_fields", resolved=_slot(()), observation=S.observe("extra_fields", (), how="call_arg")),
    ]
    a = S.aggregate(five)
    b = S.aggregate(list(reversed(five)))
    assert a["weakest"] == b["weakest"] == "incomplete", (a["weakest"], b["weakest"])
    assert a["n_by_status"] == {"complete": 2, "defaulted": 3, "derived": 1,
                               "mismatch": 1, "incomplete": 1}, a["n_by_status"]
    return "顺序无关，五态齐现"


def case_G2_not_applicable_counted():
    recs = _recs()
    recs[4] = S.evaluate("max_tree_depth", resolved=_slot(10),
                         observation=S.observe("max_tree_depth", 10), kernel="HMC")
    recs[5] = S.evaluate("chain_method", resolved=_slot("parallel", "default", "lib"),
                         observation=S.observe("chain_method", "parallel"),
                         num_chains=1)
    a = _agg(recs)
    assert a["n_not_applicable"] == 2, a["n_not_applicable"]
    assert a["n_by_status"]["derived"] == 2
    return "两项不适用 ⇒ derived ×2"


def case_G3_require_all_optional():
    a = S.aggregate(_recs()[:3], require_all=False)
    assert a["n"] == 3 and a["weakest"] == "complete", a
    _must_raise(lambda: S.aggregate(_recs()[:3]), "缺项")
    _must_raise(lambda: S.aggregate(_recs() + [_recs()[0]]), "多次")
    return "缺项默认 raise，可显式放宽"


def case_G4_override_only_counts_override():
    """`override_fields` 只收执行条件偏移，不收普通改值。"""
    recs = _recs()
    recs[0] = S.evaluate("n_warmup", resolved=_slot(800),
                         observation=S.observe("n_warmup", 400))
    a = _agg(recs)
    assert a["n_override"] == 0 and a["override_fields"] == []
    recs[6] = S.evaluate("progress_bar", resolved=_slot(True, "cli", "KW"),
                         observation=S.observe("progress_bar", False),
                         reason_code="forced_by_env")
    a2 = _agg(recs)
    assert a2["n_override"] == 1 and a2["override_fields"] == ["progress_bar"]
    return "两类 mismatch 分开计数"


def case_G5_reason_codes_deduped():
    recs = _recs()
    recs[7] = S.evaluate("extra_fields", resolved=_slot(("num_steps",)),
                         observation=S.observe("extra_fields", (), how="call_arg"))
    a = _agg(recs)
    assert a["reason_codes"] == {"extra_fields": "value_differs"}, a["reason_codes"]
    assert "extra_fields" in a["why"]
    return "原因码只带有值的那几项"


# ---------------------------------------------------------------------------
# X：护栏
# ---------------------------------------------------------------------------

_PROD = ("bm_core.py", "bm_top.py", "bm_real.py", "bm_select.py", "bmdrt.py",
         "nested_fit.py", "bm_check.py", "bm_experiments.py", "bm_identity.py",
         "exec_identity.py", "capture_contract.py", "patch_bundle.py",
         "patch_bundle_io.py")


def case_X1_not_imported_by_production():
    """纯函数层**不被生产模块 import**：它只产出记录，不参与拟合。"""
    bad = []
    for name in _PROD:
        p = _ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "sampler_provenance" in txt:
            bad.append(name)
    assert not bad, f"这些生产模块引用了本层：{bad}"
    return f"{len(_PROD)} 个生产模块零引用"


def case_X2_no_heavy_imports():
    """不 import numpyro / os / time / random，不 open()：它必须能在任何地方算。"""
    import ast

    src = (_ROOT / "sampler_provenance.py").read_text(encoding="utf-8")
    tree = ast.parse(src)                     # 只看真语句，不误伤 docstring 里的词
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    banned_mods = {"numpyro", "jax", "os", "time", "random", "subprocess",
                   "pathlib", "json", "arviz", "numpy"}
    hit = sorted(mods & banned_mods)
    assert not hit, f"纯函数层不许 import 这些：{hit}"
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "open" not in calls, "纯函数层不许 open()"
    assert "Path" not in calls, "纯函数层不许碰路径"
    return f"只 import {sorted(mods - {'annotations'})}，无 I/O"


def case_X3_inputs_not_mutated():
    recs = _recs()
    before = json.dumps(recs, sort_keys=True, default=str)
    S.aggregate(recs)
    after = json.dumps(recs, sort_keys=True, default=str)
    assert before == after, "aggregate 改了入参"
    slot = _slot(800, "tier", "BUDGETS.quick")
    snap = dict(slot)
    S.evaluate("n_warmup", resolved=slot, observation=S.observe("n_warmup", 800))
    assert slot == snap, "evaluate 改了 resolved 入参"
    return "纯函数：入参一个字段都没动"


def case_X4_json_safe():
    """记录必须能 `allow_nan=False` 直出——NaN 在这一层就被挡住了。"""
    txt = json.dumps(S.aggregate(_recs()), sort_keys=True, allow_nan=False)
    assert "NaN" not in txt and "Infinity" not in txt
    return f"{len(txt)} 字节，无 NaN/Infinity"


def case_X5_field_order_stable():
    """字段顺序是载荷的一部分：前五项必须还是那五项。"""
    assert S.CORE_FIELDS == ("n_warmup", "n_samples", "num_chains",
                             "target_accept", "max_tree_depth")
    assert S.EXEC_FIELDS == ("chain_method", "progress_bar", "extra_fields")
    assert S.FIELDS == S.CORE_FIELDS + S.EXEC_FIELDS
    assert tuple(S.OBSERVE_AT) == S.FIELDS
    assert set(S.HOW_BY_FIELD) == set(S.FIELDS)
    return "8 项顺序与契约一致"


CASES = [
    # P
    ("P1 通道白名单", case_P1_channel_whitelist),
    ("P2 契约观测点固定", case_P2_observe_at_is_fixed),
    ("P3 origin 必填 / 哨兵值", case_P3_origin_required),
    ("P4 观测 NaN", case_P4_observed_nan),
    ("P5 原因码闭集", case_P5_reason_codes_closed),
    ("P6 不许自报不适用", case_P6_not_applicable_cannot_self_declare),
    ("P7 元组无序比较", case_P7_tuple_order_insensitive),
    ("P8 树深预算", case_P8_tree_budget),
    ("P9 适用性缺上下文", case_P9_applicability_needs_context),
    ("P10 where 必填", case_P10_where_required),
    # M
    ("M1 八项全显式", case_M1_all_explicit),
    ("M2 八项全默认", case_M2_all_default),
    ("M3 部分显式", case_M3_partial_explicit),
    ("M4 --quick 记改后值", case_M4_quick_override_recorded),
    ("M5a 五参数被改", case_M5a_core_value_tampered),
    ("M5b chain_method", case_M5b_chain_method),
    ("M5c progress_bar + CI", case_M5c_progress_bar_env),
    ("M5d extra_fields", case_M5d_extra_fields),
    ("M6 观测缺失", case_M6_observation_missing),
    ("M7 属性改名", case_M7_attribute_renamed),
    ("M8 两次事件独立", case_M8_two_events_independent),
    ("M9 设备数改链方式", case_M9_sequential_override),
    ("M10 缺键落函数默认", case_M10_key_missing_tier),
    ("M11 换 HMC", case_M11_kernel_hmc),
    ("M12 单链", case_M12_single_chain),
    ("M13 B 路径声明未送达", case_M13_b_path_declared_not_delivered),
    # G
    ("G1 最弱项与顺序无关", case_G1_weakest_independent_of_order),
    ("G2 不适用计数", case_G2_not_applicable_counted),
    ("G3 require_all", case_G3_require_all_optional),
    ("G4 偏移单列", case_G4_override_only_counts_override),
    ("G5 原因码只带有值的", case_G5_reason_codes_deduped),
    # X
    ("X1 生产零引用", case_X1_not_imported_by_production),
    ("X2 无重依赖", case_X2_no_heavy_imports),
    ("X3 不改入参", case_X3_inputs_not_mutated),
    ("X4 JSON 直出", case_X4_json_safe),
    ("X5 字段顺序稳定", case_X5_field_order_stable),
]


def main() -> int:
    ok, skip, bad = 0, [], []
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except Skip as exc:
            skip.append(name)
            print(f"[skip] {name}：{exc}")
        except AssertionError as exc:
            bad.append((name, str(exc)))
            print(f"[FAIL] {name}：{exc}")
        except BaseException as exc:                 # noqa: BLE001
            bad.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERR ] {name}：{type(exc).__name__}: {exc}")
    total = len(CASES)
    print(f"\n{ok}/{total} 通过"
          + (f"，跳过 {len(skip)}" if skip else "")
          + ("；未通过：" + ", ".join(n for n, _ in bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
