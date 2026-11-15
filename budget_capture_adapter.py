# -*- coding: utf-8 -*-
"""捕获事件 → `budget_params` 六键的**拒绝式适配层**（身份链 P0 / G2）。

这一层要解决的问题（第 4 步复裁 §4 S2 / §5 G2）
------------------------------------------------
`sampler_capture.capture_at_construction()` 产出的事件里，每项有
`{name, requested, resolved, observed, observed_status, ..., status, why, judged}`
共 11 键；`budget_params` 的每项要的是
`{value, source, origin, requested, transformed, where}` 共 6 键。
两者**不是同一个形状**，而且：

* `sampler_provenance.ORIGINS`（`cli/config_file/default/tier/derived`）与
  `budget_params.BUDGET_ORIGINS`（七值）**交集为空**；
* 事件里的 `resolved.origin` 是按代码路径写死的字面量 `derived` / `default`，
  **不是**上游来源的真实记录；
* 事件级 `upstream_origin_captured` 硬编码为 `False`；`status=complete` 只表示
  「构造实参 → 对象属性」这一段闭合，**不表示整条来源链闭合**。

所以本层**不做任何来源升级**：五项一律产出 `source="unavailable"` /
`value=None` / `origin="unknown"`，观测到的数值单独放进 `observations` 明账块，
只能作追溯，**不是身份来源**。这是红线「不能把 `unavailable` 改成 `recorded`、
不能用推断来源填历史缺口」的机器实现，不是一句注释。

边界
----
* 纯函数：零生产 import、不写盘、不读环境、不跑采样、不构造任何身份号。
* 不 import `budget_params` / `exec_identity` / `sampler_capture` /
  `sampler_provenance`（既有 AST 守护）；词表在本文件显式复刻，
  由 `test_budget_capture_adapter.py` 逐项交叉核验。
* 算号规则的唯一权威是 `exec_identity.py`；本层的 `canonical_json` /
  `budget_evidence_sha256` 与它同构，也由测试逐位对齐。
* 上游来源链（S1/S2）闭合之前，任何 `source="recorded"` 都不成立，
  真实运行经 `exec_identity.evaluate()` 仍应 `unconfirmed` 且两号 `None`。
  **这是正确结果，不是适配失败。**

    python budget_capture_adapter.py      # 内置自检
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

ADAPTER_VERSION = "1.0"

#: 与 `budget_params.py` / `exec_identity.py` 逐项对齐（测试交叉核验）。
BUDGET_KEYS = ("n_warmup", "n_samples", "num_chains", "target_accept",
               "max_tree_depth")
CELL_KEYS = ("value", "source", "origin", "requested", "transformed", "where")
BUDGET_SOURCE_STATES = ("recorded", "derived", "unavailable")
BUDGET_ORIGINS = (
    "cli_explicit", "module_constant", "programmatic_override",
    "function_default", "library_default", "runtime_derived", "unknown")

#: 与 `sampler_provenance.ORIGINS` 对齐。**与 BUDGET_ORIGINS 交集为空**。
SP_ORIGINS = ("cli", "config_file", "default", "tier", "derived")

#: C5：SP.ORIGINS → BUDGET_ORIGINS 的映射。空集表示「拒绝映射」。
#: 不许静默挑一个相近的值（例如把 `default` 挑成 `function_default`）——
#: 那会把「按路径推断的出处」伪装成「已捕获的上游来源」。
ORIGIN_MAPPING: dict[str, str] = {}

#: `budget_evidence` 恰好五键，由身份层校验。
EVIDENCE_KEYS = ("upstream_origin_captured", "capture_version",
                 "capture_site", "evidence_sha256", "event_sha256")

_OBSERVED_OK = ("captured", "attribute_missing", "hook_not_reached",
                "capture_error", "not_applicable")
_STATUS_OK = ("complete", "defaulted", "derived", "mismatch", "incomplete")
_MISSING = object()


# ---------------------------------------------------------------------------
# 规范化（与 exec_identity.canonical_json 同构）
# ---------------------------------------------------------------------------

def _plain(obj: Any, *, path: str = "") -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        f = float(obj)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"{path or '<value>'} 是 {f!r}：NaN/Inf 不能进证据摘要")
        return f
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{path or '<value>'} 的键必须是字符串，收到 {k!r}")
            out[k] = _plain(v, path=f"{path}.{k}" if path else k)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(v, path=f"{path}[{i}]") for i, v in enumerate(obj)]
    raise ValueError(
        f"{path or '<value>'} 的类型 {type(obj).__name__} 不支持进证据摘要")


def canonical_json(obj: Any) -> str:
    """按键名码位排序、无空格、`allow_nan=False` 的规范 JSON。

    与 `exec_identity.canonical_json` 同构；本层**只**对自己构造的、全为
    标准类型的 `budget_params` / `observations` 块调用，不碰原始事件对象。
    """
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def budget_evidence_sha256(block: Mapping) -> str:
    """六键预算块的证据指纹（与 `exec_identity.budget_evidence_sha256` 同构）。

    先把块规整成 `BUDGET_KEYS × CELL_KEYS` 的固定顺序，再算摘要：
    改任一单元格（包括只在追溯字段里改）都会换指纹，防止「验证完再篡改块」。
    """
    if not isinstance(block, Mapping):
        raise ValueError("budget_params 必须是映射")
    missing = [k for k in BUDGET_KEYS if k not in block]
    if missing:
        raise ValueError(f"budget_params 缺五项核心键 {missing}")
    extra = [k for k in block if k not in BUDGET_KEYS]
    if extra:
        raise ValueError(f"budget_params 出现未登记键 {extra}")
    norm = {}
    for k in BUDGET_KEYS:
        cell = block[k]
        if not isinstance(cell, Mapping):
            raise ValueError(f"budget_params[{k!r}] 必须是六键复合块")
        absent = [c for c in CELL_KEYS if c not in cell]
        over = [c for c in cell if c not in CELL_KEYS]
        if absent or over:
            raise ValueError(
                f"budget_params[{k!r}] 键必须恰好是 {list(CELL_KEYS)}；缺 {absent}、多 {over}")
        norm[k] = {c: cell[c] for c in CELL_KEYS}
    return _sha256(canonical_json(norm))


# ---------------------------------------------------------------------------
# 事件 → 六键
# ---------------------------------------------------------------------------

def _check_event(event: Any) -> dict:
    if not isinstance(event, Mapping):
        raise ValueError("捕获事件必须是映射")
    absent = [k for k in ("capture_version", "site", "scope",
                          "upstream_origin_captured", "judged", "fields")
              if k not in event]
    if absent:
        raise ValueError(f"捕获事件缺字段 {absent}")
    if event["upstream_origin_captured"] is not False:
        raise ValueError(
            "捕获事件的 upstream_origin_captured 不是 False："
            "本轮上游来源链未闭合，适配层不接收任何自称已闭合的事件")
    if event["judged"] is not True:
        raise ValueError("捕获事件未经 sampler_provenance 判定（judged 非 True）")
    fields = event["fields"]
    if not isinstance(fields, Mapping):
        raise ValueError("捕获事件 fields 必须是映射")
    absent = [k for k in BUDGET_KEYS if k not in fields]
    if absent:
        raise ValueError(f"捕获事件 fields 缺五项预算 {absent}")
    for k in BUDGET_KEYS:
        rec = fields[k]
        if not isinstance(rec, Mapping):
            raise ValueError(f"捕获事件 fields[{k!r}] 必须是逐项记录")
        if rec.get("judged") is not True or "resolved" not in rec:
            raise ValueError(f"捕获事件 fields[{k!r}] 缺判定或 resolved")
        if not isinstance(rec["resolved"], Mapping) or "value" not in rec["resolved"]:
            raise ValueError(f"捕获事件 fields[{k!r}].resolved 缺 value")
        if rec.get("observed_status") not in _OBSERVED_OK:
            raise ValueError(f"捕获事件 fields[{k!r}].observed_status 非法")
        if rec.get("status") not in _STATUS_OK:
            raise ValueError(f"捕获事件 fields[{k!r}].status 非法")
    return {
        "capture_version": event["capture_version"],
        "site": event["site"],
        "scope": event["scope"],
    }


def adapt_capture_event(event: Any, *, upstream_claim: Any = _MISSING) -> dict:
    """把**真实捕获事件**适配成可落盘的 `budget_params` + `budget_evidence`。

    `upstream_claim` 只作为**反例入口**存在：任何试图从外部声明上游已闭合的
    调用一律 raise。上游证据只能来自捕获事件本身，不接受 runner 自觉。

    返回的三块用途互不混同：
      * `budget_params` —— 合规六键块，五项 `source="unavailable"`，是身份输入；
      * `budget_evidence` —— 显式证据门槛字段，身份层会重算两个指纹；
      * `observations` —— 观测值明账，**不是身份来源**，不得灌进哈希。
    """
    if upstream_claim is not _MISSING:
        raise ValueError(
            "upstream_claim 不得从外部传入：上游来源证据只能由生产捕获事件给出，"
            "不接受调用方自称已闭合（红线：不许把推断值升级成 recorded）")
    meta = _check_event(event)
    fields = event["fields"]

    observations: dict[str, dict] = {}
    for k in BUDGET_KEYS:
        rec = fields[k]
        resolved = rec["resolved"]
        observed = rec.get("observed")
        observations[k] = {
            "resolved_value": resolved.get("value"),
            "observed_value": (None if not isinstance(observed, Mapping)
                               else observed.get("value")),
            "resolved_origin_literal": resolved.get("origin"),
            "capture_status": rec.get("status"),
            "observed_status": rec.get("observed_status"),
        }

    # 拒绝式适配：值确定 ≠ 来源确定。五项一律 unavailable / value=None。
    budget_params = {
        k: {"value": None, "source": "unavailable", "origin": "unknown",
            "requested": None, "transformed": [], "where": None}
        for k in BUDGET_KEYS
    }
    evidence = {
        "upstream_origin_captured": False,
        "capture_version": meta["capture_version"],
        "capture_site": meta["site"],
        "evidence_sha256": budget_evidence_sha256(budget_params),
        "event_sha256": _sha256(canonical_json(observations)),
    }
    return {
        "adapter_version": ADAPTER_VERSION,
        "budget_params": budget_params,
        "budget_evidence": evidence,
        "observations": observations,
        "promotion_allowed": False,
        "note": (
            "值已捕获但上游来源链未闭合 ⇒ source=unavailable；"
            "观测值只在 observations 里作追溯，不是身份来源。"
            "经 exec_identity.evaluate() 应得 unconfirmed 且两号 None。"),
    }


# ---------------------------------------------------------------------------
# 内置自检（不跑采样、不 import 任何生产模块）
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    ok = True

    def chk(label: str, cond: Any, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}" + (f"：{detail}" if detail else ""))

    def raises(fn, *keys: str) -> None:
        try:
            fn()
        except ValueError as exc:
            msg = str(exc)
            chk("异常含关键词 " + ",".join(keys), all(x in msg for x in keys), msg[:70])
            return
        raise AssertionError("本该 raise 却通过了")

    ev = {
        "capture_version": "1.0", "site": "bm_core.fit",
        "scope": "construction_boundary", "upstream_origin_captured": False,
        "judged": True,
        "fields": {k: {"resolved": {"value": v, "origin": "derived", "where": "ctor"},
                       "observed": {"value": v, "how": "attr", "where": "mcmc"},
                       "observed_status": "captured", "status": "complete",
                       "judged": True}
                   for k, v in zip(BUDGET_KEYS, (8, 8, 1, 0.9, 10))},
    }
    out = adapt_capture_event(ev)
    chk("五项齐全", list(out["budget_params"]) == list(BUDGET_KEYS))
    chk("每项恰好六键",
        all(list(c) == list(CELL_KEYS) for c in out["budget_params"].values()))
    chk("五项一律 unavailable / value=None",
        all(c["source"] == "unavailable" and c["value"] is None
            for c in out["budget_params"].values()))
    chk("origin 一律 unknown（不挑选、不升级）",
        all(c["origin"] == "unknown" for c in out["budget_params"].values()))
    chk("观测值只在 observations",
        out["observations"]["max_tree_depth"]["resolved_value"] == 10
        and out["budget_params"]["max_tree_depth"]["value"] is None)
    chk("词表交集为空", not (set(SP_ORIGINS) & set(BUDGET_ORIGINS)),
        str(set(SP_ORIGINS) & set(BUDGET_ORIGINS)))
    chk("映射表为空（拒绝映射）", ORIGIN_MAPPING == {})
    chk("证据五项齐全", list(out["budget_evidence"]) == list(EVIDENCE_KEYS))
    chk("证据上游标志 False", out["budget_evidence"]["upstream_origin_captured"] is False)
    chk("指纹可复算",
        out["budget_evidence"]["evidence_sha256"]
        == budget_evidence_sha256(out["budget_params"]))
    chk("promotion_allowed 恒 False", out["promotion_allowed"] is False)
    raises(lambda: adapt_capture_event(ev, upstream_claim=True), "upstream_claim")
    raises(lambda: adapt_capture_event(dict(ev, upstream_origin_captured=True)),
           "upstream_origin_captured")
    raises(lambda: adapt_capture_event(dict(ev, judged=False)), "judged")
    bad = dict(ev)
    bad["fields"] = dict(ev["fields"], n_warmup=dict(ev["fields"]["n_warmup"],
                                                     resolved={"origin": "derived"}))
    raises(lambda: adapt_capture_event(bad), "value")
    tampered = dict(out["budget_params"])
    tampered["n_warmup"] = dict(tampered["n_warmup"], source="recorded")
    chk("篡改块会换指纹",
        budget_evidence_sha256(tampered)
        != out["budget_evidence"]["evidence_sha256"])
    print(f"\n自检 {'通过' if ok else '失败'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
