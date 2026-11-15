# -*- coding: utf-8 -*-
"""execution_harness.py —— F-02：单次执行链的绑定 / 组装 / 落盘（harness 侧）。

**只做一件事**：在拟合执行过程中，把实际捕获事件与当前执行实例绑定，
组装正式 v1.5 身份记录并可靠落盘、重读复核。

明确不做：决定拓扑 / 选择 K / 调整采样预算 / 自动补全缺失来源 /
根据拟合结果改变身份字段 / 运行完整 EIS 拟合。

证据不足时**照常落盘**，只是 `identity.status = "unconfirmed"`、
`run_id` / `workflow_id` 均为 `None`。失败不会被写成成功。

`capture_real_event()` 只构造真实 numpyro MCMC 对象并调用生产捕获接口
`sampler_capture.capture_at_construction()`，**不调用 MCMC.run()**。

    python docs/round_2026-11-15/execution_harness.py      # 内置自检
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import exec_identity as E                                   # noqa: E402
import sampler_capture as SC                                # noqa: E402

HARNESS_VERSION = "1.0"

#: 冻结的唯一执行路径。`bm_top.fit_top` 只作第二站点兼容验证对象。
FROZEN_PATH = {
    "primary_site": E.FROZEN_PRIMARY_SITE,
    "construction_point":
        "bm_core.py:610 MCMC(kern, num_warmup=, num_samples=, num_chains=, progress_bar=)",
    "capture_interface": "sampler_capture.capture_at_construction",
    "record_format": "exec_identity_1.5 + capture_attestation_v1",
    "compat_sites": list(E.FROZEN_COMPAT_SITES),
    "excluded": [
        "nested_fit.py 独立研究接口（bmdrt.py 未接、默认 voigt）",
        "evidence/nested_topology/m35_cond_n2.py:116（K3 独立实验路径）",
        "bm_select.py retry / refit 路径",
        "其余 CLI 与程序化入口",
    ],
}

SCENARIOS = ("ok", "sampler_error", "evidence_missing", "persist_failure")


# ---------------------------------------------------------------------------
# 执行实例与真实捕获
# ---------------------------------------------------------------------------

def build_execution_instance(*, data_sha256: str, candidate_id: str,
                             nonce: str, site: str | None = None) -> dict:
    """绑定锚点：数据身份 + 候选 + 冻结站点 + 本次唯一 nonce。"""
    site = site or E.FROZEN_PRIMARY_SITE
    if site not in E.FROZEN_EXECUTION_SITES:
        raise ValueError(f"site 不在冻结路径内：{site!r}")
    if not isinstance(nonce, str) or not nonce.strip():
        raise ValueError("nonce 必须是非空字符串")
    return {"data_sha256": data_sha256, "candidate_id": candidate_id,
            "site": site, "nonce": nonce}


def capture_real_event(site: str, *, cfg: Mapping | None = None) -> dict:
    """调用**生产捕获接口**取一条真实事件（构造 MCMC 对象，不 run、不拟合）。"""
    if site not in SC.CALL_SITES:
        raise ValueError(f"site 未接入生产捕获：{site!r}")
    c = dict(cfg or {"num_chains": 1, "num_warmup": 8, "num_samples": 8,
                     "target_accept": 0.9, "max_tree_depth": 10})
    m = SC._mcmc(kernel="NUTS", progress=False, **c)
    passed = dict(n_warmup=c["num_warmup"], n_samples=c["num_samples"],
                  num_chains=c["num_chains"], target_accept=c["target_accept"],
                  max_tree_depth=c["max_tree_depth"], progress_bar=False,
                  extra_fields=("diverging",))
    return SC.capture_at_construction(
        m, site=site, passed=passed,
        run_call=dict(extra_fields=("diverging",)))


# ---------------------------------------------------------------------------
# 组装正式记录
# ---------------------------------------------------------------------------

def assemble_record(*, conditions: Mapping, event: Mapping | None,
                    instance: Mapping, execution_status: str = "ok") -> dict:
    """把真实捕获事件与执行实例绑定，组装正式 v1.5 记录。

    `conditions` 是**冻结的执行计划**给出的条件字段（candidate/data/rng/config/
    code/env/workflow/版本）；harness 不猜、不补、不按结果改。事件里的
    `budget_params` / `budget_evidence` 只做搬运，不推断来源。
    """
    rec = copy.deepcopy(dict(conditions))
    ev = event if isinstance(event, Mapping) else {}
    bp = ev.get("budget_params")
    be = ev.get("budget_evidence")
    obs = ev.get("budget_observations")

    rec["budget_params"] = copy.deepcopy(dict(bp)) if isinstance(bp, Mapping) else None
    rec[E.BUDGET_EVIDENCE_FIELD] = copy.deepcopy(dict(be)) if isinstance(be, Mapping) else None
    rec["budget_observations"] = copy.deepcopy(dict(obs)) if isinstance(obs, Mapping) else None
    rec[E.EXECUTION_INSTANCE_FIELD] = copy.deepcopy(dict(instance))
    rec["execution_status"] = execution_status
    rec["capture_site"] = ev.get("site")
    rec["capture_errors"] = ev.get("budget_adaptation_error")
    rec["frozen_path"] = FROZEN_PATH["primary_site"]
    rec["harness_version"] = HARNESS_VERSION

    # 只有「证据齐全 + 执行正常」才由受控接口签发；否则不签发 ⇒ 无号。
    if (isinstance(bp, Mapping) and isinstance(be, Mapping)
            and execution_status == "ok"):
        rec[E.ATTESTATION_FIELD] = E.issue_capture_attestation(
            event_sha256=be["event_sha256"],
            budget_sha256=be["evidence_sha256"], instance=instance)
    else:
        rec[E.ATTESTATION_FIELD] = None
    return rec


def finalize(record: Mapping) -> dict:
    """落盘前算身份。身份字段只是 `evaluate()` 的结果，不另行裁决。"""
    out = copy.deepcopy(dict(record))
    ev = E.evaluate(out)
    out["identity"] = dict(ev)
    bad = [x for x in ev["reasons"]
           if x.startswith("attestation_") or x.startswith("execution_instance_")]
    # 「根本没签发」优先于「签了但验证失败」：缺证据 / 采样异常属于前者，
    # 是如实落盘而不是凭证被拒，测试与审计都靠这个区分。
    if not isinstance(out.get(E.ATTESTATION_FIELD), Mapping):
        out["attestation_status"] = "absent"
    elif bad:
        out["attestation_status"] = "rejected"
    else:
        out["attestation_status"] = "verified"
    return out


# ---------------------------------------------------------------------------
# 落盘 / 审计 / 重读复核
# ---------------------------------------------------------------------------

def _json_obj(record: Mapping) -> Any:
    """走身份层的规范化，保证落盘的是纯 JSON（无 tuple / numpy / NaN）。"""
    return json.loads(E.canonical_json(record))


def _audit(audit_path: str | Path | None, record: Mapping, outcome: str) -> None:
    if audit_path is None:
        return
    inst = record.get(E.EXECUTION_INSTANCE_FIELD) or {}
    row = {
        "outcome": outcome,
        "instance_token": E.execution_instance_token(inst) if inst else None,
        "nonce": inst.get("nonce"),
        "event_sha256": (record.get(E.BUDGET_EVIDENCE_FIELD) or {}).get("event_sha256"),
        "identity_status": (record.get("identity") or {}).get("status"),
        "execution_status": record.get("execution_status"),
        "attestation_status": record.get("attestation_status"),
    }
    p = Path(audit_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def seen_instance_tokens(audit_path: str | Path | None) -> set[str]:
    """审计日志里已出现过的执行实例指纹 ⇒ 检测重复提交 / 跨运行复用。"""
    if audit_path is None or not Path(audit_path).is_file():
        return set()
    out: set[str] = set()
    for line in Path(audit_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tok = json.loads(line).get("instance_token")
        if tok:
            out.add(tok)
    return out


def seen_event_shas(audit_path: str | Path | None) -> set[str]:
    """审计日志里已出现过的捕获事件指纹。"""
    if audit_path is None or not Path(audit_path).is_file():
        return set()
    out: set[str] = set()
    for line in Path(audit_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha = json.loads(line).get("event_sha256")
        if sha:
            out.add(sha)
    return out


def check_reuse(record: Mapping, audit_path: str | Path | None) -> None:
    """两道复用检测：实例 token 重复提交、A 次的捕获事件被 B 次执行使用。"""
    inst = record.get(E.EXECUTION_INSTANCE_FIELD) or {}
    tok = E.execution_instance_token(inst)
    if tok in seen_instance_tokens(audit_path):
        raise ValueError("执行实例已使用过：拒绝重复提交 / 凭证跨运行复用")
    es = (record.get(E.BUDGET_EVIDENCE_FIELD) or {}).get("event_sha256")
    if es and es in seen_event_shas(audit_path):
        raise ValueError("捕获事件已在另一次执行中使用：拒绝错误关联")


def persist(record: Mapping, path: str | Path, *,
            audit_path: str | Path | None = None,
            fail: bool = False) -> Path:
    """原子落盘（tmp + os.replace）。失败时**不留下半条成功记录**。"""
    p = Path(path)
    if fail:
        _audit(audit_path, record, outcome="persist_failed")
        raise OSError("注入的落盘失败")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(_json_obj(record), ensure_ascii=False,
                              sort_keys=True, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, p)
    _audit(audit_path, record, outcome="persisted")
    return p


def load_and_verify(path: str | Path) -> tuple[dict, dict]:
    """重读落盘记录：复核关键字段、两个指纹与身份计算结果。"""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    checks: dict[str, Any] = {}
    key_fields = ("identity_contract_version", E.RUN_ID_CONFIG_KEYS_VERSION_FIELD,
                  "candidate_id", "data", "code", "env", "budget_params",
                  E.BUDGET_EVIDENCE_FIELD, E.EXECUTION_INSTANCE_FIELD,
                  E.ATTESTATION_FIELD, "identity", "execution_status")
    checks["key_fields_present"] = all(k in rec for k in key_fields)
    bp, be = rec.get("budget_params"), rec.get(E.BUDGET_EVIDENCE_FIELD)
    if isinstance(bp, Mapping) and isinstance(be, Mapping):
        recomputed = E.budget_evidence_sha256(bp)
        checks["evidence_sha256_recomputed"] = recomputed
        checks["evidence_sha256_matches"] = recomputed == be.get("evidence_sha256")
    else:
        checks["evidence_sha256_matches"] = None
    ev = E.evaluate(rec)
    checks["identity_recomputed"] = dict(ev)
    checks["identity_matches_persisted"] = ev == rec.get("identity")
    checks["fingerprints_consistent"] = (
        checks["key_fields_present"]
        and checks["identity_matches_persisted"]
        and checks["evidence_sha256_matches"] is not False)
    return rec, checks


# ---------------------------------------------------------------------------
# 单次执行链（四种情况，不跑完整拟合）
# ---------------------------------------------------------------------------

def run_once(*, conditions: Mapping, site: str, cfg: Mapping | None,
             data_sha256: str, candidate_id: str, nonce: str,
             out_dir: str | Path, scenario: str = "ok") -> dict:
    """跑一条完整执行链，返回结果供测试断言。"""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario 必须是 {SCENARIOS}")
    out_dir = Path(out_dir)
    rec_path = out_dir / f"{scenario}.json"
    audit_path = out_dir / "audit.jsonl"
    result: dict[str, Any] = {"scenario": scenario, "path": str(rec_path),
                              "audit_path": str(audit_path)}

    instance = build_execution_instance(
        data_sha256=data_sha256, candidate_id=candidate_id, nonce=nonce, site=site)
    event = capture_real_event(site, cfg=cfg)

    if scenario == "evidence_missing":
        event = dict(event)
        event.pop("budget_evidence", None)      # 证据缺失：不自动补
    exec_status = "sampler_error" if scenario == "sampler_error" else "ok"

    rec = assemble_record(conditions=conditions, event=event,
                          instance=instance, execution_status=exec_status)
    result["event_judged"] = bool(event.get("judged"))
    result["capture_observations"] = (event.get("budget_observations") or {}) if \
        isinstance(event.get("budget_observations"), Mapping) else {}

    try:
        check_reuse(rec, audit_path)
    except ValueError as exc:
        result["reuse_rejected"] = str(exc)
        return result

    final = finalize(rec)
    result["execution_status"] = final["execution_status"]
    result["attestation_status"] = final["attestation_status"]
    result["identity"] = dict(final["identity"])

    if scenario == "persist_failure":
        try:
            persist(final, rec_path, audit_path=audit_path, fail=True)
        except OSError as exc:
            result["persist_error"] = str(exc)
            result["record_written"] = rec_path.is_file()
        return result

    persist(final, rec_path, audit_path=audit_path)
    result["record_written"] = rec_path.is_file()
    _, checks = load_and_verify(rec_path)
    result["reload_checks"] = checks
    return result


def _selftest() -> bool:
    ok = True

    def chk(label: str, cond: Any) -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}")

    chk("冻结主路径 = bm_core.fit", FROZEN_PATH["primary_site"] == "bm_core.fit")
    chk("兼容站点仅 bm_top.fit_top", FROZEN_PATH["compat_sites"] == ["bm_top.fit_top"])
    chk("排除 4 类入口", len(FROZEN_PATH["excluded"]) == 4)
    try:
        build_execution_instance(data_sha256="a" * 64, candidate_id="c",
                                 nonce="n", site="nested_fit.run")
        chk("非冻结站点被拒", False)
    except ValueError:
        chk("非冻结站点被拒", True)
    print("\n自检", "通过" if ok else "有 FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
