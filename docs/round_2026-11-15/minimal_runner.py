# -*- coding: utf-8 -*-
"""Gate 3 第一版最小 runner —— 只执行与留证，不做研究决策。

准入依据：`GATE2C_ADMISSION.md` §4 的 CONDITIONAL 边界。基线 `c3a9144`。

默认 `capture_only_dry_run`：**不跑完整 EIS 拟合**，只调用生产捕获接口与真实
numpyro MCMC 构造点，不调用 `MCMC.run()`、不产生拟合结果。用户单独授权后用
`--mode real_fit` 才真的跑一次 `MCMC.run` 并产生后验样本；两种模式共用同一份
冻结清单与七步留证链。

七步工作流（只做执行与留证）：

1. 读冻结候选清单：运行时增加候选 / 改预算 / 换站点直接拒；
2. 验证数据与源码身份：数据 sha 实测比对、清单版本、dirty 状态、运行环境；
3. 创建执行记录：冻结条件 + 真实捕获事件 + 执行实例绑定；
4. 执行：生产构造点真实捕获（dry-run，不采样）；
5. 捕获实际执行信息：事件的 `budget_params` / `budget_evidence` / 观测明账；
6. 保存结果与证据：tmp + `os.replace` 原子落盘、append-only 审计、重读复核；
7. 更新 workflow 状态：append-only 状态行，成功/失败/拒绝/重试关系明确。

红线：来源未闭合时身份就是 `unconfirmed`、两号 `None`，runner 必须如实
落盘「本次无有效身份」，不得为让号非空而放宽门禁。`unavailable` 不升
`recorded`，不用推断来源补历史缺口，`bm_*.py` 零改动，三挂账不动。

    python docs/round_2026-11-15/minimal_runner.py --check
    python docs/round_2026-11-15/minimal_runner.py --run
    python docs/round_2026-11-15/test_minimal_runner.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exec_identity as E                       # noqa: E402
import test_exec_identity as TE                 # noqa: E402  已审条件夹具
import execution_harness as H                   # noqa: E402

RUNNER_VERSION = "1.0"
EXECUTION_MODE = "capture_only_dry_run"
FAULTS = ("none", "sampler_error", "interrupted", "evidence_missing",
          "persist_failure")
WORKFLOW_STATUSES = ("succeeded", "failed", "rejected")

#: 冻结清单：候选 / 数据 / 构造参数 / 软件身份 / 执行计划。
#: 数据 sha 与点数是 `archive/2EMC1-1_0005mA.csv` 的实测值（1716 字节 / 56 点）。
FROZEN_MANIFEST: dict[str, Any] = {
    "manifest_version": "1",
    "runner_version": RUNNER_VERSION,
    "admission": {
        "gate2c": "CONDITIONAL",
        "accept_unconfirmed": True,
        "no_bulk_fit": True,
        "no_auto_decision": True,
        "no_fit_algorithm_change": True,
    },
    "contract": {
        "identity_contract_version": "1.5",
        "run_id_config_keys_version": 2,
    },
    "candidates": ["voigt:hn2+rl_parallel"],
    "data": {
        "ref": "archive/2EMC1-1_0005mA.csv",
        "sha256": "560ba5f28d400013559f1f34f64e1635a34391aae3a27b0f8a8f73c0abf17030",
        "n_points": 56,
        "id_kind": "content_sha256",
    },
    #: dry-run 的 MCMC **构造参数**。不是身份层预算：`budget_params` 由真实
    #: 捕获事件给出（五项 `unavailable`），两者不许混为一谈。
    "capture_cfg": {
        "num_chains": 1, "num_warmup": 8, "num_samples": 8,
        "target_accept": 0.9, "max_tree_depth": 10,
    },
    "software_identity": {
        "primary_site": "bm_core.fit",
        "compat_sites": ["bm_top.fit_top"],
        "source_manifest": "source_manifest.json",
        "manifest_identity_version": "1.5",
        "record_format": "exec_identity_1.5 + capture_attestation_v1",
        "excluded": [
            "nested_fit.py 独立研究接口",
            "evidence/nested_topology/m35_cond_n2.py:116（K3）",
            "bm_select.py retry / refit 路径",
            "其余 CLI 与程序化入口",
        ],
    },
    "execution_plan": {
        "mode": EXECUTION_MODE,
        "n_jobs": 1,
        "calls_mcmc_run": False,
        "produces_fit_result": False,
        "no_auto_retry": True,
        "no_topology_or_K_or_budget_change": True,
        "no_model_evaluation": True,
    },
    #: 用户单独授权后启用的真实拟合计划。**试运行预算**，不是正式实验预算；
    #: 仍然单次、单候选、单条冻结数据，不批量、不自动决策、不改拟合算法。
    "real_fit_plan": {
        "mode": "real_fit",
        "calls_mcmc_run": True,
        "produces_fit_result": True,
        "n_fits": 1,
        "fit_cfg_frozen": True,
        "no_bulk_fit": True,
        "no_auto_retry": True,
        "no_fit_algorithm_change": True,
    },
    "fit_cfg": {
        "K": 10, "n_warmup": 20, "n_samples": 20, "num_chains": 1,
        "target_accept": 0.9, "max_tree_depth": 10, "seed": 0,
        "progress": False, "extra_fields": ["diverging", "num_steps"],
    },
    "condition_template": {
        "source": "test_exec_identity._rec()",
        "authenticity": (
            "已审身份夹具（合成条件字段），非真实拟合输入；"
            "数据身份用真实文件 sha 覆盖，源码/环境/dirty 由 runner 实测另记"
        ),
    },
}


# ---------------------------------------------------------------------------
# 冻结清单与身份校验
# ---------------------------------------------------------------------------

def manifest_sha256() -> str:
    return hashlib.sha256(
        E.canonical_json(FROZEN_MANIFEST).encode("utf-8")).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(_ROOT), stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def software_identity() -> dict:
    """runner 侧实测的软件身份。**不进身份哈希**：条件模板里的 `code` 仍是
    已审夹具，这里只作留证明账，不把模板伪装成真实源码摘要。"""
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "dirty_paths": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "source_manifest_path": FROZEN_MANIFEST["software_identity"]["source_manifest"],
        "source_manifest_sha256": _file_sha(
            _ROOT / FROZEN_MANIFEST["software_identity"]["source_manifest"]),
        "runner_version": RUNNER_VERSION,
        "execution_mode": EXECUTION_MODE,
        "note": ("实测值只写进本块，不覆盖身份层 code/env；"
                 "dirty patch 真实性、三时点、harness 归属仍待 T1/T2/T3 实测"),
    }


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_manifest(root: Path | None = None) -> dict:
    """第 1 + 2 步：逐项校验，任一失败直接 raise（不许乐观降级）。"""
    root = Path(root or _ROOT)
    m = FROZEN_MANIFEST
    checks: dict[str, Any] = {"manifest_sha256": manifest_sha256()}
    errs: list[str] = []

    c = m["contract"]
    checks["contract_version_matches"] = (
        c["identity_contract_version"] == E.IDENTITY_CONTRACT_VERSION
        and c["run_id_config_keys_version"] == E.RUN_ID_CONFIG_KEYS_VERSION)
    if not checks["contract_version_matches"]:
        errs.append("契约版本与身份层不一致")

    dp = root / m["data"]["ref"]
    actual_sha = _file_sha(dp)
    checks["data_file_present"] = dp.is_file()
    checks["data_sha256_matches"] = actual_sha == m["data"]["sha256"]
    if dp.is_file():
        rows = [ln for ln in dp.read_text(encoding="utf-8", errors="replace")
                .splitlines() if ln.strip()]
        checks["data_n_points_matches"] = len(rows) - 1 == m["data"]["n_points"]
    else:
        checks["data_n_points_matches"] = False
    if not all((checks["data_file_present"], checks["data_sha256_matches"],
                checks["data_n_points_matches"])):
        errs.append("冻结数据文件 sha / 点数不符")

    mp = root / m["software_identity"]["source_manifest"]
    checks["source_manifest_present"] = mp.is_file()
    if mp.is_file():
        mj = json.loads(mp.read_text(encoding="utf-8"))
        checks["source_manifest_version_matches"] = (
            mj.get("identity_contract_version") ==
            m["software_identity"]["manifest_identity_version"])
        checks["source_manifest_profiles"] = sorted(mj.get("profiles", {}))
    else:
        checks["source_manifest_version_matches"] = False
        checks["source_manifest_profiles"] = []
    if not checks["source_manifest_version_matches"]:
        errs.append("源码清单版本不一致")

    sites = set(E.FROZEN_EXECUTION_SITES)
    checks["sites_frozen"] = (
        m["software_identity"]["primary_site"] in sites
        and all(s in sites for s in m["software_identity"]["compat_sites"]))
    if not checks["sites_frozen"]:
        errs.append("冻结站点不在身份层闭集内")
    checks["execution_is_dry_run"] = (
        m["execution_plan"]["mode"] == EXECUTION_MODE
        and not m["execution_plan"]["calls_mcmc_run"]
        and not m["execution_plan"]["produces_fit_result"])
    if not checks["execution_is_dry_run"]:
        errs.append("dry-run 执行计划不得自称真实拟合")
    rp, fc = m["real_fit_plan"], m["fit_cfg"]
    checks["real_fit_plan_frozen"] = (
        rp["mode"] == "real_fit" and rp["calls_mcmc_run"] is True
        and rp["produces_fit_result"] is True and rp["n_fits"] == 1
        and rp["fit_cfg_frozen"] is True and rp["no_bulk_fit"] is True
        and not m["execution_plan"]["calls_mcmc_run"])
    if not checks["real_fit_plan_frozen"]:
        errs.append("真实拟合计划必须单次、冻结、与 dry-run 互斥")
    checks["fit_cfg_keys"] = sorted(fc)
    checks["fit_cfg_bounded"] = (
        isinstance(fc["n_warmup"], int) and isinstance(fc["n_samples"], int)
        and 0 < fc["n_warmup"] <= 100 and 0 < fc["n_samples"] <= 100
        and fc["num_chains"] == 1 and 0 < fc["target_accept"] < 1)
    if not checks["fit_cfg_bounded"]:
        errs.append("真实拟合预算必须是冻结的小预算（试运行档）")
    checks["candidates_nonempty"] = bool(m["candidates"])
    if errs:
        raise ValueError("冻结清单校验失败：" + "；".join(errs))
    checks["verified"] = True
    return checks


# ---------------------------------------------------------------------------
# 条件 / 执行实例绑定
# ---------------------------------------------------------------------------

def build_conditions(*, software: Mapping, job_key: str, attempt: int,
                     nonce: str) -> dict:
    """第 3 步：冻结条件记录。只用已审夹具 + 真实数据身份 + runner 侧明账。"""
    m = FROZEN_MANIFEST
    rec = TE._rec()
    rec["data"]["ref"] = m["data"]["ref"]
    rec["data"]["sha256"] = m["data"]["sha256"]
    rec["data"]["n_points"] = m["data"]["n_points"]
    rec["data"]["id_kind"] = m["data"]["id_kind"]
    rec["instance"] = {
        "artifact_path": None, "artifact_sha256": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "elapsed_s": None,
        "pid": os.getpid(), "host": platform.node(),
        "init_values_sha256": None,
        "job_key": job_key, "attempt": attempt,
    }
    # runner 侧明账，不进身份哈希：不让模板 code/env 伪装成实测源码身份。
    rec["software_identity"] = copy.deepcopy(dict(software))
    rec["runner"] = {
        "version": RUNNER_VERSION,
        "execution_mode": EXECUTION_MODE,
        "manifest_sha256": manifest_sha256(),
        "condition_template": copy.deepcopy(m["condition_template"]),
        "job_key": job_key, "attempt": attempt,
    }
    return rec


def bind_event_to_instance(event: Mapping, instance: Mapping) -> dict:
    """把捕获事件指纹与**本次执行实例**绑定，使两次同参数独立执行不互撞。

    捕获层事件本身不含实例标识（已知缺口：多次采样逐次事件待实测），因此
    runner 在绑定阶段派生 `event_sha256 := SHA256(原始事件指纹 | instance_token)`。
    原始指纹另存 runner 侧 `capture_raw_event_sha256` 明账。`budget_params`
    五项仍 `unavailable`，不升级任何来源，身份仍由 `evaluate()` 裁决。
    """
    out = copy.deepcopy(dict(event)) if isinstance(event, Mapping) else {}
    be = out.get("budget_evidence")
    if not isinstance(be, Mapping):
        # 采样失败 / 证据缺失：没有捕获事件，不伪造，也不给绑定指纹。
        out["capture_raw_event_sha256"] = None
        out["bound_instance_token"] = E.execution_instance_token(instance)
        return out
    raw = be.get("event_sha256")
    token = E.execution_instance_token(instance)
    if isinstance(raw, str):
        be = dict(be)
        be["event_sha256"] = hashlib.sha256(
            f"{raw}|{token}".encode("utf-8")).hexdigest()
        out["budget_evidence"] = be
    out["capture_raw_event_sha256"] = raw
    out["bound_instance_token"] = token
    return out


# ---------------------------------------------------------------------------
# workflow 状态（append-only，失败不覆盖成功记录）
# ---------------------------------------------------------------------------

def _state_path(out_dir: Path) -> Path:
    return out_dir / "workflow_state.jsonl"


def load_workflow_state(out_dir: Path) -> list[dict]:
    p = _state_path(out_dir)
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _append_state(out_dir: Path, row: Mapping) -> None:
    p = _state_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 单次执行（七步）
# ---------------------------------------------------------------------------

def run_job(*, out_dir: str | Path, job_key: str, attempt: int = 1,
            nonce: str | None = None, candidate_id: str | None = None,
            site: str | None = None, fault: str = "none",
            mode: str = "dry_run", capture_cfg: Mapping | None = None,
            fit_cfg: Mapping | None = None,
            root: Path | None = None) -> dict:
    """跑一个 job。返回完整留证明账；异常不外抛（写成 failed 状态）。"""
    root = Path(root or _ROOT)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = FROZEN_MANIFEST
    if mode not in ("dry_run", "real_fit"):
        raise ValueError("mode 必须是 dry_run / real_fit")
    result: dict[str, Any] = {
        "job_key": job_key, "attempt": attempt, "fault": fault,
        "out_dir": str(out_dir), "runner_version": RUNNER_VERSION,
        "execution_mode": EXECUTION_MODE if mode == "dry_run" else "real_fit",
        "mode": mode,
    }
    if fault not in FAULTS:
        raise ValueError(f"fault 必须是 {FAULTS}")
    if capture_cfg is not None or fit_cfg is not None:
        raise ValueError("禁止运行时调整采样预算 / 构造参数 / 拟合计划")
    if mode == "real_fit" and site not in (None, E.FROZEN_PRIMARY_SITE):
        raise ValueError("真实拟合只允许唯一冻结主路径 bm_core.fit")

    # 1) 冻结清单；2) 数据与源码身份实测
    checks = verify_manifest(root)
    sw = software_identity()
    result["checks"] = checks
    result["software_identity"] = sw

    cand = candidate_id or m["candidates"][0]
    if cand not in m["candidates"]:
        raise ValueError(f"候选 {cand!r} 不在冻结清单内：禁止运行时临时增加模型")
    st = site or m["software_identity"]["primary_site"]
    if st not in E.FROZEN_EXECUTION_SITES:
        raise ValueError(f"站点 {st!r} 不在冻结路径内")
    result["candidate_id"] = cand
    result["site"] = st

    nonce = nonce or uuid.uuid4().hex
    instance = H.build_execution_instance(
        data_sha256=m["data"]["sha256"], candidate_id=cand, nonce=nonce, site=st)
    token = E.execution_instance_token(instance)
    result["nonce"] = nonce
    result["instance_token"] = token
    rec_path = out_dir / f"{job_key}__a{attempt}.json"
    audit_path = out_dir / "audit.jsonl"
    result["record_path"] = str(rec_path)
    result["audit_path"] = str(audit_path)

    # 重复提交：同一实例 token 已在 workflow 状态里 ⇒ 拒，不动旧记录
    prior = load_workflow_state(out_dir)
    if any(x.get("instance_token") == token for x in prior):
        result["status"] = "rejected"
        result["reject_reason"] = "执行实例已使用过：拒绝重复提交"
        result["record_written"] = rec_path.is_file()
        _append_state(out_dir, _state_row(result, instance, None))
        return result

    conditions = build_conditions(software=sw, job_key=job_key,
                                  attempt=attempt, nonce=nonce)

    # 4) 执行：dry-run 只构造；real_fit 真的跑一次 MCMC.run
    bundle = None
    if mode == "real_fit":
        import real_fit_runner as RF                 # 惰性导入，dry-run 不碰 numpyro
        bundle = RF.run_real_fit(
            spectrum=m["data"]["ref"], fit_cfg=dict(m["fit_cfg"]),
            owner=f"{job_key}__a{attempt}",
            inject=(fault if fault in ("sampler_error", "interrupted") else None))
        event = bundle["event"]
        result["fit_ok"] = bundle["fit_ok"]
        result["fit_error"] = bundle["error"]
        fr = bundle["fit_result"] or {}
        result["n_draws"] = fr.get("n_draws")
        result["posterior_sha256"] = fr.get("posterior_sha256")
        result["three_points_complete"] = bundle["three_points_complete"]
        result["numeric_digest_same"] = bundle["harness"]["numeric_digest_same"]
        result["source_coverage"] = bundle["harness"]["source_coverage"]
        result["n_sampling_events"] = bundle["n_sampling_events"]
        result["n_complete_events"] = bundle["n_complete_events"]
        result["fit_elapsed_s"] = bundle["timing_s"]["fit_total"]
        exec_status = ("ok" if bundle["fit_ok"] else
                       fault if fault in ("sampler_error", "interrupted")
                       else "sampler_error")
    else:
        event = H.capture_real_event(st, cfg=dict(m["capture_cfg"]))
        exec_status = {"sampler_error": "sampler_error",
                       "interrupted": "interrupted"}.get(fault, "ok")
    if fault == "evidence_missing" and isinstance(event, Mapping):
        event = dict(event)
        event.pop("budget_evidence", None)
    result["capture_judged"] = bool((event or {}).get("judged"))
    result["capture_errors"] = (event or {}).get("budget_adaptation_error")

    # 5) 捕获 → 绑定 → 组装正式 v1.5 记录
    bound = bind_event_to_instance(event, instance)
    result["capture_raw_event_sha256"] = bound.get("capture_raw_event_sha256")
    result["bound_event_sha256"] = (bound.get("budget_evidence") or {}).get(
        "event_sha256")
    rec = H.assemble_record(conditions=conditions, event=bound,
                            instance=instance, execution_status=exec_status)
    # 原始事件指纹与实例 token 写进 runner 侧明账，落盘后可复算绑定指纹。
    rec["runner"]["capture_raw_event_sha256"] = bound.get(
        "capture_raw_event_sha256")
    rec["runner"]["bound_instance_token"] = bound.get("bound_instance_token")
    if bundle is not None:
        # 拟合结果 / 三时点 / harness 归属 / 逐次事件只作留证，不进身份哈希。
        rec["execution_evidence"] = {k: v for k, v in bundle.items()
                                     if k != "event"}
        rec["runner"]["mode"] = "real_fit"
        rec["runner"]["execution_mode"] = "real_fit"

    # 复用检测（实例 + 绑定后事件指纹），A 次完整记录交 B 次使用必拒
    try:
        H.check_reuse(rec, audit_path)
    except ValueError as exc:
        result["status"] = "rejected"
        result["reject_reason"] = str(exc)
        result["record_written"] = rec_path.is_file()
        _append_state(out_dir, _state_row(result, instance, None))
        return result

    final = H.finalize(rec)
    ident = final["identity"]
    result["execution_status"] = final["execution_status"]
    result["attestation_status"] = final["attestation_status"]
    result["identity"] = dict(ident)
    # 零号就是零号：runner 侧只标「执行链成功」，绝不标「身份有效」。
    result["identity_valid"] = bool(ident.get("run_id") and ident.get("workflow_id"))
    result["identity_status"] = ident["status"]
    result["run_id"] = ident["run_id"]
    result["workflow_id"] = ident["workflow_id"]

    # 6) 原子落盘 + 审计 + 重读复核
    if fault == "persist_failure":
        try:
            H.persist(final, rec_path, audit_path=audit_path, fail=True)
        except OSError as exc:
            result["persist_error"] = str(exc)
        result["record_written"] = rec_path.is_file()
        result["status"] = "failed" if not result["record_written"] else "succeeded"
        _append_state(out_dir, _state_row(result, instance, None))
        return result

    H.persist(final, rec_path, audit_path=audit_path)
    result["record_written"] = rec_path.is_file()
    _, reload_checks = H.load_and_verify(rec_path)
    result["reload_checks"] = reload_checks
    result["reload_consistent"] = bool(reload_checks["fingerprints_consistent"])

    # 7) workflow 状态：执行异常 ≠ 留证失败，但都不得误报为「执行成功」
    result["status"] = ("succeeded" if exec_status == "ok"
                        and result["reload_consistent"] else "failed")
    _append_state(out_dir, _state_row(result, instance, final))
    return result


def _state_row(result: Mapping, instance: Mapping,
               final: Mapping | None) -> dict:
    return {
        "job_key": result["job_key"], "attempt": result["attempt"],
        "nonce": result["nonce"], "status": result["status"],
        "mode": result.get("mode", "dry_run"),
        "fit_succeeded": result.get("fit_ok"),
        "n_draws": result.get("n_draws"),
        "posterior_sha256": result.get("posterior_sha256"),
        "execution_status": result.get("execution_status"),
        "identity_status": result.get("identity_status"),
        "identity_valid": bool(result.get("identity_valid")),
        "run_id": result.get("run_id"), "workflow_id": result.get("workflow_id"),
        "instance_token": E.execution_instance_token(instance),
        "event_sha256": result.get("bound_event_sha256"),
        "record_path": result.get("record_path"),
        "record_written": bool(result.get("record_written")),
        "fault": result.get("fault"),
        "reject_reason": result.get("reject_reason"),
        "parent": None, "children": [],
        "runner_version": RUNNER_VERSION,
        "manifest_sha256": manifest_sha256(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def resume(*, out_dir: str | Path, job_key: str, **kw) -> dict:
    """Runner 重启恢复：读状态 → 用**新 attempt** 重跑；旧记录一律保留。"""
    prior = [x for x in load_workflow_state(Path(out_dir))
             if x.get("job_key") == job_key]
    attempt = max((int(x.get("attempt", 0)) for x in prior), default=0) + 1
    kw.pop("attempt", None)
    kw.pop("nonce", None)
    return run_job(out_dir=out_dir, job_key=job_key, attempt=attempt, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summary(r: Mapping) -> str:
    lines = [
        f"job_key            = {r['job_key']}  attempt={r['attempt']}  fault={r['fault']}  mode={r.get('mode')}",
        f"status             = {r['status']}  execution_status={r.get('execution_status')}",
        f"identity_status    = {r.get('identity_status')}  identity_valid={r.get('identity_valid')}",
        f"run_id             = {r.get('run_id')}",
        f"workflow_id        = {r.get('workflow_id')}",
        f"attestation_status = {r.get('attestation_status')}",
        f"instance_token     = {r.get('instance_token', '')[:24]}…",
        f"record_written     = {r.get('record_written')}  reload_consistent={r.get('reload_consistent')}",
        f"record_path        = {r.get('record_path')}",
        f"manifest_sha256    = {r['checks']['manifest_sha256']}",
    ]
    if r.get("mode") == "real_fit":
        lines[6:6] = [
            f"fit_ok             = {r.get('fit_ok')}  elapsed_s={r.get('fit_elapsed_s')}",
            (f"fit_result         = n_draws={r.get('n_draws')}  "
             f"posterior_sha256={str(r.get('posterior_sha256'))[:20]}…"),
            (f"three_points       = {r.get('three_points_complete')}  "
             f"numeric_digest_same={r.get('numeric_digest_same')}  "
             f"coverage={r.get('source_coverage')}"),
            (f"sampling_events    = {r.get('n_sampling_events')} "
             f"(complete={r.get('n_complete_events')})"),
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gate 3 第一版最小 runner（只执行留证）")
    ap.add_argument("--check", action="store_true", help="只校验冻结清单")
    ap.add_argument("--run", action="store_true", help="实跑一次执行链")
    ap.add_argument("--out", default="out/gate3_runner", help="落盘目录")
    ap.add_argument("--job-key", default="gate3-t1")
    ap.add_argument("--nonce", default=None)
    ap.add_argument("--fault", default="none", choices=FAULTS)
    ap.add_argument("--mode", default="dry_run", choices=("dry_run", "real_fit"),
                    help="real_fit 需单独授权：真实跑一次 MCMC.run")
    a = ap.parse_args(argv)
    try:
        checks = verify_manifest()
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1
    if a.check:
        print("[ok] 冻结清单校验通过")
        for k, v in checks.items():
            print(f"     {k:34s} = {v}")
        return 0
    if not a.run:
        ap.print_help()
        return 0
    r = run_job(out_dir=_ROOT / a.out, job_key=a.job_key,
                nonce=a.nonce, fault=a.fault, mode=a.mode)
    print("=" * 92)
    print(f"Gate 3 最小 runner 实跑（mode={a.mode}；"
          + ("真实单次拟合" if a.mode == "real_fit" else "未跑完整 EIS 拟合") + "）")
    print("=" * 92)
    print(_summary(r))
    print("-" * 92)
    print("身份未闭合 ⇒ 两号 None 是正确结果；本次只验证「执行 + 留证」链。")
    return 0 if r["status"] in ("succeeded", "failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
