# -*- coding: utf-8 -*-
"""Gate 3 第一版最小 runner 的试运行验收：T1 / T2 / T3 + 冻结禁令。

不跑完整 EIS 拟合；每个用例都在临时目录里走 runner 的真实执行链。

- T1 单次执行：执行、身份、结果、证据全部闭合（零号如实落盘）。
- T2 重复执行：同条件两次独立执行，基础条件锚点相同、实例不同、不合并。
- T3 失败与恢复：采样异常 / 中断 / 证据缺失 / 落盘失败 / 重复提交 /
  Runner 重启；不覆盖原始记录、不把失败误报为成功。

    python docs/round_2026-11-15/test_minimal_runner.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exec_identity as E                       # noqa: E402
import execution_harness as H                   # noqa: E402
import minimal_runner as M                      # noqa: E402


def _run(d: Path, *, tag: str, **kw) -> dict:
    kw.setdefault("job_key", tag)
    kw.setdefault("nonce", f"nonce-{tag}")
    return M.run_job(out_dir=d, root=_ROOT, **kw)


def _record(d: Path, r: Mapping) -> dict:      # noqa: F821
    return json.loads(Path(r["record_path"]).read_text(encoding="utf-8"))


def _states(d: Path) -> list[dict]:
    return M.load_workflow_state(d)


def _anchor(rec: dict) -> str:
    """基础条件身份锚点：run_id 载荷的规范化摘要。零号时也能比条件是否相同。"""
    return hashlib.sha256(
        E.canonical_json(E.run_id_payload(rec)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# T1 单次执行
# ---------------------------------------------------------------------------

def case_T1_single_execution():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t1")
        assert r["status"] == "succeeded", r
        assert r["checks"]["verified"] and r["checks"]["data_sha256_matches"]
        assert r["record_written"] and r["reload_consistent"], r
        assert r["execution_mode"] == M.EXECUTION_MODE
        assert r["identity_status"] == "unconfirmed", r["identity"]
        assert r["run_id"] is None and r["workflow_id"] is None, r["identity"]
        assert r["identity_valid"] is False
        assert r["attestation_status"] == "verified"
        st = _states(d)
        assert len(st) == 1 and st[0]["status"] == "succeeded", st
        assert st[0]["identity_valid"] is False and st[0]["run_id"] is None
        assert r["software_identity"]["git_commit"]
        assert r["checks"]["manifest_sha256"] == M.manifest_sha256()
        return "T1 单次执行：7 步闭合、零号如实落盘、状态行独立"


def case_T1_record_fields():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t1-fields")
        rec, checks = H.load_and_verify(r["record_path"])
        assert checks["key_fields_present"] and checks["fingerprints_consistent"]
        assert checks["identity_matches_persisted"]
        assert rec["data"]["sha256"] == M.FROZEN_MANIFEST["data"]["sha256"]
        assert rec["data"]["ref"] == M.FROZEN_MANIFEST["data"]["ref"]
        assert rec["runner"]["execution_mode"] == M.EXECUTION_MODE
        assert rec["runner"]["manifest_sha256"] == M.manifest_sha256()
        assert "condition_template" in rec["runner"]
        assert "git_commit" in rec["software_identity"]
        bud = rec["budget_params"]
        assert all(bud[k]["source"] == "unavailable" and bud[k]["value"] is None
                   for k in E.BUDGET_KEYS), bud
        assert rec[E.BUDGET_EVIDENCE_FIELD]["upstream_origin_captured"] is False
        raw = rec["runner"]["capture_raw_event_sha256"]
        bound = rec[E.BUDGET_EVIDENCE_FIELD]["event_sha256"]
        assert isinstance(raw, str) and len(raw) == 64
        assert raw != bound
        want = hashlib.sha256(
            f"{raw}|{E.execution_instance_token(rec[E.EXECUTION_INSTANCE_FIELD])}"
            .encode("utf-8")).hexdigest()
        assert bound == want, "事件指纹必须可由原始指纹 + 实例 token 复算"
        return "T1 记录字段：真实数据 sha、五项 unavailable、事件绑定可复算"


def case_T1_no_fit_result():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rec = _record(d, _run(d, tag="t1-no-fit"))
        plan = M.FROZEN_MANIFEST["execution_plan"]
        assert plan["calls_mcmc_run"] is False and plan["produces_fit_result"] is False
        for forbidden in ("posterior", "fit_result", "samples", "mcmc_samples"):
            assert forbidden not in rec, f"dry-run 不得产生 {forbidden}"
        assert rec["execution_status"] == "ok"
        return "T1 不产生拟合结果：dry-run 边界成立"


def case_T1_zero_id_not_marked_valid():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rec = _record(d, _run(d, tag="t1-zero"))
        ident = rec["identity"]
        assert ident["run_id"] is None and ident["workflow_id"] is None
        assert ident["status"] == "unconfirmed"
        st = _states(d)[0]
        assert not st["identity_valid"] and st["run_id"] is None
        assert "budget_not_recorded" in " ".join(ident["reasons"]), ident["reasons"]
        return "T1 零号不得写成有效身份；五项 budget_not_recorded 明账"


# ---------------------------------------------------------------------------
# T2 重复执行
# ---------------------------------------------------------------------------

def case_T2_repeat_same_conditions():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = M.run_job(out_dir=d, root=_ROOT, job_key="t2", attempt=1, nonce="t2-n1")
        b = M.run_job(out_dir=d, root=_ROOT, job_key="t2", attempt=2, nonce="t2-n2")
        assert a["status"] == b["status"] == "succeeded", (a["status"], b["status"])
        ra, rb = _record(d, a), _record(d, b)
        assert _anchor(ra) == _anchor(rb), "同条件两次执行的基础条件身份必须相同"
        assert a["instance_token"] != b["instance_token"], "两次实际执行实例必须不同"
        assert a["record_path"] != b["record_path"]
        assert Path(a["record_path"]).is_file() and Path(b["record_path"]).is_file(), \
            "不许因条件相同而覆盖/合并两次运行"
        assert a["run_id"] is None and b["run_id"] is None
        st = _states(d)
        assert len(st) == 2 and {x["attempt"] for x in st} == {1, 2}
        assert all(x["status"] == "succeeded" and not x["identity_valid"] for x in st)
        return "T2 同条件两次：锚点相同、实例不同、两记录并存、不合并"


def case_T2_bound_event_sha_differs():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = M.run_job(out_dir=d, root=_ROOT, job_key="t2b", attempt=1, nonce="t2b-n1")
        b = M.run_job(out_dir=d, root=_ROOT, job_key="t2b", attempt=2, nonce="t2b-n2")
        assert a["capture_raw_event_sha256"] == b["capture_raw_event_sha256"], \
            "同参数捕获事件本身不含实例标识（已知缺口，不伪装成已解决）"
        assert a["bound_event_sha256"] != b["bound_event_sha256"], \
            "绑定层必须让两次执行的事件指纹不同，否则 check_reuse 会误拒"
        return "T2 明账：捕获层无实例标识；区分由 runner 绑定层承担"


def case_T2_resume_keeps_old_record():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = _run(d, tag="t2-resume")
        before = Path(a["record_path"]).read_bytes()
        b = M.resume(out_dir=d, job_key="t2-resume", root=_ROOT, fault="none")
        assert b["attempt"] == 2 and b["status"] == "succeeded", b
        assert b["record_path"] != a["record_path"]
        assert Path(a["record_path"]).read_bytes() == before, "旧记录不得被覆盖"
        assert Path(b["record_path"]).is_file()
        assert {x["attempt"] for x in _states(d)} == {1, 2}
        return "T2 恢复：新 attempt 生成新文件，旧记录逐位不变"


# ---------------------------------------------------------------------------
# T3 失败与恢复
# ---------------------------------------------------------------------------

def case_T3_sampler_error():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t3-sampler", fault="sampler_error")
        assert r["status"] == "failed" and r["execution_status"] == "sampler_error"
        assert r["attestation_status"] == "absent"
        assert r["record_written"] and r["reload_consistent"]
        assert r["run_id"] is None
        st = _states(d)[0]
        assert st["status"] == "failed" and not st["identity_valid"], \
            "失败任务不得误报为成功"
        return "T3 采样异常：记录保留、状态 failed、不误报成功"


def case_T3_interrupted():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t3-interrupt", fault="interrupted")
        assert r["status"] == "failed" and r["execution_status"] == "interrupted"
        assert r["record_written"] and r["run_id"] is None
        assert _states(d)[0]["status"] == "failed"
        return "T3 拟合中断：如实落盘 failed，不产生有效身份"


def case_T3_evidence_missing():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t3-evidence", fault="evidence_missing")
        assert r["status"] == "succeeded" and r["reload_consistent"], r
        assert r["attestation_status"] == "absent"
        assert r["run_id"] is None and r["workflow_id"] is None
        rec = _record(d, r)
        assert rec[E.BUDGET_EVIDENCE_FIELD] is None
        assert rec[E.ATTESTATION_FIELD] is None
        return "T3 证据缺失：不自动补全，缺证据照常落盘 unconfirmed"


def case_T3_persist_failure():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        r = _run(d, tag="t3-persist", fault="persist_failure")
        assert r["status"] == "failed" and r["record_written"] is False, r
        assert "persist_error" in r
        st = _states(d)[0]
        assert st["status"] == "failed", "落盘失败不得标成功"
        audit = (d / "audit.jsonl").read_text(encoding="utf-8")
        assert "persist_failed" in audit
        return "T3 落盘失败：无半条成功记录、审计留痕、状态 failed"


def case_T3_duplicate_submit_rejected():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = M.run_job(out_dir=d, root=_ROOT, job_key="t3-dup", attempt=1,
                      nonce="t3-dup-n")
        before = Path(a["record_path"]).read_bytes()
        b = M.run_job(out_dir=d, root=_ROOT, job_key="t3-dup", attempt=1,
                      nonce="t3-dup-n")
        assert b["status"] == "rejected" and "重复提交" in b["reject_reason"], b
        assert Path(a["record_path"]).read_bytes() == before, "重复提交不得覆盖"
        st = _states(d)
        assert [x["status"] for x in st] == ["succeeded", "rejected"], st
        return "T3 同一任务重复提交：拒，旧记录逐位不变"


def case_T3_restart_cross_process():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = _run(d, tag="t3-restart")
        before = Path(a["record_path"]).read_bytes()
        code = (
            "import json,sys; sys.path.insert(0,r'" + str(_HERE) + "'); "
            "sys.path.insert(0,r'" + str(_ROOT) + "'); "
            "import minimal_runner as M; "
            "r=M.resume(out_dir=r'" + str(d) + "',job_key='t3-restart',root=r'" +
            str(_ROOT) + "'); "
            "print(json.dumps({k:r.get(k) for k in "
            "('attempt','status','record_path','record_written','checks')},"
            "ensure_ascii=False,default=str))"
        )
        out = subprocess.check_output([sys.executable, "-c", code],
                                      cwd=str(_ROOT)).decode("utf-8").strip()
        b = json.loads(out.splitlines()[-1])
        assert b["attempt"] == 2 and b["status"] == "succeeded", b
        assert b["record_written"] and b["record_path"] != a["record_path"]
        assert Path(a["record_path"]).read_bytes() == before
        assert len(_states(d)) == 2
        return "T3 Runner 重启：跨进程读状态、新 attempt 恢复、旧记录保留"


def case_T3_cross_run_event_reuse_rejected():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        a = _run(d, tag="t3-reuse")
        rec = _record(d, a)
        forged = copy.deepcopy(rec)
        forged[E.EXECUTION_INSTANCE_FIELD]["nonce"] = "another-run"
        try:
            H.check_reuse(forged, d / "audit.jsonl")
        except ValueError as exc:
            assert "已使用过" in str(exc) or "另一次执行" in str(exc), str(exc)
            return "T3 跨运行复用：A 的完整记录交 B 次执行被拒"
        raise AssertionError("A 次捕获事件被 B 次执行复用却未被拒")


# ---------------------------------------------------------------------------
# 冻结禁令（Gate 2C §4 不允许的扩展）
# ---------------------------------------------------------------------------

def _raises(fn, needle: str) -> None:
    try:
        fn()
    except (ValueError, TypeError) as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(f"应被拒：{needle}")


def case_freeze_extra_candidate_rejected():
    with tempfile.TemporaryDirectory() as td:
        _raises(lambda: _run(Path(td), tag="f-cand",
                             candidate_id="nested:layers3"), "不在冻结清单")
        return "禁令：运行时临时增加候选被拒"


def case_freeze_budget_override_rejected():
    with tempfile.TemporaryDirectory() as td:
        _raises(lambda: _run(Path(td), tag="f-budget",
                             capture_cfg={"num_chains": 4}), "禁止运行时调整")
        return "禁令：运行时调整采样预算 / 构造参数被拒"


def case_freeze_site_outside_rejected():
    with tempfile.TemporaryDirectory() as td:
        _raises(lambda: _run(Path(td), tag="f-site",
                             site="nested_fit.run"), "不在冻结路径")
        return "禁令：外推到未冻结站点被拒"


def case_freeze_unknown_fault_rejected():
    with tempfile.TemporaryDirectory() as td:
        _raises(lambda: _run(Path(td), tag="f-fault",
                             fault="auto_retry"), "fault 必须是")
        return "禁令：自动重试等未冻结行为被拒"


def case_freeze_manifest_tamper_rejected():
    saved = M.FROZEN_MANIFEST["data"]["sha256"]
    try:
        M.FROZEN_MANIFEST["data"]["sha256"] = "0" * 64
        exc = _last_error(M.verify_manifest)
        assert exc is not None and "冻结数据文件 sha" in str(exc), exc
    finally:
        M.FROZEN_MANIFEST["data"]["sha256"] = saved
    assert M.verify_manifest()["verified"]
    return "禁令：篡改冻结数据 sha ⇒ 清单校验直接失败（无乐观降级）"


def _last_error(fn):
    try:
        fn()
    except Exception as exc:                    # noqa: BLE001
        return exc
    return None


CASES = [
    ("T1 单次执行链全闭合", case_T1_single_execution),
    ("T1 记录字段与事件绑定可复算", case_T1_record_fields),
    ("T1 不产生拟合结果", case_T1_no_fit_result),
    ("T1 零号不得写成有效身份", case_T1_zero_id_not_marked_valid),
    ("T2 同条件两次独立执行", case_T2_repeat_same_conditions),
    ("T2 绑定事件指纹区分两次执行", case_T2_bound_event_sha_differs),
    ("T2 恢复保留旧记录", case_T2_resume_keeps_old_record),
    ("T3 采样异常", case_T3_sampler_error),
    ("T3 拟合中断", case_T3_interrupted),
    ("T3 证据缺失", case_T3_evidence_missing),
    ("T3 结果写入失败", case_T3_persist_failure),
    ("T3 同一任务重复提交", case_T3_duplicate_submit_rejected),
    ("T3 Runner 重启跨进程恢复", case_T3_restart_cross_process),
    ("T3 跨运行复用被拒", case_T3_cross_run_event_reuse_rejected),
    ("禁令 临时增加候选", case_freeze_extra_candidate_rejected),
    ("禁令 运行时调整预算", case_freeze_budget_override_rejected),
    ("禁令 外推未冻结站点", case_freeze_site_outside_rejected),
    ("禁令 未冻结的自动行为", case_freeze_unknown_fault_rejected),
    ("禁令 篡改冻结清单", case_freeze_manifest_tamper_rejected),
]


def main() -> int:
    ok = 0
    for i, (label, fn) in enumerate(CASES, 1):
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {i:02d}/{len(CASES)} {label}：{msg}")
        except Exception as exc:                # noqa: BLE001
            print(f"[FAIL] {i:02d}/{len(CASES)} {label}："
                  f"{type(exc).__name__}: {exc}")
    print(f"\n试运行验收：{ok}/{len(CASES)} 通过"
          + ("" if ok == len(CASES) else "  ⛔ 有 FAIL"))
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
