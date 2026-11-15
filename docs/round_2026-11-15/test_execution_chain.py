# -*- coding: utf-8 -*-
"""test_execution_chain.py —— F-01/F-02 单次执行链测试（不跑完整 EIS 拟合）。

四种情况：正常结束 / 采样异常 / 证据缺失 / 落盘失败。
三个必留测试：伪造证据 / 跨运行复用 / 落盘后恢复。
外加 F-01 三分（内容完整性 / 来源真实性 / 执行关联性）单独断言、
受控签发正对照、第二站点兼容验证、冻结路径与失败不误报成功。

    python docs/round_2026-11-15/test_execution_chain.py
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exec_identity as E                                   # noqa: E402
import execution_harness as H                               # noqa: E402
import test_exec_identity as TE                             # noqa: E402

_CANDIDATE = "voigt:hn2+rl_parallel"
_CFG_CORE = {"num_chains": 1, "num_warmup": 8, "num_samples": 8,
             "target_accept": 0.9, "max_tree_depth": 10}
_CFG_TOP = {"num_chains": 3, "num_warmup": 12, "num_samples": 24,
            "target_accept": 0.85, "max_tree_depth": 7}
_RESULTS: dict[str, dict] = {}


def _conditions(data_sha: str | None = None) -> dict:
    """冻结的执行计划给出的条件字段；预算/凭证由 harness 覆盖。"""
    r = TE._rec()
    if data_sha is not None:
        r["data"]["sha256"] = data_sha
    return r


def _run(tag: str, *, scenario: str | None = None,
         site: str = E.FROZEN_PRIMARY_SITE,
         cfg: dict | None = None, nonce: str | None = None) -> dict:
    sc = scenario or tag
    with tempfile.TemporaryDirectory() as d:
        out = H.run_once(conditions=_conditions(), site=site,
                         cfg=cfg or _CFG_CORE, data_sha256=TE._DATA,
                         candidate_id=_CANDIDATE,
                         nonce=nonce or f"nonce-{tag}",
                         out_dir=Path(d) / tag, scenario=sc)
        # 结果里的路径在临时目录外已失效，但所有断言都在目录内完成
        out["_audit_text"] = (Path(out["audit_path"]).read_text(encoding="utf-8")
                              if Path(out["audit_path"]).is_file() else "")
    _RESULTS[tag] = out
    return out


# ---------------------------------------------------------------------------
# 四种情况
# ---------------------------------------------------------------------------

def case_T1_ok():
    r = _run("ok")
    assert r["event_judged"] is True, "真实捕获事件没判定"
    assert r["record_written"] is True
    assert r["execution_status"] == "ok"
    assert r["attestation_status"] == "verified", r["attestation_status"]
    # 来源未闭合 ⇒ 如实零号，这是正确的保守行为
    assert r["identity"]["status"] == "unconfirmed", r["identity"]
    assert r["identity"]["run_id"] is None and r["identity"]["workflow_id"] is None
    c = r["reload_checks"]
    assert c["fingerprints_consistent"] is True and c["evidence_sha256_matches"] is True
    assert c["identity_matches_persisted"] is True
    assert '"outcome": "persisted"' in r["_audit_text"]
    return "T1 正常结束：真实捕获→签发→落盘→重读一致；unconfirmed 零号（正确结果）"


def case_T2_sampler_error():
    r = _run("sampler_error")
    assert r["execution_status"] == "sampler_error"
    assert r["attestation_status"] == "absent", "采样异常时不得签发凭证"
    assert r["identity"]["run_id"] is None
    assert r["record_written"] is True, "失败也必须留完整记录"
    assert r["reload_checks"]["fingerprints_consistent"] is True
    assert '"execution_status": "sampler_error"' in r["_audit_text"]
    return "T2 采样异常：不签发、照常落盘、unconfirmed 零号、审计留痕"


def case_T3_evidence_missing():
    r = _run("evidence_missing")
    assert r["attestation_status"] == "absent"
    assert r["identity"]["run_id"] is None
    assert r["record_written"] is True
    assert r["reload_checks"]["fingerprints_consistent"] is True
    # 证据块缺失，但捕获到的观测值不能丢（只是不进身份）
    with tempfile.TemporaryDirectory() as d:
        out = H.run_once(conditions=_conditions(), site=E.FROZEN_PRIMARY_SITE,
                         cfg=_CFG_CORE, data_sha256=TE._DATA,
                         candidate_id=_CANDIDATE, nonce="nonce-T3",
                         out_dir=Path(d) / "t3", scenario="evidence_missing")
        rec, _ = H.load_and_verify(out["path"])
    assert rec[E.BUDGET_EVIDENCE_FIELD] is None
    assert isinstance(rec["budget_observations"], dict) and \
        len(rec["budget_observations"]) == len(E.BUDGET_KEYS)
    return "T3 证据缺失：不自动补、观测值保留、落盘 unconfirmed 零号"


def case_T4_persist_failure():
    r = _run("persist_failure")
    assert r["persist_error"], "没触发注入的落盘失败"
    assert r["record_written"] is False, "落盘失败却留下了记录文件"
    assert '"outcome": "persist_failed"' in r["_audit_text"]
    assert r["identity"]["run_id"] is None, "落盘失败不该产生可追溯成功身份"
    return "T4 落盘失败：无成功记录文件、审计记 persist_failed、不误报成功"


# ---------------------------------------------------------------------------
# 三个必留测试
# ---------------------------------------------------------------------------

def case_forged_evidence():
    """自造一份内部完全一致的预算证据 + 自造凭证 ⇒ 不能获得可信身份。"""
    bud, evi = TE._budget_cells(), None
    evi = TE._budget_evidence(bud)
    inst = {"data_sha256": TE._DATA, "candidate_id": _CANDIDATE,
            "site": E.FROZEN_PRIMARY_SITE, "nonce": "forged"}
    fake = {"version": E.ATTESTATION_VERSION, "issued_by": E.ATTESTATION_ISSUER,
            "event_sha256": evi["event_sha256"],
            "budget_sha256": evi["evidence_sha256"],
            "instance_token": E.execution_instance_token(inst),
            "signature": "0" * 64}
    rec = TE._rec()
    rec[E.ATTESTATION_FIELD] = fake
    rec[E.EXECUTION_INSTANCE_FIELD] = inst
    out = E.evaluate(rec)
    assert out["status"] == "unconfirmed", out
    assert out["run_id"] is None and out["workflow_id"] is None
    assert "attestation_signature_invalid" in out["reasons"], out["reasons"]
    # 连凭证都不带也不行
    bare = TE._rec()
    bare[E.ATTESTATION_FIELD] = None
    o2 = E.evaluate(bare)
    assert o2["run_id"] is None and "attestation_missing" in o2["reasons"]
    return "F-01a 伪造证据：自洽字典 + 自造签名 / 无凭证，全部零号"


def case_cross_run_reuse():
    """A 次执行的捕获事件与凭证交给 B 次执行 ⇒ 必须拒绝错误关联。"""
    # ① 凭证层面的错关联：B 换了数据，仍贴 A 的凭证
    a = TE._rec()
    b = TE._rec()
    b["data"]["sha256"] = TE._sha("data/b.csv")
    b[E.ATTESTATION_FIELD] = copy.deepcopy(a[E.ATTESTATION_FIELD])
    b[E.EXECUTION_INSTANCE_FIELD] = copy.deepcopy(a[E.EXECUTION_INSTANCE_FIELD])
    out = E.evaluate(b)
    assert out["run_id"] is None, "跨运行凭证竟发了号"
    assert "attestation_data_binding_mismatch" in out["reasons"], out["reasons"]

    # ② 事件层面的错关联：同一捕获事件被第二次执行使用
    with tempfile.TemporaryDirectory() as d:
        audit = Path(d) / "audit.jsonl"
        cond = _conditions()
        event = H.capture_real_event(E.FROZEN_PRIMARY_SITE, cfg=_CFG_CORE)
        instA = H.build_execution_instance(data_sha256=TE._DATA,
                                          candidate_id=_CANDIDATE, nonce="A")
        recA = H.finalize(H.assemble_record(conditions=cond, event=event,
                                            instance=instA))
        H.persist(recA, Path(d) / "a.json", audit_path=audit)
        instB = H.build_execution_instance(data_sha256=TE._DATA,
                                          candidate_id=_CANDIDATE, nonce="B")
        recB = H.assemble_record(conditions=cond, event=event, instance=instB)
        try:
            H.check_reuse(recB, audit)
            raise AssertionError("A 的事件被 B 复用却没被拒")
        except ValueError as exc:
            assert "捕获事件" in str(exc), str(exc)
        # ③ 同一执行实例重复提交
        recA2 = H.assemble_record(conditions=cond, event=event, instance=instA)
        try:
            H.check_reuse(recA2, audit)
            raise AssertionError("重复提交同一实例却没被拒")
        except ValueError as exc:
            assert "执行实例" in str(exc), str(exc)
    return "F-01b 跨运行复用：凭证错关联 / 事件被复用 / 实例重复提交，三重拒绝"


def case_reload_restore():
    """执行完成后重新读取身份记录，关键字段、指纹、身份计算结果一致。"""
    with tempfile.TemporaryDirectory() as d:
        out = H.run_once(conditions=_conditions(), site=E.FROZEN_PRIMARY_SITE,
                         cfg=_CFG_CORE, data_sha256=TE._DATA,
                         candidate_id=_CANDIDATE, nonce="nonce-reload",
                         out_dir=Path(d) / "reload", scenario="ok")
        rec, checks = H.load_and_verify(out["path"])
        assert checks["key_fields_present"] and checks["fingerprints_consistent"]
        assert checks["evidence_sha256_matches"] is True
        assert checks["identity_matches_persisted"] is True
        assert rec[E.ATTESTATION_FIELD]["signature"] == \
            E.attestation_signature(rec[E.ATTESTATION_FIELD])
        assert rec[E.EXECUTION_INSTANCE_FIELD]["nonce"] == "nonce-reload"
        # **跨进程**重读：纯函数 + 项目密钥 ⇒ 新解释器里身份计算逐位一致
        code = ("import json,sys;sys.path.insert(0,sys.argv[1]);"
                "import exec_identity as E;"
                "r=json.load(open(sys.argv[2],encoding='utf-8'));"
                "print(json.dumps(E.evaluate(r),sort_keys=True))")
        raw = subprocess.check_output(
            [sys.executable, "-c", code, str(_ROOT), str(out["path"])],
            cwd=str(_ROOT)).decode("utf-8")
        reloaded = json.loads(raw)
        assert reloaded == rec["identity"], (reloaded, rec["identity"])
        assert reloaded["run_id"] is None
    return "F-01c 落盘后恢复：关键字段/两个指纹/身份计算一致，且跨进程重算一致"


# ---------------------------------------------------------------------------
# F-01 三分 / 正对照 / 第二站点 / 冻结路径
# ---------------------------------------------------------------------------

def case_three_parts_integrity():
    r = TE._rec()
    r["budget_params"]["n_warmup"]["value"] = 9999     # 不更新指纹
    out = E.evaluate(r)
    assert out["run_id"] is None
    assert any("budget_evidence_sha_mismatch" in x or
               "attestation_budget_sha_mismatch" in x for x in out["reasons"]), \
        out["reasons"]
    return "F-01 内容完整性：改预算值不重算指纹 ⇒ 零号"


def case_three_parts_authenticity():
    r = TE._rec()
    r[E.ATTESTATION_FIELD]["signature"] = "f" * 64
    out = E.evaluate(r)
    assert out["run_id"] is None and "attestation_signature_invalid" in out["reasons"]
    r2 = TE._rec()
    r2[E.ATTESTATION_FIELD]["issued_by"] = "caller/self"
    assert E.evaluate(r2)["run_id"] is None
    return "F-01 来源真实性：签名/签发者被改 ⇒ 零号（只认受控接口）"


def case_three_parts_binding():
    r = TE._rec()
    r[E.EXECUTION_INSTANCE_FIELD]["nonce"] = "another-run"
    out = E.evaluate(r)
    assert out["run_id"] is None and \
        "attestation_instance_token_mismatch" in out["reasons"], out["reasons"]
    r2 = TE._rec()
    r2[E.EXECUTION_INSTANCE_FIELD]["site"] = "nested_fit.run"
    assert E.evaluate(r2)["run_id"] is None
    return "F-01 执行关联性：nonce/站点被改 ⇒ 零号"


def case_controlled_issuance_positive():
    """正对照：受控接口签发的合规记录仍能发 R2-（门槛不是一律拒）。"""
    r = TE._rec()
    assert r[E.ATTESTATION_FIELD]["signature"] == \
        E.attestation_signature(r[E.ATTESTATION_FIELD])
    out = E.evaluate(r)
    assert out["status"] == "recorded", out["reasons"]
    assert out["run_id"].startswith("R2-") and out["workflow_id"].startswith("W2-")
    return "受控签发正对照：合规记录仍发 R2-/W2-，门槛只拒未签发记录"


def case_version_layer_separation():
    """Gate 2C 版本裁决回归锁：门槛世代独立，号的计算世代不因门槛变化而升。"""
    r = TE._rec()
    a = r[E.ATTESTATION_FIELD]
    body = json.loads(E._attestation_body(a))
    assert body["version"] == E.ATTESTATION_VERSION, "门槛版本没进签名消息"
    # 改门槛版本 ⇒ 版本非法 + 签名失配 ⇒ 零号（比顶层版本字段更强）
    r2 = TE._rec()
    r2[E.ATTESTATION_FIELD]["version"] = "9"
    o = E.evaluate(r2)
    assert o["run_id"] is None and "attestation_version_invalid" in o["reasons"]
    assert "attestation_signature_invalid" in o["reasons"], o["reasons"]
    # 两层都不进哈希载荷
    for payload in (E.run_id_payload(r), E.workflow_payload(r)):
        assert E.ATTESTATION_FIELD not in payload
        assert E.EXECUTION_INSTANCE_FIELD not in payload
    assert set(E.run_id_payload(r)) == {
        "v", "config_keys_version", "rerun_attempt", "rerun_lineage",
        "candidate_id", "data", "rng_plan", "budget", "config", "code", "env"}, \
        "载荷形状变了 ⇒ 必须按 D-4 升契约版本并全换号"
    assert E.IDENTITY_CONTRACT_VERSION == "1.5" and E.RUN_ID_CONFIG_KEYS_VERSION == 2
    assert E.evaluate(r)["run_id"].startswith("R2-")
    return "版本裁决：门槛世代被签名覆盖，载荷形状未变 ⇒ 保持 1.5/2（不做表面换号）"


def case_compat_site():
    r = _run("compat", scenario="ok", site="bm_top.fit_top", cfg=_CFG_TOP)
    assert r["event_judged"] is True
    assert r["record_written"] is True
    assert r["identity"]["status"] == "unconfirmed" and r["identity"]["run_id"] is None
    assert r["reload_checks"]["fingerprints_consistent"] is True
    return "第二站点 bm_top.fit_top 兼容验证：3链12/24 真实捕获、落盘、零号"


def case_frozen_path():
    assert H.FROZEN_PATH["primary_site"] == "bm_core.fit"
    assert H.FROZEN_PATH["compat_sites"] == ["bm_top.fit_top"]
    assert len(H.FROZEN_PATH["excluded"]) == 4
    for site in ("nested_fit.run", "m35_cond_n2.py:116", "bm_select.retry"):
        try:
            H.build_execution_instance(data_sha256=TE._DATA,
                                       candidate_id=_CANDIDATE,
                                       nonce="n", site=site)
            raise AssertionError(f"非冻结站点 {site} 竟被接受")
        except ValueError:
            pass
    return "冻结路径：主路径 bm_core.fit、兼容站点 1 个、4 类入口排除且不纳入"


def case_failures_not_success():
    for s in ("sampler_error", "evidence_missing", "persist_failure"):
        r = _RESULTS.get(s) or _run(s)
        assert r["identity"]["run_id"] is None, f"{s} 竟发了号"
        assert r["identity"]["workflow_id"] is None, f"{s} 竟发了工作流号"
    assert _RESULTS["sampler_error"]["execution_status"] == "sampler_error"
    assert _RESULTS["evidence_missing"]["attestation_status"] == "absent"
    assert _RESULTS["persist_failure"]["record_written"] is False
    return "失败不误报成功：三种失败/缺证据场景两号皆 None，状态如实落盘"


CASES = [
    ("T1 正常结束", case_T1_ok),
    ("T2 采样异常", case_T2_sampler_error),
    ("T3 证据缺失", case_T3_evidence_missing),
    ("T4 落盘失败", case_T4_persist_failure),
    ("F-01a 伪造证据", case_forged_evidence),
    ("F-01b 跨运行复用", case_cross_run_reuse),
    ("F-01c 落盘后恢复", case_reload_restore),
    ("F-01 内容完整性", case_three_parts_integrity),
    ("F-01 来源真实性", case_three_parts_authenticity),
    ("F-01 执行关联性", case_three_parts_binding),
    ("受控签发正对照", case_controlled_issuance_positive),
    ("版本裁决回归锁", case_version_layer_separation),
    ("第二站点兼容验证", case_compat_site),
    ("冻结路径与排除清单", case_frozen_path),
    ("失败不误报成功", case_failures_not_success),
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
    print(f"\n{ok}/{len(CASES)} 通过" +
          (f"；未通过：{'、'.join(n for n, _ in bad)}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
