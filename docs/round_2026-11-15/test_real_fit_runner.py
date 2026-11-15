# -*- coding: utf-8 -*-
"""真实 EIS 单次拟合的试运行验收（runner real_fit 模式）。

只跑一次冻结的 20/20 拟合（模块内缓存复用），另用一次 tiny 拟合验证
`SamplingEventLog` 多次采样逐次事件、一次注入异常验证失败不误报成功。
其余用例是纯函数 / 记录级复核，不再产生拟合结果。

    python docs/round_2026-11-15/test_real_fit_runner.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_contract as C                       # noqa: E402
import exec_identity as E                          # noqa: E402
import execution_harness as H                      # noqa: E402
import minimal_runner as M                         # noqa: E402
import real_fit_runner as RF                       # noqa: E402

_TINY = {"K": 10, "n_warmup": 8, "n_samples": 8, "num_chains": 1,
         "target_accept": 0.9, "max_tree_depth": 10, "seed": 0,
         "progress": False, "extra_fields": ("diverging", "num_steps")}
_CACHE: dict = {}
_TMP: str | None = None


def _once() -> dict:
    if _CACHE:
        return _CACHE
    global _TMP
    _TMP = tempfile.mkdtemp(prefix="gate3-realfit-")
    d = Path(_TMP)
    r = M.run_job(out_dir=d, root=_ROOT, job_key="real-eis-test",
                  nonce="real-eis-test-nonce", mode="real_fit")
    rec = json.loads(Path(r["record_path"]).read_text(encoding="utf-8"))
    _CACHE.update(result=r, record=rec, out_dir=str(d),
                  bytes=Path(r["record_path"]).stat().st_size)
    return _CACHE


# ---------------------------------------------------------------------------
# R1–R5：单次真实拟合与留证链
# ---------------------------------------------------------------------------

def case_R1_single_real_fit():
    c = _once()
    r, rec = c["result"], c["record"]
    assert r["mode"] == "real_fit" and r["fit_ok"] is True, r
    assert r["status"] == "succeeded" and r["reload_consistent"], r
    assert r["n_draws"] == M.FROZEN_MANIFEST["fit_cfg"]["n_samples"]
    assert r["posterior_sha256"] and len(r["posterior_sha256"]) == 64
    assert r["record_written"] and c["bytes"] < 200_000, \
        f"记录未压缩：{c['bytes']} 字节"
    assert r["run_id"] is None and r["workflow_id"] is None, r["identity"]
    assert r["identity_status"] == "unconfirmed" and not r["identity_valid"]
    assert rec["runner"]["mode"] == "real_fit"
    assert "execution_evidence" in rec
    assert all(rec["budget_params"][k]["source"] == "unavailable"
               for k in E.BUDGET_KEYS)
    assert rec[E.BUDGET_EVIDENCE_FIELD]["upstream_origin_captured"] is False
    _, checks = H.load_and_verify(Path(r["record_path"]))
    assert checks["fingerprints_consistent"] and checks["identity_matches_persisted"]
    return ("R1 真实单次拟合：20 draws、记录闭合、零号如实落盘、"
            f"{c['bytes']} 字节")


def case_R2_posterior_and_result_sha_recomputable():
    fr = _once()["record"]["execution_evidence"]["fit_result"]
    assert fr["n_draws"] == 20 and len(fr["posterior_keys"]) >= 8
    assert isinstance(fr["posterior_sha256"], str) and len(fr["posterior_sha256"]) == 64
    d = {k: v for k, v in fr.items() if k != "result_sha256"}
    want = hashlib.sha256(E.canonical_json(d).encode("utf-8")).hexdigest()
    assert want == fr["result_sha256"], "落盘后复算结果指纹不一致"
    assert all(isinstance(v, float) for v in fr["posterior_mean"].values())
    return "R2 后验摘要与结果指纹落盘后可逐位复算"


def case_R3_three_points():
    three = _once()["record"]["execution_evidence"]["three_points"]
    assert set(three) == set(RF.POINTS)
    for p in RF.POINTS:
        s = three[p]
        assert s is not None and s["n_total_modules"] > 0
        assert isinstance(s["files"], dict) and s["state_counts"]
        want = hashlib.sha256(E.canonical_json(
            {"files": s["files"], "state_counts": s["state_counts"]})
            .encode("utf-8")).hexdigest()
        assert want == s["sha256"], f"{p} 快照指纹不可复算"
    assert _once()["result"]["three_points_complete"] is True
    return "R3 三时点齐：项目侧文件表 + 状态计数，指纹可复算"


def case_R4_harness_ownership():
    h = _once()["record"]["execution_evidence"]["harness"]
    assert h["numeric_digest_same"] is True, "harness 不该改 numeric 摘要"
    assert h["numeric_digest_with_harness"] == h["numeric_digest_without_harness"]
    assert len(h["harness_paths"]) >= 6
    assert "capture_contract.py" in h["harness_modules"]
    # 如实：本次实测是 undeclared_loaded，不许写成 complete。
    assert h["source_coverage"] in ("complete", "undeclared_loaded", "external_path")
    if h["source_coverage"] == "undeclared_loaded":
        assert h["undeclared_modules"], "判 undeclared 却列不出未声明模块"
    return (f"R4 harness 归属：numeric 摘要逐位相同；"
            f"coverage={h['source_coverage']}（未声明项如实明账）")


def case_R5_sampling_events():
    e = _once()["record"]["execution_evidence"]
    assert e["n_sampling_events"] == 1 and e["n_complete_events"] == 1
    ev = e["sampling_events"][0]
    assert ev["status"] == "ok" and ev["owner"].startswith("real-eis-test")
    eff = ev["effective"]
    cfg = M.FROZEN_MANIFEST["fit_cfg"]
    assert (eff["num_warmup"] == cfg["n_warmup"]
            and eff["num_samples"] == cfg["n_samples"]
            and eff["num_chains"] == cfg["num_chains"]
            and eff["_max_tree_depth"] == cfg["max_tree_depth"]), eff
    return "R5 逐次事件：1 条 ok，入口生效参数与冻结 fit_cfg 一致"


# ---------------------------------------------------------------------------
# R6–R10：失败、复用、禁令与多次采样
# ---------------------------------------------------------------------------

def case_R6_sampler_error_injection():
    with tempfile.TemporaryDirectory() as td:
        r = M.run_job(out_dir=Path(td), root=_ROOT, job_key="real-eis-fail",
                      nonce="real-eis-fail-nonce", mode="real_fit",
                      fault="sampler_error")
        assert r["fit_ok"] is False and r["fit_error"], r
        assert r["n_draws"] is None and r["posterior_sha256"] is None
        assert r["n_complete_events"] == 0, "异常采样不得算完整事件"
        assert r["status"] == "failed" and r["execution_status"] == "sampler_error"
        assert r["record_written"] and r["reload_consistent"]
        assert r["run_id"] is None and r["workflow_id"] is None
        st = M.load_workflow_state(Path(td))[0]
        assert st["status"] == "failed" and st["fit_succeeded"] is False
        return "R6 采样异常注入：无后验结果、事件标 error、failed 不误报成功"


def _raises(fn, needle: str) -> None:
    try:
        fn()
    except (ValueError, TypeError) as exc:
        assert needle in str(exc), str(exc)
        return
    raise AssertionError(f"应被拒：{needle}")


def case_R7_freeze_guards_real_fit():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _raises(lambda: M.run_job(out_dir=d, root=_ROOT, job_key="g1",
                                  mode="real_fit", fit_cfg=_TINY),
                "禁止运行时调整")
        _raises(lambda: M.run_job(out_dir=d, root=_ROOT, job_key="g2",
                                  mode="real_fit", site="bm_top.fit_top"),
                "唯一冻结主路径")
        _raises(lambda: M.run_job(out_dir=d, root=_ROOT, job_key="g3",
                                  mode="auto"), "mode 必须是")
        return "R7 禁令：运行时改 fit_cfg / 换站点 / 未登记模式全拒"


def case_R8_multi_sampling_events_no_overwrite():
    """C-3：两次独立拟合各一条事件，不覆盖；同一 owner 不 reset 必须报错。"""
    a = _once()["record"]["execution_evidence"]
    b = RF.run_real_fit(spectrum=M.FROZEN_MANIFEST["data"]["ref"],
                        fit_cfg=_TINY, owner="multi-test-2")
    assert a["n_complete_events"] == b["n_complete_events"] == 1
    assert a["sampling_events"][0]["event_id"] != b["sampling_events"][0]["event_id"]
    assert b["three_points_complete"] and b["fit_ok"]
    log = C.SamplingEventLog()
    with log.session("dup-owner"):
        pass
    _raises(lambda: _reuse(log), "reset")
    log.reset()
    with log.session("dup-owner"):
        pass
    return "R8 多次采样逐次事件：两轮各 1 条不覆盖；同 owner 不 reset 必拒"


def _reuse(log: C.SamplingEventLog):
    with log.session("dup-owner"):
        pass


def case_R9_cross_run_reuse_rejected():
    c = _once()
    forged = copy.deepcopy(c["record"])
    forged[E.EXECUTION_INSTANCE_FIELD]["nonce"] = "another-run"
    try:
        H.check_reuse(forged, Path(c["out_dir"]) / "audit.jsonl")
    except ValueError as exc:
        assert "已使用过" in str(exc) or "另一次执行" in str(exc)
        return "R9 跨运行复用：A 次完整记录交 B 次执行被拒"
    raise AssertionError("A 次记录被 B 次执行复用却未被拒")


def case_R10_evidence_missing_zero_id():
    rec = copy.deepcopy(_once()["record"])
    rec[E.BUDGET_EVIDENCE_FIELD] = None
    rec[E.ATTESTATION_FIELD] = None
    o = E.evaluate(rec)
    assert o["status"] == "unconfirmed" and o["run_id"] is None \
        and o["workflow_id"] is None, o
    return "R10 真实拟合记录去掉证据块：仍零号，不自动补来源"


CASES = [
    ("R1 单次真实拟合与留证链", case_R1_single_real_fit),
    ("R2 后验与结果指纹可复算", case_R2_posterior_and_result_sha_recomputable),
    ("R3 三时点快照", case_R3_three_points),
    ("R4 harness 归属", case_R4_harness_ownership),
    ("R5 逐次采样事件", case_R5_sampling_events),
    ("R6 采样异常注入", case_R6_sampler_error_injection),
    ("R7 冻结禁令", case_R7_freeze_guards_real_fit),
    ("R8 多次采样逐次事件不覆盖", case_R8_multi_sampling_events_no_overwrite),
    ("R9 跨运行复用被拒", case_R9_cross_run_reuse_rejected),
    ("R10 证据缺失零号", case_R10_evidence_missing_zero_id),
]


def main() -> int:
    ok = 0
    try:
        for i, (label, fn) in enumerate(CASES, 1):
            try:
                msg = fn()
                ok += 1
                print(f"[ok] {i:02d}/{len(CASES)} {label}：{msg}")
            except Exception as exc:            # noqa: BLE001
                print(f"[FAIL] {i:02d}/{len(CASES)} {label}："
                      f"{type(exc).__name__}: {exc}")
    finally:
        if _TMP:
            shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n真实拟合试运行验收：{ok}/{len(CASES)} 通过"
          + ("" if ok == len(CASES) else "  ⛔ 有 FAIL"))
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
