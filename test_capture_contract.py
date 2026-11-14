# -*- coding: utf-8 -*-
"""test_capture_contract.py —— 采集契约（C-1 harness / C-2 路径 / C-3 采样事件）反例矩阵。

测的是**语义**：每一格问"这个写法该不该被当成同一个身份""这次采样该记在谁头上"。
身份算法本身不动——`exec_identity.py` 一行没改，所以这里还要证明**桥接后仍然
`recorded`**（I 组）：harness 声明进去之后既不再制造假的 `undeclared_loaded`，
也不会把编排代码的变化带进 `run_id`。

写这一组的原因是上一步的实测（`docs/round_2026-11-14/SNAPSHOT_PROBE.md`）暴露了
四处**不报错**的错：runner 自身被当成未声明、`D:/...` 与 `bm_top.py` 永不相等、
单槽位快照被后来的采样覆盖、钩子状态跨运行泄漏。四条都不会崩，只会让记录
看起来完整却是错的，所以每一条都要有一个会失败的格子。

    python test_capture_contract.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import capture_contract as C          # noqa: E402
import exec_identity as E             # noqa: E402
import test_exec_identity as TE       # noqa: E402  # 复用记录夹具，避免第二份会漂的副本


class Skip(Exception):
    """当前平台/权限做不到的格子：显式跳过，不算通过也不假装通过。"""


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def _root() -> Path:
    r = Path(tempfile.mkdtemp(prefix="cap_test_"))
    (r / "sub").mkdir()
    (r / "bm_top.py").write_text("# x\n", encoding="utf-8")
    (r / "main.py").write_text("# x\n", encoding="utf-8")
    (r / "core.py").write_text("# x\n", encoding="utf-8")
    (r / "sub" / "m.py").write_text("# x\n", encoding="utf-8")
    return r


def _cleanup(r: Path) -> None:
    shutil.rmtree(r, ignore_errors=True)


def _harness(path: str = "run_identity.py", **over) -> dict:
    kw = dict(sha256=TE._sha("harness"), compute_role="cli_only",
              why_not_numeric="只做采集与编排，不进阻抗模型计算",
              evidence="I6：把它当 numeric 会改变 run_id")
    kw.update(over)
    return C.harness_item(path, **kw)


def _must_raise(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (ValueError, RuntimeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"应当报错，实际没有：{fn.__name__}")


def _bridge(files, declared, root) -> dict:
    """采集层输出 → 完整记录（把 `classify_runtime` 的结果贴进 `code`）。"""
    rec = TE._rec(items=list(declared))
    cr = C.classify_runtime(files, declared, root=root)
    code = rec["code"]
    code["runtime_loaded_modules"] = cr["runtime_loaded_modules"]
    code["external_modules"] = [{"resolved": e["resolved"], "state": e["state"]}
                                for e in cr["external_modules"]]
    code["undeclared_modules"] = cr["undeclared_modules"]
    code["declared_not_loaded"] = cr["declared_not_loaded"]
    code["source_coverage"] = ("external_path" if cr["external_modules"] else
                               "undeclared_loaded" if cr["undeclared_modules"]
                               else "complete")
    return rec


# ---------------------------------------------------------------------------
# P 组：路径规范化（C-2）
# ---------------------------------------------------------------------------
def case_P1_abs_vs_rel_same_identity():
    r = _root()
    try:
        a = C.normalize_path(str(r / "bm_top.py"), r)
        b = C.normalize_path("bm_top.py", r)
        assert a["state"] == b["state"] == "project", (a["state"], b["state"])
        assert a["path"] == b["path"] == "bm_top.py", (a["path"], b["path"])
        return ("绝对 Windows 路径与相对路径归一到同一身份（P-1 的核心："
                "更早的实现会让两者永不相等，于是全量 undeclared）")
    finally:
        _cleanup(r)


def case_P2_other_spellings():
    r = _root()
    try:
        want = "bm_top.py"
        got = {s: C.normalize_path(s, r)["path"] for s in
               ("./bm_top.py", "sub/../bm_top.py", "bm_top.py",
                (r / "bm_top.py").as_posix())}
        assert set(got.values()) == {want}, got
        return f"四种写法全部归一为 {want}：{sorted(got)}"
    finally:
        _cleanup(r)


def case_P3_escape_is_external():
    r = _root()
    try:
        a = C.normalize_path("../outside.py", r)
        b = C.normalize_path(str(r.parent / "outside.py"), r)
        assert a["state"] == b["state"] == "external", (a["state"], b["state"])
        assert a["path"] is None and b["path"] is None
        return "越界路径判为 external（不强行映射进声明集），path=None"
    finally:
        _cleanup(r)


def case_P4_symlink_inside():
    r = _root()
    try:
        link = r / "alias.py"
        try:
            os.symlink(r / "bm_top.py", link)
        except (OSError, NotImplementedError) as exc:
            raise Skip(f"本机不能建符号链接：{exc}")
        n = C.normalize_path("alias.py", r)
        assert n["state"] == "project", n
        assert n["path"] == "bm_top.py", n["path"]
        assert n["is_link"] is True, n
        return "仓库内符号链接解析到目标文件，身份是目标（is_link=True）"
    finally:
        _cleanup(r)


def case_P5_symlink_escape():
    r = _root()
    outside = Path(tempfile.mkdtemp(prefix="cap_out_"))
    try:
        (outside / "h.py").write_text("# x\n", encoding="utf-8")
        try:
            os.symlink(outside / "h.py", r / "esc.py")
        except (OSError, NotImplementedError) as exc:
            raise Skip(f"本机不能建符号链接：{exc}")
        n = C.normalize_path("esc.py", r)
        assert n["state"] == "symlink_escape", n
        assert n["path"] is None
        return "字面在仓库内、realpath 指向仓库外 ⇒ symlink_escape（单独一档）"
    finally:
        _cleanup(r)
        shutil.rmtree(outside, ignore_errors=True)


def case_P6_case_only_collision():
    r = _root()
    try:
        # 已存在的文件：realpath 会归一到磁盘上的真实大小写，两种写法是同一个身份
        a = C.normalize_path("bm_top.py", r)
        b = C.normalize_path("BM_TOP.py", r)
        assert a["path"] == b["path"] == "bm_top.py", (a["path"], b["path"])
        # 都不存在的一对名字：只能靠 normcase 判，平台语义不同
        s = C.normalize_set(["Ghost.py", "ghost.py"], r)
        if os.name != "nt":                           # POSIX：两个不同的文件
            assert not s["collisions"] and s["n"] == 2, (s["collisions"], s["n"])
            note = "POSIX 两个独立文件 ⇒ 保留两项"
        else:                                         # Windows：同一个文件
            kinds = [c["kind"] for c in s["collisions"]]
            assert kinds == ["case_collision"], kinds
            assert s["n"] == 1, s["n"]
            note = "Windows 同一文件 ⇒ case_collision，只留一项"
        return f"已存在文件归一到真实大小写；不存在的一对：{note}"
    finally:
        _cleanup(r)


def case_P7_duplicate_strict():
    r = _root()
    try:
        msg = _must_raise(C.normalize_set, ["bm_top.py", "./bm_top.py"], r, strictly=True)
        loose = C.normalize_set(["bm_top.py", "./bm_top.py"], r)
        assert loose["n"] == 1 and loose["collisions"][0]["kind"] == "duplicate"
        return f"声明集严格模式报错（{msg}）；运行时宽松模式只记 duplicate、不丢条目"
    finally:
        _cleanup(r)


def case_P8_invalid_forms():
    r = _root()
    try:
        bad = [C.normalize_path(x, r)["state"] for x in ("", "   ", None, 3)]
        assert bad == ["invalid"] * 4, bad
        root_itself = C.normalize_path(".", r)
        assert root_itself["state"] == "invalid" and root_itself["why"] == "root_itself"
        return "空串/空白/非字符串 ⇒ invalid；仓库根本身 ⇒ invalid(root_itself)"
    finally:
        _cleanup(r)


def case_P9_external_project_copy():
    """仓库外那份**同名**文件才是 `external_path`（site-packages 里的项目副本）。"""
    r = _root()
    outside = Path(tempfile.mkdtemp(prefix="cap_out_"))
    try:
        (outside / "core.py").write_text("# copy\n", encoding="utf-8")
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
                    {"path": "core.py", "role": "module", "sha256": TE._sha("core")}]
        cr = C.classify_runtime({str(r / "main.py"): ["main"],
                                 str(outside / "core.py"): ["core"]},
                                declared, root=r)
        paths = [e["path"] for e in cr["runtime_loaded_modules"]]
        assert paths == ["main.py"], paths
        assert [e["modules"] for e in cr["external_modules"]] == [["core"]], \
            cr["external_modules"]
        assert cr["counts"]["external"] == 1, cr["counts"]
        return "仓库外的同名 core.py ⇒ external_modules（真实加载的那份在仓库外）"
    finally:
        _cleanup(r)
        shutil.rmtree(outside, ignore_errors=True)


def case_P10_unrelated_outside_ignored():
    """仓库外一个**从没声明过**的模块不是 `external_path`，只是被忽略并计数。"""
    r = _root()
    outside = Path(tempfile.mkdtemp(prefix="cap_out_"))
    try:
        (outside / "vendored.py").write_text("# dep\n", encoding="utf-8")
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")}]
        cr = C.classify_runtime({str(r / "main.py"): ["main"],
                                 str(outside / "vendored.py"): ["vendored"]},
                                declared, root=r,
                                tp_roots=(str(outside),))
        assert cr["external_modules"] == [], cr["external_modules"]
        assert cr["counts"]["external"] == 0, cr["counts"]
        assert cr["n_ignored"] >= 1 and cr["n_third_party"] >= 1, cr
        return ("没声明过的外部模块只进 n_ignored（numpy/sitecustomize 属于这一类），"
                "否则每条记录都会带几千个假 external")
    finally:
        _cleanup(r)
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# H 组：harness 归属（C-1）
# ---------------------------------------------------------------------------
def case_H1_harness_refuses_numeric():
    msg = _must_raise(_harness, compute_role="numeric")
    assert "numeric" in msg
    return f"harness 不许 numeric（{msg}）——编排代码的改动不该换 run_id"


def case_H2_harness_requires_reason():
    a = _must_raise(_harness, why_not_numeric="")
    b = _must_raise(_harness, evidence="   ")
    return f"缺理由/缺证据都拒收：{a}｜{b}"


def case_H3_harness_requires_posix_rel():
    msg = _must_raise(_harness, path="D:/eis训练数据/bmdrt/run.py")
    assert "POSIX" in msg
    return f"绝对路径拒收（{msg}）"


def case_H4_harness_item_shape():
    it = _harness()
    assert it["runtime_scope"] == "harness" and it["role"] == "entry"
    assert it["compute_role"] == "cli_only" and it["workflow_scope"] is True
    assert C.is_harness(it) and not C.is_harness({"path": "x.py"})
    return "harness 项字段齐全：runtime_scope=harness / compute_role 非 numeric / workflow_scope 默认 True"


def case_H5_declared_harness_not_undeclared():
    r = _root()
    try:
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
                    {"path": "core.py", "role": "module", "sha256": TE._sha("core")},
                    _harness()]
        cr = C.classify_runtime({str(r / "main.py"): ["main"],
                                 str(r / "core.py"): ["core"],
                                 str(r / "run_identity.py"): ["__main__"]},
                                declared, root=r)
        cls = {e["path"]: e["runtime_class"] for e in cr["runtime_loaded_modules"]}
        assert cls == {"main.py": "project", "core.py": "project",
                       "run_identity.py": "harness"}, cls
        assert cr["undeclared_modules"] == []
        assert cr["harness_modules"] == ["run_identity.py"]
        return "已声明的 harness 归 harness 类，不算 undeclared，但**留在** runtime 里可见"
    finally:
        _cleanup(r)


def case_H6_undeclared_harness_still_fails():
    r = _root()
    try:
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")}]
        cr = C.classify_runtime({str(r / "main.py"): ["main"],
                                 str(r / "run_identity.py"): ["__main__"]},
                                declared, root=r)
        assert cr["undeclared_modules"] == ["run_identity.py"], cr["undeclared_modules"]
        cls = {e["path"]: e["runtime_class"] for e in cr["runtime_loaded_modules"]}
        assert cls["run_identity.py"] == "undeclared"
        return "没声明的 harness 照样 undeclared（排除只能靠声明，不能靠文件名过滤）"
    finally:
        _cleanup(r)


def case_H7_invalid_path_not_dropped():
    r = _root()
    try:
        cr = C.classify_runtime({None: ["weird"], str(r / "main.py"): ["main"]},
                                [{"path": "main.py", "role": "entry",
                                  "sha256": TE._sha("main")}], root=r)
        bad = [e for e in cr["runtime_loaded_modules"] if e.get("invalid")]
        assert bad and bad[0]["runtime_class"] == "undeclared", cr
        return "解析不出路径的条目记成 undeclared(invalid)，不消失、不变成 complete"
    finally:
        _cleanup(r)


# ---------------------------------------------------------------------------
# E 组：采样事件归属（C-3）
# ---------------------------------------------------------------------------
def case_E1_two_events_no_overwrite():
    log = C.SamplingEventLog()
    with log.session("R1-a"):
        with log.event(snapshot={"files": {"a.py": ["a"]}}, effective={"n": 1}):
            pass
        with log.event(snapshot={"files": {"a.py": ["a"], "b.py": ["b"]}},
                       effective={"n": 2}):
            pass
    ev = log.events("R1-a")
    assert [e["event_id"] for e in ev] == ["R1-a#0", "R1-a#1"]
    assert len(ev[0]["snapshot"]["files"]) == 1 and len(ev[1]["snapshot"]["files"]) == 2
    assert [e["effective"]["n"] for e in ev] == [1, 2]
    return "两次采样各自独立（首版单槽位会让 event#0 变成第二次的快照）"


def case_E2_sessions_isolated():
    log = C.SamplingEventLog()
    with log.session("R1-a"):
        with log.event(snapshot={"files": {}}, effective={"n": 1}):
            pass
    with log.session("R1-b"):
        with log.event(snapshot={"files": {}}, effective={"n": 2}):
            pass
    assert [e["event_id"] for e in log.events("R1-a")] == ["R1-a#0"]
    assert [e["event_id"] for e in log.events("R1-b")] == ["R1-b#0"]
    assert [e["owner"] for e in log.events()] == ["R1-a", "R1-b"]
    return "两个会话的事件不串（Profile C 的两族对拍各认自己的采样）"


def case_E3_nested_session():
    log = C.SamplingEventLog()
    with log.session("parent"):
        with log.event(snapshot={"files": {}}, effective={"n": 0}):
            pass
        with log.session("child"):
            assert log.current() == "child"
            with log.event(snapshot={"files": {}}, effective={"n": 1}):
                pass
        assert log.current() == "parent"
        with log.event(snapshot={"files": {}}, effective={"n": 2}):
            pass
    assert [e["event_id"] for e in log.events("parent")] == ["parent#0", "parent#1"]
    assert [e["event_id"] for e in log.events("child")] == ["child#0"]
    return "嵌套会话正确（父的两次采样不会被 with_voigt 的子采样顶掉）"


def case_E4_unclosed_is_not_complete():
    log = C.SamplingEventLog()
    with log.session("R1-a"):
        with log.event(snapshot={"files": {}}, effective={}):
            pass
        log._sessions["R1-a"]["events"].append(          # 模拟进程被杀：事件没关
            {"event_id": "R1-a#1", "owner": "R1-a", "idx": 1, "seq": 99,
             "started": True, "status": "running", "error": None,
             "snapshot": {}, "effective": {}})
    s = log.summary("R1-a")
    assert s["n_events"] == 2 and s["n_ok"] == 1 and s["n_running"] == 1, s
    assert [e["event_id"] for e in log.complete("R1-a")] == ["R1-a#0"]
    return "未关闭的事件是 running，complete() 不返回它（没跑完不许当证据）"


def case_E5_exception_marks_error():
    log = C.SamplingEventLog()
    try:
        with log.session("R1-a"):
            with log.event(snapshot={"files": {}}, effective={}):
                raise RuntimeError("采样炸了")
    except RuntimeError:
        pass
    s = log.summary("R1-a")
    assert s["n_error"] == 1 and s["n_ok"] == 0, s
    assert log.events("R1-a")[0]["error"].startswith("RuntimeError")
    assert log.complete("R1-a") == []
    return "异常采样标 error 且原样抛出；它不进 complete()"


def case_E6_no_session_refuses():
    log = C.SamplingEventLog()
    msg = _must_raise(lambda: log.event(snapshot={}, effective={}).__enter__())
    assert "归属" in msg
    return f"没有会话时采样事件直接报错（{msg}），不落到'最近一个'"


def case_E7_no_silent_reuse():
    log = C.SamplingEventLog()
    with log.session("R1-a"):
        with log.event(snapshot={"files": {}}, effective={}):
            pass
    msg = _must_raise(lambda: log.session("R1-a").__enter__())
    assert "reset" in msg
    log.reset()
    with log.session("R1-a"):
        with log.event(snapshot={"files": {}}, effective={}):
            pass
    assert [e["event_id"] for e in log.events("R1-a")] == ["R1-a#0"]
    return f"同一 owner 复用必须 reset（{msg}）；reset 后序号从头"


def case_E8_seq_monotonic():
    log = C.SamplingEventLog()
    for o in ("A", "B"):
        with log.session(o):
            for _ in range(2):
                with log.event(snapshot={"files": {}}, effective={}):
                    pass
    seqs = [e["seq"] for e in log.events()]
    assert seqs == sorted(set(seqs)) and seqs[0] == 0, seqs
    return f"全局 seq 单调且不重复：{seqs}（局部 idx 每会话从 0 起）"


# ---------------------------------------------------------------------------
# I 组：桥接到身份层（证明"声明即可，不用改算法"）
# ---------------------------------------------------------------------------
def case_I1_harness_not_in_numeric_digest():
    base = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
            {"path": "core.py", "role": "module", "sha256": TE._sha("core")}]
    with_h = base + [_harness()]
    d0, d1 = E.declared_source_tree_sha256(base), E.declared_source_tree_sha256(with_h)
    n0, n1 = E.numeric_source_tree_sha256(base), E.numeric_source_tree_sha256(with_h)
    w0, w1 = E.workflow_source_tree_sha256(base), E.workflow_source_tree_sha256(with_h)
    assert d0 != d1, "declared 层必须变：改 runner 要能被看见"
    assert n0 == n1, "numeric 层必须不变：编排代码不该换 run_id"
    assert w0 != w1, "workflow 层必须变：harness 影响编排"
    return "harness 进 declared 与 workflow、不进 numeric（三层各司其职）"


def case_I2_declared_harness_records():
    r = _root()
    try:
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
                    {"path": "core.py", "role": "module", "sha256": TE._sha("core")},
                    _harness()]
        rec = _bridge({str(r / "main.py"): ["main"], str(r / "core.py"): ["core"],
                       str(r / "run_identity.py"): ["__main__"]}, declared, r)
        try:
            v = E.evaluate(rec)
        except TypeError as exc:                     # 夹具字段不足时如实报
            raise Skip(f"夹具不满足 v1.4 记录形状：{exc}")
        assert v["status"] == "recorded", (v["status"], v["reasons"])
        assert not any(x.startswith("undeclared_loaded") for x in v["reasons"]), v
        return f"harness 声明后整条记录 recorded（run_id={v['run_id'][:12]}…），无 undeclared"
    finally:
        _cleanup(r)


def case_I3_undeclared_harness_derives():
    r = _root()
    try:
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
                    {"path": "core.py", "role": "module", "sha256": TE._sha("core")}]
        rec = _bridge({str(r / "main.py"): ["main"], str(r / "core.py"): ["core"],
                       str(r / "run_identity.py"): ["__main__"]}, declared, r)
        v = E.evaluate(rec)
        assert v["status"] == "derived", (v["status"], v["reasons"])
        assert any(x.startswith("undeclared_loaded:run_identity.py")
                   for x in v["reasons"]), v["reasons"]
        return "同一份代码不声明 harness ⇒ derived + undeclared_loaded（这就是修复前的默认命运）"
    finally:
        _cleanup(r)


def case_I4_external_derives():
    r = _root()
    outside = Path(tempfile.mkdtemp(prefix="cap_out_"))
    try:
        (outside / "main.py").write_text("# x\n", encoding="utf-8")
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")}]
        rec = _bridge({str(r / "main.py"): ["main"],
                       str(outside / "main.py"): ["main"]}, declared, r)
        v = E.evaluate(rec)
        assert v["status"] == "derived", (v["status"], v["reasons"])
        assert "external_path" in v["reasons"], v["reasons"]
        return "仓库外模块 ⇒ derived + external_path（明确降级，不强行映射）"
    finally:
        _cleanup(r)
        shutil.rmtree(outside, ignore_errors=True)


def case_I5_runtime_labels_not_hashed():
    r = _root()
    try:
        declared = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")}]
        rec = _bridge({str(r / "main.py"): ["main"]}, declared, r)
        a = E.evaluate(rec)
        stripped = dict(rec)
        stripped["code"] = dict(rec["code"])
        stripped["code"]["runtime_loaded_modules"] = [
            {"path": e["path"]} for e in rec["code"]["runtime_loaded_modules"]]
        b = E.evaluate(stripped)
        assert a["run_id"] == b["run_id"] and a["workflow_id"] == b["workflow_id"]
        assert a["status"] == b["status"] == "recorded", (a["status"], b["status"])
        return "runtime_class/role/modules 这些新标签不进哈希（同号同状态）"
    finally:
        _cleanup(r)


def case_I6_numeric_harness_would_change_id():
    """I6 是 H1 的证据：把 harness 当 numeric，`run_id` 真的会变。"""
    base = [{"path": "main.py", "role": "entry", "sha256": TE._sha("main")},
            {"path": "core.py", "role": "module", "sha256": TE._sha("core")}]
    bad = base + [{"path": "run_identity.py", "role": "entry",
                   "sha256": TE._sha("harness")}]
    r0 = TE._ids(TE._rec(items=base))[0]
    r1 = TE._ids(TE._rec(items=bad))[0]
    assert r0 != r1, "runner 被当 numeric 时 run_id 必须变——否则 H1 的理由不成立"
    return f"harness 当 numeric ⇒ run_id {r0[:10]}… → {r1[:10]}…（换号且不报错）"


# ---------------------------------------------------------------------------
CASES = [
    ("P1 绝对/相对同一身份", case_P1_abs_vs_rel_same_identity),
    ("P2 四种写法归一", case_P2_other_spellings),
    ("P3 越界判外部", case_P3_escape_is_external),
    ("P4 链接指向仓库内", case_P4_symlink_inside),
    ("P5 链接逃出仓库", case_P5_symlink_escape),
    ("P6 只差大小写", case_P6_case_only_collision),
    ("P7 重复路径", case_P7_duplicate_strict),
    ("P8 非法写法", case_P8_invalid_forms),
    ("P9 仓库外同名文件", case_P9_external_project_copy),
    ("P10 无关外部只计数", case_P10_unrelated_outside_ignored),
    ("H1 harness 拒绝 numeric", case_H1_harness_refuses_numeric),
    ("H2 harness 必须给理由", case_H2_harness_requires_reason),
    ("H3 harness 必须相对路径", case_H3_harness_requires_posix_rel),
    ("H4 harness 字段形状", case_H4_harness_item_shape),
    ("H5 声明后不算未声明", case_H5_declared_harness_not_undeclared),
    ("H6 未声明照样失败", case_H6_undeclared_harness_still_fails),
    ("H7 解析不出的不丢", case_H7_invalid_path_not_dropped),
    ("E1 两次采样不覆盖", case_E1_two_events_no_overwrite),
    ("E2 会话隔离", case_E2_sessions_isolated),
    ("E3 嵌套会话", case_E3_nested_session),
    ("E4 未关闭不算完整", case_E4_unclosed_is_not_complete),
    ("E5 异常标 error", case_E5_exception_marks_error),
    ("E6 无会话拒收", case_E6_no_session_refuses),
    ("E7 复用必须 reset", case_E7_no_silent_reuse),
    ("E8 seq 单调", case_E8_seq_monotonic),
    ("I1 harness 不进 numeric", case_I1_harness_not_in_numeric_digest),
    ("I2 声明后 recorded", case_I2_declared_harness_records),
    ("I3 未声明则 derived", case_I3_undeclared_harness_derives),
    ("I4 外部则 derived", case_I4_external_derives),
    ("I5 新标签不进哈希", case_I5_runtime_labels_not_hashed),
    ("I6 numeric harness 换号", case_I6_numeric_harness_would_change_id),
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
