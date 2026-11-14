# -*- coding: utf-8 -*-
"""test_patch_bundle.py —— 补丁包契约的反例矩阵（纯函数 15 格 + 真实 Git 往返 7 格）。

测的是**能不能声称"工作树可完整重建"**，不是"打包脚本跑通了"。
每一格都对着一种"看起来成功、其实恢复不出来"的情形：

· 只打 tracked diff（未跟踪/ignored 一个没进）——最常见的假成功；
· 文件少了却说"打包了大部分"——本轮 P0，必须是硬失败；
· bundle 被改过一个字符——静态校验必须报 `corrupt`，不能靠重建时偶然发现；
· 路径穿越/大小写冲突/盘符——写入侧的安全边界；
· 同一工作树生成两次——必须逐字节相同，否则摘要没有意义。

I/O 格子在临时目录里现建一个**真 Git 仓库**（不是 mock）：只要 fake 掉 Git，
"patch 能不能 apply""diff 沉默时怎么办"就全测不到了。

    python test_patch_bundle.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import patch_bundle as PB          # noqa: E402
import patch_bundle_io as PIO      # noqa: E402
import exec_identity as EI         # noqa: E402
import capture_contract as CC      # noqa: E402

_GIT_ID = ("-c", "user.name=pb_test", "-c", "user.email=pb@test", "-c",
           "core.autocrlf=false", "-c", "core.filemode=false", "-c",
           "commit.gpgsign=false")


class Skip(Exception):
    """当前平台做不到的格子：显式跳过，不算通过。"""


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _tmp(prefix: str = "pb_test_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def _gitq(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *_GIT_ID, *args], cwd=str(root),
                          capture_output=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} 失败："
                             f"{proc.stderr.decode('utf-8', 'replace')[:300]}")
    return proc.stdout.decode("utf-8", "replace")


def _repo(files: dict[str, str] | None = None, *, ignore: str = "out/\n") -> Path:
    r = _tmp("pb_repo_")
    (r / ".gitignore").write_text(ignore, encoding="utf-8", newline="")
    base = files if files is not None else {"a.py": "A = 1\n", "b.py": "B = 2\n",
                                            "sub/c.py": "C = 3\n"}
    for rel, text in base.items():
        p = r / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
    _gitq(r, "init", "-q", "-b", "main", ".")
    _gitq(r, "add", "-A")
    _gitq(r, "commit", "-qm", "base")
    return r


def _cleanup(*paths: Path) -> None:
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


def _decl(*paths: str) -> list[dict]:
    return [{"path": p, "role": "module"} for p in paths]


def _must_raise(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (PB.BundleError, PIO.BundleIOError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError(f"应当报错，实际没有：{getattr(fn, '__name__', fn)}")


def _blob(path: str, text: str, *, why: str = "untracked", mode=PB.MODE_FILE) -> dict:
    data = text.encode("utf-8")
    return PB.make_file_record(path, mode, PB.content_sha256(data), source="blob",
                               why=why, content=data)


def _base(path: str, text: str) -> dict:
    return PB.make_file_record(path, PB.MODE_FILE,
                               PB.content_sha256(text.encode("utf-8")), source="base")


def _bundle(**over) -> dict:
    kw = dict(base={"commit": "0" * 40},
              declaration=_decl("a.py", "b.py"),
              files=[_base("a.py", "A = 1\n"), _blob("b.py", "B = 2\n")],
              patches=[], capture={"tool": "test"})
    kw.update(over)
    return PB.assemble(**kw)


# ---------------------------------------------------------------------------
# 纯函数格
# ---------------------------------------------------------------------------

def case_P1_blob_roundtrip():
    b = _bundle()
    v = PB.verify(b)
    assert v["verdict"] == "complete", v
    rec = [f for f in b["files"] if f["path"] == "b.py"][0]
    assert PB.decode_blob(rec) == b"B = 2\n"
    assert rec["bundle_digest_kind"] == "path_mode_content", rec["bundle_digest_kind"]
    return "blob 内容可原样取回，且摘要覆盖路径+mode+内容"


def case_P2_no_content_kind():
    b = _bundle()
    rec = [f for f in b["files"] if f["path"] == "a.py"][0]
    assert rec["bundle_digest_kind"] == "path_mode_only", rec["bundle_digest_kind"]
    assert "content_b64" not in rec
    return "base/patch 记录不带内容，摘要显式降为 path_mode_only（不冒充验过内容）"


def case_P3_mode_and_path_in_digest():
    txt = "print(1)\n"
    d644 = PB.file_digest("x.py", PB.MODE_FILE, txt.encode())
    d755 = PB.file_digest("x.py", PB.MODE_EXEC, txt.encode())
    d_other = PB.file_digest("y.py", PB.MODE_FILE, txt.encode())
    assert len({d644, d755, d_other}) == 3
    assert PB.file_digest("x.py", "644", txt.encode()) == d644
    return "同内容不同 mode / 不同路径 ⇒ 三个不同摘要（内容相同不算同一个文件）"


def case_P4_missing_declared_is_incomplete():
    b = PB.assemble(base={"commit": "0" * 40},
                    declaration=_decl("a.py", "b.py", "c.py"),
                    files=[_base("a.py", "A = 1\n"), _blob("b.py", "B = 2\n")])
    v = PB.verify(b)
    assert v["verdict"] == "incomplete", v["verdict"]
    assert v["missing"] == ["c.py"], v["missing"]
    assert v["reasons"][0]["code"] == "declared_missing"
    assert [i["path"] for i in PB.tier_items(b)] == ["a.py", "b.py"]
    return "少一个声明文件 ⇒ incomplete（不是 complete，也不是 warn）；身份项少一项"


def case_P5_tamper_blob_content():
    b = _bundle()
    t = json.loads(json.dumps(b))
    t["files"][1]["content_b64"] = "QiA9IDkK"
    v = PB.verify(t)
    assert v["verdict"] == "corrupt", v["verdict"]
    codes = {r["code"] for r in v["reasons"]}
    assert {"tree_digest_mismatch", "file_digest_mismatch"} <= codes, codes
    return "改 blob 内容 ⇒ 两道都报（文件摘要 + 整体摘要）"


def case_P6_tamper_any_field():
    b = _bundle()
    for mut in ("capture", "counts", "base", "declaration", "missing"):
        t = json.loads(json.dumps(b))
        if mut == "capture":
            t["capture"]["tool"] = "other"
        elif mut == "counts":
            t["counts"]["files"] = 99
        elif mut == "base":
            t["base"]["commit"] = "1" * 40
        elif mut == "declaration":
            t["declaration"][0]["compute_role"] = "plot_only"
        else:
            t["missing"] = ["a.py"]
        v = PB.verify(t)
        assert v["verdict"] == "corrupt", (mut, v["verdict"])
    return "改任何一个顶层字段都会撞上 tree_digest（含 capture / counts / base）"


def case_P7_tamper_patch():
    patch = PB.make_patch_record("unstaged", "diff --git a/a.py b/a.py\n", paths=["a.py"])
    b = _bundle(patches=[patch],
                files=[_base("a.py", "A = 1\n"), _blob("b.py", "B = 2\n")])
    assert PB.verify(b)["verdict"] == "complete"
    t = json.loads(json.dumps(b))
    t["patches"][0]["text"] += "+# evil\n"
    codes = {r["code"] for r in PB.verify(t)["reasons"]}
    assert "patch_digest_mismatch" in codes, codes
    return "patch 正文被改动 ⇒ patch_digest_mismatch（不只看 tree_digest）"


def case_P8_undeclared_file_refused():
    msg = _must_raise(PB.assemble, base={"commit": "0" * 40},
                      declaration=_decl("a.py"),
                      files=[_base("a.py", "A\n"), _blob("b.py", "B\n")])
    assert "b.py" in msg
    b = _bundle()
    t = json.loads(json.dumps(b))
    t["files"].append(_blob("extra.py", "E\n"))
    v = PB.verify(t)
    assert v["verdict"] == "corrupt", v["verdict"]
    assert any(r["code"] == "file_not_declared" for r in v["reasons"])
    return "非声明文件不许进 bundle：assemble 拒绝，verify 也报 file_not_declared"


def case_P9_bad_paths_refused():
    bad = ["/abs.py", "a\\b.py", "a/../b.py", "./a.py", "a//b.py", "", "CON.py",
           "a.py ", "D:/x.py", "a/./b.py", "a.py\x00"]
    for p in bad:
        _must_raise(PB.check_rel_path, p)
    _must_raise(PB.assemble, base={"commit": "0" * 40},
                declaration=_decl("a.py", "a.py"), files=[])
    _must_raise(PB.assemble, base={"commit": "0" * 40},
                declaration=_decl("A.py", "a.py"), files=[])
    assert PB.check_rel_path("sub/bm_top.py") == "sub/bm_top.py"
    return "穿越/绝对/反斜杠/盘符/保留名/重复/只差大小写 全部拒绝"


def case_P10_bridge_with_capture_contract():
    """规范化下来的项目路径必须能过 bundle 的写入侧检查（不是第二套规则）。"""
    root = _tmp("pb_bridge_")
    try:
        (root / "sub").mkdir()
        (root / "sub" / "m.py").write_text("# x\n", encoding="utf-8")
        raw = str(root / "sub" / "m.py")
        r = CC.normalize_path(raw, root)
        assert r["state"] == "project", r
        assert PB.check_rel_path(r["path"]) == r["path"]
        assert PB.check_rel_path("sub/m.py") == r["path"]
        return "capture_contract 归一出的 D:/… 路径与 bundle 的写入侧检查同解"
    finally:
        _cleanup(root)


def case_P11_determinism():
    b1 = _bundle()
    b2 = PB.assemble(base={"commit": "0" * 40},
                     declaration=list(reversed(_decl("a.py", "b.py"))),
                     files=[_blob("b.py", "B = 2\n"), _base("a.py", "A = 1\n")],
                     patches=[], capture={"tool": "test"})
    assert PB.bundle_bytes(b1) == PB.bundle_bytes(b2), "同状态不同入参顺序必须同字节"
    b3 = PB.assemble(base={"commit": "0" * 40},
                     declaration=_decl("a.py", "b.py"),
                     files=[_base("a.py", "A = 1\n"), _blob("b.py", "B = 3\n")],
                     patches=[], capture={"tool": "test"})
    assert b3["tree_digest"] != b1["tree_digest"], "差一个字节就必须换摘要"
    assert json.loads(PB.bundle_bytes(b1).decode())["tree_digest"] == b1["tree_digest"]
    return "顺序无关、内容敏感；落盘再读回摘要不变"


def case_P12_mode_normalization():
    assert PB.normalize_mode(0o644) == PB.MODE_FILE
    assert PB.normalize_mode(0o755) == PB.MODE_EXEC
    assert PB.normalize_mode("755") == PB.MODE_EXEC
    _must_raise(PB.normalize_mode, "120000")
    _must_raise(PB.normalize_mode, 0o120000)
    _must_raise(PB.normalize_mode, 0o160000)
    _must_raise(PB.normalize_mode, True)
    return "mode 归一为 644/755；符号链接与 submodule 的 mode 直接报错"


def case_P13_compare_digests():
    a = {"x": "1", "y": "2"}
    assert PB.compare_digests(a, dict(a))["match"]
    r = PB.compare_digests(a, {"x": "1", "y": "3", "z": "4"})
    assert not r["match"]
    assert r["mismatched"] == [{"path": "y", "expected": "2", "actual": "3"}], r
    assert r["extra"] == ["z"] and r["missing"] == [], r
    assert not PB.compare_digests(a, {"x": "1"})["match"]
    return "逐项比较：多一个/少一个/值不同 都不算 match"


def case_P14_parse_garbage():
    _must_raise(PB.parse_bundle, b"\x00\x01not json")
    _must_raise(PB.parse_bundle, b"[1,2,3]")
    _must_raise(PB.parse_bundle, b'{"a":NaN}')
    return "非法 JSON / 非对象顶层 ⇒ BundleError（不是静默当成空 bundle）"


def case_P15_tier_bridge():
    b = PB.assemble(
        base={"commit": "0" * 40},
        declaration=[{"path": "n.py", "role": "module"},
                     {"path": "p.py", "role": "module", "compute_role": "plot_only"},
                     {"path": "w.py", "role": "module", "workflow_scope": False}],
        files=[_base("n.py", "N\n"), _base("p.py", "P\n"), _base("w.py", "W\n")])
    items = PB.tier_items(b)
    tiers = {L: EI.source_tree_sha256(items, layer=L)
             for L in ("declared", "numeric", "workflow")}
    assert len(set(tiers.values())) == 3, tiers
    b2 = json.loads(json.dumps(b))
    w = [f for f in b2["files"] if f["path"] == "w.py"][0]
    w["sha256"] = "f" * 64
    b2["tree_digest"] = PB.manifest_digest(b2)
    items2 = PB.tier_items(b2)
    t2 = {L: EI.source_tree_sha256(items2, layer=L) for L in tiers}
    assert t2["declared"] != tiers["declared"] and t2["numeric"] != tiers["numeric"]
    assert t2["workflow"] == tiers["workflow"], "w.py 不在 workflow 层，改它不该动 workflow"
    return "tier_items 直接喂 exec_identity：三层互不相等，改 n.py 只动 declared/numeric"


# ---------------------------------------------------------------------------
# 真实 Git 往返格
# ---------------------------------------------------------------------------

def _roundtrip(root: Path, decl: list[dict], wd: Path | None = None) -> dict:
    wd = wd or _tmp("pb_rt_")
    return PIO.roundtrip(root, declaration=decl, workdir=wd)


def case_G1_clean_roundtrip():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        rep = _roundtrip(r, _decl("a.py", "b.py", "sub/c.py"), wd)
        rb = rep["rebuild"]
        assert rep["complete"], rep["verdict"]
        assert rb["files"]["compare"]["match"] and rb["tiers"]["compare"]["match"]
        srcs = {f["source"] for f in rep["bundle"]["files"]}
        assert srcs == {"base"}, srcs
        assert not rep["bundle"]["patches"]
        return (f"干净树：19→3 个文件全部 source=base，重建 {rb['n_extracted']} 个文件后"
                "逐文件 sha256 与三层摘要全同")
    finally:
        _cleanup(r, wd)


def case_G2_dirty_roundtrip():
    """staged + unstaged + 未跟踪 + ignored，四类同时存在。"""
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        (r / "a.py").write_text("A = 10  # unstaged\n", encoding="utf-8", newline="")
        (r / "b.py").write_text("B = 20  # staged\n", encoding="utf-8", newline="")
        _gitq(r, "add", "b.py")
        (r / "sub" / "c.py").write_text("C = 30  # staged+unstaged\n",
                                        encoding="utf-8", newline="")
        _gitq(r, "add", "sub/c.py")
        (r / "sub" / "c.py").write_text("C = 31  # 再改一次\n",
                                        encoding="utf-8", newline="")
        (r / "new.py").write_text("NEW = 1  # untracked\n", encoding="utf-8", newline="")
        (r / "out").mkdir(exist_ok=True)
        (r / "out" / "k.json").write_text('{"k": 1}\n', encoding="utf-8", newline="")
        decl = _decl("a.py", "b.py", "sub/c.py", "new.py", "out/k.json")
        rep = _roundtrip(r, decl, wd)
        b = rep["bundle"]
        by = {f["path"]: f for f in b["files"]}
        assert by["a.py"]["source"] == "patch" and by["a.py"]["patch_ref"] == "unstaged", by["a.py"]
        assert by["b.py"]["source"] == "patch" and by["b.py"]["patch_ref"] == "staged", by["b.py"]
        assert by["sub/c.py"]["patch_ref"] == "staged+unstaged", by["sub/c.py"]
        assert by["new.py"]["source"] == "blob" and by["new.py"]["why_blob"] == "untracked"
        assert by["out/k.json"]["why_blob"] == "ignored", by["out/k.json"]
        assert rep["complete"], rep["verdict"]
        assert rep["rebuild"]["tiers"]["compare"]["match"]
        return ("staged/unstaged/未跟踪/ignored 四类同时重建成功；"
                "`sub/c.py` 两段 patch 都认领（staged+unstaged）")
    finally:
        _cleanup(r, wd)


def case_G3_deleted_declared_refuses():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        (r / "b.py").unlink()
        rep = _roundtrip(r, _decl("a.py", "b.py"), wd)
        assert not rep["complete"]
        assert rep["structural"]["verdict"] == "incomplete", rep["structural"]["verdict"]
        assert rep["structural"]["missing"] == ["b.py"], rep["structural"]["missing"]
        assert rep["rebuild"]["verdict"] == "not_rebuilt", rep["rebuild"]["verdict"]
        return "删掉一个声明文件 ⇒ incomplete 且**不重建**（不能拿'打了大部分'冒充完整）"
    finally:
        _cleanup(r, wd)


def case_G4_tampered_bundle_on_disk():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        rep = _roundtrip(r, _decl("a.py", "b.py"), wd)
        path = wd / "b.json"
        path.write_bytes(PB.bundle_bytes(rep["bundle"]))
        assert PB.verify(PB.parse_bundle(path.read_bytes()))["verdict"] == "complete"
        t = json.loads(path.read_text(encoding="utf-8"))
        rec = [f for f in t["files"] if f["source"] == "blob"]
        target = rec[0] if rec else t["files"][0]
        target["sha256"] = "0" * 64
        path.write_text(json.dumps(t), encoding="utf-8")
        v = PB.verify(PB.parse_bundle(path.read_bytes()))
        assert v["verdict"] == "corrupt", v["verdict"]
        codes = {x["code"] for x in v["reasons"]}
        assert "tree_digest_mismatch" in codes, codes
        return "落盘后被改一个 sha256 ⇒ corrupt（完整性不只靠内存里的对象）"
    finally:
        _cleanup(r, wd)


def case_G5_refuse_nonempty_dest():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        rep = _roundtrip(r, _decl("a.py"), wd)
        dest = Path(rep["rebuild"]["dest"])
        assert dest.exists()
        _must_raise(PIO.rebuild, rep["bundle"], dest=dest, repo=r)
        return "重建目标已存在内容 ⇒ 拒绝（不覆盖，避免'看起来验证过'）"
    finally:
        _cleanup(r, wd)


def case_G6_numeric_drift_is_visible():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        decl = _decl("a.py", "b.py")
        rep1 = _roundtrip(r, decl, wd / "one" if False else _tmp("pb_rt1_"))
        t1 = rep1["rebuild"]["tiers"]["expected"]
        (r / "a.py").write_text("A = 999\n", encoding="utf-8", newline="")
        rep2 = _roundtrip(r, decl, _tmp("pb_rt2_"))
        t2 = rep2["rebuild"]["tiers"]["expected"]
        assert t1["numeric"] != t2["numeric"], (t1, t2)
        assert t1["declared"] != t2["declared"]
        assert PB.compare_digests(t1, t2)["match"] is False
        return "改一个 numeric 文件 ⇒ 三层摘要立刻不同（bundle 跟工作树走，不跟 commit 走）"
    finally:
        _cleanup(r, wd)


def case_G7_same_state_same_bytes():
    r = _repo()
    wd = _tmp("pb_rt_")
    try:
        (r / "a.py").write_text("A = 7\n", encoding="utf-8", newline="")
        (r / "out").mkdir(exist_ok=True)
        (r / "out" / "k.json").write_text("{}\n", encoding="utf-8", newline="")
        decl = _decl("a.py", "b.py", "out/k.json")
        b1 = PIO.collect(r, declaration=decl)
        b2 = PIO.collect(r, declaration=decl)
        assert PB.bundle_bytes(b1) == PB.bundle_bytes(b2)
        return "同一工作树连采两次 ⇒ 逐字节相同（含 blob/patch/digest）"
    finally:
        _cleanup(r, wd)


def case_G8_diff_silent_escrow():
    """diff 沉默但字节不同 ⇒ 必须兜底存内容，而不是"没在 diff 里就算没改"。

    **这一格是故障注入，不是真复现**：真实触发要靠 clean/smudge filter 或
    `core.autocrlf` 的归一，而那两者在本仓库配置下（`autocrlf=false`、
    无 `.gitattributes`）造不出来。这里把 `collect_patches` 换成"永远返回空"，
    等价于"Git 对这份改动什么都没报"——采集侧此时唯一的正确行为是**不猜**
    （不能因为 diff 没有就说没改），而是把内容装进 bundle 并写明理由。
    """
    r = _repo()
    orig = PIO.collect_patches
    wd = _tmp("pb_rt_")
    try:
        (r / "a.py").write_text("A = 5  # 改了但 diff 假装没看见\n",
                                encoding="utf-8", newline="")
        PIO.collect_patches = lambda root, paths: []
        b = PIO.collect(r, declaration=_decl("a.py", "b.py"))
        rec = [f for f in b["files"] if f["path"] == "a.py"][0]
        assert rec["source"] == "blob", rec["source"]
        assert rec["why_blob"] == "diff_silent_but_bytes_differ", rec.get("why_blob")
        assert b["capture"]["notes"][0]["path"] == "a.py", b["capture"]["notes"]
        rep = PIO.rebuild(b, dest=wd / "rebuilt", repo=r)
        assert rep["verdict"] == "matched", rep["verdict"]
        return "diff 沉默 ⇒ 兜底存内容（why=diff_silent_but_bytes_differ），重建仍 matched"
    finally:
        PIO.collect_patches = orig
        _cleanup(r, wd)


def case_P16_recomputed_tree_digest():
    """改内容**并把整体摘要重算一遍**：只剩每文件摘要能抓住。

    这一格存在的理由：真实场景里"改 bundle"的人往往也会把摘要重算（工具、脚本、
    或者干脆是另一个版本的打包器）。只有一个总摘要时，这种改法完全看不出来。
    """
    b = _bundle()
    t = json.loads(json.dumps(b))
    t["files"][1]["content_b64"] = "QiA9IDkK"          # "B = 9\n"
    t["tree_digest"] = PB.manifest_digest(t)           # 顺手把总摘要改成自洽的
    v = PB.verify(t)
    assert v["verdict"] == "corrupt", v["verdict"]
    codes = {r["code"] for r in v["reasons"]}
    assert codes == {"file_digest_mismatch"}, codes
    return "重算 tree_digest 也挡不住：每文件摘要仍然报 file_digest_mismatch"


def case_X1_not_imported_by_production():
    """身份/采集/打包这三层**不许被生产模块 import**（本层的存在前提）。

    `patch_bundle_io` 会 import `exec_identity`（要算三层摘要）；如果反过来生产拟合
    代码 import 了这里，`run_id` 的语义就会和证据工具链缠在一起——那时"改一行采集"
    可能悄悄改变一次拟合的身份。这条护栏很便宜，但一旦破了很难回头。
    """
    mine = {"patch_bundle", "patch_bundle_io", "capture_contract", "exec_identity"}
    offenders = []
    for f in sorted(_ROOT.glob("*.py")):
        if f.stem in mine or f.name.startswith("test_"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in mine:
            if f"import {m}" in text or f"from {m}" in text:
                offenders.append(f"{f.name} → {m}")
    assert not offenders, offenders
    return "生产模块一个都没 import 身份/采集/打包三层（只有测试与 docs 的 harness 会）"


CASES = [
    ("P1 blob 往返", case_P1_blob_roundtrip),
    ("P2 无内容档标注", case_P2_no_content_kind),
    ("P3 mode/路径进摘要", case_P3_mode_and_path_in_digest),
    ("P4 缺声明文件", case_P4_missing_declared_is_incomplete),
    ("P5 改 blob 内容", case_P5_tamper_blob_content),
    ("P6 改任意字段", case_P6_tamper_any_field),
    ("P7 改 patch 正文", case_P7_tamper_patch),
    ("P8 非声明文件", case_P8_undeclared_file_refused),
    ("P9 非法路径", case_P9_bad_paths_refused),
    ("P10 与采集契约同解", case_P10_bridge_with_capture_contract),
    ("P11 确定性", case_P11_determinism),
    ("P12 mode 归一", case_P12_mode_normalization),
    ("P13 摘要比较", case_P13_compare_digests),
    ("P14 非法 JSON", case_P14_parse_garbage),
    ("P15 身份三层桥接", case_P15_tier_bridge),
    ("P16 重算总摘要", case_P16_recomputed_tree_digest),
    ("G1 干净树往返", case_G1_clean_roundtrip),
    ("G2 脏树往返", case_G2_dirty_roundtrip),
    ("G3 缺文件拒绝重建", case_G3_deleted_declared_refuses),
    ("G4 落盘后被改", case_G4_tampered_bundle_on_disk),
    ("G5 非空目标拒绝", case_G5_refuse_nonempty_dest),
    ("G6 数值漂移可见", case_G6_numeric_drift_is_visible),
    ("G7 同状态同字节", case_G7_same_state_same_bytes),
    ("G8 diff 沉默兜底", case_G8_diff_silent_escrow),
    ("X1 生产层不引用", case_X1_not_imported_by_production),
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
