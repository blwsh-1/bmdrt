# -*- coding: utf-8 -*-
"""补丁包的 I/O 层：从真实工作树采集、在独立目录重建。

判定规则全在 `patch_bundle.py`（纯函数、可单测）；这里只负责三件事：
调 Git、读写文件、把结果拼成报告。分成两层是因为**平台细节会漂**：
换行归一、exec 位、符号链接、`git apply` 的上下文匹配，这些都会随时间
和机器变化，而"什么算完整重建"不该跟着变。

采集口径（每一步都能说出为什么）
--------------------------------
**base 必须逐字节比对，不能靠"没在 diff 里就是没改"**。`git diff` 沉默的原因
至少有三种：`core.autocrlf` 归一、clean/smudge filter、`core.fileMode`。
三者都可能让工作树字节与 commit blob 不同而 diff 什么都不报。所以每个声明文件
都比一次 `sha256(磁盘字节) == sha256(commit blob)`，相等才敢标 `source="base"`。

**patch 只覆盖声明集**。`git diff -- <声明路径>`：bundle 声称的范围就是声明集，
把无关路径的修改卷进来，会让 `git apply` 因为 bundle 之外的改动而失败，
也会让"恢复工作树"悄悄带上没声明的代码。

**重建必须在独立目录**。直接在当前工作树上"验证恢复"证明不了 bundle 有恢复能力
（原文件还在，改不动也"看起来对"）。

模式保真度的诚实边界
--------------------
Windows 上 `st_mode` 的 exec 位不可用（`core.filemode=false` 时 git 也不记），
所以重建后的 mode 比较在 Windows 上**标为跳过**并写明原因，而不是拿
"两边都读到 644"冒充验过。身份三层摘要不含 mode，所以这条不影响身份结论。

    python patch_bundle_io.py capture   --root . --profile B --out out/pb/b.json
    python patch_bundle_io.py verify    --bundle out/pb/b.json
    python patch_bundle_io.py rebuild   --bundle out/pb/b.json --dest /tmp/rebuilt
    python patch_bundle_io.py roundtrip --root . --profile B --workdir /tmp/pb_rt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import patch_bundle as PB          # noqa: E402
import exec_identity as EI         # noqa: E402


class BundleIOError(RuntimeError):
    """采集/重建过程中做不到的事（不是"不完整"，是"这次根本没法做"）。"""


#: 每一条 git 调用都带这三个 `-c`：全局配置里 `core.autocrlf` 可能是 `true`
#: （Windows 安装默认），它会让 diff 与工作树字节**看起来一致、实际不同**。
_GIT_CFG = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
            "-c", "core.quotepath=false")

#: 未跟踪文件按"有没有被 ignore 规则命中"分两档，理由要能说清。
_BLOB_WHY = {"untracked": "untracked", "ignored": "ignored",
             "silent": "diff_silent_but_bytes_differ"}


def _git(root: str | os.PathLike, *args: str, text: bool = True,
         check: bool = True) -> Any:
    cmd = ["git", *_GIT_CFG, *args]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, env=env)
    if check and proc.returncode != 0:
        raise BundleIOError(
            f"git {' '.join(args)} 失败（rc={proc.returncode}）："
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:400]}")
    return proc.stdout.decode("utf-8", "surrogateescape") if text else proc.stdout


def repo_root(path: str | os.PathLike) -> str:
    out = _git(path, "rev-parse", "--show-toplevel").strip()
    if not out:
        raise BundleIOError(f"{path} 不在 Git 工作树里")
    return os.path.realpath(out)


def head_commit(root: str | os.PathLike) -> str:
    return _git(root, "rev-parse", "HEAD").strip()


def _in_commit(root: str, commit: str, path: str) -> bool:
    proc = subprocess.run(
        ["git", *_GIT_CFG, "cat-file", "-e", f"{commit}:{path}"],
        cwd=root, capture_output=True)
    return proc.returncode == 0


def _blob(root: str, commit: str, path: str) -> bytes:
    return _git(root, "cat-file", "blob", f"{commit}:{path}", text=False)


def _git_mode(root: str, commit: str, path: str) -> str | None:
    """commit 里这个文件的 git mode（`100644`/`100755`）。"""
    out = _git(root, "ls-tree", commit, "--", path).strip()
    if not out:
        return None
    return out.split()[0]


def _index_mode(root: str, path: str) -> str | None:
    out = _git(root, "ls-files", "-s", "--", path).strip()
    if not out:
        return None
    return out.split()[0]


def _fs_mode(path: Path) -> str:
    """磁盘上的 mode。Windows 上 exec 位不可用，恒为 `100644`——见模块 docstring。"""
    try:
        st = path.lstat()
    except FileNotFoundError:
        return PB.MODE_FILE
    return PB.MODE_EXEC if (st.st_mode & 0o111) else PB.MODE_FILE


def _is_ignored(root: str, path: str) -> bool:
    proc = subprocess.run(["git", *_GIT_CFG, "check-ignore", "-q", "--", path],
                          cwd=root, capture_output=True)
    return proc.returncode == 0


def collect_patches(root: str, paths: Sequence[str]) -> list[dict]:
    """staged 与 unstaged 两段 diff，**限定在声明集内**。

    顺序不能反：`git diff --cached` 是 HEAD→index，`git diff` 是 index→工作树。
    先 staged 后 unstaged 才复现"工作树现在长什么样"；反过来会因为上下文
    对不上而 apply 失败（或更糟：能 apply 但结果不对）。
    """
    spec = ["--", *sorted(set(paths))]
    common = ("--binary", "--no-color", "--no-ext-diff", "--full-index", "--no-renames")
    out: list[dict] = []
    for name, args in (("staged", ("diff", "--cached", *common)),
                       ("unstaged", ("diff", *common))):
        text = _git(root, *args, *spec)
        if not text.strip():
            continue
        touched = _patch_paths(text)
        out.append(PB.make_patch_record(name, text, paths=touched))
    return out


def _patch_paths(text: str) -> list[str]:
    """从 diff 头里取 b/ 侧路径，用来核对"这段 patch 到底碰了哪些声明文件"。"""
    found: set[str] = set()
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" ")
            if len(parts) >= 4:
                b = parts[3]
                if b.startswith("b/"):
                    found.add(b[2:])
        elif line.startswith("+++ b/"):
            found.add(line[6:])
    return sorted(p for p in found if p and p != "/dev/null")


def collect(root: str | os.PathLike, *, declaration: Sequence[Mapping],
            base_commit: str | None = None,
            capture_meta: Mapping[str, Any] | None = None) -> dict:
    """采集一份 bundle。返回的 bundle 已经过 `verify()`，判定写在 `_verify` 里。"""
    rt = repo_root(root)
    commit = base_commit or head_commit(rt)
    decl = PB.make_declaration(declaration)
    if base_commit:
        # 允许从"当前仓库里存在的任意 commit"取基准，但必须真的存在
        proc = subprocess.run(["git", *_GIT_CFG, "cat-file", "-e", f"{commit}^{{commit}}"],
                              cwd=rt, capture_output=True)
        if proc.returncode != 0:
            raise BundleIOError(f"仓库里没有 commit {commit}")

    paths = [d["path"] for d in decl]
    patches = collect_patches(rt, paths)
    pnames = {p["name"]: set(p["paths"]) for p in patches}

    files: list[dict] = []
    notes: list[dict] = []
    for d in decl:
        p = d["path"]
        full = Path(rt) / p
        if not full.exists() or not full.is_file():
            if full.is_symlink():
                raise BundleIOError(
                    f"{p} 是符号链接：bundle 只声称能重建普通文件（链接会指向声明集之外）")
            continue      # 缺文件不进 files ⇒ assemble 记进 missing ⇒ verify 判 incomplete
        content = full.read_bytes()
        sha = PB.content_sha256(content)
        in_base = _in_commit(rt, commit, p)
        gm = _git_mode(rt, commit, p) if in_base else None
        im = _index_mode(rt, p)
        mode = PB.normalize_mode(gm or im or _fs_mode(full))

        if in_base and PB.content_sha256(_blob(rt, commit, p)) == sha and mode == gm:
            files.append(PB.make_file_record(p, mode, sha, source="base"))
            continue
        if in_base:
            refs = [n for n in ("staged", "unstaged") if p in pnames.get(n, set())]
            if refs:
                files.append(PB.make_file_record(p, mode, sha, source="patch",
                                                 patch_ref="+".join(refs)))
                continue
            # 到这里说明：commit 里有、字节不一样、两段 diff 都没提它。
            # 最可能的解释就是"git 认为没变而文件其实变了"（换行/filter）。
            # 不能猜，兜底存内容，并把这个理由记下来——这是本轮最该被看见的一档。
            files.append(PB.make_file_record(
                p, mode, sha, source="blob", content=content,
                why=_BLOB_WHY["silent"]))
            notes.append({"path": p, "note": "diff_silent_but_bytes_differ"})
            continue
        why = _BLOB_WHY["ignored"] if _is_ignored(rt, p) else _BLOB_WHY["untracked"]
        files.append(PB.make_file_record(p, mode, sha, source="blob",
                                         content=content, why=why))

    meta = {"tool": "patch_bundle_io", "tool_version": "1.0",
            "git_cfg": {"core.autocrlf": "false"},
            "notes": notes}
    meta.update(dict(capture_meta or {}))
    # 注意：**不往 bundle 里塞任何额外键**。`tree_digest` 覆盖除自身以外的全部字段，
    # 加一个 `_verify` 就会让摘要对不上；那正是本轮要防的"看起来完整"的错。
    return PB.assemble(base={"commit": commit}, declaration=decl, files=files,
                       patches=patches, capture=meta)


# ---------------------------------------------------------------------------
# 重建
# ---------------------------------------------------------------------------

def _extract_archive(root: str, commit: str, dest: Path) -> int:
    """`git archive` 到 tar，再**自己**解包：不信任成员名，拒绝链接与穿越。"""
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest.parent / (dest.name + ".tar")
    _git(root, "archive", "--format=tar", f"--output={tar_path}", commit)
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith("/") or ".." in name.split("/"):
                raise BundleIOError(f"tar 成员路径不安全：{name!r}")
            if m.issym() or m.islnk():
                raise BundleIOError(f"tar 成员是链接（{name!r}）：不在重建承诺范围内")
            if m.isdir():
                (dest / name).mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                raise BundleIOError(f"tar 成员不是普通文件：{name!r}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                raise BundleIOError(f"tar 成员读不出内容：{name!r}")
            target.write_bytes(src.read())
            if os.name != "nt":
                os.chmod(target, 0o755 if (m.mode & 0o111) else 0o644)
            n += 1
    tar_path.unlink(missing_ok=True)
    return n


def _apply(root: str, dest: Path, patches: Sequence[Mapping]) -> list[dict]:
    out: list[dict] = []
    for rec in patches:
        name = rec["name"]
        pf = dest.parent / f"{dest.name}.{name}.patch"
        pf.write_text(rec["text"], encoding="utf-8", newline="")
        proc = subprocess.run(
            ["git", *_GIT_CFG, "apply", "--binary", "--whitespace=nowarn",
             str(pf)], cwd=str(dest), capture_output=True)
        out.append({"name": name, "rc": proc.returncode,
                    "stderr": proc.stderr.decode("utf-8", "replace").strip()[:300]})
        pf.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise BundleIOError(
                f"patch {name} 应用失败（rc={proc.returncode}）：{out[-1]['stderr']}")
    return out


def _write_blobs(dest: Path, files: Sequence[Mapping]) -> int:
    n = 0
    for rec in files:
        if rec.get("source") != "blob":
            continue
        data = PB.decode_blob(rec)
        target = dest / rec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if os.name != "nt":
            os.chmod(target, 0o755 if rec["mode"] == PB.MODE_EXEC else 0o644)
        n += 1
    return n


def _tiers(items: Iterable[Mapping]) -> dict:
    items = list(items)
    return {
        "declared": EI.source_tree_sha256(items, layer="declared"),
        "numeric": EI.source_tree_sha256(items, layer="numeric"),
        "workflow": EI.source_tree_sha256(items, layer="workflow"),
    }


def rebuild(bundle: Mapping[str, Any], *, dest: str | os.PathLike,
            repo: str | os.PathLike) -> dict:
    """在 `dest`（必须是空/新目录）里从 bundle 重建，并逐项比对。

    返回的报告分两段，**不合并成一个布尔**：
    `files`（每个文件的 sha256/mode）与 `tiers`（身份三层摘要）。
    只在两段都 `match` 时才说 `matched`——"三层相同但少了文件"这种情形
    在三层摘要里看不见（少一个非声明文件不影响，少一个声明文件会让摘要直接变，
    但少一个 *未参与* 某层的文件只影响那一层），所以逐文件比较是必需的。
    """
    d = Path(dest)
    if d.exists() and any(d.iterdir()):
        raise BundleIOError(f"{dest} 不是空目录：重建必须在新目录里做（不覆盖工作树）")
    v = PB.verify(bundle)
    if v["verdict"] != "complete":
        return {"attempted": False, "structural": v, "verdict": "not_rebuilt",
                "why": f"bundle 结构判定为 {v['verdict']}，不重建"}
    base = bundle["base"]["commit"]
    rt = repo_root(repo)

    n_arch = _extract_archive(rt, base, d)
    applied = _apply(rt, d, bundle.get("patches") or [])
    n_blob = _write_blobs(d, bundle.get("files") or [])

    exp_sha, exp_mode, act_sha, act_mode = {}, {}, {}, {}
    absent: list[str] = []
    for rec in bundle["files"]:
        p = rec["path"]
        exp_sha[p] = rec["sha256"]
        exp_mode[p] = rec["mode"]
        f = d / p
        if not f.is_file():
            absent.append(p)
            continue
        data = f.read_bytes()
        act_sha[p] = PB.content_sha256(data)
        act_mode[p] = _fs_mode(f) if os.name != "nt" else PB.MODE_FILE

    digests = PB.compare_digests(exp_sha, act_sha)
    modes = PB.compare_digests(exp_mode, act_mode)
    mode_check = "checked" if os.name != "nt" else "skipped_windows"
    if mode_check != "checked":
        modes["checked"] = False
    tiers_exp = _tiers(PB.tier_items(bundle))
    tiers_act = _tiers([dict(it, sha256=act_sha[it["path"]])
                        for it in PB.tier_items(bundle) if it["path"] in act_sha])
    tier_cmp = PB.compare_digests(tiers_exp, tiers_act)

    files_ok = digests["match"] and not absent
    modes_ok = True if mode_check != "checked" else modes["match"]
    verdict = "matched" if (files_ok and modes_ok and tier_cmp["match"]) else "mismatched"
    return {"attempted": True, "dest": str(d), "structural": v,
            "n_extracted": n_arch, "n_blob_written": n_blob, "patches_applied": applied,
            "files": {"compare": digests, "absent_after_rebuild": absent,
                      "n_checked": len(act_sha)},
            "modes": {"compare": modes, "mode_check": mode_check,
                      "note": None if mode_check == "checked" else
                      "Windows 上 exec 位不可用：mode 比较跳过，不作为判定依据"},
            "tiers": {"expected": tiers_exp, "actual": tiers_act, "compare": tier_cmp},
            "verdict": verdict}


def roundtrip(root: str | os.PathLike, *, declaration: Sequence[Mapping],
              base_commit: str | None = None, workdir: str | os.PathLike,
              capture_meta: Mapping[str, Any] | None = None) -> dict:
    """采集 → 静态校验 → 独立目录重建 → 逐项比对，一次跑完。"""
    bundle = collect(root, declaration=declaration, base_commit=base_commit,
                     capture_meta=capture_meta)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    dest = wd / "rebuilt"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    rep = rebuild(bundle, dest=dest, repo=root)
    structural = PB.verify(bundle)
    complete = bool(structural["verdict"] == "complete" and rep["verdict"] == "matched")
    return {"bundle": bundle, "structural": structural, "rebuild": rep,
            "complete": complete,
            "verdict": {"structural": structural["verdict"],
                        "rebuild": rep["verdict"], "complete": complete}}


# ---------------------------------------------------------------------------
# 声明集来源：source_manifest.json + 可选 harness
# ---------------------------------------------------------------------------

def declaration_from_manifest(root: str | os.PathLike, profile: str) -> list[dict]:
    src = Path(root) / "source_manifest.json"
    m = json.loads(src.read_text(encoding="utf-8"))
    prof = m["profiles"][profile]
    items = []
    for e in prof["declared_source_set"]:
        item = {"path": e["path"], "role": e.get("role", "module")}
        for k in ("compute_role", "workflow_scope", "why_not_numeric",
                  "why_not_workflow", "evidence", "reason"):
            if k in e:
                item[k] = e[k]
        items.append(item)
    if len(items) != prof["n_files"]:
        raise BundleIOError(
            f"声明集条目数 {len(items)} ≠ manifest 的 n_files={prof['n_files']}")
    return items


def harness_item(path: str, compute_role: str = "cli_only") -> dict:
    import capture_contract as CC
    return CC.harness_item(
        path, sha256="0" * 64, compute_role=compute_role,
        why_not_numeric="只做采集/编排，不进阻抗模型计算",
        evidence="C-1：harness 进 declared/workflow，不进 numeric")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load(path: str | os.PathLike) -> dict:
    return PB.parse_bundle(Path(path).read_bytes())


#: bundle 顶层允许的键。多一个就是"有字段没进摘要"，少一个就是"缺章节"。
BUNDLE_KEYS = frozenset({"format", "version", "base", "declaration", "files",
                         "patches", "missing", "counts", "capture", "tree_digest"})


def _write(bundle: Mapping[str, Any], path: str | os.PathLike) -> None:
    extra = set(bundle) - BUNDLE_KEYS
    if extra:
        raise BundleIOError(f"bundle 有摘要之外的键 {sorted(extra)}：落盘前必须去掉")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(PB.bundle_bytes(bundle))


def _decl(args) -> list[dict]:
    items = declaration_from_manifest(args.root, args.profile) if args.profile else []
    for spec in args.harness or ():
        path, _, role = spec.partition(":")
        items.append(harness_item(path, role or "cli_only"))
    if not items:
        raise BundleIOError("没有声明集：给 --profile 或 --harness")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="补丁包采集/校验/重建")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="从工作树采集 bundle")
    c.add_argument("--root", default=".")
    c.add_argument("--profile", default=None)
    c.add_argument("--harness", action="append")
    c.add_argument("--base", default=None)
    c.add_argument("--out", required=True)

    v = sub.add_parser("verify", help="只校验 bundle 自身")
    v.add_argument("--bundle", required=True)

    r = sub.add_parser("rebuild", help="在独立目录重建并比对")
    r.add_argument("--bundle", required=True)
    r.add_argument("--root", default=".")
    r.add_argument("--dest", required=True)

    t = sub.add_parser("roundtrip", help="采集+重建+比对")
    t.add_argument("--root", default=".")
    t.add_argument("--profile", default=None)
    t.add_argument("--harness", action="append")
    t.add_argument("--base", default=None)
    t.add_argument("--workdir", required=True)
    t.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "capture":
        b = collect(args.root, declaration=_decl(args), base_commit=args.base)
        _write(b, args.out)
        v = PB.verify(b)
        print(json.dumps(v, ensure_ascii=False))
        return 0 if v["verdict"] == "complete" else 1
    if args.cmd == "verify":
        print(json.dumps(PB.verify(_load(args.bundle)), ensure_ascii=False))
        return 0
    if args.cmd == "rebuild":
        rep = rebuild(_load(args.bundle), dest=args.dest, repo=args.root)
        print(json.dumps(rep, ensure_ascii=False))
        return 0 if rep["verdict"] == "matched" else 1
    rep = roundtrip(args.root, declaration=_decl(args), base_commit=args.base,
                    workdir=args.workdir)
    if args.out:
        _write(rep["bundle"], args.out)
    print(json.dumps({"verdict": rep["verdict"],
                      "tiers_equal": rep["rebuild"]["tiers"]["compare"]["match"],
                      "files_equal": rep["rebuild"]["files"]["compare"]["match"]},
                     ensure_ascii=False))
    return 0 if rep["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
