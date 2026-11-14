# -*- coding: utf-8 -*-
"""补丁包契约（patch bundle contract）：纯函数层。

这一层只回答一个问题：**一份 bundle 是否足以声称"工作树可完整重建"**。
它不读 Git、不碰磁盘、不打时间戳——`collect`（采集）与 `rebuild`（恢复）
在 I/O 层 `patch_bundle_io.py`。这样拆是因为"打包"和"重建"都可能有一堆
平台细节（换行、exec 位、符号链接），而**判定规则**不该随这些细节漂移。

为什么不能只有 `git diff`
------------------------
三条，每一条都在真实仓库里会踩到：

1. **`git diff` 不包含未跟踪文件**。声明集里的新模块只要还没 `git add`，
   diff 里一个字都没有，而它恰恰可能是这次运行的数值代码。
2. **`git diff` 也不包含 ignored 文件**。`out/` 被 ignore，但 `config.json`
   一类的声明项可能落在 ignore 规则内（或将来落进去）。
3. **diff 可能"沉默"而字节不同**。换行归一（`core.autocrlf`）、clean/smudge
   filter、`core.fileMode` 都会让 `git diff` 什么都不报、工作树字节却和
   commit 里的 blob 不相等。只看 diff 的打包器会漏掉它，然后在重建后
   复算出一个不同的源码摘要——**不报错，只是身份错了**。

所以 bundle 对每个声明文件记一个 `source`，三选一，必须能解释每一份字节从哪来：

| `source` | 含义 | bundle 里存什么 |
|---|---|---|
| `base` | 与 base commit 的 blob **逐字节相同**（比对过 sha256） | 只存 sha256 + git mode |
| `patch` | 由 staged/unstaged diff 恢复 | 只存 sha256 + 引用哪个 patch |
| `blob` | 其余一切：未跟踪、ignored、**diff 沉默但字节不同** | 存完整内容（base64）+ mode |

`blob` 是**兜底**，不是"未跟踪专用"。宁可多存几 KB 内容，也不能让某一份字节
没有出处——这正是"打包了大部分文件"不能冒充"可完整重建"的地方。

完整性 vs 真实性
----------------
`tree_digest` 能挡**意外损坏**（改了一个字符、编码转换、手工编辑），挡不住
**蓄意伪造**：能改内容的人也能重算摘要。真实的"这份 bundle 是谁打的"要靠
外部锚点（签名、远端记录），本轮不做，也不假装做了。

    python patch_bundle.py        # 跑内置自检
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import struct
from typing import Any, Iterable, Mapping, Sequence

PATCH_BUNDLE_VERSION = "1.0"
#: `pb1` 是**载荷布局**的版本：字段增删改名都要换号（同 `IDENTITY_CONTRACT_VERSION`）。
PATCH_BUNDLE_FORMAT = "pb1"

#: 每份字节的出处。见模块 docstring 的三行表。
FILE_SOURCES = ("base", "patch", "blob")

MODE_FILE = "100644"
MODE_EXEC = "100755"
SUPPORTED_MODES = (MODE_FILE, MODE_EXEC)

#: 声明项允许带进 bundle 的字段。**其余字段一律不进**：bundle 是身份输入，
#: 不是注释仓库；`sha256` 也不在这里——它在文件记录里（"应该有什么" vs "字节是什么"）。
DECL_KEYS = ("path", "role", "compute_role", "workflow_scope", "runtime_scope",
             "why_not_numeric", "why_not_workflow", "evidence", "reason")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_WIN_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)}


class BundleError(ValueError):
    """bundle 结构性非法：不是"不完整"，是"这个 bundle 根本不成形"。"""


# ---------------------------------------------------------------------------
# 基本量：mode / 路径 / 摘要
# ---------------------------------------------------------------------------

def normalize_mode(mode: Any) -> str:
    """把各种 mode 写法归一成 git 的 `100644` / `100755`。

    只接受**普通文件**。`120000`（符号链接）与 `160000`（submodule）直接报错：
    符号链接在重建时会写出一条指向别处的路径，最容易变成"落在仓库里的外部代码"；
    submodule 是另一个仓库，bundle 无法声称能重建它。二者都不该被当成"差不多能恢复"。
    """
    if isinstance(mode, bool):
        raise BundleError(f"mode 不能是 bool：{mode!r}")
    if isinstance(mode, int):
        if mode in (0o120000, 0o160000):
            raise BundleError(
                f"mode {oct(mode)} 不是普通文件（符号链接/submodule）："
                "bundle 只声称能重建普通文件")
        if mode & 0o111:
            return MODE_EXEC
        if mode & 0o100000 or mode <= 0o777:
            return MODE_FILE
        raise BundleError(f"无法解释的 mode：{oct(mode)}")
    if isinstance(mode, str):
        s = mode.strip()
        if s in SUPPORTED_MODES:
            return s
        if s in ("644", "755"):
            return "100" + s
        if s in ("0o644", "0o755"):
            return "100" + s[2:]
        if s in ("120000", "160000"):
            raise BundleError(f"mode {s!r} 不是普通文件（符号链接/submodule）")
    raise BundleError(f"mode 必须是 {SUPPORTED_MODES} 之一或其整数形式，收到 {mode!r}")


def check_rel_path(path: Any) -> str:
    """声明项/文件记录的路径：必须是**已规范化**的 POSIX 相对路径。

    比 `capture_contract.normalize_path` 更严，但不是第二套规则：那一步负责把
    任意写法（`D:/…`、反斜杠、`..`、符号链接）解析成相对路径，这一步负责**拒绝**
    任何在重建时可能写到声明范围之外的形状。写入侧的安全边界必须靠字面检查，
    不能靠"上游应该已经转好了"。

    跨平台一致：只差大小写的两个路径在 Windows 上写不出来，所以**一律拒绝**，
    而不是按平台行为分叉——bundle 是证据，同一份 bundle 在任何机器上都得同判。
    """
    if not isinstance(path, str):
        raise BundleError(f"path 必须是字符串，收到 {type(path).__name__}")
    if not path:
        raise BundleError("path 不能为空")
    if "\x00" in path:
        raise BundleError(f"path 含 NUL：{path!r}")
    if "\\" in path:
        raise BundleError(f"path 必须用 POSIX 分隔符，收到反斜杠：{path!r}")
    if path.startswith("/"):
        raise BundleError(f"path 必须是相对路径，收到绝对路径：{path!r}")
    if _DRIVE.match(path):
        raise BundleError(f"path 不许带盘符：{path!r}")
    segs = path.split("/")
    for seg in segs:
        if seg in ("", ".", ".."):
            raise BundleError(f"path 含空段/`.`/`..`：{path!r}")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in seg):
            raise BundleError(f"path 含控制字符：{path!r}")
        if seg != seg.rstrip(" ."):
            raise BundleError(f"path 段以空格或点结尾（Windows 无法原样落盘）：{path!r}")
        if seg.split(".")[0].lower() in _WIN_RESERVED:
            raise BundleError(f"path 段是 Windows 保留名 {seg!r}：{path!r}")
    return path


def content_sha256(content: bytes) -> str:
    """裸内容摘要。**这个值同时是 `exec_identity` 的源码哈希**（同口径：bytes 的 sha256）。"""
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise BundleError(f"content 必须是 bytes，收到 {type(content).__name__}")
    return hashlib.sha256(bytes(content)).hexdigest()


def file_digest(path: Any, mode: Any, content: bytes) -> str:
    """bundle 层的文件摘要：**路径 + mode + 内容**一起哈希。

    为什么不能只哈希内容：两个文件内容一样但一个是可执行脚本、一个是数据文件，
    重建后行为不同；同一个文件在两条路径上出现，也是两个身份。分域 + 长度前缀
    是为了让 `("a", "b")` 与 `("ab", "")` 这类拼接不撞值。
    """
    p = check_rel_path(path)
    m = normalize_mode(mode)
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise BundleError(f"content 必须是 bytes，收到 {type(content).__name__}")
    h = hashlib.sha256()
    for part in (b"pb1", b"file", p.encode("utf-8"), m.encode("ascii"), bytes(content)):
        h.update(struct.pack("<Q", len(part)))
        h.update(part)
    return h.hexdigest()


def canonical_json(obj: Any) -> bytes:
    """确定性序列化：键排序、ASCII 转义、无空格、拒绝 NaN/Infinity。

    `ensure_ascii=True` 不是为了好看：UTF-8 直接落盘时，不同平台/编辑器的换行与
    BOM 处理会让"同一份 bundle"出现两种字节。ASCII-only 让摘要跨机器可复算。
    """
    try:
        text = json.dumps(obj, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"bundle 里有无法确定性序列化的值：{exc}") from exc
    return text.encode("utf-8")


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """整体摘要：**排除 `tree_digest` 自身**再哈希（否则自指，永远算不出来）。"""
    body = {k: v for k, v in manifest.items() if k != "tree_digest"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------

def make_declaration(items: Iterable[Mapping]) -> list[dict]:
    """规范化声明集：字段白名单 + 路径检查 + 重复/大小写冲突检查。

    声明集是 bundle 的**分母**。它必须自带，不能重建时再去问工作树"该有哪些文件"——
    那样一份被删掉的声明文件会静默消失，于是"少了东西"看起来像"本来就没有"。
    这正是 `declared_missing` 能成为硬失败的前提。
    """
    out: list[dict] = []
    seen: dict[str, dict] = {}
    by_case: dict[str, str] = {}
    for it in items or ():
        if not isinstance(it, Mapping):
            raise BundleError(f"声明项必须是映射，收到 {type(it).__name__}")
        path = check_rel_path(it.get("path"))
        if path in seen:
            raise BundleError(f"声明集路径重复：{path}")
        lk = path.lower()
        if lk in by_case:
            raise BundleError(
                f"声明集两条路径只差大小写（{by_case[lk]!r} vs {path!r}）："
                "大小写不敏感的文件系统上写不出两个文件，不能当两个身份")
        rec: dict[str, Any] = {"path": path}
        for k in DECL_KEYS[1:]:
            if k in it and it[k] is not None:
                rec[k] = it[k]
        if "role" not in rec:
            rec["role"] = "module"
        if not isinstance(rec["role"], str) or not rec["role"]:
            raise BundleError(f"{path}: role 必须是非空字符串")
        if "compute_role" in rec and not isinstance(rec["compute_role"], str):
            raise BundleError(f"{path}: compute_role 必须是字符串")
        if "workflow_scope" in rec and not isinstance(rec["workflow_scope"], bool):
            raise BundleError(f"{path}: workflow_scope 必须是 bool")
        if "runtime_scope" in rec and not isinstance(rec["runtime_scope"], str):
            raise BundleError(f"{path}: runtime_scope 必须是字符串")
        seen[path] = rec
        by_case[lk] = path
        out.append(rec)
    if not out:
        raise BundleError("声明集为空：一份不声明任何文件的 bundle 什么都没声称")
    return sorted(out, key=lambda x: x["path"])


def make_file_record(path: Any, mode: Any, sha256: str, *, source: str,
                     why: str | None = None, patch_ref: str | None = None,
                     content: bytes | None = None) -> dict:
    """一条文件记录。`sha256` 是**内容**摘要（身份用），`bundle_digest` 是路径+mode+内容。

    `source="blob"` 必须给 `content`，且 `sha256` 必须与它相符——这一条是**采集侧
    的自检**：如果采集算出的摘要和它要存的内容对不上，那是采集 bug，不能等重建时
    才发现（那时已经很难分清是 bundle 坏了还是 Git 的行为不同）。
    """
    p = check_rel_path(path)
    m = normalize_mode(mode)
    if source not in FILE_SOURCES:
        raise BundleError(f"source 必须是 {FILE_SOURCES} 之一，收到 {source!r}")
    if not isinstance(sha256, str) or not _HEX64.match(sha256):
        raise BundleError(f"{p}: sha256 必须是 64 位小写十六进制，收到 {sha256!r}")

    rec: dict[str, Any] = {"path": p, "mode": m, "source": source, "sha256": sha256}
    if source == "blob":
        if content is None:
            raise BundleError(f"{p}: source=blob 必须带内容（未跟踪/ignored/diff 沉默都在这里）")
        real = content_sha256(content)
        if real != sha256:
            raise BundleError(f"{p}: 记录 sha256={sha256} 与所存内容的 sha256={real} 不符")
        rec["bytes"] = len(content)
        rec["content_b64"] = base64.b64encode(bytes(content)).decode("ascii")
        if not (why or "").strip():
            raise BundleError(f"{p}: source=blob 必须写 why（未跟踪/ignored/diff 沉默要能分辨）")
        rec["why_blob"] = why
    elif source == "patch":
        if not (patch_ref or "").strip():
            raise BundleError(f"{p}: source=patch 必须写 patch_ref（不能只说'在某个 diff 里'）")
        rec["patch_ref"] = patch_ref
    if content is None:
        # 内容不在 bundle 里（base/patch）就**不能**算"路径+mode+内容"的摘要——
        # 拿空内容代进去，会把"内容根本没验"伪装成"验过了"。只盖路径与 mode，
        # 并显式标出这一档，让读到的人知道内容要靠重建来证。
        rec["bundle_digest_kind"] = "path_mode_only"
        rec["bundle_digest"] = file_digest(p, m, b"")
    else:
        rec["bundle_digest_kind"] = "path_mode_content"
        rec["bundle_digest"] = file_digest(p, m, content)
    return rec


def make_patch_record(name: str, text: str, *, paths: Sequence[str]) -> dict:
    """一段 patch 的记录。`text` 是 `git diff` 的**原样输出**（`--binary` 保证 ASCII）。"""
    if not isinstance(name, str) or not name or "/" in name:
        raise BundleError(f"patch 名必须是单段字符串，收到 {name!r}")
    if not isinstance(text, str):
        raise BundleError(f"patch {name}: text 必须是 str（原样输出，不做换行转换）")
    data = text.encode("utf-8")
    return {"name": name, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data), "paths": sorted({check_rel_path(p) for p in paths}),
            "text": text}


def assemble(*, base: Mapping[str, Any], declaration: Iterable[Mapping],
             files: Iterable[Mapping], patches: Iterable[Mapping] = (),
             capture: Mapping[str, Any] | None = None) -> dict:
    """把声明集、文件记录、patch 组成一份 bundle，并盖上 `tree_digest`。

    这里做的是**结构性**检查（每一份字节都有出处、声明与文件一一对得上）。
    "全部声明文件都在"是 `verify` 的事——`assemble` 允许缺，但会把缺的写进
    `missing`，让判定显式变成 `incomplete` 而不是"打包成功"。
    """
    if not isinstance(base, Mapping):
        raise BundleError("base 必须是映射")
    commit = base.get("commit")
    if not isinstance(commit, str) or not _COMMIT.match(commit):
        raise BundleError(f"base.commit 必须是 7–64 位小写十六进制 commit，收到 {commit!r}")
    base_rec = {"commit": commit}
    for k in ("subject", "branch", "describe"):
        if base.get(k):
            base_rec[k] = str(base[k])

    decl = make_declaration(declaration)
    decl_paths = {d["path"] for d in decl}
    decl_case = {p.lower(): p for p in decl_paths}

    plist = [make_patch_record(p["name"], p["text"], paths=p.get("paths", ()))
             if "sha256" not in p else dict(p) for p in (patches or ())]
    pnames = set()
    for p in plist:
        nm = p.get("name")
        if not isinstance(nm, str) or not nm:
            raise BundleError(f"patch 记录缺 name：{p!r}")
        if nm in pnames:
            raise BundleError(f"patch 名重复：{nm}")
        pnames.add(nm)

    flist: list[dict] = []
    fseen: set[str] = set()
    for f in files or ():
        if not isinstance(f, Mapping):
            raise BundleError(f"文件记录必须是映射，收到 {type(f).__name__}")
        p = check_rel_path(f.get("path"))
        if p in fseen:
            raise BundleError(f"文件记录重复：{p}")
        if p not in decl_paths:
            raise BundleError(
                f"文件 {p!r} 不在声明集里：bundle 的文件集**必须**等于声明集，"
                "多出来的文件不许被顺手算进源码身份")
        rec = dict(f)
        normalize_mode(rec.get("mode"))
        if rec.get("source") == "blob":
            if "content_b64" not in rec:
                raise BundleError(f"{p}: blob 记录缺 content_b64（用 make_file_record 构造）")
        elif rec.get("source") == "patch":
            refs = str(rec.get("patch_ref") or "").split("+")
            unknown = [r for r in refs if r not in pnames]
            if not refs or unknown:
                raise BundleError(
                    f"{p}: patch_ref={rec.get('patch_ref')!r} 里有不存在的 patch {unknown}")
        fseen.add(p)
        flist.append(rec)
    _ = decl_case

    files_sorted = sorted(flist, key=lambda x: x["path"])
    missing = sorted(decl_paths - fseen)
    counts = {"declared": len(decl), "files": len(files_sorted), "missing": len(missing),
              "base": sum(1 for f in files_sorted if f.get("source") == "base"),
              "patch": sum(1 for f in files_sorted if f.get("source") == "patch"),
              "blob": sum(1 for f in files_sorted if f.get("source") == "blob"),
              "patches": len(plist),
              "blob_bytes": sum(int(f.get("bytes") or 0) for f in files_sorted)}

    manifest: dict[str, Any] = {
        "format": PATCH_BUNDLE_FORMAT,
        "version": PATCH_BUNDLE_VERSION,
        "base": base_rec,
        "declaration": decl,
        "files": files_sorted,
        "patches": sorted(plist, key=lambda x: x["name"]),
        "missing": missing,
        "counts": counts,
        "capture": dict(capture or {}),
    }
    manifest["tree_digest"] = manifest_digest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def decode_blob(rec: Mapping[str, Any]) -> bytes:
    """取回 blob 内容并当场校验（长度、sha256、bundle_digest）。"""
    p = rec.get("path")
    try:
        data = base64.b64decode(str(rec.get("content_b64")), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BundleError(f"{p}: content_b64 不是合法 base64：{exc}") from exc
    if "bytes" in rec and int(rec["bytes"]) != len(data):
        raise BundleError(f"{p}: 记录 bytes={rec['bytes']} 与实际 {len(data)} 不符")
    real = content_sha256(data)
    if real != rec.get("sha256"):
        raise BundleError(f"{p}: 内容 sha256={real} 与记录 {rec.get('sha256')} 不符")
    dg = file_digest(p, rec.get("mode"), data)
    if dg != rec.get("bundle_digest"):
        raise BundleError(f"{p}: bundle_digest 与内容不符（记录 {rec.get('bundle_digest')}）")
    return data


def verify(bundle: Mapping[str, Any]) -> dict:
    """静态校验：**只看 bundle 自身**，不需要工作树、不需要 Git。

    三种判定，优先级从高到低：

    · `corrupt`：bundle 自身坏了（摘要对不上、base64 解不开、字段类型不对）。
      这时"缺不缺文件"没有意义——先修 bundle。
    · `incomplete`：结构完整但有声明文件没有记录（`declared_missing`）。
      **这是硬失败**，不是警告：少一个文件的 bundle 重建出来的工作树
      一定不是原来那个，而它看起来"几乎一样"。
    · `complete`：结构完整、每个声明文件都有出处、每份内嵌内容自洽。

    `complete` **不等于**"重建过并且对得上"——那是 I/O 层往返的事。
    本函数永远不返回"重建成功"这类字眼，避免把静态完整说成动态可恢复。
    """
    reasons: list[dict] = []
    if not isinstance(bundle, Mapping):
        return {"verdict": "corrupt", "reasons": [{"code": "not_mapping"}],
                "n_declared": 0, "n_files": 0, "n_missing": 0, "missing": []}
    if bundle.get("format") != PATCH_BUNDLE_FORMAT:
        reasons.append({"code": "format_mismatch", "detail": bundle.get("format")})
    if bundle.get("version") != PATCH_BUNDLE_VERSION:
        reasons.append({"code": "version_mismatch", "detail": bundle.get("version")})
    stored = bundle.get("tree_digest")
    if not isinstance(stored, str) or not _HEX64.match(stored):
        reasons.append({"code": "tree_digest_missing"})
    else:
        try:
            real = manifest_digest(bundle)
        except BundleError as exc:
            reasons.append({"code": "not_serializable", "detail": str(exc)})
            real = None
        if real is not None and real != stored:
            reasons.append({"code": "tree_digest_mismatch",
                            "detail": {"stored": stored, "recomputed": real}})

    try:
        decl = make_declaration(bundle.get("declaration") or ())
    except BundleError as exc:
        reasons.append({"code": "declaration_invalid", "detail": str(exc)})
        decl = []
    decl_paths = {d["path"] for d in decl}

    patches = bundle.get("patches") or []
    pnames = set()
    for p in patches if isinstance(patches, list) else []:
        if not isinstance(p, Mapping):
            reasons.append({"code": "patch_not_mapping"})
            continue
        nm = p.get("name")
        pnames.add(nm)
        text = p.get("text")
        if not isinstance(text, str):
            reasons.append({"code": "patch_text_invalid", "detail": nm})
            continue
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() != p.get("sha256"):
            reasons.append({"code": "patch_digest_mismatch", "detail": nm})
        if "bytes" in p and int(p["bytes"]) != len(data):
            reasons.append({"code": "patch_bytes_mismatch", "detail": nm})

    files = bundle.get("files") or []
    if not isinstance(files, list):
        reasons.append({"code": "files_not_list"})
        files = []
    seen: set[str] = set()
    for f in files:
        if not isinstance(f, Mapping):
            reasons.append({"code": "file_not_mapping"})
            continue
        try:
            p = check_rel_path(f.get("path"))
        except BundleError as exc:
            reasons.append({"code": "file_path_invalid", "detail": str(exc)})
            continue
        if p in seen:
            reasons.append({"code": "file_duplicate", "detail": p})
            continue
        seen.add(p)
        if p not in decl_paths:
            reasons.append({"code": "file_not_declared", "detail": p})
        src = f.get("source")
        if src not in FILE_SOURCES:
            reasons.append({"code": "file_source_invalid", "detail": {"path": p, "source": src}})
            continue
        try:
            normalize_mode(f.get("mode"))
        except BundleError as exc:
            reasons.append({"code": "file_mode_invalid", "detail": str(exc)})
        sha = f.get("sha256")
        if not isinstance(sha, str) or not _HEX64.match(sha):
            reasons.append({"code": "file_sha_invalid", "detail": p})
        if src == "blob":
            try:
                decode_blob(f)
            except BundleError as exc:
                reasons.append({"code": "file_digest_mismatch", "detail": str(exc)})
        elif src == "patch":
            ref = str(f.get("patch_ref") or "")
            unknown = [r for r in ref.split("+") if r not in pnames]
            if unknown:
                reasons.append({"code": "patch_ref_unknown",
                                "detail": {"path": p, "ref": ref, "unknown": unknown}})

    missing = sorted(decl_paths - seen)
    if missing:
        reasons.append({"code": "declared_missing", "detail": missing})
    declared_but_not_files = [d["path"] for d in decl if d["path"] not in seen]

    corrupt = any(r["code"] not in ("declared_missing",) for r in reasons)
    verdict = "corrupt" if corrupt else ("incomplete" if missing else "complete")
    return {"verdict": verdict, "reasons": reasons,
            "n_declared": len(decl), "n_files": len(seen),
            "n_missing": len(missing), "missing": missing,
            "declared_without_record": declared_but_not_files,
            "tree_digest": bundle.get("tree_digest")}


# ---------------------------------------------------------------------------
# 供身份层与往返比较使用
# ---------------------------------------------------------------------------

def tier_items(bundle: Mapping[str, Any]) -> list[dict]:
    """把 bundle 还原成 `exec_identity.source_tree_sha256` 的输入。

    只出 `path / sha256 / compute_role / workflow_scope` 四项：
    `exec_identity` 就是按这四项分层的。**声明项里没写的字段这里也不补**——
    "不写就是 numeric / 进 workflow"的保守默认由身份层负责，bundle 不许替它填。
    缺文件的声明项**不出现在这里**（它没有 sha256），由 `verify` 报 `declared_missing`。
    """
    decl = {d["path"]: d for d in (bundle.get("declaration") or [])}
    out: list[dict] = []
    for f in bundle.get("files") or []:
        p = f.get("path")
        d = decl.get(p)
        if d is None:
            raise BundleError(f"文件 {p!r} 不在声明集里，不能进身份")
        item = {"path": p, "sha256": f.get("sha256")}
        for k in ("compute_role", "workflow_scope"):
            if k in d:
                item[k] = d[k]
        out.append(item)
    return sorted(out, key=lambda x: x["path"])


def compare_digests(expected: Mapping[str, str], actual: Mapping[str, str]) -> dict:
    """逐项比较两组摘要（身份三层、或每个文件的摘要）。

    判据是**逐项相同**，不是"大部分相同"：`missing`/`extra`/`mismatched` 任一非空
    都是 `match=False`。聚合量（总数、总字节）相同也可能是不同文件，只看聚合会漏。
    """
    exp = {str(k): str(v) for k, v in dict(expected or {}).items()}
    act = {str(k): str(v) for k, v in dict(actual or {}).items()}
    missing = sorted(set(exp) - set(act))
    extra = sorted(set(act) - set(exp))
    mismatched = [{"path": k, "expected": exp[k], "actual": act[k]}
                  for k in sorted(set(exp) & set(act)) if exp[k] != act[k]]
    return {"match": not (missing or extra or mismatched),
            "n_expected": len(exp), "n_actual": len(act),
            "missing": missing, "extra": extra, "mismatched": mismatched,
            "n_mismatched": len(mismatched)}


def bundle_bytes(bundle: Mapping[str, Any]) -> bytes:
    """落盘字节：确定性 JSON + 末尾一个换行。**摘要算的是正文，不含这个换行。**"""
    return canonical_json(bundle) + b"\n"


def _reject_constant(name: str) -> Any:
    raise BundleError(f"bundle 里有非标准 JSON 常量 {name}：写入侧用 allow_nan=False，"
                      "读入侧也不接受 NaN/Infinity")


def parse_bundle(data: bytes | str) -> dict:
    """读回 bundle。JSON 都解不开就是 `corrupt`，直接抛 `BundleError`。

    `NaN`/`Infinity` 一并拒绝：Python 的 `json` 默认接受它们，但 `canonical_json`
    默认不产出（`allow_nan=False`）。写入侧产不出、读入侧却接受，等于给"手工
    编辑过的 bundle"留了一道后门——它会在摘要处才炸，错误信息还会指向别的地方。
    """
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        obj = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, BundleError) as exc:
        raise BundleError(f"bundle 不是合法 JSON：{exc}") from exc
    if not isinstance(obj, dict):
        raise BundleError("bundle 顶层必须是对象")
    return obj


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    from pathlib import Path

    ok = 0

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            raise AssertionError(msg)
        ok += 1

    a = b"print(1)\n"
    chk(file_digest("a.py", MODE_FILE, a) == file_digest("a.py", "644", a),
        "mode 644/100644 应同摘要")
    chk(file_digest("a.py", MODE_FILE, a) != file_digest("a.py", MODE_EXEC, a),
        "同内容不同 mode 摘要必须不同")
    chk(file_digest("a.py", MODE_FILE, a) != file_digest("b.py", MODE_FILE, a),
        "同内容不同路径摘要必须不同")

    for bad in ("/abs.py", "a\\b.py", "a/../b.py", "./a.py", "a//b.py",
                "", "CON.py", "a.py ", "D:/x.py"):
        try:
            check_rel_path(bad)
        except BundleError:
            ok += 1
        else:
            raise AssertionError(f"应当拒绝路径 {bad!r}")

    decl = [{"path": "a.py", "role": "module"}, {"path": "b.py", "role": "module"}]
    f1 = make_file_record("a.py", MODE_FILE, content_sha256(a), source="base")
    f2 = make_file_record("b.py", MODE_FILE, content_sha256(b"x\n"), source="blob",
                          why="untracked", content=b"x\n")
    b = assemble(base={"commit": "a" * 40}, declaration=decl, files=[f1, f2],
                 patches=[], capture={"tool": "selftest"})
    v = verify(b)
    chk(v["verdict"] == "complete", f"应 complete，实际 {v}")
    chk(decode_blob(b["files"][1]) == b"x\n", "blob 应能取回")

    b2 = assemble(base={"commit": "a" * 40}, declaration=decl, files=[f1], patches=[])
    v2 = verify(b2)
    chk(v2["verdict"] == "incomplete", f"缺文件必须 incomplete，实际 {v2['verdict']}")
    chk(v2["missing"] == ["b.py"], v2["missing"])

    tampered = json.loads(json.dumps(b))
    tampered["files"][1]["content_b64"] = base64.b64encode(b"y\n").decode()
    chk(verify(tampered)["verdict"] == "corrupt", "改内容必须 corrupt")

    tampered2 = json.loads(json.dumps(b))
    tampered2["capture"]["tool"] = "other"
    chk(verify(tampered2)["verdict"] == "corrupt", "改任何字段都必须被 tree_digest 抓到")

    chk(bundle_bytes(b) == bundle_bytes(assemble(
        base={"commit": "a" * 40}, declaration=decl, files=[f2, f1], patches=[],
        capture={"tool": "selftest"})), "字段顺序不同也必须同字节")

    items = tier_items(b)
    chk([i["path"] for i in items] == ["a.py", "b.py"], items)
    chk(all("compute_role" not in i for i in items), "声明里没写就不许替它补默认值")

    tmp = Path(tempfile.mkdtemp(prefix="pb_self_"))
    p = tmp / "b.json"
    p.write_bytes(bundle_bytes(b))
    chk(parse_bundle(p.read_bytes())["tree_digest"] == b["tree_digest"], "往返后摘要应不变")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print(f"patch_bundle 自检 {ok} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
