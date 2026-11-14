# -*- coding: utf-8 -*-
"""采集契约（capture contract）：路径规范化、harness 归属、采样事件归属。

这个模块只解决**采集**问题，不碰身份算法：`exec_identity.py` 的哈希载荷、
状态阶梯、覆盖判定一行没改。它产出的东西是 `exec_identity.evaluate()` 的**输入**，
而且必须是它已经能吃的形状（POSIX 相对路径的 `runtime_loaded_modules`、
`external_modules`、`declared_source_set`）。

三条规则（本轮裁决，见 `docs/round_2026-11-14/CAPTURE_CONTRACT.md`）
------------------------------------------------------------------
**C-1 harness 归属**。runner 自己的文件会出现在 `sys.modules` 里（`__main__`），
它的 realpath 在仓库根内 ⇒ 必然进入运行时集合。若不在声明集里，**每次**运行都是
`undeclared_loaded` ⇒ 永久 `derived`。解法不是"采集时按文件名过滤"（那是无痕排除），
而是把它**显式声明为 harness 项**：源码身份照进 `declared_source_tree_sha256`
（改 runner 能被看见），但 `compute_role` 必须非 `numeric`（编排代码的变化不换
`run_id`），影响工作流时由 `workflow_scope=True` 进 `workflow_id`。
**过滤器不参与判定**：没声明的 harness 照样是 `undeclared`，一行都不会被悄悄丢掉。

**C-2 路径规范化**。两侧都走同一条规则：`realpath` → 判断是否在仓库根内 →
在则转 POSIX 相对路径，不在则**明确记为外部**（不强行映射进声明集）。只校验
"不是绝对路径、没有反斜杠"不等于路径身份一致：`D:/.../bmdrt/bm_top.py` 与
`bm_top.py` 用那条规则都"合法"，却永不相等，于是**全量未声明**——错得像
"代码库整个没声明"，而不像"格式错了"。

**C-3 采样事件归属**。一次工作流可能有多次 `MCMC.run`（`with_voigt=True`、
Profile C 的两族对拍）。单槽位快照会让所有记录共用**最后一次**快照；正确做法是
每次调用一条事件记录（序号、入口快照、生效参数、所属会话、开始/结束状态），
由会话（owner）认领。会话必须显式开、显式关；未关的事件是 `running`，
**不算完整采样**。进程内重复运行必须 `reset()`，否则第二次会读到第一次的事件。

纯函数 vs 有状态
----------------
前两节是纯函数（只读文件系统的 stat，无 I/O 副作用、无时间戳）；
`SamplingEventLog` 有状态但没有 I/O——它只是把"哪次采样属于谁"这件事记清楚。

    python capture_contract.py        # 跑内置自检
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CAPTURE_CONTRACT_VERSION = "1.0"

#: 运行时条目的归属分类。`undeclared` 不是"丢掉"，是"照实记下来等着降级"。
RUNTIME_CLASSES = ("project", "harness", "external", "undeclared")
#: 路径规范化后的状态。`symlink_escape` 单独留一档：它"看起来在仓库里"。
PATH_STATES = ("project", "external", "symlink_escape", "invalid")
#: harness 项允许的 `compute_role`（都不进 `numeric_source_tree_sha256`）。
HARNESS_COMPUTE_ROLES = ("cli_only", "decision", "plot_only", "reference_only",
                         "budget_source")


# ---------------------------------------------------------------------------
# C-2 路径规范化
# ---------------------------------------------------------------------------

def _norm_root(root: str | os.PathLike) -> str:
    return os.path.realpath(os.fspath(root))


def _under(path: str, root: str) -> bool:
    """`path` 是否在 `root` 之下（大小写不敏感的比较在 Windows 上必要）。"""
    r = os.path.normcase(root)
    p = os.path.normcase(path)
    if p == r:
        return True
    return p.startswith(r + os.sep)


def normalize_path(raw: Any, root: str | os.PathLike) -> dict:
    """把任意写法解析成**仓库根下的 POSIX 相对路径**，或明确判为外部。

    接受绝对路径、反斜杠、`..`、符号链接——这正是它存在的理由：调用点拿到的
    就是 `D:\\...\\bm_top.py` 这类形状，要求调用点自己拼字符串只会制造两套规则。

    返回 `{raw, resolved, path, state, why, exists, is_link, case_only}`：

    · `state="project"`：`path` 是 POSIX 相对路径，可直接与声明集比较；
    · `state="external"`：在仓库根之外（`../x.py`、`D:/other/x.py`），`path=None`；
    · `state="symlink_escape"`：**字面在仓库内**但 realpath 指向仓库外——
      单独一档，因为它最容易被误判成"项目模块"；
    · `state="invalid"`：空串/非字符串/指向根目录本身。

    相对路径按**仓库根**解释（`bm_top.py` ⇒ `<root>/bm_top.py`），不按 cwd：
    否则同一条记录在不同目录下跑会得到不同的身份。
    """
    root_real = _norm_root(root)
    if not isinstance(raw, str):
        return {"raw": raw, "resolved": None, "path": None, "state": "invalid",
                "why": f"not_str:{type(raw).__name__}", "exists": False,
                "is_link": False, "case_only": False}
    if not raw.strip():
        return {"raw": raw, "resolved": None, "path": None, "state": "invalid",
                "why": "empty", "exists": False, "is_link": False,
                "case_only": False}

    p = raw if os.path.isabs(raw) else os.path.join(root_real, raw)
    ab = os.path.abspath(p)
    resolved = os.path.realpath(p)
    exists = os.path.exists(resolved)

    same_norm = os.path.normcase(ab) == os.path.normcase(resolved)
    case_only = bool(same_norm and ab != resolved)
    is_link = not same_norm

    if not _under(resolved, root_real):
        state = "symlink_escape" if _under(ab, root_real) else "external"
        return {"raw": raw, "resolved": resolved, "path": None, "state": state,
                "why": "realpath_outside_root" if state == "external"
                       else "link_target_outside_root",
                "exists": exists, "is_link": is_link, "case_only": case_only}

    rel = os.path.relpath(resolved, root_real).replace(os.sep, "/")
    if rel in ("", "."):
        return {"raw": raw, "resolved": resolved, "path": None, "state": "invalid",
                "why": "root_itself", "exists": exists, "is_link": is_link,
                "case_only": case_only}
    return {"raw": raw, "resolved": resolved, "path": rel, "state": "project",
            "why": None, "exists": exists, "is_link": is_link,
            "case_only": case_only}


def _entry_path(entry: Any) -> Any:
    return entry.get("path") if isinstance(entry, Mapping) else entry


def normalize_set(entries: Iterable[Any], root: str | os.PathLike, *,
                  strictly: bool = False) -> dict:
    """对一整组路径做规范化，并检出**规范化后的冲突**。

    冲突三类，都是"两个不同写法其实是同一个身份"或"同一身份被记了两次"：

    · `duplicate`：规范化后 path 相同（声明集里出现两次 ⇒ 分母不可复算）；
    · `case_collision`：只差大小写（Windows 上是同一个文件，POSIX 上不是）；
    · `alias`：不同 raw 指向同一个 realpath（符号链接指向仓库内另一个已声明文件）。

    `strictly=True` 时前两类直接 `raise`（声明集用）；运行时集合用宽松模式，
    因为同一文件被两个模块名加载是**正常现象**，记下 `alias` 就行。
    """
    root_real = _norm_root(root)
    by_path: dict[str, dict] = {}
    by_norm: dict[str, str] = {}
    external: list[dict] = []
    invalid: list[dict] = []
    collisions: list[dict] = []
    aliases: list[dict] = []

    for e in entries or ():
        r = normalize_path(_entry_path(e), root_real)
        if r["state"] == "invalid":
            invalid.append({"entry": e, **r})
            continue
        if r["state"] in ("external", "symlink_escape"):
            external.append({"entry": e, **r})
            continue
        rel = r["path"]
        nk = os.path.normcase(rel)
        if rel in by_path:
            collisions.append({"kind": "duplicate", "path": rel,
                               "raws": [by_path[rel]["raw"], r["raw"]]})
            if strictly:
                raise ValueError(f"声明集路径重复（规范化后）：{rel}")
            continue
        if nk in by_norm:
            collisions.append({"kind": "case_collision", "path": rel,
                               "other": by_norm[nk],
                               "raws": [by_norm[nk], rel]})
            if strictly:
                raise ValueError(
                    f"声明集路径只差大小写（{by_norm[nk]!r} vs {rel!r}）："
                    "在大小写不敏感的文件系统上是同一个文件，不能当两个身份")
            continue
        by_norm[nk] = rel
        by_path[rel] = {"path": rel, "state": "project",
                        "raw": r["raw"], "resolved": r["resolved"],
                        "exists": r["exists"], "is_link": r["is_link"],
                        "case_only": r["case_only"], "entry": e}

    # alias：不同 raw 落到同一 realpath（上面的 duplicate 已覆盖 path 相同的情形，
    # 这里只留 raw 不同的证据，供报告里点名"同一文件被两种写法声明"）
    seen_res: dict[str, str] = {}
    for rel, it in by_path.items():
        seen_res.setdefault(os.path.normcase(it["resolved"]), rel)
    for c in collisions:
        if c["kind"] == "duplicate" and len(set(c["raws"])) > 1:
            aliases.append({"path": c["path"], "raws": c["raws"]})

    return {"by_path": by_path, "external": external, "invalid": invalid,
            "collisions": collisions, "aliases": aliases,
            "paths": sorted(by_path), "n": len(by_path)}


# ---------------------------------------------------------------------------
# C-1 harness 归属
# ---------------------------------------------------------------------------

def harness_item(path: Any, *, sha256: str, compute_role: str,
                 why_not_numeric: str, evidence: str, role: str = "entry",
                 workflow_scope: bool = True,
                 why_not_workflow: str | None = None) -> dict:
    """构造一个 harness 声明项（runner / 采集脚本 / 编排脚本）。

    **为什么不允许 `compute_role="numeric"`**：harness 会改候选枚举、准入、重跑和
    编排，这些东西一改就该换 `workflow_id`，但不该换 `run_id`——`run_id` 说的是
    "这一次拟合的条件"。把它塞进 `numeric` 层，等于"改一行日志格式，所有拟合换号"，
    而换号是**静默**的：没人会发现，只会看到同一条件两个号。

    **为什么三个字段都必须显式给**：`compute_role` 缺失时 `exec_identity` 默认按
    `numeric` 算（§3.1.3 的保守默认）。默认值在这里正好是**错的那一边**，所以
    不许省略、不许猜。`why_not_numeric`+`evidence` 是 `check_declared_set` 的既有
    要求，提前在这里报错，比等到 `evaluate()` 里变成 `declared_set_invalid` 好。

    `path` 必须是**已规范化**的 POSIX 相对路径：本函数不猜 cwd，也不接受绝对路径
    （用 `normalize_path` 转换后再传）。
    """
    if not isinstance(path, str) or not path:
        raise ValueError("harness_item: path 必须是非空字符串（先过 normalize_path）")
    if os.path.isabs(path) or "\\" in path or path.startswith("/"):
        raise ValueError(
            f"harness_item: path 必须是 POSIX 相对路径，收到 {path!r}："
            "绝对路径与声明集永不相等，会让每次运行都 undeclared_loaded")
    if role not in ("entry", "module"):
        raise ValueError(f"harness_item: role 只能是 entry/module，收到 {role!r}")
    if compute_role == "numeric":
        raise ValueError(
            "harness_item: compute_role 不许是 numeric——harness 的改动不该换 run_id；"
            "请显式给一个非 numeric 档并写明 why_not_numeric")
    if compute_role not in HARNESS_COMPUTE_ROLES:
        raise ValueError(
            f"harness_item: compute_role 非法 {compute_role!r}，"
            f"可选 {HARNESS_COMPUTE_ROLES}")
    for name, val in (("why_not_numeric", why_not_numeric), ("evidence", evidence)):
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"harness_item: {name} 不能为空（降级必须有理由与证据）")
    if workflow_scope is False and not (why_not_workflow or "").strip():
        raise ValueError("harness_item: workflow_scope=False 必须写 why_not_workflow")

    out = {"path": path, "role": role, "sha256": sha256,
           "compute_role": compute_role, "runtime_scope": "harness",
           "why_not_numeric": why_not_numeric, "evidence": evidence,
           "workflow_scope": bool(workflow_scope)}
    if why_not_workflow:
        out["why_not_workflow"] = why_not_workflow
    return out


def is_harness(item: Mapping) -> bool:
    return (isinstance(item, Mapping)
            and item.get("runtime_scope") == "harness")


# ---------------------------------------------------------------------------
# 运行时集合的归属分类
# ---------------------------------------------------------------------------

def third_party_roots() -> tuple[str, ...]:
    """标准库与 site-packages 的根：这些**不是**"外部项目模块"。

    契约 §3 表的 `external_path` 说的是"模块 realpath 在声明根之外（editable install
    指向别处 / site-packages 里有一份**项目副本**）"——那是一份跑错位置的项目代码，
    要降级。`numpy` 落在 site-packages 里是**正常**的，把它记成 external 会让每条
    记录都带 2000 个"外部模块"，`external_path` 立刻失去意义。

    所以判据要两步：先在仓库根内找（project），再看是不是第三方（忽略），
    剩下的才算"项目代码跑出了仓库"。
    """
    import site
    import sysconfig

    roots: set[str] = set()
    paths = sysconfig.get_paths()
    for k in ("stdlib", "platstdlib", "purelib", "platlib"):
        p = paths.get(k)
        if p:
            roots.add(os.path.realpath(p))
    try:
        for p in site.getsitepackages():
            roots.add(os.path.realpath(p))
    except Exception:                                  # noqa: BLE001
        pass
    try:
        roots.add(os.path.realpath(site.getusersitepackages()))
    except Exception:                                  # noqa: BLE001
        pass
    return tuple(sorted(roots))


def is_third_party(path: str | None, roots: Sequence[str]) -> bool:
    if not path:
        return False
    p = os.path.normcase(os.path.realpath(path))
    for r in roots:
        rn = os.path.normcase(r)
        if p == rn or p.startswith(rn + os.sep):
            return True
    return False


def classify_runtime(files: Any, declared: Sequence[Mapping], *,
                     root: str | os.PathLike,
                     tp_roots: Sequence[str] | None = None) -> dict:
    """把运行时模块文件分成 project / harness / external，并把未声明的留下来。

    `files` 支持两种形状：
    · `{raw_file: [module_name, ...]}`（推荐，保留"同一文件多个模块名"的证据）；
    · `[raw_file, ...]`（只关心集合时用）。

    **未声明的文件不进"被过滤"那一类**：它进 `runtime_class="undeclared"` 且**留在**
    `runtime_loaded_modules` 里，于是 `exec_identity` 的集合复算会得到
    `undeclared_loaded`。这是 C-1 的关键：排除只能通过**声明**实现，
    采集层不许按文件名静默丢掉任何东西。

    第三方（`site-packages` / 标准库 / 解释器注入的 `sitecustomize`）**不进任何一档**，
    只计数：它们不是"项目代码跑错了位置"。那怎么区分"第三方的 numpy"和"site-packages
    里那份项目副本"？**按名字**：只有当仓库外那个文件的 basename（或模块名）与声明集
    里的某项对得上，才算 `external_path`——"我声明了 `bm_top.py`，但真正加载的那份在
    仓库外面"。对不上的（`numpy`、`sitecustomize`、标准库 `.pyd`）进 `n_ignored`，
    在汇总里可见但不成条目。

    外部路径不进 `runtime_loaded_modules`（其 path 不可能是 POSIX 相对路径），改走
    `external_modules`，由 `exec_identity` 判 `external_path`。
    """
    if isinstance(files, Mapping):
        pairs = [(k, list(v) if not isinstance(v, str) else [v])
                 for k, v in files.items()]
    elif isinstance(files, Iterable) and not isinstance(files, (str, bytes)):
        pairs = [(f, []) for f in files]
    else:
        raise ValueError("classify_runtime: files 必须是映射或可迭代对象")

    tp = tuple(tp_roots) if tp_roots is not None else third_party_roots()
    decl = normalize_set([{"path": _entry_path(d)} for d in declared], root,
                         strictly=False)
    decl_paths = set(decl["paths"])
    decl_by = {_entry_path(d): d for d in (declared or ()) if _entry_path(d)}
    decl_base = {os.path.normcase(os.path.basename(p)) for p in decl_paths}
    decl_stem = {os.path.normcase(os.path.splitext(os.path.basename(p))[0])
                 for p in decl_paths}

    grouped: dict[str, dict] = {}
    ext: list[dict] = []
    n_tp, n_ignored = 0, 0
    for raw, mods in pairs:
        r = normalize_path(raw, root)
        if r["state"] == "project":
            pass
        elif r["state"] == "invalid":
            # 路径解释不了就**不能消失**（否则"有一项根本没法解释"会变成 complete）
            grouped.setdefault("\u0000invalid:" + str(r["raw"]), {
                "path": str(r["raw"]), "runtime_class": "undeclared",
                "modules": sorted(set(mods)), "invalid": True,
                "why": r["why"]})
            continue
        else:
            base = os.path.normcase(os.path.basename(r["resolved"] or str(raw)))
            stem = os.path.normcase(os.path.splitext(base)[0])
            named_like = (base in decl_base) or any(
                os.path.normcase(m) in decl_stem for m in mods)
            if not named_like:
                n_ignored += 1
                if is_third_party(r["resolved"], tp):
                    n_tp += 1
                continue
            item = {"resolved": r["resolved"], "state": r["state"],
                    "modules": sorted(set(mods)), "raw": r["raw"],
                    "why": "basename 与声明项同名但 realpath 在仓库外"}
            for g in ext:
                if g["resolved"] == item["resolved"]:
                    g["modules"] = sorted(set(g["modules"]) | set(item["modules"]))
                    break
            else:
                ext.append(item)
            continue
        rel = r["path"]
        cell = grouped.get(rel)
        if cell is None:
            d = decl_by.get(rel)
            if d is None:
                cls = "undeclared"
            else:
                cls = "harness" if is_harness(d) else "project"
            cell = {"path": rel, "runtime_class": cls,
                    "role": (d or {}).get("role"),
                    "compute_role": (d or {}).get("compute_role"),
                    "modules": []}
            grouped[rel] = cell
        cell["modules"] = sorted(set(cell["modules"]) | set(mods))

    entries = sorted(grouped.values(), key=lambda x: x["path"])
    counts = {c: 0 for c in RUNTIME_CLASSES}
    for e in entries:
        counts[e["runtime_class"]] += 1
    counts["external"] = len(ext)
    return {
        "runtime_loaded_modules": entries,
        "external_modules": sorted(ext, key=lambda x: x["resolved"]),
        "counts": counts,
        "n_third_party": n_tp,
        "n_ignored": n_ignored,
        "third_party_roots": list(tp),
        "harness_modules": sorted(e["path"] for e in entries
                                  if e["runtime_class"] == "harness"),
        "undeclared_modules": sorted(e["path"] for e in entries
                                     if e["runtime_class"] == "undeclared"),
        "declared_paths": sorted(decl_paths),
        "declared_not_loaded": sorted(decl_paths - {e["path"] for e in entries}),
        "declared_problems": {"collisions": decl["collisions"],
                              "invalid": decl["invalid"],
                              "external": decl["external"]},
    }


# ---------------------------------------------------------------------------
# C-3 采样事件归属
# ---------------------------------------------------------------------------

class SamplerOwnerRequired(RuntimeError):
    """采样发生时没有打开的会话：事件没有归属，不能记在"上一次"头上。"""


class SamplingEventLog:
    """每次 `MCMC.run` 一条事件，事件归属由**显式会话**决定。

    三件事是这类的全部目的：

    1. **不覆盖**。第二次采样不许冲掉第一次的快照（单槽位覆盖是本轮实测踩到的
       错配来源：后两次记录都顶着第一次的 `sampler_entry`）。
    2. **不猜归属**。没有会话就 `raise`，不落到"最近一个"或"最后一个"。
    3. **不把没跑完的当证据**。事件未 `close` 时是 `running`，`complete()` 不返回它；
       异常的采样是 `error`，也不返回。两者都能在 `summary()` 里看见。

    会话必须显式 `reset()` 后才能复用同一个 owner：进程内重复运行如果忘了重置，
    第二次会读到第一次的事件——那是个不报错的错，所以这里把它变成会报的错。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._stack: list[str] = []
        self._seq = 0

    # -- 会话 --------------------------------------------------------------
    @contextmanager
    def session(self, owner: str):
        """打开一个归属会话。`owner` 是运行实例键（如 `run_id` 或子任务名）。"""
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("session: owner 必须是非空字符串")
        if owner in self._sessions:
            raise ValueError(
                f"session: owner {owner!r} 已经用过；同一进程内重复跑同一实例，"
                "必须先 reset()——否则这一次会读到上一次的事件")
        self._sessions[owner] = {"owner": owner, "events": [], "closed": False}
        self._stack.append(owner)
        try:
            yield self._sessions[owner]
        finally:
            self._sessions[owner]["closed"] = True
            if self._stack and self._stack[-1] == owner:
                self._stack.pop()

    def current(self) -> str | None:
        return self._stack[-1] if self._stack else None

    # -- 事件 --------------------------------------------------------------
    @contextmanager
    def event(self, *, snapshot: Any, effective: Any,
              sampler: str = "numpyro.infer.MCMC.run"):
        """记录一次采样调用。异常退出时标 `error` 并原样抛出。"""
        owner = self.current()
        if owner is None:
            raise SamplerOwnerRequired(
                "采样事件没有归属：必须先 with log.session(owner) 打开会话")
        sess = self._sessions[owner]
        idx = len(sess["events"])
        ev = {"event_id": f"{owner}#{idx}", "owner": owner, "idx": idx,
              "seq": self._seq, "sampler": sampler,
              "started": True, "status": "running", "error": None,
              "snapshot": snapshot, "effective": effective}
        self._seq += 1
        sess["events"].append(ev)
        try:
            yield ev
        except BaseException as exc:                  # noqa: BLE001
            ev["status"] = "error"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            if ev["status"] == "running":
                ev["status"] = "ok"

    # -- 读取 --------------------------------------------------------------
    def events(self, owner: str | None = None) -> list[dict]:
        if owner is None:
            out = [e for s in self._sessions.values() for e in s["events"]]
            return sorted(out, key=lambda e: e["seq"])
        return list((self._sessions.get(owner) or {}).get("events") or [])

    def complete(self, owner: str) -> list[dict]:
        """**完整**的采样事件（`status == "ok"`）。`running`/`error` 都不算。"""
        return [e for e in self.events(owner) if e["status"] == "ok"]

    def summary(self, owner: str) -> dict:
        ev = self.events(owner)
        by = {"ok": 0, "error": 0, "running": 0}
        for e in ev:
            by[e["status"]] = by.get(e["status"], 0) + 1
        sess = self._sessions.get(owner) or {}
        return {"owner": owner, "session_closed": bool(sess.get("closed")),
                "n_events": len(ev), "n_ok": by["ok"], "n_error": by["error"],
                "n_running": by["running"],
                "event_ids": [e["event_id"] for e in ev],
                "n_files_per_event": [len((e["snapshot"] or {}).get("files", {})
                                          if isinstance(e["snapshot"], Mapping)
                                          else e["snapshot"] or []) for e in ev]}

    def reset(self) -> None:
        """清空**全部**状态。进程内复用日志对象前必须调用。"""
        self._sessions.clear()
        self._stack.clear()
        self._seq = 0


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def _selftest() -> bool:
    import tempfile
    ok = True
    root = Path(tempfile.mkdtemp(prefix="cap_selftest_"))
    try:
        (root / "sub").mkdir()
        (root / "bm_top.py").write_text("x", encoding="utf-8")
        (root / "sub" / "m.py").write_text("x", encoding="utf-8")
        cases = [
            ("相对路径", str(root / "bm_top.py"), "project", "bm_top.py"),
            ("正斜杠绝对", (root / "bm_top.py").as_posix(), "project", "bm_top.py"),
            ("点斜杠", "./bm_top.py", "project", "bm_top.py"),
            ("上一级", "sub/../bm_top.py", "project", "bm_top.py"),
            ("子目录", "sub/m.py", "project", "sub/m.py"),
            ("越界", "../outside.py", "external", None),
        ]
        for label, raw, want_state, want_path in cases:
            r = normalize_path(raw, root)
            good = r["state"] == want_state and r["path"] == want_path
            ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {label}: {r['state']} {r['path']}")

        # 同一身份的两种写法必须归一
        a = normalize_path(str(root / "bm_top.py"), root)["path"]
        b = normalize_path("bm_top.py", root)["path"]
        c = normalize_path("sub/../bm_top.py", root)["path"]
        good = a == b == c == "bm_top.py"
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] 三种写法归一：{a} / {b} / {c}")

        s = normalize_set(["bm_top.py", "./bm_top.py"], root)
        good = len(s["collisions"]) == 1 and s["collisions"][0]["kind"] == "duplicate"
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] 重复检出：{s['collisions']}")

        items = [{"path": "bm_top.py", "role": "module", "sha256": "a" * 64,
                  "compute_role": "numeric"},
                 harness_item("sub/m.py", sha256="b" * 64, compute_role="cli_only",
                              why_not_numeric="只采集", evidence="本文件")]
        cr = classify_runtime({str(root / "bm_top.py"): ["bm_top"],
                               str(root / "sub" / "m.py"): ["__main__"],
                               str(root / "sub" / "h.py"): ["h"]}, items, root=root)
        cls = {e["path"]: e["runtime_class"] for e in cr["runtime_loaded_modules"]}
        good = (cls == {"bm_top.py": "project", "sub/h.py": "undeclared",
                        "sub/m.py": "harness"})
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] 归属分类：{cls}")

        log = SamplingEventLog()
        with log.session("R1-x"):
            with log.event(snapshot={"files": {"a": ["a"]}}, effective={"n": 1}):
                pass
            with log.event(snapshot={"files": {"a": ["a"], "b": ["b"]}},
                           effective={"n": 2}):
                pass
        ev = log.events("R1-x")
        good = (len(ev) == 2 and ev[0]["event_id"] == "R1-x#0"
                and len(ev[0]["snapshot"]["files"]) == 1
                and len(ev[1]["snapshot"]["files"]) == 2)
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] 两次采样不覆盖："
              f"{[len(e['snapshot']['files']) for e in ev]}")
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n自检{'通过' if ok else '失败'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selftest() else 1)
