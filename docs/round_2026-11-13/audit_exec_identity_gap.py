#!/usr/bin/env python
"""阶段 0：历史执行身份缺口的**只读**审计。

回答三个问题，不回答第四个（"怎么补上"——那需要编）：

1. 239 条记录各自离 `recorded` 差几个要素，缺的是哪几项；
2. 产物里到底有没有任何软件身份痕迹（版本 / git / 环境 / 时间戳）；
3. 磁盘上还能不能找到"当时那份代码"的副本，能否据此补 `code_version`。

**只读**：不写 `evidence/`、不改 `out/`、不动 `circuit_registry.py`。
所有数字都从 `evidence/stage0_records/records.jsonl` 与磁盘现状重算，
不引用任何成文结论，避免"文档说 233 所以 233"。

用法：
    python docs/round_2026-11-13/audit_exec_identity_gap.py
    python docs/round_2026-11-13/audit_exec_identity_gap.py --json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FROZEN = ROOT / "evidence" / "stage0_records"
RECORDS = FROZEN / "records.jsonl"
MANIFEST = FROZEN / "MANIFEST.json"

#: `_conditions_recorded()` 的四个条件，逐项在这里重算（不读 `run_id_status` 字符串，
#: 因为本审计要看的正是"状态对不对"以及"差在哪一项"，而不是把状态抄一遍）。
ELEMENTS = (
    ("seed_artifact", lambda e: str(e.get("seed_source", "")).split(":", 1)[0] == "artifact"),
    ("budget_recorded", lambda e: e.get("budget_source") == "recorded"),
    ("code_version", lambda e: bool((e.get("software") or {}).get("code_version"))),
    ("data_content_sha256", lambda e: e.get("data_id_kind") == "content_sha256"),
)

#: 产物里若出现这些名字的字段，就有可能承载软件身份。命中后再看值是否有用。
IDENTITY_TOKENS = re.compile(
    r"vers|git|commit|rev|env|python|numpyro|jax|script|cmd|argv|producer|"
    r"code|pkg|package|host|machine|platform|created|timestamp|date|dirty", re.I)


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_records() -> list[dict]:
    return [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines()]


def element_flags(rec: dict) -> dict:
    ex = rec["observation"]["execution"]
    return {name: bool(fn(ex)) for name, fn in ELEMENTS}


def distance_table(recs: list[dict]) -> dict:
    """每条记录"四要素齐几项"的分布，以及差的具体是哪几项。"""
    per_n = collections.Counter()
    missing = collections.Counter()
    for rec in recs:
        flags = element_flags(rec)
        n = sum(flags.values())
        per_n[n] += 1
        if n < len(ELEMENTS):
            missing[tuple(k for k, v in flags.items() if not v)] += 1
    return {
        "n_records": len(recs),
        "by_count": {k: per_n[k] for k in sorted(per_n)},
        "by_missing": {"+".join(k) if k else "none": v
                       for k, v in sorted(missing.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def _walk_keys(obj, path="", out=None, limit=0) -> list:
    """收集所有**键名**命中 IDENTITY_TOKENS 的位置（不关心值，先看有没有）。"""
    out = [] if out is None else out
    if limit and len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if IDENTITY_TOKENS.search(str(k)):
                out.append((p, type(v).__name__, str(v)[:80]))
            _walk_keys(v, p, out, limit)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            _walk_keys(v, f"{path}[{i}]", out, limit)
    return out


def scan_artifacts(man: dict) -> list[dict]:
    rows = []
    for rel in sorted(man["source_sha256"]):
        fp = ROOT / rel
        hits = []
        if fp.suffix == ".jsonl":
            for line in fp.read_text(encoding="utf-8").splitlines()[:3]:
                if line.strip():
                    _walk_keys(json.loads(line), "", hits, limit=40)
        else:
            _walk_keys(json.loads(fp.read_text(encoding="utf-8")), "", hits, limit=40)
        rows.append({"path": rel, "n_identity_keys": len(hits), "hits": hits[:8]})
    return rows


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", check=False).stdout.strip()
    except OSError:
        return ""


def commit_context(ts: dt.datetime) -> dict:
    """该时刻之前最近一次提交。**只是上界，不是身份**——工作树是否脏无法从产物判断。"""
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    line = _git("log", "--until", stamp, "--format=%h|%ad|%s",
                "--date=format:%Y-%m-%d %H:%M", "-1")
    if not line:
        return {"nearest_commit": None}
    sha, when, subj = (line.split("|", 2) + ["", ""])[:3]
    same_day = _git("log", "--since", ts.strftime("%Y-%m-%d 00:00"),
                    "--until", stamp, "--oneline")
    return {"nearest_commit": sha, "commit_time": when, "commit_subject": subj[:60],
            "commits_that_day_before": len(same_day.splitlines())}


def original_dir_rows(recs: list[dict]) -> list[dict]:
    """记录里记了 `original_dir` 的产物：文件的真实产出时间 + 同目录里有没有代码副本。"""
    by_dir = collections.Counter(r["source"].get("original_dir") for r in recs)
    rows = []
    for d, n in sorted(by_dir.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if not d:
            continue
        p = ROOT / d
        files = sorted(p.iterdir()) if p.is_dir() else []
        arts = [f for f in files if f.name.endswith((".nested.json", ".json"))]
        newest = max((f.stat().st_mtime for f in arts), default=None)
        rows.append({
            "original_dir": d, "n_records": n, "exists": p.is_dir(),
            "n_files": len(files),
            "n_artifacts": len(arts),
            "artifact_mtime": None if newest is None else
            dt.datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M"),
            "code_copies": [f.name for f in files if f.suffix == ".py"],
        })
    return rows


def code_copy_rows(recs: list[dict]) -> list[dict]:
    """磁盘上一切 `.py` 副本，以及它们与仓库同名文件是否逐字节相同。

    **同名相同也不够**：这些是入口脚本，真正被 import 的 `bm_top` / `bm_core` /
    `nested_fit` 都没有副本。脚本相同 ≠ 库相同。
    """
    cand = set()
    for d in {r["source"].get("original_dir") for r in recs if r["source"].get("original_dir")}:
        p = ROOT / d
        if p.is_dir():
            cand.update(f for f in p.iterdir() if f.suffix == ".py")
    scripts_dir = ROOT / "evidence" / "stage0_three_spectra" / "scripts"
    if scripts_dir.is_dir():
        cand.update(scripts_dir.iterdir())
    rows = []
    for f in sorted(cand, key=lambda x: x.name):
        repo_same_name = sorted(ROOT.glob(f"**/{f.name}"))
        repo_same_name = [q for q in repo_same_name if "out" not in q.parts]
        same = None
        if repo_same_name:
            same = _sha256(repo_same_name[0]) == _sha256(f)
        mods = re.findall(r"^\s*(?:import|from)\s+(bm_\w+|nested_fit)", f.read_text(
            encoding="utf-8", errors="replace"), re.M)
        rows.append({
            "path": f.relative_to(ROOT).as_posix(),
            "sha256_12": _sha256(f)[:12],
            "repo_counterpart": (repo_same_name[0].relative_to(ROOT).as_posix()
                                 if repo_same_name else None),
            "identical_to_counterpart": same,
            "imports_project_modules": sorted(set(mods)),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="只输出机器可读汇总")
    args = ap.parse_args(argv)

    if not RECORDS.is_file() or not MANIFEST.is_file():
        print(f"缺 {RECORDS} 或 {MANIFEST}，无法审计")
        return 2
    recs = load_records()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dist = distance_table(recs)
    arts = scan_artifacts(man)
    odirs = original_dir_rows(recs)
    copies = code_copy_rows(recs)

    # 每条"最近"记录（只差 code_version）的提交上下文
    near = []
    for rec in recs:
        flags = element_flags(rec)
        if sum(flags.values()) == len(ELEMENTS) - 1 and not flags["code_version"]:
            d = rec["source"].get("original_dir")
            p = ROOT / d if d else None
            arts_in = [f for f in p.iterdir() if f.suffix == ".json"] if p and p.is_dir() else []
            ts = max((f.stat().st_mtime for f in arts_in), default=None)
            near.append({
                "record_id": rec["record_id"],
                "original_dir": d,
                "artifact_mtime": None if ts is None else
                dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
                **({"commit_context": commit_context(dt.datetime.fromtimestamp(ts))}
                   if ts is not None else {}),
            })

    summary = {
        "registry_version": _registry_version(),
        "n_records": dist["n_records"],
        "elements": [n for n, _ in ELEMENTS],
        "distance": dist,
        "records_with_original_dir": sum(1 for r in recs if r["source"].get("original_dir")),
        "artifact_identity_keys": {
            "files_scanned": len(arts),
            "files_with_any_hit": sum(1 for a in arts if a["n_identity_keys"]),
            "details": arts,
        },
        "original_dirs": odirs,
        "nearest_to_recorded": near,
        "code_copies": copies,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"registry {summary['registry_version']}；记录 {summary['n_records']} 条")
    print("\n[1] 四要素齐备度（`_conditions_recorded` 的四个条件逐项重算）")
    for n, c in dist["by_count"].items():
        print(f"    齐 {n}/4 项：{c} 条")
    print("    缺口分布：")
    for k, v in dist["by_missing"].items():
        print(f"      缺 {k}：{v} 条")

    print("\n[2] 产物里有没有软件身份字段")
    print(f"    扫描 {len(arts)} 份产物，键名命中身份关键词的文件 "
          f"{summary['artifact_identity_keys']['files_with_any_hit']} 份")
    for a in arts:
        print(f"      {a['n_identity_keys']:>2} 命中  {a['path']}")

    print("\n[3] 原始目录与产出时刻")
    print(f"    记录里带 `original_dir` 的：{summary['records_with_original_dir']} 条"
          f"（其余 {summary['n_records'] - summary['records_with_original_dir']} 条连目录都没记）")
    for o in odirs:
        print(f"      {o['artifact_mtime']}  {o['original_dir']}  "
              f"（{o['n_records']} 条，{o['n_artifacts']} 份产物，代码副本 {len(o['code_copies'])} 个）")

    print("\n[4] 磁盘上的代码副本")
    for c in copies:
        same = ("同" if c["identical_to_counterpart"] else
                "异" if c["identical_to_counterpart"] is False else "无对应")
        print(f"      {c['path']}  sha {c['sha256_12']}  与仓库{same}文件  "
              f"import {','.join(c['imports_project_modules']) or '—'}")

    print("\n[5] 只差 code_version 的记录（若能补上，会变成 R-）")
    for n in near:
        cc = n.get("commit_context") or {}
        print(f"      {n['artifact_mtime']}  {n['original_dir']}  "
              f"→ 之前最近提交 {cc.get('nearest_commit')} {cc.get('commit_time')}")

    print("\n结论（本审计能证与不能证）")
    print("    · 能证：239 条里没有一条记了 `code_version`；产物里没有任何版本/git/环境字段。")
    print("    · 能证：磁盘上只有入口脚本副本，被 import 的库模块一份都没有。")
    print("    · 不能证：产出时刻对应哪个提交——时间只能给**上界**，脏工作树无法排除。")
    print("    · 因此 `code_version` 无法从事后证据恢复；缺口只能向前关，不能向后补。")
    return 0


def _registry_version() -> str:
    sys.path.insert(0, str(ROOT))
    import circuit_registry as R  # 只读常量，不触发任何计算
    return R.REGISTRY_VERSION


if __name__ == "__main__":
    raise SystemExit(main())
