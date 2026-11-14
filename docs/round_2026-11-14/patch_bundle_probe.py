# -*- coding: utf-8 -*-
"""真实工作树的补丁包往返实测（② 的证据脚本）。

四个场景，都不用 mock：

A 干净工作树（profile B 的 19 个声明文件）：证明 clean 树上 bundle 全是 `base`，
  重建后逐文件摘要与身份三层摘要全同。
B 干净工作树 + 3 个 harness 项（22 个）：证明 harness 声明也能进 bundle 并被重建，
  同时**不改变** numeric 层的口径（harness 的 compute_role 非 numeric）。
C 同一 commit 的**脏克隆**（staged / unstaged / 未跟踪 / ignored 四类同时存在）：
  证明四类来源都能被正确归属并重建。
D 脏克隆里删掉一个声明文件：证明 `incomplete` 而且**不重建**（P0）。

在克隆里做脏状态，不在用户的工作树上做——这条不是洁癖：脏状态需要改文件、
删文件，而"验证工具"不该拥有改动被测对象的权力。

    python docs/round_2026-11-14/patch_bundle_probe.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import patch_bundle as PB          # noqa: E402
import patch_bundle_io as PIO      # noqa: E402
import exec_identity as EI         # noqa: E402

OUT = ROOT / "out" / "pb"
HARNESS = [
    ("capture_contract.py", "cli_only"),
    ("exec_identity.py", "reference_only"),      # 分类待裁，见报告 §5
    ("docs/round_2026-11-14/snapshot_probe.py", "cli_only"),
]


def _git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *_PIO_CFG, *args], cwd=str(root), capture_output=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败："
                         f"{proc.stderr.decode('utf-8', 'replace')[:300]}")
    return proc.stdout.decode("utf-8", "replace")


_PIO_CFG = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
            "-c", "core.quotepath=false", "-c", "user.name=pb_probe",
            "-c", "user.email=pb@probe", "-c", "commit.gpgsign=false")


def _decl(extra: list[dict] | None = None) -> list[dict]:
    items = PIO.declaration_from_manifest(ROOT, "B")
    for path, role in HARNESS:
        items.append(PIO.harness_item(path, role))
    items.extend(extra or [])
    return items


def _tiers(items) -> dict:
    return {L: EI.source_tree_sha256(items, layer=L)
            for L in ("declared", "numeric", "workflow")}


def _slim(rep: dict) -> dict:
    b = rep["bundle"]
    rb = rep["rebuild"]
    return {
        "verdict": rep["verdict"],
        "tree_digest": b["tree_digest"],
        "bundle_bytes": len(PB.bundle_bytes(b)),
        "base": b["base"]["commit"][:12],
        "counts": b["counts"],
        "sources": {f["path"]: {"source": f["source"],
                                "patch_ref": f.get("patch_ref"),
                                "why_blob": f.get("why_blob")}
                    for f in b["files"] if f["source"] != "base"},
        "patches": [{"name": p["name"], "bytes": p["bytes"], "paths": p["paths"]}
                    for p in b["patches"]],
        "notes": b["capture"].get("notes"),
        "tiers": rb["tiers"]["expected"],
        "tiers_equal": rb["tiers"]["compare"]["match"],
        "files_equal": rb["files"]["compare"]["match"],
        "n_files_checked": rb["files"]["n_checked"],
        "n_extracted": rb.get("n_extracted"),
        "mode_check": rb["modes"]["mode_check"],
        "structural": {k: rep["structural"][k] for k in
                       ("verdict", "n_declared", "n_files", "n_missing", "missing")},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pb_probe_"))
    t0 = time.time()
    report: dict = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "repo_commit": _git(ROOT, "rev-parse", "HEAD").strip(),
                    "python": sys.version.split()[0], "scenarios": {}}
    try:
        # ---- A / B：干净工作树 -------------------------------------------
        decl_a = PIO.declaration_from_manifest(ROOT, "B")
        rep_a = PIO.roundtrip(ROOT, declaration=decl_a, workdir=work / "A")
        report["scenarios"]["A_clean_profile_B"] = _slim(rep_a)

        decl_b = _decl()
        rep_b = PIO.roundtrip(ROOT, declaration=decl_b, workdir=work / "B")
        report["scenarios"]["B_clean_with_harness"] = _slim(rep_b)
        thin = _tiers(PB.tier_items(rep_b["bundle"]))
        report["scenarios"]["B_clean_with_harness"]["numeric_same_as_A"] = (
            thin["numeric"] == report["scenarios"]["A_clean_profile_B"]["tiers"]["numeric"])
        report["scenarios"]["B_clean_with_harness"]["declared_changed_vs_A"] = (
            thin["declared"] != report["scenarios"]["A_clean_profile_B"]["tiers"]["declared"])

        # ---- C / D：同一 commit 的脏克隆 ---------------------------------
        clone = work / "clone"
        _git(ROOT, "clone", "-q", "--no-local", str(ROOT), str(clone))
        base_clone = _git(clone, "rev-parse", "HEAD").strip()

        (clone / "bm_ad.py").write_text("# probe: unstaged\n", encoding="utf-8", newline="")
        (clone / "bm_core.py").write_text("# probe: staged\n", encoding="utf-8", newline="")
        _git(clone, "add", "bm_core.py")
        (clone / "bm_corr.py").write_text("# probe: staged\n", encoding="utf-8", newline="")
        _git(clone, "add", "bm_corr.py")
        (clone / "bm_corr.py").write_text("# probe: staged + unstaged\n",
                                          encoding="utf-8", newline="")
        (clone / "probe_new_module.py").write_text(
            "NEW = 1  # 未跟踪，但被声明成 harness\n", encoding="utf-8", newline="")
        (clone / "out").mkdir(exist_ok=True)
        (clone / "out" / "probe_ignored.json").write_text(
            '{"probe": "ignored but declared"}\n', encoding="utf-8", newline="")

        extra = [PIO.harness_item("probe_new_module.py", "cli_only"),
                 {"path": "out/probe_ignored.json", "role": "config"}]
        decl_c = _decl(extra)
        rep_c = PIO.roundtrip(clone, declaration=decl_c, workdir=work / "C")
        report["scenarios"]["C_dirty_clone"] = _slim(rep_c)
        report["scenarios"]["C_dirty_clone"]["clone_base"] = base_clone[:12]
        report["scenarios"]["C_dirty_clone"]["tiers_differ_from_clean"] = (
            rep_c["rebuild"]["tiers"]["expected"]["numeric"]
            != report["scenarios"]["A_clean_profile_B"]["tiers"]["numeric"])

        # D：删掉一个声明文件
        (clone / "bm_ws.py").unlink()
        b_d = PIO.collect(clone, declaration=decl_c)
        v_d = PB.verify(b_d)
        rep_d = PIO.rebuild(b_d, dest=work / "D" / "rebuilt", repo=clone)
        report["scenarios"]["D_deleted_declared"] = {
            "structural": {k: v_d[k] for k in
                           ("verdict", "n_declared", "n_files", "n_missing", "missing")},
            "reasons": [r["code"] for r in v_d["reasons"]],
            "rebuild_verdict": rep_d["verdict"],
            "rebuild_attempted": rep_d["attempted"],
            "why": rep_d.get("why"),
        }

        report["elapsed_s"] = round(time.time() - t0, 1)
        (OUT / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v["verdict"] for k, v in report["scenarios"].items()
                          if isinstance(v, dict) and "verdict" in v}, ensure_ascii=False))
        print(f"report → {OUT / 'report.json'}  ({report['elapsed_s']} s)")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
