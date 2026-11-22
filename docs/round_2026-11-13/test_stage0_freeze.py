# -*- coding: utf-8 -*-
"""test_stage0_freeze.py —— 冻结快照的完整性测试（审计 D4 / D5）。

`test_circuit_registry.py` 测的是**契约**（schema 能不能表达、乱填会不会报错）。
这个文件测的是**证据快照本身**：`--check` 到底会不会发现漂移。

它必须发现以下几类（12 条用例）：

| 编号 | 篡改 | 期望 |
|---|---|---|
| D0 | 什么都不改 | 通过（基线；基线不过，后面所有"失败"都不可信） |
| D4a | `MANIFEST.registry_digest` 改成**未登记**值 | 失败，并指出 digest |
| D4b | `MANIFEST.registry_version` 改成**未登记**值 | 失败 |
| D4c | `MANIFEST` 代际改成**另一个已登记**代际 | 通过、但单列代际差（不阻断） |
| D5a | 原始证据文件字节变了 | 失败，并指出 source_sha256 |
| D5b | 原始证据字节没变 | 通过（不能把"没漂移"判成漂移） |
| D5c | 冻结副本 `records.jsonl` 被改 | 失败 |
| D5d | `MANIFEST.json` 被删 | 失败（查不了 ≠ 没问题） |
| D5e | 原始**谱**（仓库外文件）内容变了 | 失败（data_sha256） |
| D5f | 清单**少一个** `files` 条目，其他一字不动 | 失败（删字段也算漂移） |
| D6 | 按记录**自报**代际复算 `run_id` | 通过（239/239 可复算 = 历史身份仍算数） |
| D6b | 改掉一个 `run_id` 再复算 | 失败（复算审计不能恒真） |

D0 的语义（2026-11-22 订正）：通过 = **完整性**通过。冻结快照属 0.7.0/`b2675c82`、
现役是 0.8.0/`aeb6aca1`，这个代际差**按设计不阻断**（`_KNOWN_GENERATIONS`），
否则 `--check` 永久红、判据失效。D4a/D4b 挡的仍是"改成不存在的代际"。
D4c 是这条新语义的正面用例：改成**另一个已登记**代际不算篡改——**这是已知口子**，
要靠 `GENERATION.json` 给 MANIFEST 一个外部锚点才关得上。

做法：把 `evidence/stage0_records/` 复制到临时目录再改那一份，**不动仓库里的冻结副本**。

用法：python docs/round_2026-11-13/test_stage0_freeze.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import circuit_registry as R  # noqa: E402
import map_stage0_records as M  # noqa: E402


def _run(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = M.main(argv)
    return rc, buf.getvalue()


def _copy_frozen() -> Path:
    d = Path(tempfile.mkdtemp(prefix="stage0_frozen_"))
    dst = d / "stage0_records"
    shutil.copytree(M.FROZEN, dst)
    return dst


def _edit_manifest(frozen: Path, **kw) -> None:
    p = frozen / "MANIFEST.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    man.update(kw)
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8", newline="\n")


def case_baseline_passes():
    """基线：未改动的冻结副本必须通过，否则后面所有"失败"都不可信。"""
    rc, out = _run(["--check"])
    assert rc == 0, f"基线快照没通过（rc={rc}）：\n{out}"
    assert "通过" in out, out
    return "未改动的冻结快照 --check 返回 0"


def case_d4a_registry_digest_tampered():
    frozen = _copy_frozen()
    _edit_manifest(frozen, registry_digest="0000000000000000")
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 2, f"改了 registry_digest 却通过（rc={rc}）：\n{out}"
    assert "registry_digest" in out, out
    return "MANIFEST.registry_digest 被改 ⇒ 失败（注册表漂移可发现）"


def case_d4b_registry_version_tampered():
    frozen = _copy_frozen()
    _edit_manifest(frozen, registry_version="0.0.1")
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 2, f"改了 registry_version 却通过（rc={rc}）：\n{out}"
    assert "registry_version" in out, out
    return "MANIFEST.registry_version 被改 ⇒ 失败"


def case_d4c_known_generation_gap_passes():
    """改成**另一个已登记**代际 ⇒ 完整性仍通过，代际差单列。

    这是 2026-11-22 新语义的正面用例：`--check` 不能因为"快照属于旧代际"就红，
    否则永久红。同时它把**已知口子**钉在测试里——把声明换成另一个真实存在过的
    代际抓不到，因为 MANIFEST 没有外部锚点（`GENERATION.json` 才关得上）。
    """
    frozen = _copy_frozen()
    cur = (R.REGISTRY_VERSION, R.registry_digest())
    _edit_manifest(frozen, registry_version=cur[0], registry_digest=cur[1])
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 0, f"改成已登记代际却报失败（rc={rc}）：\n{out}"
    assert "已登记" in out and "完整性通过" in out, out
    return "代际改成另一已登记代际 ⇒ 完整性通过、代际差单列（已知口子，待 GENERATION.json）"


def case_d5a_source_bytes_changed():
    """原始证据字节变了 ⇒ 必须失败（拿一份改过的副本冒充原文件）。"""
    man = json.loads((M.FROZEN / "MANIFEST.json").read_text(encoding="utf-8"))
    rel, want = sorted(man["source_sha256"].items())[0]
    tmp = Path(tempfile.mkdtemp(prefix="stage0_src_"))
    fake = tmp / "tampered.json"
    fake.write_bytes((_ROOT / rel).read_bytes() + b"\n")
    probe = {"registry_version": R.REGISTRY_VERSION,
             "registry_digest": R.registry_digest(),
             "source_sha256": {rel: want}, "files": {}}
    probs, cross = M.verify_frozen(tmp, {}, probe, {rel: fake})
    assert any("source_sha256" in p for p in probs), probs
    assert not cross, cross
    return f"原始证据 {Path(rel).name} 字节变化 ⇒ verify_frozen 报 source_sha256 不符"


def case_d5b_source_unchanged_passes():
    """没变的证据不能被判成漂移（假阳性同样会让冻结检查失去意义）。"""
    man = json.loads((M.FROZEN / "MANIFEST.json").read_text(encoding="utf-8"))
    rel, want = sorted(man["source_sha256"].items())[0]
    probe = {"registry_version": R.REGISTRY_VERSION,
             "registry_digest": R.registry_digest(),
             "source_sha256": {rel: want}, "files": {}}
    probs, cross = M.verify_frozen(M.FROZEN, {}, probe, {rel: _ROOT / rel})
    assert not probs, probs
    assert not cross, cross
    return "未变的原始证据不报警"


def case_d5e_data_file_changed():
    """原始**谱**（不是产物）字节变了也要抓到：这类文件在仓库外，容易漏。"""
    man = json.loads((M.FROZEN / "MANIFEST.json").read_text(encoding="utf-8"))
    rel, want = sorted(man["data_sha256_files"].items())[0]
    tmp = Path(tempfile.mkdtemp(prefix="stage0_data_"))
    fake = tmp / "spectrum.csv"
    fake.write_bytes(Path(rel).read_bytes() + b"\n")
    probe = {"registry_version": R.REGISTRY_VERSION,
             "registry_digest": R.registry_digest(),
             "data_sha256_files": {str(fake): want}, "files": {}}
    probs, cross = M.verify_frozen(tmp, {}, probe, {})
    assert any("data_sha256" in p for p in probs), probs
    assert not cross, cross
    return f"原始谱 {Path(rel).name} 内容变化 ⇒ verify_frozen 报 data_sha256 不符"


def case_d5c_frozen_copy_tampered():
    frozen = _copy_frozen()
    p = frozen / "records.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 2, f"冻结副本被改却通过（rc={rc}）：\n{out}"
    assert "records.jsonl" in out, out
    return "冻结 records.jsonl 被改 ⇒ 失败（重算字节与磁盘字节双向比）"


def case_manifest_missing():
    frozen = _copy_frozen()
    (frozen / "MANIFEST.json").unlink()
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 2, f"MANIFEST 缺失却通过（rc={rc}）：\n{out}"
    return "MANIFEST 缺失 ⇒ 失败（不许把查不了当成没问题）"


def case_d5f_manifest_files_entry_removed():
    """清单少一个 `files` 条目（其他一字不动）必须失败。

    单看 `verify_frozen` 的第 2 层是**发现不了**的：它遍历的是清单自己的 `files`，
    少一条就少查一条。真正抓住它的是第 1 层——重新生成的 `MANIFEST.json` 文本
    与冻结副本字节不同。这个用例就是为了钉住"删字段 ≠ 没问题"。
    """
    frozen = _copy_frozen()
    p = frozen / "MANIFEST.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    drop = sorted(man["files"])[0]
    man["files"].pop(drop)
    p.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8", newline="\n")
    rc, out = _run(["--check", "--frozen", str(frozen)])
    assert rc == 2, f"清单少了 files[{drop}] 却通过（rc={rc}）：\n{out}"
    assert "MANIFEST.json" in out, out
    return f"删掉 MANIFEST.files[{drop}] ⇒ 失败（删字段也算漂移，不是'少查一条'）"


def case_d6_runids_recomputable():
    """按记录**自报**的代际，239 条 run_id 必须能全部复算。

    这条修的是 L2：`_check_execution()` 会用当前 `REGISTRY_VERSION` 覆盖记录里
    的版本号，于是冻结记录一律复算不符（实测 D-4d61cf10ae → D-9751ecfac8）。
    审计侧按自报代际复算，才能真正回答"这 239 条身份还算不算数"。
    """
    ok, bad, skip, unver = M.recompute_run_ids(M.FROZEN / "records.jsonl")
    assert not bad, f"复算不符 {len(bad)} 条：{bad[:3]}"
    assert not unver, f"{unver} 条没有 registry_version，不能算通过"
    assert ok > 0, "一条都没复算出来"
    return f"{ok} 条按自报代际复算全部一致（跳过 {skip}）"


def case_d6b_recompute_detects_wrong_runid():
    """复算审计不能是恒真：改掉一个 run_id 必须报不通过。"""
    frozen = _copy_frozen()
    p = frozen / "records.jsonl"
    lines = p.read_text(encoding="utf-8").split("\n")
    rec = json.loads(lines[0])
    old = rec["observation"]["execution"]["run_id"]
    rec["observation"]["execution"]["run_id"] = old[:-4] + "beef"
    lines[0] = json.dumps(rec, ensure_ascii=False)
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    ok, bad, _, unver = M.recompute_run_ids(p)
    assert len(bad) == 1, f"改了 run_id 却没报（bad={bad}）"
    assert not unver, unver
    return f"run_id 被改 ⇒ 复算报不符（{old} → 期望 {ok} 条通过、1 条不通过）"


CASES = [
    ("D0 基线通过", case_baseline_passes),
    ("D4a registry_digest 被改", case_d4a_registry_digest_tampered),
    ("D4b registry_version 被改", case_d4b_registry_version_tampered),
    ("D4c 代际改成另一已登记代际", case_d4c_known_generation_gap_passes),
    ("D5a 原始证据被改", case_d5a_source_bytes_changed),
    ("D5b 原始证据未变", case_d5b_source_unchanged_passes),
    ("D5e 原始谱被改", case_d5e_data_file_changed),
    ("D5c 冻结副本被改", case_d5c_frozen_copy_tampered),
    ("D5d MANIFEST 缺失", case_manifest_missing),
    ("D5f 清单少一个 files 条目", case_d5f_manifest_files_entry_removed),
    ("D6 run_id 按自报代际可复算", case_d6_runids_recomputable),
    ("D6b 复算审计能抓错号", case_d6b_recompute_detects_wrong_runid),
]


def main() -> int:
    ok, bad = 0, []
    for name, fn in CASES:
        try:
            msg = fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except AssertionError as exc:
            bad.append(name)
            print(f"[FAIL] {name}：{exc}")
    print(f"\n{ok}/{len(CASES)} 通过" + (f"；未通过：{', '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
