"""重算 P1 的 before/after 溯源快照（修掉 2026-11-09 首次执行时的假记录）。

为什么需要这个脚本：`p_after_e2.snapshot()` 早期用的是裸 `import bm_kk`，
命中 `sys.modules` 缓存 —— 拍 before、`git apply`、再拍 after，两次拿到同一份
旧字节码，`n_changed` 恒为 0。补丁本身是**生效**的（`bm_check` 前后都 rc=0，
影子对拍 0 处不一致），但溯源表错写成"什么都没变"。

本脚本不改 `p_after_e2.py` 的流程，只做一件事：**用两个独立子进程**分别加载
旧/新 `bm_kk.py` 各拍一次快照，再 diff。子进程隔离保证不可能复用旧模块。
旧记录保留为 `p1_before.json.stale` / `p1_after.json.stale`，不删除。

用法：PY evidence/ws_crit/p1_recompute.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BAK = HERE / "patches" / "bm_kk.py.pre_p1"
SRC = ROOT / "bm_kk.py"
PY = sys.executable

SNAP_CODE = r"""
import json, sys
sys.path.insert(0, r"{root}")
sys.path.insert(0, r"{here}")
import p_after_e2 as P
rows = P.snapshot()
open(r"{out}", "w", encoding="utf-8").write(
    json.dumps(rows, ensure_ascii=False, indent=1))
print(len(rows))
"""


def take(tag: str) -> dict:
    out = HERE / f"_snap_{tag}.json"
    code = SNAP_CODE.format(root=ROOT, here=HERE, out=out)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"快照 {tag} 失败：\n{r.stderr[-1500:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    if not BAK.exists():
        raise SystemExit(f"缺备份 {BAK}")
    # **按字节搬运**：`read_text`/`write_text` 会做换行归一化，写回时把 CRLF
    # 变成 LF，整文件 diff 就全是噪声、sha 也对不上。溯源脚本必须是字节级无害的。
    patched = SRC.read_bytes()
    old = BAK.read_bytes()
    if patched == old:
        raise SystemExit("当前 bm_kk.py 与备份相同 → 补丁没打在树上，先确认状态")

    # 1) 旧代码快照
    SRC.write_bytes(old)
    try:
        before = take("before")
    finally:
        SRC.write_bytes(patched)

    # 2) 新代码快照（已经是当前内容，独立子进程重新加载）
    after = take("after")

    # 3) 旧记录留档，再写新记录
    for name in ("p1_before.json", "p1_after.json"):
        p = HERE / name
        if p.exists():
            shutil.copy2(p, p.with_suffix(".json.stale"))
    (HERE / "p1_before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "p1_after.json").write_text(
        json.dumps(after, ensure_ascii=False, indent=1), encoding="utf-8")

    import hashlib

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    FIELDS = ("amp_ok", "phase_ok", "is_candidate", "f_cut", "slope_z", "slope_im")
    changes = []
    for tag in before:
        if tag not in after:
            continue
        for k in FIELDS:
            va, vb = before[tag].get(k), after[tag].get(k)
            same = va == vb
            if (isinstance(va, float) and isinstance(vb, float)
                    and va != va and vb != vb):        # 双 nan
                same = True
            if not same:
                changes.append(dict(tag=tag, field=k, before=va, after=vb))

    prov = dict(
        sha_before=sha(BAK), sha_after=sha(SRC),
        n_spectra=len(before), n_changed=len({c["tag"] for c in changes}),
        n_field_changes=len(changes), changes=changes,
        recomputed_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
        recompute_reason=(
            "首次自动执行时 snapshot() 用裸 import，命中 sys.modules 缓存，"
            "before/after 拿到同一份旧字节码 → 原记录 n_changed=0 是假的。"
            "本文件由 p1_recompute.py 用两个独立子进程重新测算；"
            "原记录保留为 *.json.stale。补丁的生效性另有两条独立证据："
            "影子实现对拍 0/24 不一致、bm_check 前后均 rc=0。"),
    )
    (HERE / "p1_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"语料 {len(before)} 条，字段级变化 {len(changes)} 处，"
          f"涉及 {prov['n_changed']} 条谱")
    for c in changes:
        print(f"  {c['tag']:28s} {c['field']:12s} {c['before']} → {c['after']}")
    print(f"sha before {prov['sha_before'][:12]}  after {prov['sha_after'][:12]}")
    print(f"已写 p1_before.json / p1_after.json / p1_provenance.json")
    for f in HERE.glob("_snap_*.json"):
        f.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
