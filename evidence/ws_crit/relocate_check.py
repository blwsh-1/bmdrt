# -*- coding: utf-8 -*-
"""迁移对拍：生产 `bm_ws` vs 探索期 `evidence/ws_crit/bm_wscrit`。

目的
----
`bm_ws.py` 是从 `bm_wscrit.py` 复制派生出来的（只改文件头定位）。本脚本证明
**搬迁没有改变任何判据**：对 31 条谱（E2 的 18 条真实 AZ31 + examples 合成谱）
逐条比较 `ws_diagnostics()` 的全部字段，再对 8 组已落盘的 npz 后验比较
`rpc_share()` / `ws_share_low()` 的全部字段。

判据：数值字段 `abs(a-b) <= 1e-12`（或同时 nan）、字符串与布尔逐项相等。

产物：`relocate_check.json`（差异清单；空列表 = 逐位一致）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/eis训练数据/bmdrt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evidence" / "ws_crit"))

import bm_ws                                   # noqa: E402  生产版
import bm_wscrit as OLD                        # noqa: E402  探索版
import p_after_e2 as P                         # noqa: E402  语料


def _eq(a, b, tol=1e-12):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k], tol) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))
    return a == b


GROUPS = [
    "evidence/b2x/k1/example_caseF_wo/real_example_caseF_wo_samples.npz",
    "evidence/b2x/k3/example_caseF_wo/real_example_caseF_wo_samples.npz",
    "evidence/b2x/k1ws/example_caseF_wo/real_example_caseF_wo_samples.npz",
    "evidence/b2x/k1wo/example_caseF_wo/real_example_caseF_wo_samples.npz",
    "out/_caseE_diff/k1_Ws/example_caseE/real_example_caseE_samples.npz",
    "out/_caseE_diff/k1_Wo/example_caseE/real_example_caseE_samples.npz",
    "out/_caseE_diff/k2_Ws/example_caseE/real_example_caseE_samples.npz",
    "out/example_caseE_warburg/real_example_caseE_warburg_samples.npz",
]


def main():
    diffs = []
    n_spec = 0
    for tag, f, Z in P.corpus():
        n_spec += 1
        try:
            a = bm_ws.ws_diagnostics(f, Z)
            b = OLD.ws_diagnostics(f, Z)
        except Exception as e:                 # noqa: BLE001
            diffs.append(dict(kind="spectrum", tag=tag, error=f"{type(e).__name__}: {e}"))
            continue
        if not _eq(a, b):
            bad = [k for k in a if not _eq(a[k], b.get(k))]
            diffs.append(dict(kind="spectrum", tag=tag, fields=bad,
                              a={k: a[k] for k in bad}, b={k: b.get(k) for k in bad}))

    n_npz = 0
    for rel in GROUPS:
        p = ROOT / rel
        if not p.exists():
            diffs.append(dict(kind="npz", path=rel, error="文件不存在"))
            continue
        n_npz += 1
        d = dict(np.load(p, allow_pickle=True))
        for fn in ("rpc_share", "ws_share_low"):
            a = getattr(bm_ws, fn)(d)
            b = getattr(OLD, fn)(d)
            if not _eq(a, b):
                bad = [k for k in a if not _eq(a[k], b.get(k))]
                diffs.append(dict(kind=fn, path=rel, fields=bad,
                                  a={k: a[k] for k in bad}, b={k: b.get(k) for k in bad}))

    out = ROOT / "evidence" / "ws_crit" / "relocate_check.json"
    payload = dict(n_spectra=n_spec, n_npz=n_npz,
                   n_fields_per_spectrum=len(a) if n_spec else 0,
                   n_diffs=len(diffs), diffs=diffs)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    print(f"谱 {n_spec} 条 × 诊断字段，npz {n_npz} 组 × 2 个份额函数")
    print(f"不一致 {len(diffs)} 处 → {'逐位一致（搬迁未改判据）' if not diffs else '见 ' + str(out)}")
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())
