#!/usr/bin/env python
"""离线重算 `--strict-guards` 的效果（不重新采样）。

B1/B2 的 `--json` 里每个候选都带 `guards`（两族同一张清单）与 `elpd`、
`usable`。`_quality(strict=True)` 相对默认口径只多了一件事：把
`ess_bulk / ess_tail / bad_k / tree_cap` 里值为 `triggered` 的候选判为不可用。
所以可以**直接在已落盘的结果上**重放 `_family_state`，得到"若开启 strict
会怎样"，而不必再跑一次 30–60 min 的 MCMC。

用法
----
    python evidence/nested_topology/m4_strict_offline.py <result.json>

输出：逐 seed 的 `默认 degraded` vs `strict degraded`、新增降级 seed、
以及 guard 状态汇总。**这是重放不是重跑**：seed / 采样流完全相同，
差异只来自可用性判据，不能当成独立验证。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

STRICT_KEYS = ("ess_bulk", "ess_tail", "bad_k", "tree_cap")


def usable(c: dict, strict: bool) -> bool:
    if not c.get("usable"):
        return False
    if strict and any((c.get("guards") or {}).get(k) == "triggered"
                      for k in STRICT_KEYS):
        return False
    return True


def family_state(cands: list[dict], strict: bool, margin: float = 4.0):
    """与 bm_select._family_state 同判据，只吃 json 里的精简字段。"""
    ok = [c for c in cands if usable(c, strict)]
    if not ok:
        return True, "族内无可用候选"
    best = max(ok, key=lambda c: c["elpd"])
    lost = [c for c in cands
            if not usable(c, strict) and c["elpd"] > best["elpd"] + margin]
    return bool(lost), ("降配参赛：" + "; ".join(c["label"] for c in lost)) if lost else ""


def replay(path: Path) -> int:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    print(f"来源 {path.name} | n_seeds={d.get('n_seeds')} | "
          f"pair_mode={d.get('pair_mode')} | retry={d.get('retry')} | "
          f"strict_guards={d.get('strict_guards')}")
    print("【离线重放】只改可用性判据，不重新采样；差异不能当独立验证。")
    print(f"{'case':<22}{'seed':>5}{'默认降级':>10}{'strict降级':>12}"
          f"{'新增':>6}  说明")
    tot_new = 0
    for r in rows:
        for tr in r.get("candidate_trace") or []:
            cs = tr.get("candidates") or []
            v = [c for c in cs if c["family"] == "voigt"]
            groups = tr.get("groups") or []
            if d.get("pair_mode") == "per-branch" and groups:
                nested_groups = [[c for c in cs
                                  if c["family"] == "nested" and
                                  c["label"].startswith(g["group"] + "/")]
                                 for g in groups]
            else:
                nested_groups = [[c for c in cs if c["family"] == "nested"]]
            d0 = (family_state(v, False)[0] or
                  any(family_state(ng, False)[0] for ng in nested_groups))
            d1 = (family_state(v, True)[0] or
                  any(family_state(ng, True)[0] for ng in nested_groups))
            _, why = family_state(v, True)
            new = d1 and not d0
            tot_new += int(new)
            print(f"{r['case']:<22}{tr['seed']:>5}{str(d0):>10}{str(d1):>12}"
                  f"{int(new):>6}  {why[:46] if new else ''}")
    print(f"\nstrict 新增降级 seed：{tot_new} 个（默认口径的降级总数见下）")
    print(f"默认降级 seed 合计：{sum(len(r['degraded_seeds']) for r in rows)}")
    cnt = {fam: Counter() for fam in ("voigt", "nested")}
    for r in rows:
        for tr in r.get("candidate_trace") or []:
            for c in tr.get("candidates") or []:
                for k in STRICT_KEYS:
                    cnt[c["family"]][(c.get("guards") or {}).get(k, "?")] += 1
    for fam, c in cnt.items():
        print(f"  {fam:<7}strict 四键：" + " ".join(f"{k}={v}" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(replay(Path(sys.argv[1])))
