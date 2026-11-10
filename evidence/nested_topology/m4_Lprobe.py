"""探针：voigt 族开/关 L 对跨族 Δelpd 的影响（不改 bm_select 默认口径）。

对每个用例：
  · detect_L 自动判定（生产判据，与 bm_real.py:2532 同源）
  · voigt 跑 K=1,2，各跑 include_L=False / True 两版
  · nested 跑 2 支路 × 2 层（不涉及 L 开关，其 L_ad 关不掉）
  · 分别算「voigt(no L) 最优 vs nested 最优」与「voigt(L) 最优 vs nested 最优」的配对差

目的：量化当前 282–283 nats 的 Δ 里，有多少来自「voigt 被剥夺 L」。
只读不写生产路径，不动阈值。

用法：python evidence/nested_topology/m4_Lprobe.py [case ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bm_select as S          # noqa: E402
import bm_real as R           # noqa: E402


def _best(cands):
    ok = [c for c in cands if c["ok"]]
    return max(ok, key=lambda c: c["elpd"]) if ok else None


def main(argv):
    want = set(argv[1:]) or None
    for tag, expect, note, f, Z in S._cases():
        if want and tag not in want:
            continue
        t0 = time.time()
        has_L = R.detect_L(f, Z)
        print(f"\n=== {tag} （期望 {expect}）detect_L={has_L} ===", flush=True)

        V0, V1 = [], []
        for K in (1, 2):
            a = S.fit_voigt(f, Z, K, seed=1, include_L=False)
            b = S.fit_voigt(f, Z, K, seed=1, include_L=True)
            V0.append(a)
            V1.append(b)
            la = a["raw"].get("L")
            lb = b["raw"].get("L")
            print(f"  voigt K={K}: noL elpd={a['elpd']:8.2f} R̂={a['rhat']:.2f} | "
                  f"L elpd={b['elpd']:8.2f} R̂={b['rhat']:.2f} "
                  f"L={('%.3g H' % lb) if lb else '—'}", flush=True)

        N = [S.fit_nested(f, Z, L, branch=br, seed=1)
             for br in ("parallel", "series") for L in (1, 2)]
        for c in N:
            print(f"  nested {c['label']:<14} elpd={c['elpd']:8.2f} R̂={c['rhat']:.2f}",
                  flush=True)

        n = _best(N)
        print(f"  → nested 最优 = {n['label'] if n else '无'}", flush=True)
        for name, V in (("noL", V0), ("withL", V1)):
            v = _best(V)
            if v is None or n is None:
                print(f"  Δ({name}) : 无可用候选", flush=True)
                continue
            d, se = S.paired_diff(n["pointwise"], v["pointwise"], f=f)
            z = d / se if se and np.isfinite(se) and se > 0 else float("inf")
            sw = d >= S.SWITCH_DELTA and z >= S.SWITCH_Z
            bk = -d >= S.SWITCH_DELTA and -z >= S.SWITCH_Z
            print(f"  Δ({name}) = {d:9.2f}  dse={se:6.2f}  z={z:7.2f}  → "
                  f"{'nested' if sw else ('voigt' if bk else 'indistinguishable')}"
                  f"   [voigt 最优 {v['label']}]", flush=True)
        print(f"  耗时 {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
