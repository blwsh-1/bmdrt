#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""入口等价性核对：`nested_fit.run` 与 `circuit_runner → bm_select.fit_nested`
是不是在调**同一个** `bm_top.fit_top`，以及实参差在哪。

做法（不跑 MCMC）：把 `bm_top.fit_top` 换成只记录 kwargs 的桩，抛
`_Stop` 跳出调用栈。两条路径各抓一组实参，逐键比对。

    python entry_equiv_probe.py            # 默认 standard 档
    python entry_equiv_probe.py --budget quick

结论只有三种：SAME（逐键相同）/ DIFF（列出差异键）/ ERROR。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_ROUND = Path(__file__).resolve().parent
_ROOT = _ROUND.parents[1]
for p in (str(_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

import bm_real as R                    # noqa: E402
import bm_top as T                     # noqa: E402
import bm_select as SEL                # noqa: E402
import nested_fit as NF                # noqa: E402
import circuit_runner as RU            # noqa: E402

SPECTRUM = str(_ROOT / "out" / "fig3" / "inputs" / "az31.csv")
CAPTURED: list[dict] = []


class _Stop(Exception):
    pass


def _stub(f, Z, **kw):
    CAPTURED.append(dict(kw, _f=np.asarray(f).copy(), _Z=np.asarray(Z).copy()))
    raise _Stop


def capture(fn) -> dict | None:
    CAPTURED.clear()
    orig = T.fit_top
    T.fit_top = _stub
    try:
        fn()
    except _Stop:
        pass
    except Exception as exc:                    # noqa: BLE001
        print(f"  [warn] 非预期异常 {type(exc).__name__}: {exc}")
    finally:
        T.fit_top = orig
    return CAPTURED[0] if CAPTURED else None


def path_a(f, Z, budget: str) -> dict | None:
    """nested_fit.run —— 单族研究接口。"""
    tmp = str(_ROUND / "_tmp_a")
    return capture(lambda: NF.run(SPECTRUM, outdir=tmp, branch="parallel",
                                  layers=2, budget=budget,
                                  sigma_mode="perpoint", sigma_pct=2.0,
                                  seed=0, plot=False, verbose=False))


def path_b(f, Z, budget: str) -> dict | None:
    """circuit_runner → bm_select.fit_nested —— 跨族判决接口。"""
    sig = RU.SIGMA_PCT_PERP0INT * np.abs(Z)
    kw = dict(RU.BUDGETS[budget])
    return capture(lambda: SEL.fit_nested(f, Z, 2, branch="parallel", seed=0,
                                          sigma_floor=SEL.SIGMA_FLOOR, kw=kw,
                                          sigma=(sig, sig)))


def norm(v):
    """tuple/ndarray/float 统一成可比较的形状。

    `sigma` 走的是 `(sig, sig)` tuple 与 `np.stack([sig, sig])` 两种**等价写法**
    —— `bm_top._check_sigma` 里都先 `np.asarray(...)` 再判 `shape[0]==2`，
    归一化后完全相同。所以这里也按数组比，否则会把"同一个数组换了容器"
    误报成差异。
    """
    if isinstance(v, (np.ndarray, list, tuple)):
        try:
            a = np.asarray(v, float)
        except (TypeError, ValueError):
            return ("seq", [norm(x) for x in v])
        if a.ndim:
            return ("array", a.shape, np.round(a.ravel(), 12).tolist())
    return ("scalar", v)


#: 缺省值表：`<缺>` 与显式默认值等价时不报 DIFF。
DEFAULTS = {"sigma_floor": 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="standard")
    a = ap.parse_args(argv)

    f, Z = R.load_spectrum(SPECTRUM)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    print(f"谱 {os.path.basename(SPECTRUM)}  {f.size} 点  {a.budget} 档\n")

    A = path_a(f, Z, a.budget)
    B = path_b(f, Z, a.budget)
    if A is None or B is None:
        print("ERROR：有一条路径没有调到 fit_top")
        return 1

    keys = sorted(set(A) | set(B), key=lambda k: (k.startswith("_"), k))
    diff, same = [], []
    MISSING = "<缺>"
    for k in keys:
        va, vb = A.get(k, MISSING), B.get(k, MISSING)
        # 缺省 vs 显式默认值：语义相同，归入 same（但仍打印写法差异）
        if isinstance(va, str) and k in DEFAULTS and vb == DEFAULTS[k]:
            va = vb
        elif isinstance(vb, str) and k in DEFAULTS and va == DEFAULTS[k]:
            vb = va
        ok = norm(va) == norm(vb)
        (same if ok else diff).append((k, va, vb))

    print(f"A = nested_fit.run            （{len(A)} 个实参）")
    print(f"B = circuit_runner/fit_nested （{len(B)} 个实参）\n")
    for k, va, vb in same:
        s = va if not isinstance(va, np.ndarray) else f"<ndarray {va.shape}>"
        print(f"  [same] {k:<20} {s}")
    for k, va, vb in diff:
        print(f"  [DIFF] {k:<20} A={va!r}  B={vb!r}")
    print(f"\n{'SAME' if not diff else 'DIFF'}：{len(same)} 同名同值 / {len(diff)} 差异")
    return 0 if not diff else 3


if __name__ == "__main__":
    raise SystemExit(main())
