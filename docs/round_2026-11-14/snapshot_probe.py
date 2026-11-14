# -*- coding: utf-8 -*-
"""三时点模块快照的最小实测（契约 `EXECUTION_IDENTITY_CONTRACT.md` §3.1.2 / §18.2 第 ① 项）。

只回答一个问题
--------------
在 `before_fit` / `sampler_entry` / `after_fit` 三个时点，能不能采到**参与计算的真实
项目模块集合**，以及它与 `source_manifest.json`（Profile B）声明集的覆盖关系是什么。

不改生产代码
------------
`sampler_entry` 用 `numpyro.infer.MCMC.run` 的**类属性包装**挂接，采样调用点仍然是
生产路径 `bm_top.fit_top` → `mcmc.run(...)`。探针只在调用前后读 `sys.modules`。

口径
----
· 只记录项目代码：`sys.modules` 里 `__file__` 的 **realpath** 落在仓库根内且以
  `.py` 结尾的模块（契约 §3 表：`runtime_loaded_modules` "只含项目代码；realpath 记录"）。
· 同一文件可有多个模块名，按文件归并后连名字一起留证。
· 非 `.py` 声明条目（`config.json`）不参与模块覆盖比较，单独报。
· **探针自身**（`docs/round_*/snapshot_probe.py`）不是被拟合的代码，默认排除；
  原始（未排除）结论同时报，因为真实 runner 会遇到同一个问题（见报告 §4.1）。

用法
----
    python docs/round_2026-11-14/snapshot_probe.py                     # 进程内 3 次
    python docs/round_2026-11-14/snapshot_probe.py --fresh             # 每次新进程
    python docs/round_2026-11-14/snapshot_probe.py --no-tiny --fresh --repeat 1
    python docs/round_2026-11-14/snapshot_probe.py --two-fits          # 一次工作流两次采样
    python docs/round_2026-11-14/snapshot_probe.py --single            # 供 --fresh 调用
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISS = object()
SELF = os.path.relpath(os.path.realpath(__file__), str(ROOT)).replace("\\", "/")
POINTS = ("before_fit", "sampler_entry", "after_fit")
DEFAULT_SPECTRUM = "archive/2EMC1-1_0005mA.csv"
DEFAULT_OUT = "out/snapshot_probe/records.json"


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------

def _rel(p: str) -> str:
    return os.path.relpath(p, str(ROOT)).replace("\\", "/")


def snapshot() -> dict:
    """当前 `sys.modules` 里的项目模块集合（按 realpath 归并）。"""
    root_nc = os.path.normcase(str(ROOT)) + os.sep
    files: dict[str, list[str]] = {}
    total = 0
    for name, mod in list(sys.modules.items()):
        total += 1
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            rp = os.path.realpath(f)
        except OSError:
            continue
        if not os.path.normcase(rp).startswith(root_nc) or not rp.endswith(".py"):
            continue
        files.setdefault(_rel(rp), []).append(name)
    return {"n_total_modules": total,
            "files": {k: sorted(v) for k, v in sorted(files.items())}}


HOOK: dict = {"calls": [], "resolved": [], "missing": []}


def _grab(obj, names: tuple[str, ...], prefix: str) -> dict:
    out = {}
    for n in names:
        v = getattr(obj, n, _MISS)
        if v is _MISS:
            out[n] = None
            HOOK["missing"].append(f"{prefix}.{n}")
            continue
        HOOK["resolved"].append(f"{prefix}.{n}")
        out[n] = v if isinstance(v, (bool, int, float, str)) else repr(v)
    return out


def _effective(mcmc) -> dict:
    """采样入口处**生效**的采样参数。仅作 ③ 的观察材料，本轮不据此下结论。"""
    d = _grab(mcmc, ("num_warmup", "num_samples", "num_chains"), "MCMC")
    k = getattr(mcmc, "_kernel", None) or getattr(mcmc, "sampler", None)
    if k is not None:
        d.update(_grab(k, ("_target_accept_prob", "_max_tree_depth"), "kernel"))
    else:
        HOOK["missing"].append("MCMC._kernel")
    return d


def install_hook():
    """包装 `numpyro.infer.MCMC.run`：**每次**调用留一条带序号的记录。"""
    import numpyro.infer as ni

    orig = ni.MCMC.run
    if getattr(orig, "_snapshot_probe", False):
        return lambda: None

    def run(self, *a, **kw):
        HOOK["calls"].append({"idx": len(HOOK["calls"]),
                              "snapshot": snapshot(),
                              "effective": _effective(self)})
        return orig(self, *a, **kw)

    run._snapshot_probe = True                            # type: ignore[attr-defined]
    ni.MCMC.run = run

    def restore():
        ni.MCMC.run = orig

    return restore


# ---------------------------------------------------------------------------
# 声明集与覆盖判定
# ---------------------------------------------------------------------------

def declared(profile: str = "B") -> tuple[dict, dict]:
    m = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    p = m["profiles"][profile]
    py, other = {}, {}
    for e in p["declared_source_set"]:
        rel = e["path"]
        if rel.endswith(".py"):
            py[rel] = e.get("role")
        else:
            other[rel] = e.get("role")
    return py, other


def coverage(files_by_point: dict, py: dict, exclude: set[str]) -> dict:
    """按 §3.1.2 的四类关系判定。"""
    present = [k for k in POINTS if files_by_point.get(k)]
    sets = [set(files_by_point[k]["files"]) - exclude for k in present]
    union = set().union(*sets) if sets else set()
    undeclared = sorted(union - set(py))
    not_loaded = sorted(set(py) - union)
    missing_points = [k for k in POINTS if k not in present]
    if missing_points:
        verdict = "snapshot_missing"
        reason = "snapshot_point_declared_missing:" + ",".join(missing_points)
    elif undeclared:
        verdict, reason = "undeclared_loaded", "runtime ⊄ declared"
    else:
        verdict, reason = "complete", "runtime ⊆ declared"
    return {"verdict": verdict, "reason": reason, "n_union": len(union),
            "undeclared_modules": undeclared,
            "declared_not_loaded": not_loaded,
            "missing_points": missing_points}


# ---------------------------------------------------------------------------
# 一次运行
# ---------------------------------------------------------------------------

def run_once(spectrum: str, *, tiny: bool, two_fits: bool = False) -> dict:
    # 必须在每次运行前清空：否则进程内重复时 `calls[0]` 还是上一次的采样点，
    # 记录会**静默**顶着旧快照（不报错、值看着也合理）。
    HOOK["calls"].clear()
    HOOK["resolved"].clear()
    HOOK["missing"].clear()

    import numpy as np

    import bm_experiments as E      # noqa: F401  复现 nested_fit.run 的导入面
    import bm_real as R
    import bm_top as T
    import nested_fit as N
    from bm_select import BUDGETS    # noqa: F401

    restore = install_hook()
    t0 = time.time()
    try:
        f, Z = R.load_spectrum(str(ROOT / spectrum))
        o = np.argsort(f)
        f, Z = f[o], Z[o]
        kw = N.budget_kw("quick")
        overridden = {}
        if tiny:
            overridden = {"n_warmup": 8, "n_samples": 8,
                          "num_chains": 1, "progress": False}
            kw.update(overridden)
        sigma_top, from_data, sigma_desc = N.make_sigma(Z, "perpoint", 2.0, None)

        before = snapshot()
        r1 = T.fit_top(f, Z, branch="parallel", layers=2, seed=0,
                       sigma=sigma_top, sigma_from_data=from_data, **kw)
        t1 = time.time()
        r2 = None
        if two_fits:
            # 同进程第二次采样：模拟 `with_voigt=True` / 一次工作流两族对拍
            r2 = T.fit_top(f, Z, branch="series", layers=2, seed=0,
                           sigma=sigma_top, sigma_from_data=from_data, **kw)
        after = snapshot()
    finally:
        restore()
        t2 = time.time()

    calls = HOOK["calls"]
    return {
        "spectrum": spectrum,
        "n_points": int(f.size),
        "budget": {"profile": "quick", "kw": kw, "tiny_override": overridden},
        "sigma": {"mode": "perpoint", "pct": 2.0, "from_data": bool(from_data),
                  "desc": sigma_desc, "array_sha256": _arr_sha(sigma_top)},
        "points": {"before_fit": before,
                   "sampler_entry": calls[0]["snapshot"] if calls else None,
                   "after_fit": after},
        "sampler_calls": [{"idx": c["idx"],
                           "n_files": len(c["snapshot"]["files"]),
                           "n_total_modules": c["snapshot"]["n_total_modules"],
                           "effective": c["effective"]} for c in calls],
        "effective": calls[0]["effective"] if calls else None,
        "hook": {"n_calls": len(calls),
                 "resolved_attrs": HOOK["resolved"],
                 "missing_attrs": HOOK["missing"]},
        "metrics": {"fit1": {"rhat_max": r1.get("rhat_max"),
                             "divergences": r1.get("divergences"),
                             "sigma_mode": r1.get("sigma_mode")},
                    "fit2": None if r2 is None else
                    {"rhat_max": r2.get("rhat_max"),
                     "divergences": r2.get("divergences")}},
        "timing_s": {"to_first_fit": round(t1 - t0, 2),
                     "to_after_fit": round(t2 - t1, 2)},
    }


def _arr_sha(a) -> str | None:
    import hashlib
    if a is None:
        return None
    import numpy as np
    x = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    h = hashlib.sha256()
    h.update(b"v1\n" + np.asarray(x.shape, dtype="<i8").tobytes() + x.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def summarise(records: list[dict], py: dict, other: dict) -> dict:
    ex = {SELF}
    raw = [coverage(r["points"], py, set()) for r in records]
    exc = [coverage(r["points"], py, ex) for r in records]
    sets = [{k: set(v["files"]) for k, v in r["points"].items() if v} for r in records]
    drift = {}
    for pt in POINTS:
        seq = [s[pt] for s in sets if pt in s]
        n_uniq = len({frozenset(x) for x in seq}) if seq else None
        drift[pt] = {"n_runs": len(seq), "n_distinct_sets": n_uniq,
                     "sizes": [len(x) for x in seq],
                     "first_minus_last": sorted(seq[0] - seq[-1]) if len(seq) > 1 else [],
                     "last_minus_first": sorted(seq[-1] - seq[0]) if len(seq) > 1 else []}
    # 时点之间的差集：三时点是否真的采到了不同的东西
    point_delta = []
    for s in sets:
        if len(s) == 3:
            point_delta.append({
                "entry_minus_before": sorted(s["sampler_entry"] - s["before_fit"]),
                "after_minus_entry": sorted(s["after_fit"] - s["sampler_entry"]),
                "before_minus_after": sorted(s["before_fit"] - s["after_fit"]),
            })
    return {
        "profile": "B",
        "declared_py": len(py),
        "declared_non_module": other,
        "n_runs": len(records),
        "coverage_excluding_probe": exc,
        "coverage_raw": raw,
        "drift": drift,
        "point_delta": point_delta,
        "n_total_modules": [{k: v["n_total_modules"] for k, v in r["points"].items() if v}
                            for r in records],
        "sampler_calls": [r["sampler_calls"] for r in records],
        "effective": records[0]["effective"],
        "metrics": records[0]["metrics"],
        "timing_s": records[0]["timing_s"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", default=DEFAULT_SPECTRUM)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="每次重复起新进程")
    ap.add_argument("--single", action="store_true", help="跑一次并把 JSON 打到 stdout")
    ap.add_argument("--two-fits", action="store_true", help="同进程连跑两次采样")
    ap.add_argument("--no-tiny", dest="tiny", action="store_false",
                    help="用 quick 档原值（800/400/2），默认 tiny 8/8/1")
    ap.set_defaults(tiny=True)
    a = ap.parse_args(argv)

    if a.single:
        rec = run_once(a.spectrum, tiny=a.tiny, two_fits=a.two_fits)
        print("@@JSON@@" + json.dumps(rec, ensure_ascii=False))
        return 0

    records: list[dict] = []
    if a.fresh:
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        for _ in range(a.repeat):
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--single", "--spectrum", a.spectrum]
            if not a.tiny:
                cmd.append("--no-tiny")
            if a.two_fits:
                cmd.append("--two-fits")
            p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               cwd=str(ROOT), encoding="utf-8")
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("@@JSON@@")]
            if not line:
                print(p.stdout[-2000:], file=sys.stderr)
                print(p.stderr[-2000:], file=sys.stderr)
                return 1
            records.append(json.loads(line[-1][len("@@JSON@@"):]))
    else:
        for _ in range(a.repeat):
            records.append(run_once(a.spectrum, tiny=a.tiny, two_fits=a.two_fits))

    py, other = declared("B")
    s = summarise(records, py, other)
    s["mode"] = ("fresh" if a.fresh else "in_process") + \
                ("/two_fits" if a.two_fits else "") + \
                ("/tiny" if a.tiny else "/quick")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": s, "records": records},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"\n产物：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
