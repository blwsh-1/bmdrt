# -*- coding: utf-8 -*-
"""三时点模块快照 + 采集契约的实测（契约 §3.1.2 / §18.2 第 ① 项）。

本脚本有两个身份：
1. **测量工具**：验证三个时点能否采到真实项目模块集合（`SNAPSHOT_PROBE.md`）；
2. **采集契约的参考实现**：路径规范化、harness 归属、采样事件归属都走
   `capture_contract.py`，所以它同时是"这三条规则在真实运行里成立"的证据。

不改生产代码
------------
`sampler_entry` 用 `numpyro.infer.MCMC.run` 的**类属性包装**挂接，采样调用点仍是
生产路径 `bm_top.fit_top` → `mcmc.run(...)`。探针只在调用前后读 `sys.modules`，
不改变传给 `fit_top` 的实参（除 `--tiny` 的预算覆盖）。

三种覆盖口径（这是本脚本最主要的输出）
--------------------------------------
· `verdict_raw`：声明集**不含**探针自身 → `undeclared_loaded`（修复前的命运）；
· `verdict_harness`：把探针声明成 harness（`compute_role=cli_only`）→ `complete`；
· 两者的 `numeric_source_tree_sha256` **必须相同**：harness 不许进 `run_id`。

用法
----
    python docs/round_2026-11-14/snapshot_probe.py --fresh --repeat 3
    python docs/round_2026-11-14/snapshot_probe.py --two-fits --repeat 1
    python docs/round_2026-11-14/snapshot_probe.py --no-tiny --fresh --repeat 1
    python docs/round_2026-11-14/snapshot_probe.py --single      # 供 --fresh 调用
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import capture_contract as C       # noqa: E402
import exec_identity as E          # noqa: E402

_MISS = object()
SELF_REL = os.path.relpath(os.path.realpath(__file__),
                           os.path.realpath(ROOT)).replace("\\", "/")
POINTS = ("before_fit", "sampler_entry", "after_fit")
DEFAULT_SPECTRUM = "archive/2EMC1-1_0005mA.csv"
DEFAULT_OUT = "out/snapshot_probe/records.json"
LOG = C.SamplingEventLog()


# ---------------------------------------------------------------------------
# 快照（原样保留 __file__，规范化交给 capture_contract）
# ---------------------------------------------------------------------------

def snapshot() -> dict:
    """当前 `sys.modules` 的项目侧文件表：`{__file__: [模块名, ...]}`。

    这里**不做路径判断**：在不在仓库内、是不是符号链接，一律交给
    `capture_contract.normalize_path`——两套规则就是两套身份。
    """
    files: dict[str, list[str]] = {}
    total = 0
    for name, mod in list(sys.modules.items()):
        total += 1
        f = getattr(mod, "__file__", None)
        if not f or not str(f).endswith(".py"):
            continue          # 标准库的 .pyd/.so 不是源码；契约只记项目源码
        files.setdefault(f, []).append(name)
    return {"n_total_modules": total, "files": files}


def project_only(snap: dict) -> dict:
    """只留真实解析在仓库内的文件（用于计数与展示，不用于判定）。"""
    out: dict[str, list[str]] = {}
    for f, mods in snap["files"].items():
        n = C.normalize_path(f, ROOT)
        if n["state"] == "project":
            out.setdefault(n["path"], []).extend(mods)
    return {k: sorted(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# 挂接
# ---------------------------------------------------------------------------

HOOK: dict = {"resolved": [], "missing": []}


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
    """采样入口处**生效**的采样参数。只证明"可读"，不证明"可作身份来源"。"""
    d = _grab(mcmc, ("num_warmup", "num_samples", "num_chains"), "MCMC")
    k = getattr(mcmc, "_kernel", None) or getattr(mcmc, "sampler", None)
    if k is not None:
        d.update(_grab(k, ("_target_accept_prob", "_max_tree_depth"), "kernel"))
    else:
        HOOK["missing"].append("MCMC._kernel")
    return d


def install_hook():
    """包装 `numpyro.infer.MCMC.run`：每次调用由 `SamplingEventLog` 记一条事件。"""
    import numpyro.infer as ni

    orig = ni.MCMC.run
    if getattr(orig, "_snapshot_probe", False):
        return lambda: None

    def run(self, *a, **kw):
        with LOG.event(snapshot=snapshot(), effective=_effective(self)):
            return orig(self, *a, **kw)

    run._snapshot_probe = True                            # type: ignore[attr-defined]
    ni.MCMC.run = run
    return lambda: setattr(ni.MCMC, "run", orig)


# ---------------------------------------------------------------------------
# 声明集
# ---------------------------------------------------------------------------

def _sha_of(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


#: 采集侧的 harness 文件：**是一组，不是一个**。runner 有辅助模块是常态。
HARNESS = (
    (SELF_REL, "cli_only", "只读 sys.modules 并落证据，不进阻抗模型计算",
     "I1/I6：把它当 numeric 会改变 run_id；本文件改动只应换 workflow_id"),
    ("capture_contract.py", "cli_only", "路径/归属/事件采集工具，不参与任何数值计算",
     "test_capture_contract.py P/H/E 组；它改的是采集语义，不改模型的数"),
    ("exec_identity.py", "reference_only", "只被采集脚本引用以复算摘要，生产拟合不 import 它",
     "§3.1.1：契约层不被生产模块 import；它自己的版本号在载荷里，换号靠 identity_contract_version"),
)


def declared_pair(profile: str = "B") -> tuple[list[dict], list[dict]]:
    """返回 `(manifest 原样, manifest + harness 组)`。

    两个版本只差 harness 那几项。`numeric` 层必须**逐位相同**——这正是
    "harness 不该换 `run_id`"在真实 19 文件声明集上的检验。
    """
    m = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    base: list[dict] = []
    for e in m["profiles"][profile]["declared_source_set"]:
        rel = e["path"]
        item = {k: v for k, v in e.items() if k in
                ("path", "role", "compute_role", "workflow_scope",
                 "why_not_numeric", "evidence", "why_not_workflow")}
        item["sha256"] = _sha_of(rel)
        base.append(item)
    hs = [C.harness_item(rel, sha256=_sha_of(rel), compute_role=cr, role="entry",
                         why_not_numeric=why, evidence=ev)
          for rel, cr, why, ev in HARNESS]
    return base, base + hs


def digests(items: list[dict]) -> dict:
    return {"declared": E.declared_source_tree_sha256(items)[:16],
            "numeric": E.numeric_source_tree_sha256(items)[:16],
            "workflow": E.workflow_source_tree_sha256(items)[:16],
            "n": len(items),
            "n_numeric": sum(1 for i in items
                             if i.get("compute_role", "numeric") == "numeric")}


# ---------------------------------------------------------------------------
# 一次运行
# ---------------------------------------------------------------------------

def run_once(spectrum: str, *, tiny: bool, two_fits: bool = False,
             owner: str = "probe") -> dict:
    LOG.reset()                       # 进程内复用必须显式清零（E7）

    import numpy as np
    import bm_experiments as E_      # noqa: F401  复现 nested_fit.run 的导入面
    import bm_real as R
    import bm_top as T
    import nested_fit as N
    from bm_select import BUDGETS    # noqa: F401

    restore = install_hook()
    HOOK["resolved"].clear()
    HOOK["missing"].clear()
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
        with LOG.session(owner):
            r1 = T.fit_top(f, Z, branch="parallel", layers=2, seed=0,
                           sigma=sigma_top, sigma_from_data=from_data, **kw)
            t1 = time.time()
            r2 = None
            if two_fits:
                # 子会话：模拟 `with_voigt=True` / 一次工作流里的第二个子运行
                with LOG.session(owner + "/ref"):
                    r2 = T.fit_top(f, Z, branch="series", layers=2, seed=0,
                                   sigma=sigma_top, sigma_from_data=from_data, **kw)
        after = snapshot()
    finally:
        restore()
        t2 = time.time()

    ev = LOG.events(owner)
    third = (ev[0]["snapshot"] if ev else None)
    return {
        "spectrum": spectrum,
        "owner": owner,
        "n_points": int(f.size),
        "budget": {"profile": "quick", "kw": kw, "tiny_override": overridden},
        "sigma": {"mode": "perpoint", "pct": 2.0, "from_data": bool(from_data),
                  "desc": sigma_desc, "array_sha256": _arr_sha(sigma_top)},
        "points": {"before_fit": before,
                   "sampler_entry": third,
                   "after_fit": after},
        "events": [{k: v for k, v in e.items() if k != "snapshot"}
                   | {"n_files": len(e["snapshot"]["files"]),
                      "n_total_modules": e["snapshot"]["n_total_modules"]}
                   for e in LOG.events()],
        "summaries": [LOG.summary(o) for o in list(LOG._sessions)],
        "hook": {"n_calls": len(LOG.events()),
                 "resolved_attrs": sorted(set(HOOK["resolved"])),
                 "missing_attrs": sorted(set(HOOK["missing"]))},
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

def _union_files(rec: dict) -> dict:
    """三时点并集（raw `__file__` → 模块名），只用有快照的点。"""
    out: dict[str, list[str]] = {}
    for pt in POINTS:
        s = rec["points"].get(pt)
        if not s:
            continue
        for f, mods in s["files"].items():
            out.setdefault(f, []).extend(mods)
    return {k: sorted(set(v)) for k, v in out.items()}


def _verdict(cr: dict) -> str:
    if cr["external_modules"]:
        return "external_path"
    if cr["undeclared_modules"]:
        return "undeclared_loaded"
    return "complete"


def summarise(records: list[dict], base: list[dict], with_h: list[dict]) -> dict:
    out = []
    for rec in records:
        uf = _union_files(rec)
        raw = C.classify_runtime(uf, base, root=ROOT)
        har = C.classify_runtime(uf, with_h, root=ROOT)
        out.append({
            "spectrum": rec["spectrum"],
            "verdict_raw": _verdict(raw),
            "undeclared_raw": raw["undeclared_modules"],
            "verdict_harness": _verdict(har),
            "undeclared_harness": har["undeclared_modules"],
            "runtime_counts": har["counts"],
            "harness_modules": har["harness_modules"],
            "declared_not_loaded": har["declared_not_loaded"],
            "n_union": len(har["runtime_loaded_modules"]),
            "point_sizes": {pt: len(project_only(rec["points"][pt]))
                            for pt in POINTS if rec["points"].get(pt)},
            "n_total_modules": {pt: rec["points"][pt]["n_total_modules"]
                                for pt in POINTS if rec["points"].get(pt)},
            "events": rec["events"],
            "summaries": rec["summaries"],
            "n_calls": rec["hook"]["n_calls"],
            "timing_s": rec["timing_s"],
            "metrics": rec["metrics"],
        })

    drift = {pt: {"n_distinct": len({frozenset(project_only(r["points"][pt]))
                                     for r in records if r["points"].get(pt)}),
                  "sizes": [len(project_only(r["points"][pt]))
                            for r in records if r["points"].get(pt)]}
             for pt in POINTS}
    deltas = []
    for r in records:
        if all(r["points"].get(pt) for pt in POINTS):
            a, b, c = (set(project_only(r["points"][pt])) for pt in POINTS)
            deltas.append({"entry_minus_before": sorted(b - a),
                           "after_minus_entry": sorted(c - b)})
    return {
        "profile": "B",
        "digests_manifest_only": digests(base),
        "digests_with_harness": digests(with_h),
        "numeric_unchanged": (digests(base)["numeric"]
                              == digests(with_h)["numeric"]),
        "runs": out,
        "drift": drift,
        "point_delta": deltas,
        "hook_attrs": records[0]["hook"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", default=DEFAULT_SPECTRUM)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--two-fits", action="store_true")
    ap.add_argument("--no-tiny", dest="tiny", action="store_false")
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
                print(p.stdout[-3000:], file=sys.stderr)
                print(p.stderr[-3000:], file=sys.stderr)
                return 1
            records.append(json.loads(line[-1][len("@@JSON@@"):]))
    else:
        for _ in range(a.repeat):
            records.append(run_once(a.spectrum, tiny=a.tiny,
                                    two_fits=a.two_fits))

    base, with_h = declared_pair("B")
    s = summarise(records, base, with_h)
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
