# -*- coding: utf-8 -*-
"""真实 EIS 单次拟合的执行层（harness 侧，`bm_*.py` 零改动）。

本模块只被 `minimal_runner.run_job(mode="real_fit")` 惰性导入，
dry-run / `--check` 路径不碰 numpyro。

做四件事，全部按既有采集契约：

1. **真实拟合**：调生产入口 `bm_core.fit(f, Z, **fit_cfg)`，真的跑
   `MCMC.run`、产生后验样本；取生产代码已挂的 `mc.provenance` 作为捕获事件，
   不自己另造一份。
2. **三时点快照**：`before_fit` / `sampler_entry` / `after_fit`，
   `sampler_entry` 由 `numpyro.infer.MCMC.run` 的类属性包装取得
   （参考 `docs/round_2026-11-14/snapshot_probe.py` 的同一手法）。
3. **harness 归属**：runner / harness / 采集 / 身份层文件用
   `capture_contract.harness_item` 显式声明为 `cli_only` /
   `reference_only`；`classify_runtime` 不许按文件名过滤，并核验
   `numeric_source_tree_sha256` 在「含/不含 harness」两个声明集下逐位相同。
4. **多次采样逐次事件**：`SamplingEventLog` 每次 `MCMC.run` 一条事件、
   显式 `session(owner)` 归属、复用前必须 `reset()`；`running`/`error`
   不算完整事件。

红线：拟合结果只写 runner 侧 `execution_evidence`，**不进身份哈希**；
五项预算仍由真实捕获事件给出（`unavailable`），不升级来源；
身份由 `exec_identity.evaluate()` 裁决，本模块不代理。
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import capture_contract as C                    # noqa: E402
import exec_identity as E                      # noqa: E402

POINTS = ("before_fit", "sampler_entry", "after_fit")
_MISS = object()

#: 采集侧 harness 声明：**是一组，不是一个**。
#: `compute_role` 必须显式给；harness 改动只该换 workflow_id，不该换 run_id。
HARNESS = (
    ("docs/round_2026-11-15/minimal_runner.py", "cli_only",
     "只做冻结清单、执行编排与留证，不参与阻抗模型数值计算",
     "C-1：harness 须以 runtime_scope=harness 进声明集，不许按文件名过滤"),
    ("docs/round_2026-11-15/execution_harness.py", "cli_only",
     "绑定/组装/落盘工具，不参与任何数值计算",
     "test_execution_chain.py 14/14；改的是留证语义，不改模型的数"),
    ("docs/round_2026-11-15/real_fit_runner.py", "cli_only",
     "包装 MCMC.run 取快照与逐次事件，不改传给 bm_core.fit 的任何实参",
     "本模块 install_hook 只替换类属性，调用点仍是生产路径"),
    ("capture_contract.py", "cli_only",
     "路径/归属/事件采集工具，不参与任何数值计算",
     "test_capture_contract.py 31/31"),
    ("exec_identity.py", "reference_only",
     "只被采集脚本引用以复算摘要，生产拟合不 import 它",
     "§3.1.1：契约层不被生产模块 import"),
    ("test_exec_identity.py", "reference_only",
     "已审条件夹具来源，只提供冻结条件模板，不参与任何数值计算",
     "runner 用它构造条件字段；改它不该换 run_id，只该换 workflow_id"),
)


# ---------------------------------------------------------------------------
# 快照与事件 hook
# ---------------------------------------------------------------------------

def snapshot() -> dict:
    """当前 `sys.modules` 的项目侧文件表；路径判断全交给 `normalize_path`。"""
    files: dict[str, list[str]] = {}
    total = 0
    for name, mod in list(sys.modules.items()):
        total += 1
        f = getattr(mod, "__file__", None)
        if not f or not str(f).endswith(".py"):
            continue
        files.setdefault(f, []).append(name)
    return {"n_total_modules": total, "files": files}


def _compact(snap: Mapping | None) -> dict | None:
    """只落**项目侧**文件表；第三方/外部按契约只计数，不写全量路径。

    全量 `sys.modules` 有千余条，落盘会让单条记录膨胀到数百 KB；
    契约本来就规定第三方不进任何一档。`state_counts` 保留各状态数量，
    因此三时点之间的模块集合变化仍能被指纹看到。
    """
    if not isinstance(snap, Mapping):
        return None
    files: dict[str, dict] = {}
    counts: dict[str, int] = {}
    for raw, mods in snap["files"].items():
        n = C.normalize_path(raw, _ROOT)
        counts[n["state"]] = counts.get(n["state"], 0) + 1
        if n["state"] != "project":
            continue
        files[n["path"]] = {
            "state": n["state"],
            "modules": sorted(mods) if isinstance(mods, (list, tuple)) else [str(mods)],
        }
    out = {"n_total_modules": snap["n_total_modules"],
           "files": files, "state_counts": counts}
    out["sha256"] = hashlib.sha256(
        E.canonical_json({"files": files, "state_counts": counts})
        .encode("utf-8")).hexdigest()
    return out


def _grab(obj, names, prefix: str, hook: dict) -> dict:
    out = {}
    for n in names:
        v = getattr(obj, n, _MISS)
        if v is _MISS:
            out[n] = None
            hook["missing"].append(f"{prefix}.{n}")
            continue
        hook["resolved"].append(f"{prefix}.{n}")
        out[n] = v if isinstance(v, (bool, int, float, str)) else repr(v)
    return out


def _effective(mcmc, hook: dict) -> dict:
    d = _grab(mcmc, ("num_warmup", "num_samples", "num_chains"), "MCMC", hook)
    k = getattr(mcmc, "_kernel", None) or getattr(mcmc, "sampler", None)
    if k is not None:
        d.update(_grab(k, ("_target_accept_prob", "_max_tree_depth"), "kernel", hook))
    else:
        hook["missing"].append("MCMC._kernel")
    return d


def install_hook(log: C.SamplingEventLog, *, inject: str | None = None,
                 hook: dict | None = None):
    """包装 `numpyro.infer.MCMC.run`：每次调用由 log 记一条事件。"""
    import numpyro.infer as ni
    hook = hook if hook is not None else {"resolved": [], "missing": []}
    orig = ni.MCMC.run
    if getattr(orig, "_real_fit_runner", False):
        return lambda: None

    def run(self, *a, **kw):
        with log.event(snapshot=snapshot(), effective=_effective(self, hook)):
            if inject:
                raise RuntimeError(f"注入的采样异常：{inject}")
            return orig(self, *a, **kw)

    run._real_fit_runner = True                     # type: ignore[attr-defined]
    ni.MCMC.run = run
    return lambda: setattr(ni.MCMC, "run", orig)


# ---------------------------------------------------------------------------
# 声明集 / harness 归属
# ---------------------------------------------------------------------------

def _sha_of(rel: str) -> str:
    return hashlib.sha256((_ROOT / rel).read_bytes()).hexdigest()


def declared_pair(profile: str = "B") -> tuple[list[dict], list[dict]]:
    """`(manifest 原样, manifest + harness 组)`；内容 sha 由 runner 实测补齐。"""
    import json
    m = json.loads((_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    base: list[dict] = []
    for e in m["profiles"][profile]["declared_source_set"]:
        item = {k: v for k, v in e.items() if k in
                ("path", "role", "compute_role", "workflow_scope",
                 "why_not_numeric", "evidence", "why_not_workflow")}
        item["sha256"] = _sha_of(e["path"])
        base.append(item)
    hs = [C.harness_item(rel, sha256=_sha_of(rel), compute_role=cr,
                         role="entry", why_not_numeric=why, evidence=ev)
          for rel, cr, why, ev in HARNESS]
    return base, base + hs


def harness_evidence(snap: Mapping) -> dict:
    """C-1/C-2：运行时分类 + harness 不进 numeric 摘要的逐位核验。"""
    base, with_harness = declared_pair()
    cls = C.classify_runtime(snap["files"], with_harness, root=_ROOT)
    ext = cls["external_modules"]
    und = cls["undeclared_modules"]
    coverage = ("external_path" if ext else
                "undeclared_loaded" if und else "complete")
    return {
        "declared_n": len(base),
        "harness_paths": [h[0] for h in HARNESS],
        "numeric_digest_without_harness": E.numeric_source_tree_sha256(base),
        "numeric_digest_with_harness": E.numeric_source_tree_sha256(with_harness),
        "numeric_digest_same": (E.numeric_source_tree_sha256(base) ==
                                E.numeric_source_tree_sha256(with_harness)),
        "workflow_digest_without_harness": E.workflow_source_tree_sha256(base),
        "workflow_digest_with_harness": E.workflow_source_tree_sha256(with_harness),
        "runtime_class_counts": cls["counts"],
        "harness_modules": cls["harness_modules"],
        "undeclared_modules": und,
        "external_modules": [x["resolved"] for x in ext][:8],
        "declared_not_loaded_n": len(cls["declared_not_loaded"]),
        "n_third_party": cls["n_third_party"],
        "source_coverage": coverage,
        "note": ("harness 归属实测；条件模板 code 仍是已审夹具，"
                 "本块不进身份哈希 ⇒ 源码身份仍未闭环（G9/契约 C1–C8）"),
    }


# ---------------------------------------------------------------------------
# 拟合结果与证据
# ---------------------------------------------------------------------------

def _posterior_sha256(samples: Mapping) -> str:
    """后验指纹：固定 `<f8` / C 顺序 / 按 key 排序 / shape 进哈希。"""
    import numpy as np
    h = hashlib.sha256()
    h.update(b"posterior-v1\n")
    for k in sorted(samples):
        a = np.ascontiguousarray(np.asarray(samples[k], dtype="<f8"))
        h.update(k.encode("utf-8") + b"\n")
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def _fit_result(samples: Mapping, mc) -> dict:
    import numpy as np
    out: dict[str, Any] = {
        "n_draws": int(len(samples["R_inf"])),
        "posterior_keys": sorted(samples),
        "posterior_sha256": _posterior_sha256(samples),
        "posterior_mean": {k: float(np.mean(np.asarray(v, dtype="<f8")))
                           for k, v in sorted(samples.items())},
    }
    try:
        extra = mc.get_extra_fields() or {}
    except Exception as exc:                       # noqa: BLE001
        out["extra_fields_error"] = f"{type(exc).__name__}: {exc}"
    else:
        if "diverging" in extra:
            out["n_divergences"] = int(np.sum(np.asarray(extra["diverging"])))
        if "num_steps" in extra:
            out["num_steps_max"] = int(np.max(np.asarray(extra["num_steps"])))
    out["result_sha256"] = hashlib.sha256(
        E.canonical_json(out).encode("utf-8")).hexdigest()
    return out


def _events(log: C.SamplingEventLog, owner: str) -> list[dict]:
    out = []
    for ev in log.events(owner):
        snap = ev.get("snapshot") or {}
        out.append({
            "event_id": ev["event_id"], "owner": ev["owner"], "idx": ev["idx"],
            "seq": ev["seq"], "status": ev["status"], "error": ev["error"],
            "sampler": ev["sampler"], "effective": ev["effective"],
            "n_files": len(snap.get("files") or {}),
            "snapshot_sha256": _compact(snap)["sha256"],
        })
    return out


# ---------------------------------------------------------------------------
# 主入口：真实单次拟合
# ---------------------------------------------------------------------------

def run_real_fit(*, spectrum: str, fit_cfg: Mapping, owner: str,
                 inject: str | None = None,
                 log: C.SamplingEventLog | None = None) -> dict:
    """跑一次真实拟合，返回完整执行证据包（不含身份裁决）。"""
    import numpy as np
    import bm_real as R                            # noqa: F401
    import bm_core as CORE

    log = log or C.SamplingEventLog()
    log.reset()                                   # C-3：进程内复用必须显式清零
    hook = {"resolved": [], "missing": []}
    f, Z = R.load_spectrum(str(_ROOT / spectrum))
    o = np.argsort(f)
    f, Z = f[o], Z[o]

    kw = dict(fit_cfg)
    kw["extra_fields"] = tuple(kw.get("extra_fields") or ())
    before = snapshot()
    restore = install_hook(log, inject=inject, hook=hook)
    t0 = time.time()
    error = None
    samples = mc = None
    try:
        with log.session(owner):
            samples, mc = CORE.fit(f, Z, **kw)
    except BaseException as exc:                  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        restore()
    elapsed = round(time.time() - t0, 2)
    after = snapshot()

    evs = _events(log, owner)
    entry_snap = None
    for ev in log.events(owner):
        if ev["idx"] == 0:
            entry_snap = ev.get("snapshot")
            break
    three = {
        "before_fit": _compact(before),
        "sampler_entry": _compact(entry_snap),
        "after_fit": _compact(after),
    }
    event = getattr(mc, "provenance", None) if mc is not None else None
    if event is not None:
        event = dict(event)
        assert event.get("site") == "bm_core.fit", event.get("site")
        assert event.get("budget_adaptation_error") is None, \
            event.get("budget_adaptation_error")

    bundle: dict[str, Any] = {
        "mode": "real_fit",
        "owner": owner,
        "spectrum": spectrum,
        "n_points": int(f.size),
        "fit_cfg": {**kw, "extra_fields": list(kw["extra_fields"])},
        "fit_ok": error is None and samples is not None,
        "error": error,
        # 生产代码已挂在 mc 上的真实捕获事件，交给 runner 绑定/组装；
        # runner 落盘时会把这一键剔除，避免证据与身份记录重复。
        "event": event,
        "fit_result": _fit_result(samples, mc) if samples is not None else None,
        "three_points": three,
        "three_points_complete": all(three[p] is not None for p in POINTS),
        "harness": harness_evidence(after),
        "sampling_events": evs,
        "n_sampling_events": len(evs),
        "n_complete_events": sum(1 for e in evs if e["status"] == "ok"),
        "hook": {"n_calls": len(evs),
                 "resolved_attrs": sorted(set(hook["resolved"])),
                 "missing_attrs": sorted(set(hook["missing"]))},
        "timing_s": {"fit_total": elapsed},
        "note": ("三时点/harness/逐次事件为 runner 侧实测，不进身份哈希；"
                 "条件模板 code 仍是夹具 ⇒ 源码身份未闭环"),
    }
    return bundle


def _selftest() -> bool:
    """两次 tiny 拟合验证 C-3 逐次事件（临时用途，不写产物）。"""
    import tempfile
    cfg = {"K": 10, "n_warmup": 8, "n_samples": 8, "num_chains": 1,
           "target_accept": 0.9, "max_tree_depth": 10, "seed": 0,
           "progress": False, "extra_fields": ("diverging", "num_steps")}
    with tempfile.TemporaryDirectory():
        a = run_real_fit(spectrum="archive/2EMC1-1_0005mA.csv", fit_cfg=cfg,
                         owner="selftest-1")
        b = run_real_fit(spectrum="archive/2EMC1-1_0005mA.csv", fit_cfg=cfg,
                         owner="selftest-2")
    checks = {
        "两轮均拟合成功": a["fit_ok"] and b["fit_ok"],
        "每轮恰好一条完整事件": a["n_complete_events"] == 1 and b["n_complete_events"] == 1,
        "三时点齐": a["three_points_complete"] and b["three_points_complete"],
        "harness 不进 numeric 摘要": a["harness"]["numeric_digest_same"],
        "后验指纹可复算": bool(a["fit_result"]["posterior_sha256"]),
    }
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"[{'ok' if v else 'FAIL'}] {k}")
    print(f"real_fit_runner 自检：{sum(checks.values())}/{len(checks)} 通过")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
