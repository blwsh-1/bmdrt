# -*- coding: utf-8 -*-
"""B3b 运行期初值指纹（只读包装，不改生产代码）。

为什么需要它
------------
静态前置检查 `b3b_precheck.py` 只回答 `init_values(pk, pres, K) is not None`。
`seeded=true` 说明走确定性 `init_to_value`，但**不直接证明 NUTS 实际起步用的
参数在五个 seed 下逐项相同**。本模块在运行时捕获两层指纹：

  · `init_values_fp`  —— 传给 `init_to_value(values=...)` 的约束空间 dict；
  · `hmc_init_z_fps`  —— `HMC.init()` 返回的 `HMCState.z`，即 NUTS 真正起步的
                         unconstrained 参数（每条链一次）。

后者才是「采样器实际收到的初值」。只有同一谱、同一 K 的五个 seed 指纹一致，
才能把组间差异归因为「固定初值条件下的采样器随机流」；否则结论必须降级为
「使用确定性初值策略下的随机流敏感性」。

严格边界
--------
· 只做包装（调用原函数、原样返回），不改任何实参、不改采样器、不改初始化、
  不改实验变量；
· hook 不可用时 `level` 降级为 `init_values_only` 或 `unavailable`，如实记录，
  **不伪造运行期证据**；
· 本模块不写生产代码目录，产物只落在证据/输出目录。

用法
----
    import b3b_fingerprint as F
    status = F.install()
    F.begin_case(spectrum="...", path="...", seed=0)
    ... 调用 bmdrt.py auto ...
    F.write_case(outdir / stem, extra={"status": status})

    python evidence/m6/b3b_fingerprint.py --selfcheck   # 极小预算机制自检
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 1

_ctx = threading.local()
_records: list[dict] = []
_unattributed: list[dict] = []
_case_meta: dict = {}
_orig_fit_one = None
_orig_hmc_init = None
_fit_sig = None
_install_status: dict = {}


# --------------------------------------------------------------------------
# 规范化指纹
# --------------------------------------------------------------------------

def _fp(obj) -> str | None:
    """dict[str, array-like] → SHA256。键排序、逐项记 dtype/shape/字节。

    统一转 float64 只影响字节表示，不改变「值是否相同」；同时把原始 dtype
    名写进摘要，避免 float32/float64 的表示差异被静默抹掉。
    """
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()
    h = hashlib.sha256()
    for k in sorted(obj.keys(), key=str):
        v = obj[k]
        raw = np.asarray(v)
        a = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
        h.update(str(k).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(raw.dtype).encode("ascii"))
        h.update(b"\x00")
        h.update(str(a.shape).encode("ascii"))
        h.update(b"\x00")
        h.update(a.tobytes())
        h.update(b"\x00")
    return h.hexdigest()


def _describe(obj) -> dict:
    if not isinstance(obj, dict):
        return {}
    return {str(k): dict(dtype=str(np.asarray(v).dtype),
                         shape=list(np.asarray(v).shape))
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}


def _split_fp(z, seeded_keys) -> dict:
    """把 NUTS 实际初值 z 拆成「播种 site」与「未播种 site」两组指纹。

    `init_to_value(values=...)` 只对 values 里出现的 site 生效，其余 site 回落
    到 `init_to_uniform`，会消耗 `PRNGKey(seed)` 那条随机流。所以「seeded=true」
    只保证**部分**参数初值确定，全量指纹是否随 seed 变必须实测拆分。
    """
    if not isinstance(z, dict):
        return dict(fp_all=None, fp_seeded=None, fp_unseeded=None,
                    seeded_keys=[], unseeded_keys=[], site_fps={})
    sk = set(str(k) for k in seeded_keys)
    skeys = sorted(k for k in z if k in sk)
    ukeys = sorted(k for k in z if k not in sk)
    return dict(
        fp_all=_fp(z),
        fp_seeded=_fp({k: z[k] for k in skeys}),
        fp_unseeded=_fp({k: z[k] for k in ukeys}),
        seeded_keys=skeys,
        unseeded_keys=ukeys,
        site_fps={k: _fp({k: z[k]}) for k in sorted(z)},
    )


# --------------------------------------------------------------------------
# hook：fit_one 负责「当前是哪个 K」，HMC.init 负责「实际初值」
# --------------------------------------------------------------------------

def _wrap_fit_one(*args, **kwargs):
    global _fit_sig
    bound = _fit_sig.bind(*args, **kwargs)
    a = bound.arguments
    init = a.get("init")
    rec = dict(
        K=int(a.get("K")),
        seeded=init is not None,
        init_values_fp=_fp(init),
        init_values_params=_describe(init),
        hmc_init_z_fps=[],
        hmc_init_z_params=[],
        hmc_init_z_split=[],
        seed_arg=a.get("seed"),
    )
    _records.append(rec)
    prev = getattr(_ctx, "rec", None)
    _ctx.rec = rec
    try:
        return _orig_fit_one(*args, **kwargs)
    finally:
        _ctx.rec = prev


def _wrap_hmc_init(self, rng_key, num_warmup, init_params=None,
                   model_args=(), model_kwargs=None):
    state = _orig_hmc_init(self, rng_key, num_warmup, init_params,
                           model_args,
                           {} if model_kwargs is None else model_kwargs)
    z = getattr(state, "z", None)
    rec = getattr(_ctx, "rec", None)
    seeded_keys = list((rec or {}).get("init_values_params", {}).keys())
    split = _split_fp(z, seeded_keys)
    item = dict(fp=split["fp_all"], params=_describe(z), split=split)
    if rec is None:
        _unattributed.append(item)
    else:
        rec["hmc_init_z_fps"].append(item["fp"])
        rec["hmc_init_z_params"].append(item["params"])
        rec["hmc_init_z_split"].append(split)
    return state


def install() -> dict:
    """安装 hook。返回可 JSON 序列化的安装状态（含降级原因）。"""
    global _orig_fit_one, _orig_hmc_init, _fit_sig, _install_status
    import bm_real as R

    st = dict(schema=SCHEMA, fit_one_patched=False, hmc_init_patched=False,
              level="unavailable", note="")
    try:
        import numpyro
        st["numpyro_version"] = str(numpyro.__version__)
    except Exception as e:                                   # noqa: BLE001
        st["numpyro_version"] = None
        st["note"] = f"numpyro 不可用：{type(e).__name__}: {e}"

    _fit_sig = inspect.signature(R.fit_one)
    _orig_fit_one = R.fit_one
    R.fit_one = _wrap_fit_one
    st["fit_one_patched"] = True

    try:
        from numpyro.infer.hmc import HMC
        from numpyro.infer import NUTS
        _orig_hmc_init = HMC.init
        HMC.init = _wrap_hmc_init
        if NUTS.init is not _wrap_hmc_init:
            # 未来版本若 NUTS 自己覆盖 init，也要包上，否则捕获不到。
            NUTS.init = _wrap_hmc_init
        st["hmc_init_patched"] = (HMC.init is _wrap_hmc_init
                                  and NUTS.init is _wrap_hmc_init)
        st["level"] = "runtime_z" if st["hmc_init_patched"] else "init_values_only"
        if not st["hmc_init_patched"]:
            st["note"] = "HMC.init patch 后校验失败，只有约束空间指纹"
    except Exception as e:                                   # noqa: BLE001
        st["level"] = "init_values_only"
        st["note"] = f"无法捕获 HMCState.z：{type(e).__name__}: {e}；" \
                     "结论须限定为「确定性初值策略下的随机流敏感性」"

    _install_status = dict(st)
    return st


def uninstall() -> None:
    """恢复原函数（仅自检用）。"""
    global _orig_fit_one, _orig_hmc_init
    if _orig_fit_one is not None:
        import bm_real as R
        R.fit_one = _orig_fit_one
    if _orig_hmc_init is not None:
        from numpyro.infer.hmc import HMC
        from numpyro.infer import NUTS
        HMC.init = _orig_hmc_init
        NUTS.init = _orig_hmc_init


def begin_case(spectrum: str, path: str, seed: int) -> None:
    global _records, _unattributed, _case_meta
    _records = []
    _unattributed = []
    _case_meta = dict(spectrum=spectrum, path=path, seed=int(seed),
                      install=_install_status)
    _ctx.rec = None


def collect() -> dict:
    return dict(
        schema=SCHEMA,
        case=_case_meta,
        level=_install_status.get("level", "unavailable"),
        note=_install_status.get("note", ""),
        n_k_records=len(_records),
        by_K=[
            dict(K=r["K"], seeded=r["seeded"],
                 init_values_fp=r["init_values_fp"],
                 init_values_params=r["init_values_params"],
                 n_chains_seen=len(r["hmc_init_z_fps"]),
                 hmc_init_z_fps=r["hmc_init_z_fps"],
                 hmc_init_z_params=r["hmc_init_z_params"],
                 hmc_init_z_split=r["hmc_init_z_split"],
                 hmc_init_z_all_equal=(len(set(r["hmc_init_z_fps"])) <= 1
                                       if r["hmc_init_z_fps"] else None),
                 seed_arg=r["seed_arg"])
            for r in _records
        ],
        unattributed=[dict(fp=x["fp"], params=x["params"], split=x["split"])
                      for x in _unattributed],
    )


def write_case(path: Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    data = collect()
    if extra:
        data.update(extra)
    out = path / "init_fingerprints.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    return out


# --------------------------------------------------------------------------
# 汇总核验：同一谱、同一 K 的五个 seed 指纹必须一致
# --------------------------------------------------------------------------

def verify(cases: list[dict]) -> dict:
    """cases: [{spectrum, seed, by_K:[...]}]。逐 (谱,K) 比对 5 个 seed。"""
    groups: dict[tuple[str, int], list[dict]] = {}
    for c in cases:
        for r in c.get("by_K", []):
            key = (c["spectrum"], int(r["K"]))
            splits = r.get("hmc_init_z_split") or []
            groups.setdefault(key, []).append(dict(
                seed=int(c["seed"]),
                init_values_fp=r.get("init_values_fp"),
                z_all=tuple(r.get("hmc_init_z_fps") or ()),
                fp_seeded=tuple(s.get("fp_seeded") for s in splits),
                fp_unseeded=tuple(s.get("fp_unseeded") for s in splits),
                unseeded_keys=sorted({k for s in splits
                                      for k in (s.get("unseeded_keys") or [])}),
            ))

    def _same(col, vals):
        seq = [v[col] for v in vals]
        if not seq or not all(seq) or any(x is None for x in seq):
            return False
        return len(set(seq)) == 1

    rows = []
    for (spec, K), vals in sorted(groups.items()):
        iv_ok = _same("init_values_fp", vals)
        z_all_ok = _same("z_all", vals)
        seeded_ok = _same("fp_seeded", vals)
        rows.append(dict(
            spectrum=spec, K=K, n_seeds=len(vals),
            seeds=[v["seed"] for v in vals],
            init_values_fp_consistent=iv_ok,
            hmc_init_z_all_consistent=z_all_ok,
            hmc_init_z_seeded_consistent=seeded_ok,
            unseeded_sites=vals[0]["unseeded_keys"] if vals else [],
            init_values_fp=vals[0]["init_values_fp"] if vals else None,
        ))

    z_all = bool(rows) and all(r["hmc_init_z_all_consistent"] for r in rows)
    seeded = bool(rows) and all(r["init_values_fp_consistent"]
                                and r["hmc_init_z_seeded_consistent"] for r in rows)
    if z_all:
        level, conclusion = (
            "runtime_z_full",
            "NUTS 实际初值在 seed 0..4 下逐项一致；可把组间差异归于"
            "「固定初值条件下的采样器随机流」")
    elif seeded:
        level, conclusion = (
            "runtime_z_seeded_only",
            "仅 init_values 播种的参数初值在 seed 0..4 下逐项一致；"
            "未播种 site 的初值随随机流变化。按预注册边界，结论限定为"
            "「使用确定性初值策略下的随机流敏感性」，"
            "**不得升级为「已验证初始参数值固定」**")
    else:
        level, conclusion = (
            "degraded",
            "运行期初值指纹缺失或不一致：结论限定为"
            "「使用确定性初值策略下的随机流敏感性」，"
            "**不得升级为「已验证初始参数值固定」**")
    return dict(level=level, n_checks=len(rows),
                all_full_consistent=z_all, rows=rows,
                conclusion=conclusion)


# --------------------------------------------------------------------------
# 机制自检（极小预算，不进 B3b 结果）
# --------------------------------------------------------------------------

def selfcheck() -> int:
    import bm_core as B
    import bm_real as R

    print("=== B3b 运行期初值指纹：机制自检（极小预算，不进正式结果） ===")
    ok = True

    st = install()
    print(f"  install: level={st['level']} fit_one={st['fit_one_patched']} "
          f"hmc_init={st['hmc_init_patched']} numpyro={st.get('numpyro_version')}")
    if st["level"] != "runtime_z":
        print(f"  [FAIL] 运行期 hook 不可用：{st['note']}")
        return 1
    ok &= st["level"] == "runtime_z"

    # ① fit_one 归属逻辑：用 stub 顶替原函数，只验记录结构，不跑采样
    global _orig_fit_one
    saved = _orig_fit_one
    _orig_fit_one = lambda *a, **k: "stub"          # noqa: E731
    begin_case("selfcheck", "stub", 0)
    try:
        assert R.fit_one(None, None, 2, True, 1.0,
                         init={"R_inf": 10.0, "W": 30.0}, seed=7) == "stub"
        rec = collect()["by_K"][0]
        print(f"  ① fit_one 归属：K={rec['K']} seeded={rec['seeded']} "
              f"init_values_fp={str(rec['init_values_fp'])[:12]}…")
        ok &= rec["K"] == 2 and rec["seeded"] and rec["init_values_fp"] is not None
    finally:
        _orig_fit_one = saved

    # ② 真实极小采样：seed 0 / 1 的 NUTS 实际初值指纹必须一致
    path = ROOT / "archive" / "Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt"
    f, Z = R.load_spectrum(str(path))
    include_L = R.detect_L(f, Z)
    kk_cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    kk_opts = {k: kk_cfg["kk"][k] for k in
               ("c", "max_M", "res_tol", "auto_L", "auto_tail", "auto_cap")
               if k in kk_cfg.get("kk", {})}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        pres, pk, _all, _ = R.prescreen_stage(f, Z)
    init = R.init_values(pk, pres, 1)
    print(f"  ② 极小采样：K=1 seeded={init is not None} "
          f"init keys={sorted(init.keys()) if init else []}")

    tiny = dict(K=1, alpha=0.5, sigma=None, include_L=include_L,
                zeta_fixed=1.0, init_values=init, n_warmup=5, n_samples=5,
                num_chains=2, progress=False, target_accept=0.9,
                max_tree_depth=10, extra_fields=())
    fps, seeded_fps, unseeded_sites = {}, {}, []
    hooked_samples = None
    for seed in (0, 1):
        begin_case("selfcheck", str(path), seed)
        # 自检直接调 B.fit（不经过 fit_one），手工补一条等价的归属记录，
        # 让「播种 / 未播种」分项逻辑与正式路径一致。
        fake = dict(K=1, seeded=init is not None, init_values_fp=_fp(init),
                    init_values_params=_describe(init), hmc_init_z_fps=[],
                    hmc_init_z_params=[], hmc_init_z_split=[], seed_arg=seed)
        _records.append(fake)
        _ctx.rec = fake
        s, _mc = B.fit(f, Z, seed=seed, **tiny)
        d = collect()
        rec = d["by_K"][0]
        splits = rec["hmc_init_z_split"]
        fps[seed] = rec["hmc_init_z_fps"]
        seeded_fps[seed] = [x["fp_seeded"] for x in splits]
        unseeded_sites = sorted({k for x in splits
                                 for k in x.get("unseeded_keys", [])})
        print(f"     seed={seed}: HMC.init 捕获 {rec['n_chains_seen']} 次；"
              f"z 全量={[str(x)[:10] for x in fps[seed]]}；"
              f"播种部分={[str(x)[:10] for x in seeded_fps[seed]]}")
        ok &= rec["n_chains_seen"] == 2 and all(splits)
        if seed == 0:
            hooked_samples = s
    print(f"     未播种 site（初值由随机流给）：{unseeded_sites}")
    print(f"     跨 seed 全量初值一致：{fps.get(0) == fps.get(1)}；"
          f"播种部分一致：{seeded_fps.get(0) == seeded_fps.get(1)}")
    print("     注：一致性是**实验事实**，不是本自检的通过条件；"
          "自检只验 hook 能否捕获、分项能否计算、是否改变采样行为。")

    # ③ hook 不改变采样结果：卸载后同 seed 重跑，逐位相同
    uninstall()
    s_plain, _ = B.fit(f, Z, seed=0, **tiny)
    keys = sorted(set(hooked_samples) & set(s_plain))
    diffs = {k: float(np.max(np.abs(np.asarray(hooked_samples[k], float)
                                    - np.asarray(s_plain[k], float))))
             for k in keys
             if np.asarray(hooked_samples[k]).shape == np.asarray(s_plain[k]).shape}
    worst = max(diffs.values()) if diffs else float("nan")
    print(f"  ③ hook 不改行为：{len(keys)} 个共有数组，最大绝对差={worst:.3g}")
    ok &= bool(diffs) and worst == 0.0

    print("  === 自检：" + ("通过" if ok else "失败") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        raise SystemExit(selfcheck())
    print(__doc__)
