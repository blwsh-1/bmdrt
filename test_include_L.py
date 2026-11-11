#!/usr/bin/env python
"""对照实验：同一条谱、同一 σ 地板，只把 `include_L` 打开。

为什么需要这个对照
------------------
σ 地板扫描里 797（真值 `L-R-RCPE-RCPE`）出现极端不对称：
`rms_re=9.3e-8`（机器精度）、`rms_im=0.0256`，`σ_re/zmax=1.3e-6`
而 `σ_im/zmax=7.7e-3`，差 6000 倍。

原因不是噪声模型，是 `detect_L()` 投了 `False`：它只看最高 3 点的
`Im Z > 0`，而 797 的高频 Im 被快 CPE（f0≈2.3e5 Hz）压成负值，
尽管真值 `L1=1.18e-7 H` 在 1 MHz 处贡献 `f·L ≈ 0.118 Ω`（占 `|Z|` 的 29%）。
`jωL` 只进虚部 → 实部被精确表示、虚部留结构残差 → `σ_re→0`。

这个脚本把 `include_L` 强制打开，与扫描里的同一档位对比，用来区分两件事：

  · **σ 不对称**是不是模型缺项造成的；
  · **树深触顶（漏斗）**是不是无噪数据本身造成的（即使模型完全正确，
    两条 σ 都会往 0 推）。

用法
----
    python test_include_L.py                      # 797，K=2，floor=0/1e-4
    python test_include_L.py --only 797 --floors 0,1e-4 --K 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import test_ecm as T                                     # noqa: E402


def _cfg(outdir: Path, quick: bool, sigma_floor: float, include_L: bool) -> dict:
    import bmdrt as M
    cfg = M._deep_merge(M.DEFAULTS,
                        json.loads((ROOT / "config.json").read_text(encoding="utf-8")))
    cfg = M.resolve(cfg)
    cfg["paths"]["out"] = outdir
    cfg["batch"]["move_done"] = False
    cfg["batch"]["skip_done"] = False
    cfg["fit"]["quick"] = bool(quick)
    cfg["fit"]["sigma_floor"] = float(sigma_floor)
    cfg["fit"]["include_L"] = bool(include_L)
    return cfg


def _pick(res: dict) -> dict:
    if not res:
        return dict(ok=False)
    return dict(
        ok=True,
        include_L=res.get("include_L"),
        sigma_re=res.get("sigma_re"), sigma_im=res.get("sigma_im"),
        zmax=res.get("zmax_obs"),
        rms_re=res.get("rms_re"), rms_im=res.get("rms_im"),
        tree_cap=res.get("tree_cap"), div=res.get("divergences"),
        grade=res.get("sample_grade"),
        R_inf=res.get("R_inf"), L_uH=res.get("L_uH"),
        cover=min(res.get("cover_re") or 1.0, res.get("cover_im") or 1.0),
        comps=[(c.get("f0"), c.get("w"), c.get("phi"))
               for c in (res.get("components") or [])],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="include_L 开/关对照")
    ap.add_argument("--only", default="797", help="行号，逗号分隔")
    ap.add_argument("--floors", default="0,1e-4", help="σ 地板（相对 zmax）")
    ap.add_argument("--K", type=int, default=0, help="固定 K；0 = 用真值 K")
    ap.add_argument("--full", action="store_true", help="正式档（慢）")
    ap.add_argument("--tag", default="incL")
    a = ap.parse_args(argv)

    floors = [float(x) for x in a.floors.split(",") if x.strip()]
    idx = [int(x) for x in a.only.split(",") if x.strip()]
    df = T.load_table(T.DEFAULT_SRC)
    base = ROOT / "out" / f"ecm_{a.tag}"

    rows = []
    for i in idx:
        row = df.iloc[i]
        t = T.truth_of(row)
        t["row_Z"] = row["Z"]
        stem = f"ecm_{i:05d}_{t['circ'].replace('-', '_')}"
        csv, _, _ = T.export(t, base / "csv" / f"{stem}.csv")
        K = a.K or max(1, int(t["K"]))
        print(f"\n[{i}] {t['circ']}  真值 K={t['K']}  L={t['L'] * 1e6:.3g} µH  "
              f"用 K={K}", flush=True)
        print(f"{'include_L':10s} {'floor':>7s} {'σre/zmax':>10s} {'σim/zmax':>10s} "
              f"{'rms_re':>10s} {'rms_im':>9s} {'触顶':>6s} {'div':>4s} {'分级':>4s}")
        for L in (False, True):
            for fl in floors:
                out = base / f"L{int(L)}_floor_{fl:.0e}"
                cfg = _cfg(out, not a.full, fl, L)
                try:
                    rec = T._run_forced(csv, out, cfg, K, quiet=True)
                except Exception as e:                   # noqa: BLE001
                    print(f"{str(L):10s} {fl:7.0e}  !! {type(e).__name__}: {e}")
                    continue
                m = _pick(rec.get("result"))
                rows.append(dict(idx=i, L=L, floor=fl, m=m))
                if not m.get("ok"):
                    print(f"{str(L):10s} {fl:7.0e}  未拟合")
                    continue
                z = m["zmax"] or 1.0
                # KK 拒绝谱 ok=True 但 tree_cap 等为 None，直接格式化会 TypeError
                # 崩掉整个脚本（审查 P2-10）。逐项做 None 安全格式化。
                _nan = float("nan")
                sre, sim = m.get("sigma_re"), m.get("sigma_im")
                rre, rim = m.get("rms_re"), m.get("rms_im")
                tc, dv, gr = m.get("tree_cap"), m.get("div"), m.get("grade")
                tc_s = f"{tc:6.1%}" if isinstance(tc, (int, float)) else "   n/a"
                dv_s = f"{dv:4d}" if isinstance(dv, int) else " n/a"
                gr_s = f"{gr:4d}" if isinstance(gr, int) else " n/a"
                print(f"{str(L):10s} {fl:7.0e} "
                      f"{(sre / z if sre is not None else _nan):10.2e} "
                      f"{(sim / z if sim is not None else _nan):10.2e} "
                      f"{(rre if rre is not None else _nan):10.2e} "
                      f"{(rim if rim is not None else _nan):9.2e} "
                      f"{tc_s} {dv_s} {gr_s}", flush=True)

    md = _report(rows, base / "INCLUDE_L.md")
    print(f"\n报告：{md}")
    return 0


def _report(rows, path: Path) -> Path:
    L = ["# `include_L` 开/关对照", "",
         "> 目的：把「σ 不对称」与「树深漏斗」两件事分开归因。", "",
         "| 谱 | include_L | 地板 | σ_re/zmax | σ_im/zmax | rms_re | rms_im | 触顶 | div | 分级 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["m"]
        if not m.get("ok"):
            L.append(f"| {r['idx']} | {r['L']} | {r['floor']:.0e} | 未拟合 | | | | | | |")
            continue
        z = m["zmax"] or 1.0
        # P2-14（二轮审查）：与主打印（:112-126）同一套 None 安全格式化——
        # KK 拒绝谱 ok=True 但 tree_cap/div/grade/sigma_* 为 None，
        # 直接 f-string 格式化抛 TypeError 崩掉整个脚本。
        _na = "n/a"
        def _f(v, fmt):
            return format(v, fmt) if isinstance(v, (int, float)) else _na
        sre, sim = m.get("sigma_re"), m.get("sigma_im")
        rre, rim = m.get("rms_re"), m.get("rms_im")
        tc, dv, gr = m.get("tree_cap"), m.get("div"), m.get("grade")
        L.append(f"| {r['idx']} | {m.get('include_L')} | {r['floor']:.0e} "
                 f"| {_f(sre / z if sre is not None else None, '.2e')} "
                 f"| {_f(sim / z if sim is not None else None, '.2e')} "
                 f"| {_f(rre, '.2e')} | {_f(rim, '.2e')} "
                 f"| {_f(tc, '.1%') if isinstance(tc, (int, float)) else _na} "
                 f"| {_f(dv, 'd') if isinstance(dv, int) else _na} "
                 f"| {_f(gr, 'd') if isinstance(gr, int) else _na} |")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
