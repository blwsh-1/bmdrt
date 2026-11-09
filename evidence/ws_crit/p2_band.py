# -*- coding: utf-8 -*-
"""P2：L1 的幅度带宽该不该按端点走向分设（**影子研究，不改生产代码**）。

背景
----
N1 把 L1 的幅度判据从 `log|Z|` 换成 `log(-Im)`。换完出现一个副作用：
真 `Wo`（有限长**阻挡**）的低频段是容性尾，`-Im ∝ f^-1`，斜率 ≈ −0.97，
落在为 `Ws` 设的带宽 `[-0.85, -0.20]` 之外 → `amp_ok` 由 True 翻成 False。

但 `Wo` 的 45° 段**天生就短**（到拐点就结束，之后是电容），所以它本来就不该
用"最低 25% 段的斜率"去判。这就是本脚本要回答的问题：把带宽拆开，
换来的是不是真的灵敏度，还是只是把容性弧也放进来。

三个变体
--------
    V0  现状（N1 之后）：slope ∈ [-0.85, -0.20]
    V1  放宽下界：        slope ∈ [-1.15, -0.20]        （同时吃下 Wo 的容性尾）
    V2  按端点走向分档（调用 `bm_wscrit.terminal`）：
           to_cap               → [-1.15, -0.85]   （Wo：容性尾）
           其余（含 still_45）  → [-0.85, -0.20]   （Ws：45° 直达边缘）
           to_inductive         → 幅度判据不可用（-Im ≤ 0，取对数无意义）

三者**共用同一套相位判据**（`bm_kk.nyquist_slope_low`，45°±0.35），
所以差异只来自幅度那半边，可以直接比。

读出方式（重要）
----------------
`Wo` 的血本不是"判成 candidate"，而是"判不出 candidate 时能给出正确的家族"。
真正的家族判定在 `bm_wscrit`（`family=Wo_like`），L1 只是 `kk_check` 的前置
诊断。所以本表要连着看：如果 V1/V2 把 `wo_in` 的 candidate 抬上去，代价是
`none_2zarc`/`rq05_in` 也一起抬上去，那这笔交易就不划算。

    $PY evidence/ws_crit/p2_band.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bm_kk                                    # noqa: E402
import bm_wscrit as W                           # noqa: E402
import bench as B                               # noqa: E402

OUT = HERE / "p2_band.json"
FRAC = 0.25
NY_TOL = 0.35
BAND = dict(V0=(-0.85, -0.20), V1=(-1.15, -0.20),
            V2_ws=(-0.85, -0.20), V2_wo=(-1.15, -0.85))
NEG = ("none_2zarc", "rq05_in", "sorp_rl")


def slope_im(f, Z, frac=FRAC):
    """最低 `frac` 段的 `log(-Im) ~ log f` 斜率；`-Im ≤ 0` 时返回 None。"""
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    n_low = max(4, int(round(f.size * frac)))
    idx = np.arange(n_low)
    im = -Z[idx].imag
    if im.min() <= 0:
        return None
    return float(np.polyfit(np.log10(f[idx]), np.log10(im), 1)[0])


def in_band(s, lo_hi):
    return s is not None and lo_hi[0] <= s <= lo_hi[1]


def variants(f, Z):
    """返回 {变体名: (amp_ok, phase_ok, 说明)}。"""
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    s = slope_im(f, Z)
    ny, span, _ = bm_kk.nyquist_slope_low(f, Z, FRAC)
    ph = bool(ny is not None and abs(ny - 1.0) <= NY_TOL)
    tm = W.terminal(f, Z)
    out = {}
    out["V0"] = (in_band(s, BAND["V0"]), ph, "现状")
    out["V1"] = (in_band(s, BAND["V1"]), ph, "放宽下界")
    if tm["label"] == "to_inductive":
        out["V2"] = (False, ph, "感抗尾：幅度判据不可用")
    elif tm["label"] == "to_cap":
        out["V2"] = (in_band(s, BAND["V2_wo"]), ph, "to_cap→容性带")
    else:
        out["V2"] = (in_band(s, BAND["V2_ws"]), ph, f"{tm['label']}→Ws 带")
    return out, s, ny, tm


def run_bench(n_per=20, npts=31, fmin=0.1, fmax=1e3, noise=0.01, seed=0):
    f = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    rng = np.random.default_rng(seed)
    tab = {}
    for cls in B.CLASSES:
        hit = {k: 0 for k in ("V0", "V1", "V2")}
        for _ in range(n_per):
            Z = B.make(f, cls, rng)
            sd = noise * np.abs(Z)
            Zn = Z + sd * (rng.standard_normal(f.size)
                           + 1j * rng.standard_normal(f.size))
            v, _, _, _ = variants(f, Zn)
            for k in hit:
                hit[k] += int(v[k][0] and v[k][1])
        tab[cls] = {k: hit[k] / n_per for k in hit}
    return tab, dict(fmin=fmin, fmax=fmax, n_per=n_per, npts=npts)


def run_real(zdir=None):
    zdir = Path(zdir) if zdir else ROOT / "out/_m6_samples/z_csv"
    rows = []
    for p in sorted(Path(zdir).glob("*.csv")):
        d = np.genfromtxt(p, delimiter=",", names=True)
        nm = d.dtype.names
        f = np.asarray(d[nm[0]], float)
        Z = np.asarray(d[nm[1]], float) + 1j * np.asarray(d[nm[2]], float)
        v, s, ny, tm = variants(f, Z)
        rows.append(dict(stem=p.stem, slope_im=s, ny=ny, term=tm["label"],
                         im_ratio=tm.get("im_ratio"),
                         **{k: bool(v[k][0] and v[k][1]) for k in v}))
    return rows


def main():
    tab, meta = run_bench()
    real = run_real()

    keys = ("V0", "V1", "V2")
    print("===== 合成 9 类 × 20 条：P(candidate) =====")
    print(f"{'class':11s} " + " ".join(f"{k:>6s}" for k in keys) + "   判读")
    note = {
        "ws_vis": "Ws 可见：应三变体都高", "ws_buried": "Ws 隐没：噪声限",
        "wo_in": "**关键** Wo 在窗内：V1/V2 才可能救",
        "wo_below": "Wo 拐点在窗下：与 Ws 同形", "wshort_in": "Wshort：回平台",
        "none_2zarc": "阴例：弧下降段", "rq05_low": "阴例：窗外 R||CPE(.5)（同形）",
        "rq05_in": "阴例：窗外 CPE 峰在窗内", "sorp_rl": "阴例：R||L 吸附环",
    }
    for cls in B.CLASSES:
        r = tab[cls]
        print(f"{cls:11s} " + " ".join(f"{r[k]:6.2f}" for k in keys)
              + f"   {note.get(cls, '')}")

    spec = {k: float(np.mean([tab[c][k] for c in NEG])) for k in keys}
    print(f"\n{'三阴例均值(越低越好)':11s} " + " ".join(f"{spec[k]:6.2f}" for k in keys))

    print("\n===== 18 条真实 AZ31 谱：candidate 条数（应全为 0）=====")
    tdist = {}
    for r in real:
        tdist[r["term"]] = tdist.get(r["term"], 0) + 1
    for k in keys:
        n_hit = sum(1 for r in real if r[k])
        print(f"  {k}: {n_hit}/{len(real)} candidate")
    print(f"  端点分布：{tdist}")

    print(f"\n写入 {OUT}")
    OUT.write_text(json.dumps(dict(bench=tab, meta=meta, real=real,
                                   specificity_over_negatives=spec),
                              ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
