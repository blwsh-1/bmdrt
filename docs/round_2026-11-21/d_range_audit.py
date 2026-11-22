"""D 域审计（只读）：算每类谱的动态范围 D = max|Z|/min|Z|。

为什么这件事不需要"更多验证数据"
--------------------------------
`free` 口径把 sigma 定在 0.01*zmax。于是最低阻抗点的信噪比是恒等式

    SNR_min = |Z|min / (0.01*zmax) = 100 / D

这不是统计结论，是定义展开。D>100 时最低点的噪声就大于信号本身，
该点对似然几乎无贡献。所以"要不要守"不需要实验裁决，只需要知道
**现有谱的 D 分布**——那决定改动的影响面，而不是判据的正确性。

本脚本只读，不改任何文件；输出每类的 N、D 分位、超阈比例。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/d_range_audit.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(REPO)
sys.path.insert(0, REPO)

import bm_real as R  # noqa: E402

OUT = os.path.join("out", "noise_scale_probe")
SOURCES = [
    ("合成", os.path.join(REPO, "out", "ecm_test", "csv", "*.csv")),
    ("腐蚀", os.path.join(ROOT, "测试数据", "腐蚀", "**", "*.csv")),
    ("钝化", os.path.join(ROOT, "测试数据", "钝化", "**", "*.csv")),
    ("涂层", os.path.join(ROOT, "测试数据", "涂层", "**", "*.csv")),
    ("2E", os.path.join(REPO, "EISmeasurements", "csv_2e", "*.csv")),
    ("3E", os.path.join(REPO, "EISmeasurements", "csv_3e", "*.csv")),
    ("AZ31", os.path.join(REPO, "out", "_m6_samples", "z_csv", "*.csv")),
]


def dstats(paths, max_n=None):
    ds, snrs, bad = [], [], 0
    for p in paths[:max_n] if max_n else paths:
        try:
            f, Z = R.load_spectrum(p)
        except SystemExit:
            bad += 1
            continue
        a = np.abs(np.asarray(Z, complex))
        a = a[np.isfinite(a) & (a > 0)]
        if a.size < 10:
            bad += 1
            continue
        # 用 2% 分位而非 min，避免单点过零把 D 顶爆
        lo = float(np.percentile(a, 2))
        hi = float(a.max())
        ds.append(hi / lo)
        snrs.append(lo / (0.01 * hi))
    return np.asarray(ds), np.asarray(snrs), bad


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print("%-6s %5s %7s %7s %7s %7s %8s %8s %8s" % (
        "class", "N", "D_med", "D_p90", "D_max", "D_min",
        "D>100", "D>1e3", "SNR_med"))
    print("-" * 76)
    for name, pat in SOURCES:
        paths = sorted(glob.glob(pat, recursive=True))
        if not paths:
            print("%-6s %5s  (no files: %s)" % (name, 0, pat))
            continue
        ds, snrs, bad = dstats(paths)
        if ds.size == 0:
            print("%-6s %5s  (unreadable)" % (name, 0))
            continue
        rows.append(dict(cls=name, n=int(ds.size), unreadable=int(bad),
                         d=np.log10(ds).tolist(),
                         d_med=float(np.median(ds)),
                         d_p90=float(np.percentile(ds, 90)),
                         d_max=float(ds.max()), d_min=float(ds.min()),
                         frac_gt100=float((ds > 100).mean()),
                         frac_gt1e3=float((ds > 1000).mean()),
                         snr_med=float(np.median(snrs))))
        print("%-6s %5d %7.1f %7.1f %7.0f %7.2f %7.0f%% %7.0f%% %8.3g" % (
            name, ds.size, np.median(ds), np.percentile(ds, 90), ds.max(),
            ds.min(), 100 * (ds > 100).mean(), 100 * (ds > 1e3).mean(),
            np.median(snrs)))

    allD = np.concatenate([np.power(10, r["d"]) for r in rows])
    print("-" * 76)
    print("全部 %d 条：D 中位 %.1f，D>100 占 %.0f%%，D>1000 占 %.0f%%" % (
        allD.size, np.median(allD), 100 * (allD > 100).mean(),
        100 * (allD > 1000).mean()))

    with open(os.path.join(OUT, "d_range.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(rows=rows,
                       all_n=int(allD.size),
                       all_frac_gt100=float((allD > 100).mean()),
                       all_frac_gt1e3=float((allD > 1000).mean()),
                       formula="SNR_min = 100/D under sigma=0.01*zmax"),
                  fh, ensure_ascii=False, indent=1, allow_nan=False)
    print("->", os.path.join(OUT, "d_range.json"))
    make_fig(rows)
    return 0


# 21 条 quick/free 的 prod 口径中位 rms（docs/round_2026-11-21/BATCH_21.md 表）
PROD_RMS = {"合成": 1.0, "腐蚀": 298.5, "钝化": 2774.6, "涂层": 38.3,
            "2E": 6.4, "3E": 1.0, "AZ31": 28.9}
EN = {"合成": "synthetic", "腐蚀": "corrosion", "钝化": "passivation",
      "涂层": "coating", "2E": "2E", "3E": "3E", "AZ31": "AZ31"}


def make_fig(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]
    names = [r["cls"] for r in rows]
    for i, r in enumerate(rows):
        d = 10 ** np.asarray(r["d"])
        jit = (np.random.default_rng(i).random(d.size) - 0.5) * 0.42
        ax.plot(d, np.full(d.size, i) + jit, "o", ms=2.6, alpha=0.5,
                color="#D85A30")
        ax.plot([r["d_med"]], [i], "D", ms=9, color="#534AB7", zorder=5)
    ax.axvline(100, color="#185FA5", lw=1.4, ls="--")
    ax.axvline(1000, color="#A32D2D", lw=1.4, ls="--")
    ax.text(105, len(rows) - 0.4, "D=100\nSNR_min=1", fontsize=8,
            color="#185FA5", va="top")
    ax.text(1050, len(rows) - 0.4, "D=1000\nSNR_min=0.1", fontsize=8,
            color="#A32D2D", va="top")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([EN.get(n, n) for n in names], fontsize=9)
    ax.set_xlabel("dynamic range D = max|Z| / 2nd-pct |Z|", fontsize=10)
    ax.set_title("Where each class sits (points) and its median (diamond)",
                 fontsize=10.5, fontweight="500", loc="left")
    ax.grid(True, axis="x", alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axs[1]
    xs = np.array([r["d_med"] for r in rows])
    ys = np.array([PROD_RMS[r["cls"]] for r in rows])
    ax.plot(xs, ys, "o", ms=8, color="#D85A30")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(EN.get(r["cls"], r["cls"]), (x, y), textcoords="offset points",
                    xytext=(7, -3), fontsize=9)
    k = np.polyfit(np.log10(xs), np.log10(ys), 1)[0]
    rr = float(np.corrcoef(np.log10(xs), np.log10(ys))[0, 1])
    ax.plot(xs, 10 ** (np.log10(ys).mean()) * (xs / xs.mean()) ** k,
            color="#888780", lw=1.2, ls="--",
            label="log-log slope %.2f, r=%.2f" % (k, rr))
    ax.axvline(100, color="#185FA5", lw=1.4, ls="--")
    ax.axvline(1000, color="#A32D2D", lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median D of the class", fontsize=10)
    ax.set_ylabel("median fit error, prod/free / %", fontsize=10)
    ax.set_title("Data-side D vs fit-side error: two independent quantities",
                 fontsize=10.5, fontweight="500", loc="left")
    ax.grid(True, which="both", alpha=0.18, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_d_range.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    raise SystemExit(main())
