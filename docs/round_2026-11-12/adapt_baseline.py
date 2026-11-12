"""ADAPT-ECM 4.0.1 BM-DRT 外部基线（只读对照，不修改外部包）。

用法（必须在仓库根 `bmdrt/` 下跑，产物是 cwd 相对的 `out/...`）：

    python docs/round_2026-11-12/adapt_baseline.py --all
    python docs/round_2026-11-12/adapt_baseline.py --files archive/AZ3.txt ...

外部包路径默认 `<工作区>/_external/ADAPT-ECM-4.0.1/src/adapt_ecm`，
可用环境变量 `ADAPT_ECM_SRC` 覆盖。

**我们不 import 它的 GUI，也不改它任何一个文件**：只调用
`run_bm_drt_simple.run_bm_drt_programmatic(freq, Z)`，这是它自己为 GUI
写的无对话框入口（`run_bm_drt_simple.py:146`）。

判据口径与本项目对齐（不另造阈值）：
  · Rwp[%] = sqrt(mean(|dZ|²/|Z_obs|²))·100，实虚联合、不除 √2
  · acf    = bm_experiments.acf_max 对实/虚残差各取一次再取 max
             （与 bm_real.py:1086 相同），§13/§14 结构线 acf ≤ 0.50
  · 我们侧判定直接读已落盘的 `out/<样本>/real_<样本>.json`，不重跑拟合。

注意：本脚本只做「它报了什么 / 我们报了什么」的对照，**不把 ADAPT-ECM
的输出写进任何生产结果、不参与任何门禁或分级**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # docs/round_2026-11-12 -> 仓库根
sys.path.insert(0, str(ROOT))

_ADAPT_DEFAULT = ROOT.parent / "_external" / "ADAPT-ECM-4.0.1" / "src" / "adapt_ecm"
ADAPT_SRC = Path(os.environ.get("ADAPT_ECM_SRC", str(_ADAPT_DEFAULT)))
if not ADAPT_SRC.is_dir():
    raise SystemExit(f"找不到 ADAPT-ECM 源码目录：{ADAPT_SRC}\n"
                     "请解压 ADAPT-ECM-4.0.1.zip 或设置 ADAPT_ECM_SRC")
sys.path.insert(0, str(ADAPT_SRC))

from bm_real import load_spectrum          # noqa: E402
from bm_experiments import acf_max          # noqa: E402
import run_bm_drt_simple as A              # noqa: E402

# 我们已有产物、且判定类型有代表性的谱。AZ31-4Li.txt 是六列
# index,f,Zre,Zim,... 的导出格式，load_spectrum 会误读，故意排除。
DEFAULT_FILES = [
    "archive/AZ3.txt",
    "archive/Blank_0.1KBr_pH10_EIS_PtCu_1.2V.txt",
    "archive/2EMC1-1_0005mA.csv",
    "archive/2EMC1-2_0750mA.csv",
    "archive/2EMC2-1_0200mA.csv",
    "archive/3E-1_OER_exp1_0005mA.csv",
    "archive/AZ31-4Li_100.txt",
]


def _our_side(stem: str) -> dict:
    """读我们已落盘的判定；没有产物就明确写 unavailable，不猜。"""
    hits = sorted((ROOT / "out").glob(f"*/real_{stem}.json"))
    if not hits:
        return dict(available=False, reason="无对应 real_*.json 产物")
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    kk = d.get("kk") or {}
    gf = d.get("grade_fail")
    if kk.get("status") == "fail" or d.get("K") is None:
        verdict = "refused"          # 我们不给数
    elif gf:
        verdict = "degraded"        # 我们给数但降级
    else:
        verdict = "reported"
    diag = d.get("diag") or {}
    rhats = [v.get("rhat") for v in diag.values()
             if isinstance(v, dict) and isinstance(v.get("rhat"), (int, float))]
    return dict(
        available=True,
        path=str(hits[0].relative_to(ROOT)).replace("\\", "/"),
        verdict=verdict,
        kk_status=kk.get("status"),
        K=d.get("K"),
        grade_fail=gf,
        acf=d.get("acf"),
        divergences=d.get("divergences"),
        rhat_max=max(rhats) if rhats else None,
    )


def _bands(d: np.ndarray, z: np.ndarray) -> list[float]:
    n = d.size
    e = [0, n // 3, 2 * n // 3, n]
    return [float(np.sqrt(np.mean(np.abs(d[e[i]:e[i + 1]]) ** 2 /
                                  np.abs(z[e[i]:e[i + 1]]) ** 2)) * 100)
            for i in range(3)]           # 频率降序：高 / 中 / 低


def run_one(path: str, *, warmup: int, samples: int, seed: int,
            warburg: bool) -> dict:
    f, z = load_spectrum(path)
    t0 = time.time()
    r = A.run_bm_drt_programmatic(
        f, z, warmup=warmup, samples=samples, chains=1, seed=seed,
        include_warburg=warburg,
    )
    R = r["result"]
    Zf = R["Z_fit"]
    # `run_bm_drt_simple._clean_eis_arrays()` 会按频率**降序重排**后再返回；
    # 原始 z 可能是升序。直接相减会得到错位对齐的假残差（实测虚报 Rwp 77%）。
    # 一律用 runner 自己返回的清洗后数组做对照。
    f, z = r["freq"], r["Z"]
    assert np.array_equal(np.asarray(R["Z_exp"]), np.asarray(z)), "runner 返回的 Z 与 Z_exp 不一致"
    d = Zf - z
    az = np.abs(z)
    rwp = float(np.sqrt(np.mean(np.abs(d) ** 2 / az ** 2)) * 100)
    acf = float(max(abs(acf_max(d.real)), abs(acf_max(d.imag))))
    ci_re = float(np.median((R["Zre_fit_q95"] - R["Zre_fit_q05"]) / az))
    ci_im = float(np.median((R["Zim_fit_q95"] - R["Zim_fit_q05"]) / az))
    ps = {x["parameter"]: x["q50"] for x in R["parameter_summary"]}
    tau = np.asarray(R["tau_fine"])
    g = np.asarray(R["gamma_total_q50"])
    peak = float(tau[int(np.argmax(g))]) if g.size == tau.size else None

    out = dict(
        file=path.replace("\\", "/"),
        stem=Path(path).stem,
        n=int(f.size),
        f_min=float(f.min()), f_max=float(f.max()),
        adapt=dict(
            ok=bool(r["ok"]),
            model=R["model"], k_hn=R["k_hn"], k_warburg=R["k_warburg"],
            include_warburg=bool(R["include_warburg"]),
            warmup=R["warmup"], samples=R["samples"], chains=R["chains"],
            seed=R["seed"], elapsed_s=float(time.time() - t0),
            rwp_pct=rwp, acf=acf, bands_pct=_bands(d, z),
            ci_re_rel=ci_re, ci_im_rel=ci_im,
            gamma_peak_tau_s=peak,
            sigma_re_ohm=ps.get("sigma_re_ohm"),
            sigma_im_ohm=ps.get("sigma_im_ohm"),
            p_W_fraction=ps.get("p_W_fraction"),
            R_inf_ohm=ps.get("R_inf_ohm"),
            n_obs_im_pos=int(np.sum(z.imag > 0)),
            n_fit_im_pos=int(np.sum(Zf.imag > 0)),
            obs_im_pos_max=float(z.imag.max()) if np.any(z.imag > 0) else 0.0,
            fit_im_pos_max=float(Zf.imag.max()) if np.any(Zf.imag > 0) else 0.0,
            # runner 返回的字段里有没有任何收敛 / 质量标记？
            has_quality_flag=any(
                k in R for k in ("rhat", "divergences", "converged",
                                 "status", "warning", "quality")),
            result_keys=sorted(R.keys()),
        ),
        ours=_our_side(Path(path).stem),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warburg", choices=("on", "off"), default="on")
    ap.add_argument("--out", default="out/adapt_baseline")
    a = ap.parse_args()

    files = DEFAULT_FILES if (a.all or not a.files) else a.files
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        print(f"[START] {p}  warburg={a.warburg} "
              f"warmup={a.warmup} samples={a.samples} seed={a.seed}", flush=True)
        try:
            row = run_one(p, warmup=a.warmup, samples=a.samples,
                          seed=a.seed, warburg=(a.warburg == "on"))
        except Exception as exc:            # 单条失败不中断整批，如实记录
            row = dict(file=p, stem=Path(p).stem,
                       error=f"{type(exc).__name__}: {exc}",
                       ours=_our_side(Path(p).stem))
            print(f"  [ERROR] {row['error']}", flush=True)
        else:
            A_ = row["adapt"]
            print(f"  Rwp={A_['rwp_pct']:.2f}%  acf={A_['acf']:.3f}  "
                  f"CI_re={A_['ci_re_rel']:.3f}  "
                  f"ours={row['ours'].get('verdict', 'n/a')}  "
                  f"{A_['elapsed_s']:.0f}s", flush=True)
        (outdir / f"{row['stem']}_{a.warburg}_s{a.seed}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
        rows.append(row)

    summary = dict(
        source="ADAPT-ECM 4.0.1 / run_bm_drt_programmatic",
        settings=dict(warmup=a.warmup, samples=a.samples, seed=a.seed,
                      warburg=a.warburg, chains=1),
        n=len(rows), rows=rows,
    )
    (outdir / f"summary_{a.warburg}_s{a.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"\n[done] {len(rows)} 条 -> {outdir}")


if __name__ == "__main__":
    main()
