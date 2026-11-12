# -*- coding: utf-8 -*-
"""DRT 峰持续性判决实验：`8Li_50` vs `8Li_150`（§9.4.39 挖出的天然对照）。

**问题**：这两条谱数据质量几乎相同（KK 2.07%/2.20%、phase 0.96°/0.99°），
但 M-b acf 差 0.434（0.341 vs 0.775）。若 DRT 峰结构**不同**，说明模型缺的
正是那个峰对应的过程（真突破）；若**相同**，说明差异纯在 ECM 族覆盖不到，
DRT 无新信息（封路）。

实现纪律
--------
- **不动生产**：只 import `drt`，不改它一行。bootstrap 在这里做。
- **λ 双模式**：`fixed`（固定原谱的 λ，纯数据不确定性）/ `reselect`
  （每次按同一 discrepancy/L 曲线规则重选，含"λ 选择"的不确定性）。
  两者差异大 ⇒ 结论对 λ 敏感，不可引用（`drt.py` docstring 的警告）。
- **噪声双模型**：参数化白噪声（零假设"只有噪声"）/ 残差重采样（保留实测
  残差的经验分布，含系统偏差）。两者差异大 ⇒ 峰依赖噪声假设。
- **合成对照**：已知 2 峰真值 + 2% 噪声，验证方法能否数对；单峰 + 噪声，
  看会不会造出假峰。**先证判别力，再解释真实谱。**

用法（项目 venv，仓库根）：
    python evidence/nested_topology/m4_drt_persistence.py --quick
    python evidence/nested_topology/m4_drt_persistence.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import drt as DRT            # noqa: E402
import bm_real as R          # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")
OUT_JSON = os.path.join("out", "drt_persistence.json")
TOL_DEC = 0.30          # 峰位匹配容差（2 个网格点；m=60、跨 ~9 dec → 0.15 dec/格）
REL_PROM = 0.03         # 与 drt.peaks 默认一致
SIGNED = True           # 本批 18/18 低频是感抗（§2），非负会把负叶压成 0
PAIR = ("AZ31-8Li_50", "AZ31-8Li_150")


def _peaks_pos(res: dict) -> list[float]:
    """非负峰（容性）的 log10(τ) 列表。"""
    return [float(np.log10(p["tau_s"])) for p in DRT.peaks(res, rel=REL_PROM)
            if p["sign"] > 0]


def _persist(base: dict, boots: list[list[float]]) -> list[dict]:
    """base 每个容性峰，在多少比例的 bootstrap 样本里 ±TOL 内有峰。"""
    out = []
    pk_all = [p for p in DRT.peaks(base, rel=REL_PROM) if p["sign"] > 0]
    r_tot = sum(abs(p["R_ohm"]) for p in pk_all) or 1.0
    for p in DRT.peaks(base, rel=REL_PROM):
        if p["sign"] <= 0:
            continue
        lp = float(np.log10(p["tau_s"]))
        hit = sum(1 for ps in boots if any(abs(x - lp) <= TOL_DEC for x in ps))
        out.append(dict(
            f_peak_Hz=p["f_peak_Hz"], tau_s=p["tau_s"], R_ohm=p["R_ohm"],
            width_dec=p["width_dec"], persist=hit / max(len(boots), 1),
            r_frac=abs(p["R_ohm"]) / r_tot))
    return out


def one(f: np.ndarray, Z: np.ndarray, *, sigma_pct: float, n_boot: int,
        noise: str, lam_mode: str, seed: int = 0) -> dict:
    """单条谱：原始 DRT + bootstrap 持续性。"""
    base = DRT.drt(f, Z, sigma=sigma_pct, signed=SIGNED)
    lam0 = float(base["lam"])
    Zfit = base["Z_fit"]
    rng = np.random.default_rng(seed)
    sig_abs = sigma_pct * np.abs(Z)
    resid = Z - Zfit

    boots: list[list[float]] = []
    counts: Counter = Counter()
    for _ in range(n_boot):
        if noise == "white":
            e = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * sig_abs
        elif noise == "resid":
            k = rng.integers(0, f.size, size=f.size)   # 按点有放回
            e = resid[k]
        else:
            raise ValueError(noise)
        kw = dict(sigma=sigma_pct, signed=SIGNED)
        if lam_mode == "fixed":
            kw["lam"] = lam0
        r = DRT.drt(f, Zfit + e, **kw)
        pos = _peaks_pos(r)
        boots.append(pos)
        counts[len(pos)] += 1

    return dict(
        n_pts=int(f.size),
        lam=lam0, chi2=float(base["chi2"]), dof=int(base["dof"]),
        r_inf=float(base["R_inf"]), r_inf_true=float(base["R_inf_true"]),
        neg_pol=float(base["neg_pol"]),
        edge_low_frac=float(base["edge_low_frac"]),
        edge_high_frac=float(base["edge_high_frac"]),
        n_peaks_base=len(_peaks_pos(base)),
        peaks=_persist(base, boots),
        peak_count=dict(counts),
        peak_count_median=float(np.median([len(b) for b in boots])),
        noise=noise, lam_mode=lam_mode, n_boot=int(n_boot),
        signed=bool(SIGNED),
    )


def synth(kind: str, f: np.ndarray, sigma_pct: float = 0.02,
          seed: int = 7) -> tuple[np.ndarray, dict]:
    """合成对照：kind='two' 两峰真值；'one' 单峰真值。"""
    rng = np.random.default_rng(seed)
    w = 2 * np.pi * f
    Rs = 5.0
    if kind == "two":
        p = dict(r1=45.0, tau1=1e-3, r2=300.0, tau2=1e-1)
        Z = Rs + p["r1"] / (1 + 1j * w * p["tau1"]) \
            + p["r2"] / (1 + 1j * w * p["tau2"])
    elif kind == "one":
        p = dict(r1=300.0, tau1=1e-2)
        Z = Rs + p["r1"] / (1 + 1j * w * p["tau1"])
    else:
        raise ValueError(kind)
    e = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) \
        * sigma_pct * np.abs(Z)
    return Z + e, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="n_boot=30，先看逻辑")
    ap.add_argument("--json", default=OUT_JSON)
    ap.add_argument("--sigma-pct", type=float, default=0.02)
    a = ap.parse_args(argv)
    n_boot = 30 if a.quick else 200

    out: dict = dict(status="exploratory", signed=SIGNED,
                     tol_dec=TOL_DEC, rel_prom=REL_PROM,
                     sigma_pct=a.sigma_pct, n_boot=n_boot, cases={})

    # ---- ① 合成对照（先证判别力）----
    f_ref, _Zref = R.load_spectrum(os.path.join(SPEC_DIR, "AZ31-8Li_50.csv"))
    for kind, truth in (("two", "2 峰"), ("one", "1 峰")):
        Zs, p = synth(kind, f_ref, a.sigma_pct)
        print(f"[synth {kind}] 真值 {truth} {p}")
        d = one(f_ref, Zs, sigma_pct=a.sigma_pct, n_boot=n_boot,
                noise="white", lam_mode="reselect", seed=11)
        d["truth"] = p
        out["cases"][f"synth_{kind}"] = d
        print(f"   → base 峰数 {d['n_peaks_base']}，bootstrap 峰数中位 "
              f"{d['peak_count_median']:.0f}，分布 {d['peak_count']}")
        for pk in d["peaks"]:
            print(f"      f={pk['f_peak_Hz']:.4g} Hz  R={pk['R_ohm']:.1f} Ω  "
                  f"持续 {100*pk['persist']:.0f}%")

    # ---- ② 真实谱：先取 base λ，再做交叉 λ 检验 ----
    sp_data, sp_lam = {}, {}
    for sp in PAIR:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, sp + ".csv"))
        sp_data[sp] = (f, Z)
        sp_lam[sp] = float(DRT.drt(f, Z, sigma=a.sigma_pct,
                                   signed=SIGNED)["lam"])

    print(f"\n[交叉 λ] base λ: " +
          "  ".join(f"{s}={sp_lam[s]:.4g}" for s in PAIR))
    cross = {}
    for sp in PAIR:
        f, Z = sp_data[sp]
        for src in PAIR:
            r = DRT.drt(f, Z, sigma=a.sigma_pct, signed=SIGNED,
                        lam=sp_lam[src])
            pk = [p for p in DRT.peaks(r, rel=REL_PROM) if p["sign"] > 0]
            cross[f"{sp}|lam_of_{src}"] = dict(
                lam=float(sp_lam[src]), chi2=float(r["chi2"]),
                dof=int(r["dof"]), n_peaks=len(pk),
                peaks=[dict(f=p["f_peak_Hz"], R=p["R_ohm"]) for p in pk])
            print(f"   {sp} 用 λ={sp_lam[src]:.4g}（来自 {src}）→ "
                  f"{len(pk)} 峰，χ²={r['chi2']:.1f}/{r['dof']}")
    out["cross_lambda"] = cross

    # ---- ②b 符号自洽性：非负能否拟合？signed 的 R_inf_true 是否为负？----
    print("\n[符号自洽性]")
    sign_chk = {}
    for sp in PAIR:
        f, Z = sp_data[sp]
        row = {}
        for sg in (False, True):
            r = DRT.drt(f, Z, sigma=a.sigma_pct, signed=sg)
            pk = [p for p in DRT.peaks(r, rel=REL_PROM) if p["sign"] > 0]
            row[f"signed_{sg}"] = dict(
                lam=float(r["lam"]), chi2=float(r["chi2"]), dof=int(r["dof"]),
                chi2_over_dof=float(r["chi2"]) / max(int(r["dof"]), 1),
                n_peaks=len(pk), r_inf=float(r["R_inf"]),
                r_inf_true=float(r["R_inf_true"]), neg_pol=float(r["neg_pol"]))
        sign_chk[sp] = row
        nn, sy = row["signed_False"], row["signed_True"]
        print(f"   {sp}: 非负 χ²/dof={nn['chi2_over_dof']:.1f}（{nn['n_peaks']} 峰）"
              f" | signed χ²/dof={sy['chi2_over_dof']:.2f}（{sy['n_peaks']} 峰，"
              f"R_inf_true={sy['r_inf_true']:.1f} Ω）")
    out["sign_consistency"] = sign_chk

    # ---- ③ bootstrap 持续性 ----
    for sp in PAIR:
        f, Z = sp_data[sp]
        for noise in ("white", "resid"):
            for lam_mode in ("fixed", "reselect"):
                key = f"{sp}|{noise}|{lam_mode}"
                print(f"[real] {key}")
                d = one(f, Z, sigma_pct=a.sigma_pct, n_boot=n_boot,
                        noise=noise, lam_mode=lam_mode, seed=3)
                out["cases"][key] = d
                print(f"   base 峰数 {d['n_peaks_base']}"
                      f"（λ={d['lam']:.3g}, χ²={d['chi2']:.1f}/{d['dof']}）"
                      f"  boot 中位 {d['peak_count_median']:.0f} {d['peak_count']}")
                for pk in d["peaks"]:
                    print(f"      f={pk['f_peak_Hz']:.4g} Hz  R={pk['R_ohm']:.1f} Ω  "
                          f"持续 {100*pk['persist']:.0f}%")

    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
