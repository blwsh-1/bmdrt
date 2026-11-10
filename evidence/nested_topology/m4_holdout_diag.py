"""N-b 失败的结构诊断：残差里那个 0.5–0.7 的自相关，是「整段结构」还是「最低频一个点」。

只读 `out/m4_holdout.json` 里已存的参数，**不重新采样**（秒级）。
诊断不改变判定——N-b 已经按 §13.3 记未过；这里只回答「下一步该修什么」。

用法
----
    python evidence/nested_topology/m4_holdout_diag.py
    python evidence/nested_topology/m4_holdout_diag.py --json out/m4_holdout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm_experiments as E   # noqa: E402
import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")


def acf(x: np.ndarray) -> float:
    return float(E.acf_max(x))


def probe(spec: str, params: dict, branch: str, layers: int) -> None:
    f, Z = R.load_spectrum(os.path.join(SPEC_DIR, spec + ".csv"))
    Zm = T.z_nested(f, params, branch=branch, layers=layers)
    rel = np.abs(Zm - Z) / np.abs(Z)
    signed = (Zm - Z) / np.abs(Z)

    print(f"\n=== {spec} ===")
    order = np.argsort(f)
    lo = order[:6]
    print("  最低 6 点 (f, 相对残差, 有符号 Δre/|Z|, Δim/|Z|):")
    for i in lo:
        print(f"    {f[i]:>9.4g} Hz  {rel[i]:>7.2%}   {signed.real[i]:>+7.2%}  {signed.imag[i]:>+7.2%}")
    print(f"  最高 3 点：" + "  ".join(f"{f[i]:g}Hz:{rel[i]:.1%}" for i in order[-3:]))
    print("  acf_max 随「去掉最低 k 个点」的变化（k=0,1,2,3）：", end="")
    for k in range(4):
        idx = order[k:]
        print(f"  {acf(rel[idx]):.3f}", end="")
    print(f"   （全体中位残差 {np.median(rel):.1%}，最大 {rel.max():.1%}）")

    # 结构在哪：按频段看残差符号是否同号（同号 = 系统性偏差，不是随机噪声）
    for lo_f, hi_f, nm in ((0.01, 0.1, "低 0.01–0.1"), (0.1, 1, "低 0.1–1"),
                           (1, 100, "中 1–100"), (100, 1e5, "高 100–1e5")):
        m = (f >= lo_f) & (f < hi_f)
        if not m.any():
            continue
        same_re = float(np.mean(np.sign(signed.real[m]) == np.sign(np.median(signed.real[m]))))
        same_im = float(np.mean(np.sign(signed.imag[m]) == np.sign(np.median(signed.imag[m]))))
        print(f"    {nm:<12} n={m.sum():>2}  中位 {np.median(rel[m]):>6.2%}  "
              f"同号率 re {same_re:.0%} / im {same_im:.0%}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("out", "m4_holdout.json"))
    a = ap.parse_args(argv)
    d = json.load(open(a.json, encoding="utf-8"))
    print("说明：acf_max 是 §13.3 N-b 用的同一个指标（滞后 1–5 最大 |ρ|）。")
    print("      去掉最低 k 点的列只是**诊断**，不构成新的验收口径。")
    for r in d["rows"]:
        if "error" in r:
            print(f"\n=== {r['spec']} === 失败：{r['error']}")
            continue
        row = r["rows"][0]
        probe(r["spec"], row["params"], r["branch"], r["layers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
