"""高频残差「等效串联电感」只读估算：一个固定元件能不能解释 M-b 的高频结构？

**只读，不采样、不拟合、不改判定。** 在最高频点用 `L_eq = −Im(Z_fit − Z_obs)/ω`
给出「若只想用一个串联 L 吸收残差，需要多少 µH」，并看符号是否一致：

- 符号一致（全正或全负）⇒ 存在一个**固定**元件/频响能统一解释（测量链方向）；
- 符号翻转 ⇒ 单一固定元件解释不了（本批即此情形）。

口径与 §9.4.41 / §15 的 `K:hf_L` 一致：`res = Z_fit − Z_obs`，取 `f_max`。
`L_eq<0` 表示残差在高端**比模型更不感性**（需要容性修正），`L_eq>0` 表示模型缺感抗。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/mb_hf_l_equiv.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import bm_real as R   # noqa: E402
import bm_top as T    # noqa: E402

SPEC_DIR = os.path.join("out", "_m6_samples", "z_csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("out", "m4_holdout2_ess.json"))
    ap.add_argument("--out", default=os.path.join("out", "mb_hf_l_equiv.json"))
    a = ap.parse_args(argv)

    src = json.load(open(a.json, encoding="utf-8"))
    rows = []
    for row in src["rows"]:
        f, Z = R.load_spectrum(os.path.join(SPEC_DIR, row["spec"] + ".csv"))
        keep = np.ones(f.size, bool)
        for hz in row.get("dropped_hz") or []:
            keep &= ~np.isclose(f, float(hz), rtol=1e-6)
        f, Z = f[keep], Z[keep]
        r = row["rows"][0]
        Zm = T.z_nested(f, r["params"], branch=row["branch"], layers=row["layers"])
        res = Zm - Z
        i = int(np.argmax(f))
        w = 2 * np.pi * f[i]
        rows.append(dict(spec=row["spec"], state=row["state"], f=float(f[i]),
                         im_obs=float(Z.imag[i]), im_fit=float(Zm.imag[i]),
                         res_im=float(res.imag[i]),
                         L_eq_uH=float(-res.imag[i] / w * 1e6),
                         inductive_obs=bool(Z.imag[i] > 0)))

    v = np.array([r["L_eq_uH"] for r in rows], float)
    pos = [r["spec"] for r in rows if r["L_eq_uH"] > 0]
    neg = [r["spec"] for r in rows if r["L_eq_uH"] < 0]

    print(f"{'谱':<14}{'Im_obs':>9}{'Im_fit':>9}{'res_Im':>8}{'L_eq(µH)':>10}  实测感抗?")
    for r in rows:
        print(f"{r['spec']:<14}{r['im_obs']:>9.2f}{r['im_fit']:>9.2f}"
              f"{r['res_im']:>8.2f}{r['L_eq_uH']:>10.1f}  {'是' if r['inductive_obs'] else '否'}")
    print(f"\nL_eq：中位 {np.median(v):.1f} µH，范围 {v.min():.1f}–{v.max():.1f} µH")
    print(f"需**感抗**（L_eq>0）{len(pos)} 条：{pos}")
    print(f"需**容性修正**（L_eq<0）{len(neg)} 条")
    print("符号翻转 ⇒ 单一固定元件/单一频响解释不了全部 14 条。")

    summary = dict(median_uH=float(np.median(v)), min_uH=float(v.min()),
                   max_uH=float(v.max()), n_pos=len(pos), n_neg=len(neg),
                   sign_flips=bool(len(pos) and len(neg)), rows=rows)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
