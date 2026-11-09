# -*- coding: utf-8 -*-
"""KK 门禁能不能区分「真 Ws」与「K+1 个宽 HN」？

结论导向的实验，四组同一个频点集（caseE 真实 31 点，0.01–1e5 Hz）：

  A 真实数据                    examples/example_caseE.csv
  B 真值模型  R∞ + ZARC + Ws    （A_W=2，文献量级）
  C K=1 模型  R∞ + ZARC          （缺扩散，应被 data 侧否掉）
  D K+1 模型  R∞ + ZARC + ZARC   （宽弧替代扩散，§9.2 的 fitted 参数）

对每个 Z 数组跑 bm_kk.kk_check，看 status / 中段残差 / μ。
如果 B/C/D 都给 ok 且残差在 1e-12 以下，说明 KK 对**模型输出**恒真，
零鉴别力——KK 是数据体检，不是模型裁判。

附带量化：B 与 D 在窗口内的最大相对差异 vs 1% 噪声。
"""
import io
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import bm_kk
import bm_real
from bm_core import zarc


def zws(f, A):
    """半无限 Warburg：Z = A / (jω)^0.5。"""
    return A / np.sqrt(1j * 2 * np.pi * np.asarray(f, float))


def band(rms):
    return {k: (None if rms is None else round(float(rms[k]), 8))
            for k in ("low", "mid", "high", "all")}


def main():
    f, Z = bm_real.load_spectrum("examples/example_caseE.csv")
    o = np.argsort(f)
    f, Z = f[o], Z[o]

    R_INF, R1, F1, N1 = 10.0, 30.0, 300.0, 0.85
    A_W = 2.0
    R2, F2, N2 = 93.28, 1e-4, 0.518          # §9.2 的不可区分宽弧

    Zws = R_INF + zarc(f, R1, F1, N1) + zws(f, A_W)
    Zk1 = R_INF + zarc(f, R1, F1, N1)
    Zk2 = R_INF + zarc(f, R1, F1, N1) + zarc(f, R2, F2, N2)

    cases = [("A 真实数据", Z), ("B 真值 R∞+ZARC+Ws", Zws),
             ("C K=1  R∞+ZARC", Zk1), ("D K+1  R∞+ZARC+ZARC", Zk2)]

    rows = []
    for tag, z in cases:
        g = bm_kk.kk_check(f, z, auto_tail=True, auto_cap=True)
        rows.append(dict(tag=tag, status=g["status"], mu=g["mu"], M=g["M"],
                         rms_raw=band(g["rms_raw"]),
                         rms_clean=band(g["rms_clean"]),
                         f_cut=g["f_cut"], tail_slope=g["tail_slope"],
                         is_candidate=(g["tail"] or {}).get("is_candidate"),
                         reason=g["reason"][:110]))

    d_rel = np.abs(Zws - Zk2) / np.abs(Zws)
    diff = dict(max_rel=float(d_rel.max()),
                med_rel=float(np.median(d_rel)),
                at_max_hz=float(f[int(np.argmax(d_rel))]))

    print(f"频点 {f.size} 个，{f.min():.3g}–{f.max():.3g} Hz\n")
    hdr = f"{'case':22s} {'status':6s} {'μ':>7s} {'M':>3s} {'mid':>9s} {'low':>9s} {'cand':>5s}"
    print(hdr)
    for r in rows:
        mu = "n/a" if r["mu"] is None else f"{r['mu']:.3f}"
        print(f"{r['tag']:22s} {r['status']:6s} {mu:>7s} {str(r['M']):>3s} "
              f"{r['rms_clean']['mid']:9.2e} {r['rms_clean']['low']:9.2e} "
              f"{str(r['is_candidate']):>5s}")

    print(f"\nB vs D 窗口内相对差异：中位 {diff['med_rel']:.2e}、"
          f"最大 {diff['max_rel']:.2e}（@ {diff['at_max_hz']:.3g} Hz）")
    print("参照：E2 拟合的相对残差判据阈值 5%，实测谱噪声 ~1%")

    out = dict(n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
               rows=rows, b_vs_d=diff)
    with io.open("evidence/ws_crit/kk_k1_check.json", "w",
                 encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n写出 evidence/ws_crit/kk_k1_check.json")


if __name__ == "__main__":
    main()
