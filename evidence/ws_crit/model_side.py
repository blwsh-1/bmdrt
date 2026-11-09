# -*- coding: utf-8 -*-
"""模型侧的 Ws 判据：读**已有**后验样本，算两个份额量，不重新采样。

  rpc_share    : Wo 用。`R_W /(R_W + Σw)`，Boškoski 2024 的「相对极化贡献」。
  ws_share_low : Ws 用。`Re Z_Ws(f_min) /(Re Z_Ws(f_min) + Σw)`。
                 必须连 `f_min` 一起报——同一个 `A_W` 在不同窗口给出不同数。

全部输入都是冻结产物（`evidence/b2x/`、`out/_caseE_diff/`），本脚本只读。
"""
import json
import sys
import numpy as np

sys.path.insert(0, r"D:/eis训练数据/bmdrt")
sys.path.insert(0, r"D:/eis训练数据/bmdrt/evidence/ws_crit")

import bm_wscrit as W

GROUPS = [
    ("b2x k1 none (Wo 数据)",   "evidence/b2x/k1/example_caseF_wo/real_example_caseF_wo_samples.npz"),
    ("b2x k3 none (Wo 数据)",   "evidence/b2x/k3/example_caseF_wo/real_example_caseF_wo_samples.npz"),
    ("b2x k1 +Ws  (错元件)",     "evidence/b2x/k1ws/example_caseF_wo/real_example_caseF_wo_samples.npz"),
    ("b2x k1 +Wo  (对元件)",     "evidence/b2x/k1wo/example_caseF_wo/real_example_caseF_wo_samples.npz"),
    ("caseE k1 +Ws (真含 Ws)",   "out/_caseE_diff/k1_Ws/example_caseE/real_example_caseE_samples.npz"),
    ("caseE k1 +Wo",            "out/_caseE_diff/k1_Wo/example_caseE/real_example_caseE_samples.npz"),
    ("caseE k2 +Ws",            "out/_caseE_diff/k2_Ws/example_caseE/real_example_caseE_samples.npz"),
    ("caseE_warburg k1 +Ws",    "out/example_caseE_warburg/real_example_caseE_warburg_samples.npz"),
]


def main():
    rows = []
    print(f"{'group':24s} {'key':4s} {'rpc_med':>8s} {'rpc_90%CI':>16s} "
          f"{'P(rpc<5%)':>9s} | {'low_share':>9s} {'f_min':>7s} {'A_W':>7s}")
    for name, path in GROUPS:
        try:
            d = dict(np.load(r"D:/eis训练数据/bmdrt/" + path, allow_pickle=True))
        except Exception as e:                      # noqa: BLE001
            print(f"{name:24s} 读取失败：{e}")
            continue
        r = W.rpc_share(d)
        s = W.ws_share_low(d)
        ci = f"[{r['share_lo']:.3f},{r['share_hi']:.3f}]" if np.isfinite(r["share_med"]) else "—"
        print(f"{name:24s} {str(r['key']):4s} {r['share_med']:8.3f} {ci:>16s} "
              f"{r['p_below_5pct']:9.2f} | {s['share_med']:9.3f} {s['f_min']:7.3g} {s['A_W']:7.3f}")
        rows.append(dict(group=name, path=path, rpc=r, ws_low=s))

    out = r"D:/eis训练数据/bmdrt/evidence/ws_crit/model_side.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\n写入 {out}")
    print("\n读法：`rpc_med` 只对 Wo 有意义（Wo 的 R_W 是有限极化电阻）。"
          "\n      Ws 的 `Rw` 是窗口归一化幅值，`rpc` 会被算成 0.2%~0.9%，"
          "不能读成「扩散项不重要」——看 `low_share`，并且必须连 `f_min` 一起读。")


if __name__ == "__main__":
    main()
