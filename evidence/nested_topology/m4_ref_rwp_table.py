# -*- coding: utf-8 -*-
"""m4_ref_rwp_table.py —— 3E OER 5 mA / 50 mA 两个工作点的对照表（唯读）。

回答的问题不是"谁拟合得更好"，而是**同一个拓扑在不同工作点上是否站得住**，
以及"我们的 MCMC Rwp 与外部的 LS Rwp 差多少、差在哪"。

每个工作点有四类数：
  ① 外部工具第一名电路 + 它的 Rwp（用户截图，未复现其内部打分）
  ② BM-DRT 选出的拓扑（生产管线，standard 档）
  ③ `rwp_primary`：生产后验中位参数换成 Rwp 这把尺子（bm_ref_rwp 定义）
  ④ `rwp_ls_relative`：同拓扑、独立 LS + 逐点 2%σ 重新优化后的最优 Rwp

③→④ 的落差 = 估计目标差异；④ vs ① 的落差 = 拓扑/自由度差异。
两个落差混在一起就说不清问题，这张表就是为了把它们分开。

用法：python evidence/nested_topology/m4_ref_rwp_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bm_ref_rwp as REF                       # noqa: E402

# 外部工具（用户 2026-11-11 截图记录）。**未复现其内部打分**，只照抄报出的数。
EXTERNAL = {
    "5mA": dict(
        circuit="Rs + (Rct1|CPE1) + [CPEd|(Rct|[Rads+jωLads])] + PhaHF + CPEtail",
        n_candidates=28, rwp=0.65416, rank=1,
        note="28 选 1、无留出、CI 全空、不报 acf；3 个端部专用元件"),
    "50mA": dict(
        circuit="Rs + Lead + 2×(R|CPE)",
        n_candidates=None, rwp=0.3777, rank=1,
        note="同屏第 1 名；参数 Rs=0.432433, Llead=1.69771e-07, T2=4.186 Hz"),
}

CASES = {
    "5mA": dict(stem="3E-1_OER_exp1_0005mA",
                npz="out/3E-1_OER_exp1_0005mA/real_3E-1_OER_exp1_0005mA_samples.npz",
                js="out/3E-1_OER_exp1_0005mA/real_3E-1_OER_exp1_0005mA.json",
                c4_rwp=1.4726, similar=False,
                param_note="—“（拓扑不同，参数不可逐项对照）"),
    "50mA": dict(stem="3E-1_OER_exp1_0050mA",
                 npz="out/3E-1_OER_exp1_0050mA/real_3E-1_OER_exp1_0050mA_samples.npz",
                 js="out/3E-1_OER_exp1_0050mA/real_3E-1_OER_exp1_0050mA.json",
                 c4_rwp=0.6021, similar=True,
                 param_note="Rs 0.432433 Ω / Llead 1.6977e-7 H 与外部六位一致；"
                            "4.186 Hz 分量一致（分量顺序相反）"),
}


def load_result(c: dict):
    z = np.load(ROOT / c["npz"])
    f = np.asarray(z["f"], float)
    Z = np.asarray(z["Zre"], float) + 1j * np.asarray(z["Zim"], float)
    med = lambda k: np.median(np.asarray(z[k], float), axis=0)   # noqa: E731
    js = json.loads((ROOT / c["js"]).read_text(encoding="utf-8"))
    K = int(np.asarray(z["tau"]).shape[1])
    res = dict(
        K=K,
        w=np.asarray(med("W"), float) * np.asarray(med("pi"), float),
        f0=1.0 / (2 * np.pi * np.asarray(med("tau"), float)),
        phi=np.asarray(med("phi"), float),
        R_inf=float(med("R_inf")),
        L=float(med("L")),
        diffusion=js.get("diffusion"),
        adsorption=js.get("adsorption"),
    )
    return f, Z, res, js


def main() -> int:
    assert REF.selftest()["ok"], "参考模块自检没过，先修 bm_ref_rwp"

    rows = {}
    for tag, c in CASES.items():
        f, Z, res, js = load_result(c)
        zeta = js.get("zeta")
        blk = REF.reference_block(f, Z, res, bool(js.get("include_L")), zeta)
        # 与先前手算的 MCMC Rwp 对拍：定义没变的话应当逐位重合
        assert abs(blk["rwp_primary"] - c["c4_rwp"]) < 5e-3, (
            f"{tag}: rwp_primary={blk['rwp_primary']:.4f}% 与既有 "
            f"{c['c4_rwp']}% 不符，Rwp 定义被改过？")
        rows[tag] = dict(
            tag=tag, K=int(js["K"]), include_L=bool(js["include_L"]),
            topology="Rs + jωL + Σ HN(ζ=1)",
            n_points=int(f.size),
            acf=float(js["acf"]),
            t_fit_s=float(js["t_fit_s"]),
            rwp_primary=blk["rwp_primary"],
            rwp_ls_relative=blk["rwp_ls_relative"],
            ls_status=blk["ls_relative_refit_status"],
            ls_elapsed_s=blk.get("ls_relative_elapsed_s"),
            ls_starts=blk.get("ls_relative_refit_starts"),
            ls_at_bound=blk.get("ls_relative_at_bound"),
            external=EXTERNAL[tag],
            isomorphic=bool(c["similar"]),
            param_note=c["param_note"],
            weight_definition=blk["ls_relative_weight_definition"],
        )
        blk["rwp_primary"] = rows[tag]["rwp_primary"]

    # ---- 打印 ----
    print("\n| 项 | 5 mA | 50 mA |")
    print("|---|---|---|")
    a, b = rows["5mA"], rows["50mA"]
    for key, label in (
            ("topology", "BM-DRT 拓扑"), ("K", "K / L"),
            ("acf", "ACF（生产 MCMC 残差）"),
            ("t_fit_s", "选中 K 的拟合耗时 (s)")):
        va = f"{a[key]}" if key != "K" else f"K={a['K']} / L={a['include_L']}"
        vb = f"{b[key]}" if key != "K" else f"K={b['K']} / L={b['include_L']}"
        if key == "t_fit_s":
            va, vb = f"{a[key]:.0f}", f"{b[key]:.0f}"
        elif key == "acf":
            va, vb = f"{a[key]:.3f}", f"{b[key]:.3f}"
        print(f"| {label} | {va} | {vb} |")
    print(f"| 外部第一名电路 | {a['external']['circuit']} | {b['external']['circuit']} |")
    print(f"| 是否同构 | {'是' if a['isomorphic'] else '**否**（多 3 个端部自由度）'} "
          f"| {'是（ζ=1 HN ≡ R‖CPE）' if b['isomorphic'] else '否'} |")
    print(f"| 外部 Rwp | {a['external']['rwp']:.4f}% | {b['external']['rwp']:.4f}% |")
    print(f"| MCMC Rwp（rwp_primary） | {a['rwp_primary']:.4f}% | {b['rwp_primary']:.4f}% |")
    print(f"| LS 相对 Rwp（rwp_ls_relative） | {a['rwp_ls_relative']:.4f}% "
          f"| {b['rwp_ls_relative']:.4f}% |")
    print(f"| LS 重拟合耗时 | {a['ls_elapsed_s']:.2f} s | {b['ls_elapsed_s']:.2f} s |")
    print("\n注：耗时是**选中 K 那一次拟合**的墙钟，不含被淘汰的 K 探针。"
          "50 mA 全流程 16 min 里 914 s 花在 K=3 探针（R̂=2.303/div=90 后被剔除），"
          "那是「诚实证伪」的成本，不是选中模型的成本。")
    print("注：ACF 是生产 MCMC 残差的自相关（`acf` 字段）；LS 解自己的 acf 更高"
          "（50 mA 0.894）——Rwp 变小不会消掉残差结构，两件事。")

    out = ROOT / "out" / "m4_ref_rwp_table.json"
    out.write_text(json.dumps(dict(rows=rows), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding="utf-8")
    print(f"\n结果：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
