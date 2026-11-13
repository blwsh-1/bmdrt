# -*- coding: utf-8 -*-
"""只读验证：新增的 `branch="capacitive"` 是否**精确等于** `测试数据/` 的真值模型。

为什么需要这一步
----------------
`NOISE_ACF_TRUTH.md` §1 已用 `bm_top.z_cpe` 骨架把三类合成谱的拓扑归属
对到 1e-15，但同时指出：涂层/钝化的最内层是 `R + CPE`（**容性**），
而 `z_nested` 原有两条 branch 都是感抗 → 骨架对、元件类型不对，仍族外。
本脚本给新增的容性支路做独立对拍：真值谱**不用项目代码**重建
（直接按文件名里的 `Z_P = Q/(i·2πf)^n` 写阻抗），再与 `z_nested` 比。

边界（写死，别越线）
--------------------
· 只读：不跑 NUTS、不拟合、不改任何生产代码 / 判据 / 阈值；
· 不改 §13 `0/3`、§14 `4/14`、§17 `0/10`、§18/§19 任何结论；
· 新支路默认关，`bm_select.py` 的跨族候选池不含它；
· CSV 里含生成噪声，所以「CSV vs 真值」的 acf 只能读成**噪声结构**，
  不是模型失配；真值对拍（不用 CSV）才是元件是否装对的判据。

复跑（在仓库根 D:/eis训练数据/bmdrt）：
    python docs/round_2026-11-12/capacitive_branch_check.py --n-per-class 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bm_top as T                                      # noqa: E402
import bm_experiments as E                              # noqa: E402

DATA = Path("D:/eis训练数据/测试数据")
SUB = {
    "涂层": "涂层/Coating_Generated EIS data and Nyquist plots/"
            "Generated EIS data and Nyquist plots",
    "钝化": "钝化/Passivating_Generated EIS data and Nyquist plots/"
            "Generated EIS data and Nyquist plots",
    "腐蚀": "腐蚀/Uniform_Generated data and Nyquist plots/"
            "Generated data and Nyquist plots",
}
LAYERS = {"涂层": 2, "钝化": 1, "腐蚀": 1}


def z_p(f, Q, n):
    """真值侧的 CPE：`Z_P = Q/(i·2πf)^n`（**不调用项目代码**）。"""
    return Q / (1j * 2.0 * np.pi * np.asarray(f, float)) ** n


def truth_Z(kind, f, v):
    """按文件名拓扑串独立写阻抗公式。"""
    if kind == "涂层":                    # R1-[P2,R3-[P4,R5-P6]]
        R1, Q2, n2, R3, Q4, n4, R5, Q6, n6 = v
        p2, p4, p6 = z_p(f, Q2, n2), z_p(f, Q4, n4), z_p(f, Q6, n6)
        inner = 1.0 / (1.0 / p4 + 1.0 / (R5 + p6))
        return R1 + 1.0 / (1.0 / p2 + 1.0 / (R3 + inner))
    if kind == "钝化":                    # [P1,P2-R3]-R4
        Q1, n1, Q2, n2, R3, R4 = v
        p1, p2 = z_p(f, Q1, n1), z_p(f, Q2, n2)
        return R4 + 1.0 / (1.0 / p1 + 1.0 / (p2 + R3))
    if kind == "腐蚀":                    # R1-[R2,P3] = ZARC（voigt K=1 已覆盖）
        R1, R2, Q3, n3 = v
        p3 = z_p(f, Q3, n3)
        return R1 + 1.0 / (1.0 / R2 + 1.0 / p3)
    raise ValueError(kind)


def our_params(kind, f, v):
    """`Q_ours = Q_data / ω_ref^n`（`NOISE_ACF_TRUTH.md` §1 已对齐的换算）。"""
    w_ref = T.omega_ref(f)
    q = lambda Q, n: Q / w_ref ** n                  # noqa: E731
    if kind == "涂层":
        R1, Q2, n2, R3, Q4, n4, R5, Q6, n6 = v
        return dict(Rs=R1, R1=R3, R2=R5,
                    Q1=q(Q2, n2), n1=n2, Q2=q(Q4, n4), n2=n4,
                    Q_b=q(Q6, n6), n_b=n6)
    if kind == "钝化":
        Q1, n1, Q2, n2, R3, R4 = v
        return dict(Rs=R4, R1=R3,
                    Q1=q(Q1, n1), n1=n1,
                    Q_b=q(Q2, n2), n_b=n2)
    raise ValueError(f"{kind} 不需要容性支路（voigt K=1 已覆盖）")


def parse_params(path):
    """文件名尾部的 `[参数表]` → float 列表。"""
    tail = path.stem.rsplit("-[", 1)[1].rstrip("]")
    return [float(x.strip()) for x in tail.split(",")]


def read_csv(path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    f, zr, zi = a["freq"], a["Zreal"], a["Zimag"]
    i = np.argsort(f)                                # 源数据降序，统一转升序
    return f[i], (zr + 1j * zi)[i]


def run(n_per_class):
    out = {"note": "只读对拍；CSV 含生成噪声，真值对拍才判元件是否装对",
           "classes": {}, "ok": True}
    for kind, rel in SUB.items():
        d = DATA / rel
        files = sorted(d.glob("*.csv"))[:n_per_class]
        if not files:
            print(f"[{kind}] 找不到 CSV：{d}")
            out["ok"] = False
            continue
        rows = []
        for path in files:
            f, Z = read_csv(path)
            v = parse_params(path)
            Zt = truth_Z(kind, f, v)
            if kind == "腐蚀":
                # 真值归属是 voigt K=1，本脚本只核对「不需要新支路」
                rows.append(dict(file=path.name, covered_by="voigt K=1",
                                 new_branch_needed=False))
                continue
            p = our_params(kind, f, v)
            Zo = T.z_nested(f, p, branch="capacitive", layers=LAYERS[kind])
            rel_err = float(np.max(np.abs(Zo - Zt)) / np.max(np.abs(Zt)))
            # 与 CSV 的差 = 生成噪声；只作口径记录，不作模型判据
            res = Zo - Z
            rwp = float(np.sqrt(np.mean(np.abs(res) ** 2 / np.abs(Z) ** 2)) * 100)
            acf = float(max(E.acf_max(res.real), E.acf_max(res.imag)))
            ok = rel_err < 1e-12
            out["ok"] &= ok
            rows.append(dict(file=path.name, layers=LAYERS[kind],
                             max_rel_err_vs_truth=rel_err,
                             rwp_vs_csv_pct=rwp, acf_vs_csv=acf,
                             branch_ok=ok))
        out["classes"][kind] = rows
        errs = [r["max_rel_err_vs_truth"] for r in rows if "max_rel_err_vs_truth" in r]
        if errs:
            print(f"[{kind}] L={LAYERS[kind]}  n={len(errs)}  "
                  f"真值对拍 max rel err = {max(errs):.3e}  "
                  f"（全部 <1e-12：{all(e < 1e-12 for e in errs)}）")
            acfs = [r["acf_vs_csv"] for r in rows if "acf_vs_csv" in r]
            print(f"          CSV 噪声侧：acf 中位 {np.median(acfs):.3f}"
                  f"（min {min(acfs):.3f} / max {max(acfs):.3f}）—— 与模型无关")
        else:
            print(f"[{kind}] n={len(rows)}  → 全部由 voigt K=1 覆盖，不需要新支路")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=5)
    ap.add_argument("--out", default="out/capacitive_branch_check.json")
    a = ap.parse_args(argv)
    res = run(a.n_per_class)
    dst = ROOT / a.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结论：{'全部通过' if res['ok'] else '有失败项'}；产物 {dst}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
