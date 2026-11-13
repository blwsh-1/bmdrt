# -*- coding: utf-8 -*-
"""复杂嵌套拓扑「已知真值」数据集生成器（P0 ③）。

预注册：`CIRCUIT_TOPOLOGY_ADMISSION.md` §6 / `docs/IDENTIFIABILITY_REPORT_PREREG.md`。

**反 inverse crime（硬要求）**
--------------------------------
本脚本**独立实现**全部电路的正演，不 import `bm_top` / `bm_core` / `bm_result`，
生成过程完全不依赖拟合侧代码。项目正演只在显式 `--crosscheck` 时被调用，
用来做「两套独立实现对拍」，且该结果**不参与数据生成**。

    python docs/round_2026-11-12/truth_dataset.py
    python docs/round_2026-11-12/truth_dataset.py --crosscheck   # 独立对拍

参数口径（与 `测试数据/` 一致）
--------------------------------
    Z_P(ω) = Q_data / (iω)^n          ω = 2πf
本项目 `bm_top` 用窗口幅值口径 `Q_ours = Q_data / ω_ref^n`，
`ω_ref = 2π√(f_min·f_max)`；`--crosscheck` 里做这个换算，不在生成侧使用。

频率：60 点、0.1–1e5 Hz、**升序**（与 `load_spectrum` 一致）。
噪声是人为生成过程给的，不是真实仪器噪声；相关结构 / 重尾都在 manifest 里标注。

产物：`out/truth_dataset/*.csv` + `truth_manifest.json` + `README.md`。
本脚本只读自己、只写 `out/truth_dataset/`，不改任何生产文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "out" / "truth_dataset"

F_LO, F_HI, N_F = 0.1, 1e5, 60
F = np.logspace(math.log10(F_LO), math.log10(F_HI), N_F)
W_REF = 2 * np.pi * math.sqrt(F_LO * F_HI)

FAMILIES = (
    "nested_L1_inductive", "nested_L2_inductive",
    "nested_L1_capacitive", "nested_L2_capacitive",
    "C2_L3", "C3_mixed", "ZARC",
)
DIFFICULTIES = ("separated", "overlap", "out_of_window", "degenerate")
NOISE_LEVELS = (0.0, 0.02, 0.05)
STRUCTURES = ("white", "correlated", "heavy")
SEEDS = (0, 1)


# ---------------------------------------------------------------- 独立正演

def par(a, b):
    """并联：导纳相加。"""
    return a * b / (a + b)


def z_cpe(f, Q, n):
    """独立实现：Z_P = Q_data / (iω)^n（Q_data 口径）。"""
    return Q / (1j * 2 * np.pi * np.asarray(f)) ** n


def cpe_q(R, f0, n):
    """用「转折频率 f0 时 |Z_CPE| = R」反解 Q_data，便于控制时间常数。"""
    return R * (2 * np.pi * f0) ** n


def z_rl_parallel(f, R, tau):
    """R ∥ jωL，L = R·tau（低频短路、高频趋于 R）。"""
    return par(R, 1j * 2 * np.pi * np.asarray(f) * (R * tau))


def z_rl_series(f, R, tau):
    """R + jωL（低频趋于 R、高频发散）。"""
    return R + 1j * 2 * np.pi * np.asarray(f) * (R * tau)


def z_branch(kind, f, p):
    if kind == "parallel":
        return z_rl_parallel(f, p["R_ad"], p["tau_ad"])
    if kind == "series":
        return z_rl_series(f, p["R_L"], p["tau_L"])
    return z_cpe(f, p["Q_b"], p["n_b"])


def nested_core(layers, f, p, branch_kind):
    """嵌套骨架（**不含 Rs**）：

        L=1: CPE1 ∥ (R1 + Z_b)
        L=2: CPE1 ∥ (R1 + CPE2 ∥ (R2 + Z_b))
        L=3: 再套一层 CPE3 ∥ (R3 + Z_b)
    """
    zb = z_branch(branch_kind, f, p)
    z = p[f"R{layers}"] + zb
    for i in range(layers, 1, -1):
        z = par(z_cpe(f, p[f"Q{i}"], p[f"n{i}"]), z)
        if i - 1 >= 1:
            z = p[f"R{i-1}"] + z
    z = par(z_cpe(f, p["Q1"], p["n1"]), z)
    return z


def zarc(f, p):
    return p["Rs"] + par(p["R1"], z_cpe(f, p["Q1"], p["n1"]))


def c3_mixed(f, p):
    """C3 = 串联 (Q0∥R0) + 两层嵌套。既非纯 voigt，也非纯 nested。"""
    outer = par(z_cpe(f, p["Q0"], p["n0"]), p["R0"])
    return p["Rs"] + outer + nested_core(2, f, p, "capacitive")


# ---------------------------------------------------------------- 参数集

def make_params(family: str, difficulty: str) -> dict:
    """按「难度轴」摆时间常数，参数全部可解释、可写进 manifest。"""
    if difficulty == "separated":
        f1, f2, f3, fb = 3e3, 3e1, 3e-1, 1e-2
        n = (0.88, 0.82, 0.78)
    elif difficulty == "overlap":
        f1, f2, f3, fb = 40.0, 18.0, 9.0, 5.0
        n = (0.85, 0.83, 0.81)
    elif difficulty == "out_of_window":
        f1, f2, f3, fb = 2e4, 5e1, 2e-3, 1e-4    # 两个过程落在 f_min 以下
        n = (0.9, 0.8, 0.75)
    else:                                          # degenerate
        f1, f2, f3, fb = 12.0, 11.0, 10.0, 9.0
        n = (0.8, 0.8, 0.8)

    Rs, R1, R2, R3 = 150.0, 3e4, 1e5, 5e4
    p = dict(Rs=Rs, R1=R1, R2=R2, R3=R3,
             Q1=cpe_q(R1, f1, n[0]), n1=n[0],
             Q2=cpe_q(R2, f2, n[1]), n2=n[1],
             Q3=cpe_q(R3, f3, n[2]), n3=n[2],
             Q_b=cpe_q(R2, fb, n[2]), n_b=n[2],
             R_ad=8e4, tau_ad=1.0 / (2 * np.pi * fb),
             R_L=6e4, tau_L=1.0 / (2 * np.pi * fb),
             Q0=cpe_q(2e4, f1 / 3, 0.85), n0=0.85, R0=2e4)
    if difficulty == "degenerate":
        p["R1"] = p["R2"] = p["R3"] = 6e4        # 幅值也简并
        p["n_b"] = 0.8
    return p


def topology_string(family: str) -> str:
    return {
        "nested_L1_inductive": "Rs + [P1 ∥ (R1 + (R_ad∥L))]",
        "nested_L2_inductive": "Rs + [P1 ∥ (R1 + [P2 ∥ (R2 + (R_ad∥L))])]",
        "nested_L1_capacitive": "Rs + [P1 ∥ (R1 + Pb)]",
        "nested_L2_capacitive": "Rs + [P1 ∥ (R1 + [P2 ∥ (R2 + Pb)])]",
        "C2_L3": "Rs + [P1 ∥ (R1 + [P2 ∥ (R2 + [P3 ∥ (R3 + (R_ad∥L))])])]",
        "C3_mixed": "Rs + (P0∥R0) + nested_L2_capacitive",
        "ZARC": "Rs + (R1 ∥ P1)",
    }[family]


def forward(family: str, f, p: dict):
    if family == "ZARC":
        return zarc(f, p)
    if family == "C3_mixed":
        return c3_mixed(f, p)
    if family == "C2_L3":
        return p["Rs"] + nested_core(3, f, p, "parallel")
    L = 1 if "_L1_" in family else 2
    branch = "capacitive" if "capacitive" in family else "parallel"
    return p["Rs"] + nested_core(L, f, p, branch)


def used_params(family: str, p: dict) -> dict:
    """只把该拓扑真正用到的参数写进 manifest（别把无关参数当真值）。"""
    keys = ["Rs"]
    if family == "ZARC":
        keys += ["R1", "Q1", "n1"]
    elif family == "C3_mixed":
        keys += ["Q0", "n0", "R0", "R1", "R2", "Q1", "n1", "Q2", "n2", "Q_b", "n_b"]
    else:
        L = 3 if family == "C2_L3" else (1 if "_L1_" in family else 2)
        for i in range(1, L + 1):
            keys += [f"R{i}", f"Q{i}", f"n{i}"]
        keys += ["Q_b", "n_b"] if "capacitive" in family else ["R_ad", "tau_ad"]
    return {k: p[k] for k in keys}


# ---------------------------------------------------------------- 噪声

def add_noise(z, pct, structure, rng):
    if pct <= 0:
        return z.copy(), dict(kind="none", pct=0.0)
    sigma = pct * np.abs(z)
    if structure == "heavy":
        # Student-t(3) 标准化到单位方差，重尾但仍有限
        t = rng.standard_t(3, size=z.size) / math.sqrt(3.0)
        re, im = t * sigma, rng.standard_t(3, size=z.size) / math.sqrt(3.0) * sigma
    else:
        re, im = rng.normal(0, 1, z.size) * sigma, rng.normal(0, 1, z.size) * sigma
    out = z + re + 1j * im
    meta = dict(kind=structure, pct=pct)
    if structure == "correlated":
        t = np.linspace(0.0, 1.0, z.size)
        shape = np.sin(2 * np.pi * 1.5 * t) + 0.5 * np.sin(2 * np.pi * 3.2 * t + 0.7)
        out = out + 0.4 * sigma * shape * (1 - 0.6j)
        meta["systematic_shape"] = "0.4σ·[sin(1.5·2πt)+0.5sin(3.2·2πt+0.7)]·(1−0.6j)"
        meta["note"] = "频率相关结构由生成过程给出，不是仪器噪声；会抬高 acf"
    return out, meta


# ---------------------------------------------------------------- 交叉验证

def crosscheck(p: dict) -> dict:
    """显式调用项目正演，验证两套独立实现一致（**不参与数据生成**）。"""
    import sys
    sys.path.insert(0, str(ROOT))        # 仅 crosscheck 需要，生成侧不 import 项目模块
    import bm_top as T

    wref = T.omega_ref(F)

    def q_ours(Q_data, n):
        return Q_data / wref ** n

    base = {k: p[k] for k in ("Rs", "R1", "R2", "Q1", "n1", "Q2", "n2")}
    for k in ("Q1", "Q2"):
        base[k] = q_ours(base[k], base[k.replace("Q", "n")])
    base.update(R_ad=p["R_ad"], tau_ad=p["tau_ad"],
                 R_L=p["R_L"], tau_L=p["tau_L"],
                 L_ad=p["R_L"] * p["tau_L"])
    cap = dict(base, Q_b=q_ours(p["Q_b"], p["n_b"]), n_b=p["n_b"])
    for k in ("Q1", "Q2"):
        cap[k] = q_ours(p[k], p[k.replace("Q", "n")])

    out = {}
    for L, branch, proj in (
        (1, "parallel", T.z_nested(F, base, branch="parallel", layers=1)),
        (2, "parallel", T.z_nested(F, base, branch="parallel", layers=2)),
        (1, "capacitive", T.z_nested(F, cap, branch="capacitive", layers=1)),
        (2, "capacitive", T.z_nested(F, cap, branch="capacitive", layers=2)),
    ):
        fam = f"nested_L{L}_{branch}"
        ours = forward(fam, F, p)
        e = float(np.max(np.abs(proj - ours)) / np.max(np.abs(ours)))
        out[fam] = e
    return out


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--crosscheck", action="store_true",
                    help="额外用项目正演做独立对拍（不参与生成）")
    a = ap.parse_args()
    dest = Path(a.out)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    records, worst = [], 0.0
    for fam in FAMILIES:
        for diff in DIFFICULTIES:
            p = make_params(fam, diff)
            z0 = forward(fam, F, p)
            assert np.all(np.isfinite(z0)), f"{fam}/{diff} 正演出现非有限值"
            for pct in NOISE_LEVELS:
                structure = STRUCTURES[NOISE_LEVELS.index(pct)]
                for seed in SEEDS:
                    key = f"{fam}|{diff}|{pct}|{seed}".encode("utf-8")
                    stable_seed = int(hashlib.md5(key).hexdigest()[:8], 16)
                    rng = np.random.default_rng(stable_seed)
                    z, nmeta = add_noise(z0, pct, structure, rng)
                    stem = f"{fam}__{diff}__n{int(pct*100):02d}__s{seed}"
                    csv = dest / f"{stem}.csv"
                    with csv.open("w", encoding="utf-8", newline="") as fh:
                        fh.write("freq,Zreal,Zimag\n")
                        for fi, zi in zip(F, z):
                            fh.write(f"{fi:.10g},{zi.real:.10g},{zi.imag:.10g}\n")
                    records.append(dict(
                        id=stem, family=fam, topology=topology_string(fam),
                        dimension=("network_level" if fam == "C2_L3" else
                                   "topology_connection" if fam == "C3_mixed" else
                                   "element_type" if "capacitive" in fam else "baseline"),
                        difficulty=diff, noise=nmeta, seed=seed,
                        f_min=F_LO, f_max=F_HI, n_points=N_F,
                        truth_params=used_params(fam, p),
                        csv=csv.name,
                        csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                    ))

    manifest = dict(
        status="finalized",
        generator="docs/round_2026-11-12/truth_dataset.py",
        anti_inverse_crime=(
            "生成侧独立实现全部电路正演，不 import bm_top/bm_core；项目正演仅在 "
            "--crosscheck 时对拍，结果不参与数据生成"),
        q_convention=("Z_P = Q_data/(iω)^n；Q_ours = Q_data/ω_ref^n，"
                      "ω_ref = 2π√(f_min·f_max)"),
        f_grid=dict(f_min=F_LO, f_max=F_HI, n=N_F, order="ascending"),
        noise_note=("噪声为人为生成过程：white=i.i.d. 高斯；correlated=叠加频率相关"
                    "平滑系统项；heavy=Student-t(3) 重尾。均非真实仪器噪声"),
        n_spectra=len(records),
        families=list(FAMILIES), difficulties=list(DIFFICULTIES),
        noise_levels=list(NOISE_LEVELS), structures=list(STRUCTURES),
        seeds=list(SEEDS),
        spectra=records,
    )
    mf = dest / "truth_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                             allow_nan=False) + "\n", encoding="utf-8")

    readme = dest / "README.md"
    readme.write_text(
        "# 复杂嵌套拓扑已知真值数据集\n\n"
        f"共 **{len(records)}** 条：`{len(FAMILIES)}` 类拓扑 × "
        f"`{len(DIFFICULTIES)}` 难度 × `{len(NOISE_LEVELS)}` 噪声档 × "
        f"`{len(SEEDS)}` seed。频率 {N_F} 点、{F_LO}–{F_HI:g} Hz 升序，"
        "三列 `freq,Zreal,Zimag`。\n\n"
        "## 反 inverse crime\n\n生成侧**独立实现**所有电路正演（`truth_dataset.py` 内 "
        "`z_cpe/par/nested_core`），不引用拟合侧任何模块。`--crosscheck` 才调用 "
        "`bm_top.z_nested` 对拍，且该结果不参与数据生成。\n\n"
        "## 难度轴\n\n- `separated` 时间常数分离 >2 dec\n- `overlap` 全部挤在 1 dec 内\n"
        "- `out_of_window` 两个过程落在 `f_min` 以下\n- `degenerate` 幅值与指数同时简并\n\n"
        "## 噪声\n\n`white` / `correlated`（带频率相关平滑结构，会抬高 acf）/ "
        "`heavy`（Student-t 重尾），幅度 0 / 2% / 5%·|Z|。"
        "**是人为生成过程，不是仪器噪声**，相关结构会自带 acf，"
        "不能据此推断「模型缺结构」。\n\n"
        "## 用法（统一流程，见 `CIRCUIT_TOPOLOGY_ADMISSION.md` §6）\n\n"
        "① 用真值拓扑生成 → ② 现有 voigt + nested 拟合 → ③ 新增候选 LS 反事实比较 → "
        "④ 查残差 / 参数恢复 / 可辨识性 → ⑤ 证据通过才进 Bayesian 与模型选择。\n\n"
        "## 体检提示\n\n低频虚部符号**不是**拓扑固定标签：容性最内层在窗口内未必发散"
        "（取决于支路转折频率是否落在窗外），感性支路也可能被外层 CPE 的容性贡献盖过"
        "（本数据集 `C2_L3` 即如此）。判拓扑要看参数恢复与残差结构，不能只看 Im 符号。\n\n"
        "真值参数、拓扑串、噪声、seed、sha256 全在 `truth_manifest.json`。\n",
        encoding="utf-8")

    if a.crosscheck:
        errs = crosscheck(make_params("nested_L2_capacitive", "separated"))
        worst = max(errs.values())
        print("独立正演交叉验证（不参与生成）：")
        for k, v in errs.items():
            print(f"  {k:<24} max rel err {v:.3e}")
        assert worst < 1e-10, f"两套独立实现不一致：{worst:.3e}"

    print(f"生成 {len(records)} 条真值谱 → {dest}")
    print(f"  manifest: {mf.name} / README.md")
    if a.crosscheck:
        print(f"  独立对拍最大相对误差 {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
