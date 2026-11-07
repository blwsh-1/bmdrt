# -*- coding: utf-8 -*-
"""嵌套并联拓扑的**独立生成器**（M3 档 1 的生成侧）。

为什么必须另写一个文件
----------------------
`BM_TOP_M3_PLAN.md` §5.2 要求"生成器 → `bm_top.z_nested`"的相对误差 <1e-10。
如果生成器直接调 `z_nested`，这个检查退化成 `f(x) == f(x)`，恒等于 0，
什么也没证明（§11.3 的冒烟就是这么失效的）。

所以本模块**不 import `bm_top`**，用另一条代码路径重算同一个物理量：

| | `bm_top.z_nested` | `bm_top_gen`（本文件） |
|---|---|---|
| CPE | 闭式阻抗 `Q(ω_ref/ω)^n·e^{-j nπ/2}` | 先由 `Q` 反推导纳尺度 `Y=1/(Q·ω_ref^n)`，再算 `Y(jω)^n` |
| 组合 | 逐级串/并联，直接写阻抗式 | **逐节点导纳递推**，每级取倒数 |
| R∥L | 闭式 `R·jωτ/(1+jωτ)` | 导纳相加 `1/R + 1/(jωL)` 后取倒数 |
| R+L | `R + jωL`，`L` 直接给 | `L = R_L·τ_L`（按 v3：`τ_L` 是采样参数，`L` 才是派生量） |
| 单层 | `layers=1` 时不建 `R2` | 独立分支，**公式里根本没有 `R2` 这一项** |

两条路径同时写错的概率低；剩下的窗口靠档 2（真值反演）兜底。

四种拓扑（`BM_TOP_M3_PLAN.md` §5.1 的覆盖矩阵）
-----------------------------------------------
    gen_parallel_L2   Rs + CPE1 ∥ (R1 + CPE2 ∥ (R2 + R_ad∥L))
    gen_parallel_L1   Rs + CPE1 ∥ (R1 + R_ad∥L)              ← 不含 R2
    gen_series_L2     Rs + CPE1 ∥ (R1 + CPE2 ∥ (R2 + R_L + jωL))
    gen_series_L1     Rs + CPE1 ∥ (R1 + R_L + jωL)           ← 不含 R2

用法
----
    python bm_top_gen.py            # 档 1 自检（1-1 / 1-2 / 1-3 + 边界稳健性）
    python bm_top_gen.py --json out.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "gen_parallel_L2", "gen_parallel_L1", "gen_series_L2", "gen_series_L1",
    "gen", "add_noise", "cover_matrix",
]

BRANCHES = ("parallel", "series")
LAYERS = (1, 2)


# --------------------------------------------------------------------------
# 元件（导纳视角）
# --------------------------------------------------------------------------

def _as_omega(f) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(f, float)


def _w_ref(f, f_ref) -> float:
    """角频率形式的参考频率；`f_ref` 按 Hz 传入（与 `z_nested` 同语义）。"""
    if f_ref is None:
        fa = np.asarray(f, float)
        return float(2.0 * np.pi * np.sqrt(fa.min() * fa.max()))
    return float(2.0 * np.pi * float(f_ref))


def _Y_cpe(omega: np.ndarray, Q: float, n: float, w_ref: float) -> np.ndarray:
    """CPE 导纳，**从窗口幅值 `Q = |Z(ω_ref)|` 反推**。

    `Y(ω) = (1/(Q·ω_ref^n))·(jω)^n`，不写 `e^{-j nπ/2}` 的闭式阻抗。
    """
    y_scale = 1.0 / (Q * w_ref ** n)
    return y_scale * (1j * omega) ** n


def _Y_rl_parallel(omega: np.ndarray, R: float, L: float) -> np.ndarray:
    """R∥L 的导纳：`1/R + 1/(jωL)`。ω→0 时发散（短路）。"""
    Y = np.full(omega.shape, 1.0 / R, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y = Y + 1.0 / (1j * omega * L)
    return Y


def _Y_rl_series(omega: np.ndarray, R: float, L: float) -> np.ndarray:
    """R+L 串联的导纳：`1/(R + jωL)`。"""
    return 1.0 / (R + 1j * omega * L)


def _Z_branch(omega: np.ndarray, p: dict, branch: str) -> np.ndarray:
    """感抗支路阻抗。`parallel` 支路从导纳取倒数，`series` 支路先算阻抗。"""
    if branch == "parallel":
        L = float(p["R_ad"]) * float(p["tau_ad"])
        Y = _Y_rl_parallel(omega, float(p["R_ad"]), L)
        Z = np.zeros_like(Y)
        fin = np.isfinite(Y) & (Y != 0)
        Z[fin] = 1.0 / Y[fin]                      # ω=0 时 Y=∞ → Z=0（正确的极限）
        return Z
    if branch == "series":
        L = float(p["R_L"]) * float(p["tau_L"])    # τ 是采样参数，L 是派生量
        return 1.0 / _Y_rl_series(omega, float(p["R_L"]), L)
    raise ValueError(f"branch 只能是 'parallel'/'series'，收到 {branch!r}")


# --------------------------------------------------------------------------
# 网络（逐节点导纳递推）
# --------------------------------------------------------------------------

def _network(omega: np.ndarray, p: dict, w_ref: float,
             branch: str, layers: int) -> np.ndarray:
    """从最内层往外，每层只做"导纳相加 → 取倒数 → 串电阻"三步。"""
    if layers not in (1, 2):
        raise ValueError(f"layers 只能是 1/2，收到 {layers!r}")
    z_b = _Z_branch(omega, p, branch)

    if layers >= 2:
        # 第二层：R2 与支路串联，整体再与 CPE2 并联
        Y_node2 = 1.0 / (float(p["R2"]) + z_b) \
            + _Y_cpe(omega, float(p["Q2"]), float(p["n2"]), w_ref)
        z_mid = float(p["R1"]) + 1.0 / Y_node2
    else:
        # 单层：没有 R2，R1 直接与支路串联（v3 定义）
        z_mid = float(p["R1"]) + z_b

    Y_top = 1.0 / z_mid + _Y_cpe(omega, float(p["Q1"]), float(p["n1"]), w_ref)
    return float(p["Rs"]) + 1.0 / Y_top


def _gen(f, p: dict, f_ref=None, branch: str = "parallel", layers: int = 2):
    omega = _as_omega(f)
    return _network(omega, p, _w_ref(f, f_ref), branch, layers)


# 四个具名入口：显式写死 branch/layers，调用处不会传错组合
def gen_parallel_L2(f, p: dict, f_ref=None):
    return _gen(f, p, f_ref, "parallel", 2)


def gen_parallel_L1(f, p: dict, f_ref=None):
    return _gen(f, p, f_ref, "parallel", 1)


def gen_series_L2(f, p: dict, f_ref=None):
    return _gen(f, p, f_ref, "series", 2)


def gen_series_L1(f, p: dict, f_ref=None):
    return _gen(f, p, f_ref, "series", 1)


def gen(f, p: dict, f_ref=None, branch: str = "parallel", layers: int = 2):
    """统一入口（签名对齐 `bm_top.z_nested`，方便交叉验证时直接对拍）。"""
    return _gen(f, p, f_ref, branch, layers)


def cover_matrix() -> dict:
    """§5.1 的覆盖矩阵：哪些 (branch, layers) 组合有生成器。"""
    return {
        ("parallel", 2): gen_parallel_L2,
        ("parallel", 1): gen_parallel_L1,
        ("series", 2): gen_series_L2,
        ("series", 1): gen_series_L1,
    }


# --------------------------------------------------------------------------
# 噪声（§4 冻结口径：固定标准差、实虚独立）
# --------------------------------------------------------------------------

def add_noise(Z, sigma_frac: float, *, mode: str = "fixed", seed: int = 0,
              zmax: float | None = None) -> np.ndarray:
    """给合成谱加噪。

    `mode="fixed"`（**默认，A/B 组的口径**）
        `ε_re, ε_im ~ N(0, σ²)`，`σ = sigma_frac·zmax`，实虚各自独立。
        与 `bm_core.fit` 的同方差 Normal 似然一致。
    `mode="relative"`（**M3.5 的 `rel-perpoint-2pct` 条件专用**）
        逐点 `σ_j = sigma_frac·|Z_j^true|`，异方差。与固定 σ 口径**不是同一个
        统计模型**：生成 σ 用真值谱、拟合用 `|Z_obs|`，两者差 `O(噪声)`；
        归一化常数含 `Σ log σ_j`，所以 ELPD **不可**跨口径比较
        （`BM_TOP_M3_PLAN.md` §20.1）。也不能用来标定采样器（§4.2）。
        （历史用途：`theoretical_bound.py` 的"表达能力"对照口径。）
    """
    Z = np.asarray(Z, complex)
    rng = np.random.default_rng(seed)
    if mode == "fixed":
        s = sigma_frac * (float(zmax) if zmax is not None
                          else float(np.max(np.abs(Z))))
        s_re = s_im = s
    elif mode == "relative":
        s_re = s_im = sigma_frac * np.abs(Z)
    else:
        raise ValueError(f"mode 只能是 'fixed'/'relative'，收到 {mode!r}")
    return Z + rng.normal(0.0, s_re, Z.shape) + 1j * rng.normal(0.0, s_im, Z.shape)


# --------------------------------------------------------------------------
# 档 1 自检
# --------------------------------------------------------------------------

BASE = dict(Rs=10.0, R1=55.0, R2=600.0, Q1=200.0, n1=0.75,
            Q2=900.0, n2=0.85, R_ad=3000.0, tau_ad=1.6,
            R_L=800.0, tau_L=1.6)


def _rand_params(rng, layers: int, branch: str, n_edge: bool = False) -> dict:
    """对数均匀抽一组"物理上说得过去"的参数；`n_edge` 时把 n 顶到边界。"""
    lu = lambda lo, hi: float(10.0 ** rng.uniform(lo, hi))
    p = dict(Rs=lu(-1.0, 1.5), R1=lu(0.0, 2.5),
             Q1=lu(1.0, 3.5),
             n1=float(rng.choice([0.5, 1.0])) if n_edge else float(rng.uniform(0.5, 1.0)))
    if layers >= 2:
        p.update(R2=lu(0.0, 3.0), Q2=lu(1.0, 3.5),
                 n2=float(rng.choice([0.5, 1.0])) if n_edge else float(rng.uniform(0.5, 1.0)))
    if branch == "parallel":
        p.update(R_ad=lu(1.5, 3.5), tau_ad=lu(-2.0, 1.0))
    else:
        p.update(R_L=lu(1.5, 3.5), tau_L=lu(-2.0, 1.0))
    return p


def _with_derived(p: dict) -> dict:
    """补 `z_nested` 需要的派生量 `L_ad`（生成器内部自己从 τ 算，不吃这个）。"""
    q = dict(p)
    if "R_L" in q:
        q["L_ad"] = q["R_L"] * q["tau_L"]
    return q


def _rel_err(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def _selftest(args) -> int:
    import bm_top as T          # 只在自检里 import：模块本身保持独立

    ok = True
    rec = {"stage": "M3-1", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "python": sys.version.split()[0], "checks": {}}
    print("=" * 78)
    print("bm_top_gen 档 1 自检（结构正确性：生成器 vs bm_top.z_nested）")
    print("=" * 78)

    # 0) 结构性前提：模块级没有依赖 bm_top（用 AST 查，不受注释/文档字符串干扰）
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    mod_names = []
    for node in tree.body:                       # 只看模块顶层
        if isinstance(node, ast.Import):
            mod_names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod_names.append(node.module)
    indep = not any(m.split(".")[0] == "bm_top" for m in mod_names)
    print(f"[0] 生成器模块级不依赖 bm_top：{'是' if indep else '否（违反 §5.1）'}"
          f"（顶层 import：{', '.join(mod_names) or '无'}）")
    ok &= indep

    # 1-1) 四种拓扑 × 三套频栅 × 多组随机参数：rel err < 1e-10
    grids = {
        "logspace(-2,5,61)": np.logspace(-2, 5, 61),
        "EIS 40 点 (1–1e5 Hz)": np.logspace(0, 5, 40),
        "宽域 (-9,9,41)": np.logspace(-9, 9, 41),
    }
    rng = np.random.default_rng(20261107)
    worst = {}
    for (branch, layers), fn in cover_matrix().items():
        key = f"{branch}_L{layers}"
        e_max, e_arg = 0.0, None
        bad = 0
        for name, f in grids.items():
            for _ in range(30):
                p = _rand_params(rng, layers, branch)
                z_gen = fn(f, p)
                z_ref = T.z_nested(f, _with_derived(p), branch=branch, layers=layers)
                if not np.all(np.isfinite(z_gen)):
                    bad += 1
                e = _rel_err(z_gen, z_ref)
                if e > e_max:
                    e_max, e_arg = e, name
        passed = e_max < 1e-10 and bad == 0
        ok &= passed
        worst[key] = e_max
        print(f"[1-1] {key:<12} 30 组参数 × 3 频栅（90 次对拍）：最大 rel err {e_max:.2e}"
              f"（{e_arg}）  非有限 {bad} 个  {'✓' if passed else '✗'}")
        rec["checks"].setdefault("1-1", {})[key] = {
            "max_rel_err": e_max, "nonfinite": bad, "pass": bool(passed)}

    # 1-2) 渐近极限（超宽频率，避免窗口截断误判）
    #      注意 layers=1 的极限**不含 R2**，这是 v3 定义的直接后果
    lim = {
        ("parallel", 1): (BASE["Rs"] + BASE["R1"], BASE["Rs"]),
        ("parallel", 2): (BASE["Rs"] + BASE["R1"] + BASE["R2"], BASE["Rs"]),
        ("series", 1): (BASE["Rs"] + BASE["R1"] + BASE["R_L"], BASE["Rs"]),
        ("series", 2): (BASE["Rs"] + BASE["R1"] + BASE["R2"] + BASE["R_L"], BASE["Rs"]),
    }
    fw = np.array([1e-12, 1e12])
    for (branch, layers), (lo, hi) in lim.items():
        key = f"{branch}_L{layers}"
        z = gen(fw, _with_derived(BASE), branch=branch, layers=layers)
        e0 = abs(z[0].real - lo) / abs(lo)
        einf = abs(z[1].real - hi) / abs(hi)
        passed = e0 < 1e-6 and einf < 1e-6 and abs(z[0].imag) < 1e-6 * abs(lo)
        ok &= passed
        print(f"[1-2] {key:<12} ω→0 → {z[0].real:9.4f}（理论 {lo:g}，err {e0:.1e}）"
              f"  ω→∞ → {z[1].real:7.4f}（理论 {hi:g}，err {einf:.1e}）"
              f"  {'✓' if passed else '✗'}")
        rec["checks"].setdefault("1-2", {})[key] = {
            "z0_re": float(z[0].real), "lim0": float(lo), "err0": float(e0),
            "zinf_re": float(z[1].real), "lim_inf": float(hi), "err_inf": float(einf),
            "pass": bool(passed)}

    # 1-3) 无源性：全域 min Re ≥ 0
    f = grids["logspace(-2,5,61)"]
    for (branch, layers), fn in cover_matrix().items():
        key = f"{branch}_L{layers}"
        z = fn(f, _with_derived(BASE))
        r = float(np.min(z.real))
        passed = r >= 0.0
        ok &= passed
        print(f"[1-3] {key:<12} 无源性 min Re = {r:.4f} Ω（须 ≥ 0）{'✓' if passed else '✗'}")
        rec["checks"].setdefault("1-3", {})[key] = {"min_re": r, "pass": bool(passed)}

    # 1-4)（超出 §6.1）边界稳健性：退化参数下不得出 nan/inf
    bad = 0
    for branch in BRANCHES:
        for layers in LAYERS:
            for n_edge in (False, True):
                for _ in range(40):
                    p = _rand_params(np.random.default_rng(rng.integers(1 << 30)),
                                     layers, branch, n_edge=n_edge)
                    z = gen(np.logspace(-6, 8, 57), _with_derived(p),
                            branch=branch, layers=layers)
                    if not np.all(np.isfinite(z)):
                        bad += 1
    print(f"[1-4] 边界稳健性（n∈{{0.5,1}}、参数跨 4–5 个量级，320 组）："
          f"非有限 {bad} 个  {'✓' if bad == 0 else '✗'}")
    ok &= bad == 0
    rec["checks"]["1-4"] = {"nonfinite": bad, "pass": bool(bad == 0)}

    # 附：噪声口径自证（§4.1 固定 σ vs §4.2 相对 σ）
    #     单次抽样估 σ 的误差约 σ/√(2N)，61 点看不出问题；这里用多次重复取均值。
    f = grids["logspace(-2,5,61)"]
    Z0 = gen_parallel_L2(f, _with_derived(BASE))
    zmax = float(np.max(np.abs(Z0)))
    reps = 400
    est = np.mean([np.std((add_noise(Z0, 0.02, mode="fixed", seed=s) - Z0).real)
                   for s in range(reps)])
    tgt = 0.02 * zmax
    p_fixed = abs(est - tgt) / tgt < 0.02
    ok &= p_fixed
    print(f"[附] 固定口径：{reps} 次重复的 σ̂_re 均值 {est:.3g}"
          f"（目标 {tgt:.3g}，偏差 {abs(est-tgt)/tgt:.1%}）{'✓' if p_fixed else '✗'}")

    # 相对口径的异方差性：**这是断言，不是叙述**。
    #     σ_j = 0.02·|Z_j| 时，一组点的合并残差 std 收敛到 0.02·RMS(|Z|)
    #     （方差按 |Z|² 加权，不是按中位数），所以"确实按 |Z| 缩放"必须拿
    #     RMS 比去核。只判"高段 > 低段"不够——一个与 |Z| 无关的常数 σ
    #     在三段任意分组下也能凑出方向正确的比值。
    zabs = np.abs(Z0)
    order = np.argsort(zabs)
    lo_idx, hi_idx = order[:order.size // 3], order[-order.size // 3:]
    rms_lo = float(np.sqrt(np.mean(zabs[lo_idx] ** 2)))
    rms_hi = float(np.sqrt(np.mean(zabs[hi_idx] ** 2)))
    res_r = np.array([(add_noise(Z0, 0.02, mode="relative", seed=s) - Z0).real
                      for s in range(reps)])
    sd_lo, sd_hi = float(np.std(res_r[:, lo_idx])), float(np.std(res_r[:, hi_idx]))
    ratio_meas = sd_hi / max(sd_lo, 1e-300)
    ratio_exp = rms_hi / rms_lo
    p_order = sd_hi > sd_lo                                   # 方向
    p_ratio = ratio_meas > 1.5                                # 明显 ≠ 1
    p_scale = abs(ratio_meas / ratio_exp - 1.0) < 0.05        # 比值对得上 RMS 比
    p_abs = (abs(sd_lo / (0.02 * rms_lo) - 1.0) < 0.03 and
             abs(sd_hi / (0.02 * rms_hi) - 1.0) < 0.03)        # 绝对值也对得上
    p_rel = p_order and p_ratio and p_scale and p_abs
    ok &= p_rel
    print(f"    相对口径：σ̂ 低 |Z| 三段 {sd_lo:.3g} / 高 |Z| 三段 {sd_hi:.3g}，"
          f"比值 {ratio_meas:.1f}")
    print(f"              理论 0.02·RMS = {0.02*rms_lo:.3g} / {0.02*rms_hi:.3g}"
          f"（RMS 比 {ratio_exp:.1f}）"
          f"  [方向{'✓' if p_order else '✗'} 比值>1.5 {'✓' if p_ratio else '✗'}"
          f" 比值对 RMS {'✓' if p_scale else '✗'} 绝对量 {'✓' if p_abs else '✗'}]")
    print(f"              → 异方差确认{'✓' if p_rel else '✗'}"
          f"（只能做表达能力对照，不能标定采样器）")
    rec["noise"] = {"fixed_sigma_mean": est, "target": tgt,
                    "fixed_pass": bool(p_fixed),
                    "rel_sd_low": sd_lo, "rel_sd_high": sd_hi,
                    "rel_ratio": ratio_meas, "rel_ratio_expected": ratio_exp,
                    "rel_checks": {"order": bool(p_order), "ratio_gt_1.5": bool(p_ratio),
                                   "ratio_vs_rms": bool(p_scale),
                                   "absolute": bool(p_abs)},
                    "rel_pass": bool(p_rel)}

    print("=" * 78)
    print("档 1 全部通过" if ok else "档 1 有失败项")
    rec["pass"] = bool(ok)
    rec["worst_rel_err"] = worst

    if args.json:
        Path(args.json).write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"记录写入 {args.json}")
    return 0 if ok else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="嵌套并联拓扑独立生成器 / 档 1 自检")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    return _selftest(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
