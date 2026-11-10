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

M4 前置（2026-11-09 追加，**不影响档 1 的四个具名入口**）
--------------------------------------------------------
档 1 的合成谱够"拓扑复杂"，但不够"像真实谱"：频栅从 0.01 Hz 起、`τ_ad` 抽到
10 s（f_ad≈0.016 Hz，落在真实窗外）、没有截断、也没有"两特征刚好合并"的刀口。
`select_across_families` 真正会错的正是这些地方。本文件因此补三样：

| 新增 | 作用 |
|---|---|
| `cutoff(f, Z, f_min)` / 各入口的 `f_min=` | 模拟 E6 的窗口截断（先按全窗生成，再删点——物理上是"同一池子只测了更窄的频段"） |
| `REAL` / `EIS_GRID` | M3 A 组真值量级 + AZ31/Blank 实际频栅（0.1–1e5 Hz） |
| `gen_mixed_boundary(case)` | 三个刀口构造：环移出窗、环与第二弧合并、两弧峰位重合（R1/R2 共线） |

**这些是给选择器做压力测试的，不是新的真值来源**：刀口谱的价值在于"两族/两特征
在该谱上不可区分"，不在"它对应某个物理过程"。
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
    "cutoff", "real_grid", "REAL", "EIS_GRID", "BOUNDARY_CASES",
    "gen_mixed_boundary", "peak_hz",
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


def _gen(f, p: dict, f_ref=None, branch: str = "parallel", layers: int = 2,
         f_min: float | None = None):
    omega = _as_omega(f)
    return _network(omega, p, _w_ref(f, f_ref), branch, layers)


# 四个具名入口：显式写死 branch/layers，调用处不会传错组合
def gen_parallel_L2(f, p: dict, f_ref=None, f_min=None):
    return cutoff(f, _gen(f, p, f_ref, "parallel", 2), f_min)[1]


def gen_parallel_L1(f, p: dict, f_ref=None, f_min=None):
    return cutoff(f, _gen(f, p, f_ref, "parallel", 1), f_min)[1]


def gen_series_L2(f, p: dict, f_ref=None, f_min=None):
    return cutoff(f, _gen(f, p, f_ref, "series", 2), f_min)[1]


def gen_series_L1(f, p: dict, f_ref=None, f_min=None):
    return cutoff(f, _gen(f, p, f_ref, "series", 1), f_min)[1]


def gen(f, p: dict, f_ref=None, branch: str = "parallel", layers: int = 2,
        f_min=None):
    """统一入口（签名对齐 `bm_top.z_nested`，方便交叉验证时直接对拍）。

    `f_min` 非空时只返回 `f ≥ f_min` 的点（截断在**生成之后**做，见 `cutoff`）。
    """
    f, Z = cutoff(f, _gen(f, p, f_ref, branch, layers), f_min)
    return Z


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

# --------------------------------------------------------------------------
# M4 前置：截断 / 真实量级 / 刀口谱
# --------------------------------------------------------------------------

# M3 A 组的真值量级（`BM_TOP_M3_PLAN.md` §15），不是重新抽的一组数
REAL = dict(BASE)

# 真实频栅（从 E2/E6 的输入直接量出来的，不是"差不多"的近似）：
#   AZ31-12Li_0  50 点 0.01–1e5（7 dec）→ 截 1 Hz 36 点、截 10 Hz 29 点
#   Blank 主谱    50 点 0.1–1e4 （5 dec）→ 截 1 Hz 40 点、截 10 Hz 30 点
#   2E 系列       56 点 1–1e5
GRID_PRESETS = {
    "AZ31": (0.01, 1.0e5, 50),
    "Blank": (0.1, 1.0e4, 50),
    "2E": (1.0, 1.0e5, 56),
}


def real_grid(sample: str = "AZ31", n: int | None = None) -> np.ndarray:
    """真实样品同款频栅。`sample ∈ GRID_PRESETS`，`n` 可覆盖点数。"""
    if sample not in GRID_PRESETS:
        raise ValueError(f"sample 只能是 {tuple(GRID_PRESETS)}，收到 {sample!r}")
    lo, hi, k = GRID_PRESETS[sample]
    return np.logspace(np.log10(lo), np.log10(hi), int(n or k))


EIS_GRID = real_grid("AZ31")


def cutoff(f, Z, f_min=None):
    """按 `f ≥ f_min` 截断观测，返回 `(f_cut, Z_cut)`。

    口径要点（与 E6 一致）：**先在完整窗上生成，再删点**。物理含义是"同一个
    体系只测到 f_min 以上"，模型参数不变、`f_ref` 也不变；若改成在截断窗上
    重算 `f_ref`，CPE 的 `Q` 含义就跟着变，那不是"少测一段"而是"换了标定"，
    两者不能混。
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    if f_min is None:
        return f, Z
    keep = f >= float(f_min)
    if not keep.any():
        raise ValueError(f"f_min={f_min} 把全部 {f.size} 点都截掉了")
    return f[keep], Z[keep]


def peak_hz(R: float, Q: float, n: float, w_ref: float) -> float:
    """单个 `R∥CPE` 的 −Im 极大频率（解析：`|Y_cpe|=1/R` 处）。

    `Y_cpe(ω)=ω^n/(Q·ω_ref^n)`，令其等于 `1/R` 得
    `ω_pk=(Q·ω_ref^n/R)^{1/n}`。`n=1` 时退化为 `1/(RC)`，可直接核对。
    """
    return float((Q * w_ref ** n / R) ** (1.0 / n) / (2.0 * np.pi))


def _peak_hz_num(R: float, Q: float, n: float, w_ref: float,
                 f=None) -> float:
    """数值求峰：在密频栅上直接找 `−Im Z` 最大点（用来核 `peak_hz`，不共用公式）。"""
    if f is None:
        f = np.logspace(-6, 9, 3001)
    Y = 1.0 / R + _Y_cpe(_as_omega(f), Q, n, w_ref)
    Z = 1.0 / Y
    return float(f[int(np.argmax(-Z.imag))])


BOUNDARY_CASES = ("loop_out_of_window", "loop_merged", "arcs_merged")


def gen_mixed_boundary(case: str = "arcs_merged", *, f=None, f_min=None,
                       base: dict | None = None, noise: float | None = None,
                       seed: int = 0, f_ref=None) -> dict:
    """构造"两特征刚好分不开"的刀口谱（M4 选择器的压力测试用例）。

    三个 case 都只改一两个参数，其余用 `REAL`（M3 A 组真值），所以彼此可比：

    | case | 改什么 | 刀口在哪 | 选择器该怎样 |
    |---|---|---|---|
    | `loop_out_of_window` | `τ_ad` 推到 `f_ad=0.3·f_min` | 环峰在窗外，窗内只剩**感抗尾** | **不得**靠"K 变大"宣称发现环；`f_ad` 只能报"本窗无权评价" |
    | `loop_merged` | `τ_ad` 对齐第二弧峰位 | 感抗特征叠在弧 2 上 | L 与 K 不可分，应并列而非单选 |
    | `arcs_merged` | `Q1` 对齐第一弧峰位、`n1=n2` | 两弧同位同形 → `R1/R2` 共线 | `R1`、`R2` 只能报 `R1+R2` |

    注意 `loop_out_of_window` 的物理含义**不是"环消失了"**：固定 `R_ad` 时
    `L = R_ad·τ_ad` 随 τ 一起增大，窗内低频端反而是**大幅值感抗尾**（探针实测
    占 `|Z(f_min)|` 的 82%）。这正是 18 条真实 AZ31 谱的形状（`inductive_tail`
    18/18），所以这个 case 测的是"只有尾、没有拐点时会不会被硬判成环"。

    返回 `dict(case, f, Z, p, meta)`。`noise` 给定则按 §4.1 固定 σ 加噪
    （`σ=noise·max|Z|`，实虚独立）。

    **不要把这些谱当成"某个物理过程"**：它们只声明"在该频栅与该噪声下，
    两族/两特征近似等价"，用来检查选择器会不会在这种地方硬选。
    """
    if case not in BOUNDARY_CASES:
        raise ValueError(f"case 只能是 {BOUNDARY_CASES}，收到 {case!r}")
    if f is None:
        f = real_grid()
    f = np.asarray(f, float)
    p = dict(REAL if base is None else base)
    w_ref = _w_ref(f, f_ref)

    f_pk1 = peak_hz(p["R1"], p["Q1"], p["n1"], w_ref)
    f_pk2 = peak_hz(p["R2"], p["Q2"], p["n2"], w_ref)
    f_lo = float(f.min())
    meta = {"f_min_window": f_lo, "f_pk1_hz": f_pk1, "f_pk2_hz": f_pk2,
            "changed": {}}

    if case == "loop_out_of_window":
        target = 0.3 * f_lo
        p["tau_ad"] = 1.0 / (2.0 * np.pi * target)
        meta["changed"] = {"tau_ad": p["tau_ad"]}
    elif case == "loop_merged":
        p["tau_ad"] = 1.0 / (2.0 * np.pi * f_pk2)
        meta["changed"] = {"tau_ad": p["tau_ad"]}
    else:  # arcs_merged：把弧 1 的峰位挪到弧 2 上，并让两个 CPE 同指数
        p["n1"] = p["n2"]
        w_pk2 = 2.0 * np.pi * f_pk2
        p["Q1"] = float(p["R1"] * (w_pk2 / w_ref) ** p["n1"])
        meta["changed"] = {"Q1": p["Q1"], "n1": p["n1"]}

    meta["f_ad_hz"] = 1.0 / (2.0 * np.pi * float(p["tau_ad"]))
    meta["f_ad_over_fmin"] = meta["f_ad_hz"] / f_lo
    meta["f_pk1_over_pk2"] = meta["f_pk1_hz"] / meta["f_pk2_hz"]
    meta["L_ad_H"] = float(p["R_ad"]) * float(p["tau_ad"])

    f_full = f
    Z_full = _gen(f_full, p, f_ref, "parallel", 2)
    f_cut, Z_cut = cutoff(f_full, Z_full, f_min)
    if noise is not None:
        Z_cut = add_noise(Z_cut, float(noise), mode="fixed", seed=seed)

    meta["n_pts"] = int(f_cut.size)
    meta["zmax"] = float(np.max(np.abs(Z_cut)))
    return {"case": case, "f": f_cut, "Z": Z_cut, "p": p, "meta": meta,
            "f_full": f_full, "Z_full": Z_full}


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

    # 1-5) 截断入口 + 真实频栅：点数必须对上 E6 实测（AZ31 36/29、Blank 40/30）
    expect = {"AZ31": (36, 29), "Blank": (40, 30)}
    bad_cut = 0
    rows = []
    for name, (n1, n10) in expect.items():
        fx = real_grid(name)
        Zx = gen_parallel_L2(fx, _with_derived(REAL))
        for fmin, exp in ((None, fx.size), (1.0, n1), (10.0, n10)):
            fc, zc = cutoff(fx, Zx, fmin)
            ref = Zx if fmin is None else Zx[fx >= fmin]
            same = (fc.size == exp and fc.size == ref.size
                    and np.array_equal(zc, ref)
                    and np.allclose(fc, fx if fmin is None else fx[fx >= fmin]))
            bad_cut += (not same)
            rows.append((name, fmin, fc.size, exp, bool(same)))
    print("[1-5] 截断入口（只删点、保留值逐位相同；点数对 E6 实测）：")
    for name, fmin, n_, exp, s_ in rows:
        print(f"      {name:<6} f_min={'全窗' if fmin is None else f'{fmin:g} Hz':<8}"
              f" → {n_:2d} 点（期望 {exp:2d}）{'✓' if s_ else '✗'}")
    ok &= bad_cut == 0
    rec["checks"]["1-5"] = {"rows": [{"sample": nm, "f_min": fm, "n": n_,
                                      "expect": e, "same": s_}
                                     for nm, fm, n_, e, s_ in rows],
                            "pass": bool(bad_cut == 0)}

    # 1-6) 解析峰位 vs 数值求峰（刀口构造的几何基础；两条路径不同）
    f50 = real_grid("AZ31")
    w0 = _w_ref(f50, None)
    worst6, arg6 = 0.0, None
    for (R, Q, n) in ((600.0, 900.0, 0.85), (600.0, 900.0, 1.0),
                      (55.0, 200.0, 0.75), (3000.0, 50.0, 0.60)):
        fa, fn_ = peak_hz(R, Q, n, w0), _peak_hz_num(R, Q, n, w0)
        e = abs(fa / fn_ - 1.0)
        if e > worst6:
            worst6, arg6 = e, (R, Q, n)
    p6 = worst6 < 0.02
    ok &= p6
    print(f"[1-6] 峰位解析 vs 数值（4 组 R/Q/n）：最大相对差 {worst6:.2%}"
          f"（{arg6}）{'✓' if p6 else '✗'}")
    rec["checks"]["1-6"] = {"max_rel": worst6, "pass": bool(p6)}

    # 1-7) 三个刀口 case 的几何是否真的是"刀口"（用数值峰核，不看构造参数）
    fb = np.logspace(-6, 6, 4001)
    geo = {}
    for case in BOUNDARY_CASES:
        d = gen_mixed_boundary(case, f=f50)
        p_, m_ = d["p"], d["meta"]
        Zb = _Z_branch(_as_omega(fb), p_, "parallel")
        f_ad_num = float(fb[int(np.argmax(Zb.imag))])
        if case == "loop_out_of_window":
            passed = abs(m_["f_ad_over_fmin"] - 0.3) < 1e-9 and f_ad_num < f50.min()
            desc = (f"f_ad={m_['f_ad_hz']:.4g} Hz = {m_['f_ad_over_fmin']:.2f}·f_min，"
                    f"数值峰 {f_ad_num:.4g} Hz {'在窗外' if f_ad_num < f50.min() else '仍在窗内'}")
        elif case == "loop_merged":
            sep = abs(np.log10(f_ad_num / m_["f_pk2_hz"]))
            passed = abs(np.log10(m_["f_ad_hz"] / m_["f_pk2_hz"])) < 1e-6 and sep < 0.05
            desc = (f"f_ad={m_['f_ad_hz']:.4g} Hz vs 弧2 {m_['f_pk2_hz']:.4g} Hz，"
                    f"数值峰相隔 {sep:.3f} dec")
        else:  # arcs_merged
            p1 = {k: p_[k] for k in ("R1", "Q1", "n1")}
            p2 = {k: p_[k] for k in ("R2", "Q2", "n2")}
            pa = _peak_hz_num(p1["R1"], p1["Q1"], p1["n1"], w0)
            pb = _peak_hz_num(p2["R2"], p2["Q2"], p2["n2"], w0)
            sep = abs(np.log10(pa / pb))
            passed = (p_["n1"] == p_["n2"]) and sep < 0.05
            desc = (f"弧1 {pa:.4g} Hz vs 弧2 {pb:.4g} Hz，相隔 {sep:.3f} dec，"
                    f"n1=n2={p_['n1']:.2f}")
        geo[case] = {"pass": bool(passed), "desc": desc}
        ok &= passed
        print(f"[1-7] {case:<20} {desc}  {'✓' if passed else '✗'}")
    rec["checks"]["1-7"] = geo

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
