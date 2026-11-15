"""E1 执行器：M-b 残差结构的重复性验证（判据冻结于 `E1_PREREG.md`）。

本脚本只做三件事，**不做任何自动决策**：
1. 校验 `inbox/e1/` 的 6 条谱与频率栅格（`E1_PREREG.md` §2 R-GRID，只许删点/阻断，绝不插值）；
2. 用**唯一一个拟合入口** `m4_holdout2._fit_set()` 拟合 6 条（`standard` + 逐点 2%σ +
   `parallel/L=2` + `seed=0`），并按 M-b 口径由后验中位参数正演得到残差；
3. 按 §3/§4 的冻结规则算出 `J1/J2/J3` 并落盘全部必留产物。

**它不是验收执行器**：不判 M-a/M-b/M-c，不改 §13/§14/§17 的任何数字，不接 `bmdrt.py`。
`J1` 只等于「可再现残差结构」，`J2` 只等于「缺乏重复性证据」，两者都不涉及物理来源（§0）。

用法
----
    # 0) 先看判据引擎与栅格规则是否自洽（不需要数据，秒级）
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e1_repeatability.py --selftest

    # 1) 只检查输入是否齐（不拟合）
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e1_repeatability.py --dry

    # 2) 正式跑（6 条，约 40 s/条）
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/e1_repeatability.py

    # J3 后扩到 5 次：把 rep4/rep5 放进同一目录，重跑同一条命令即可
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))

import bm_experiments as E   # noqa: E402
import bm_real as R          # noqa: E402
import bm_top as T           # noqa: E402
import m4_holdout2 as H      # noqa: E402  唯一拟合入口（冻结，不改其常数）

PREREG = "docs/round_2026-11-15/E1_PREREG.md"
IN_ROOT = os.path.join("inbox", "e1")
OUT_ROOT = os.path.join("out", "e1")
STATES = ("8Li_50", "4Li_5")          # 通过组 / 未通过组（§1）
N_REP_FROZEN = 3                      # §1；J3 后允许 5（§6）
BANDS = ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5))

# ---- 冻结判定常数（§3，跑后不得修改）-----------------------------------------
R_HI = 0.70        # J1 门槛
R_LO = 0.30        # J2 门槛
ACF_RANGE_MAX = 0.05   # 两组共同的极差上限
# -----------------------------------------------------------------------------


# ============================ 判据引擎 =======================================
# §3 原文口径（保留，仅作"原文会怎么判"的并列输出；见 E1_PREREG §13）
def classify_orig(r_min: float, acf_range: float) -> tuple[str, str]:
    if not np.isfinite(r_min) or not np.isfinite(acf_range):
        return "J3", "undefined"
    if r_min >= R_HI and acf_range <= ACF_RANGE_MAX:
        return "J1", "reproducible"
    if r_min <= R_LO and acf_range <= ACF_RANGE_MAX:
        return "J2", "not_reproducible"
    if R_LO < r_min < R_HI:
        return "J3", "r_in_mixed_band"
    return "J3", "acf_range_too_large"


# §13 修正口径（**主判据**）：极差从"门"改为"护栏"，`min|r|` 门槛不动
def classify(r_min: float, acf_range: float,
             null_p95: float = float("inf")) -> tuple[str, str]:
    """返回 (判定, 理由)。`nan` 自然落入 J3（比较全 False）。"""
    if not np.isfinite(r_min):
        return "J3", "r_undefined"
    if not np.isfinite(acf_range):
        return "J3", "acf_range_undefined"
    guarded = np.isfinite(null_p95) and acf_range > null_p95
    if r_min >= R_HI:
        return ("J3", "acf_unstable_guard") if guarded else ("J1", "reproducible")
    if r_min <= R_LO:
        return ("J3", "acf_unstable_guard") if guarded else ("J2", "not_reproducible")
    return "J3", "r_in_mixed_band"


def surrogate_null_range(res: np.ndarray, ntry: int = 2000,
                         seed: int = 0) -> float:
    """护栏阈值：AR(1) 匹配的"三条独立复本"的 `acf` 极差 95% 分位。

    用第一条复本残差的 Re/Im 各自的 lag-1 自相关生成独立复本，
    每条按项目口径 `max(acf_re, acf_im)` 算 acf，再取 3 条极差。
    """
    rng = np.random.default_rng(seed)
    re, im = np.asarray(res.real, float), np.asarray(res.imag, float)

    def rho1(x):
        x = x - x.mean()
        d = float(x @ x)
        return float(x[:-1] @ x[1:] / d) if d > 0 else 0.0

    def gen(x, rho):
        e = rng.normal(size=x.size)
        y = np.empty(x.size)
        y[0] = e[0]
        for i in range(1, x.size):
            y[i] = rho * y[i - 1] + e[i]
        return y

    pr, pi = rho1(re), rho1(im)
    out = np.empty(ntry)
    for t in range(ntry):
        a = [max(E.acf_max(gen(re, pr)), E.acf_max(gen(im, pi))) for _ in range(3)]
        out[t] = max(a) - min(a)
    return float(np.percentile(out, 95))



def merge_states(v1: str, v2: str) -> tuple[str, str]:
    """§4：两组一致才给 J1/J2，否则 J3。"""
    if v1 == v2 == "J1":
        return "J1", "both_states_J1"
    if v1 == v2 == "J2":
        return "J2", "both_states_J2"
    return "J3", f"inconsistent({v1}/{v2})"


def pairwise_r(reps: list[np.ndarray], min_abs: bool = True) -> list[dict]:
    """所有两两组合的 Pearson r；`min_abs` 只影响调用方的聚合，不影响记录。"""
    out = []
    for i, j in itertools.combinations(range(len(reps)), 2):
        a, b = np.asarray(reps[i], float), np.asarray(reps[j], float)
        if a.size != b.size:
            out.append(dict(i=i, j=j, r=None, note="length_mismatch"))
            continue
        if a.std() <= 0 or b.std() <= 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
            out.append(dict(i=i, j=j, r=None, note="degenerate_series"))
            continue
        out.append(dict(i=i, j=j, r=float(np.corrcoef(a, b)[0, 1]), note=""))
    return out


def aggregate_r(pairs: list[dict]) -> float:
    """§3：6 个系数里取最小的 |r|；任一缺失/未定义 -> nan（-> J3）。"""
    vals = []
    for p in pairs:
        if p["r"] is None:
            return float("nan")
        vals.append(abs(p["r"]))
    return float(min(vals)) if vals else float("nan")


# ============================ 栅格规则 =======================================
def check_grid(fs: list[np.ndarray], rtol_exact: float = 1e-9,
               rtol_common: float = 1e-6, keep_frac: float = 0.90) -> dict:
    """§2 R-GRID。返回 mode + 每条谱要用的索引（只删点，不插值）。

    mode = "exact" | "common_subset" | "blocked_grid"
    """
    n = [len(f) for f in fs]
    if len(set(n)) == 1 and all(np.allclose(fs[i], fs[0], rtol=rtol_exact, atol=0)
                                for i in range(1, len(fs))):
        idx = [np.arange(n[0])] * len(fs)
        return dict(mode="exact", idx=idx, dropped_hz=[], n_kept=int(n[0]))

    ref = np.asarray(fs[0], float)
    idx = []
    for f in fs:
        f = np.asarray(f, float)
        pos = np.abs(np.log(f)[None, :] - np.log(ref)[:, None]).argmin(axis=1)
        ok = np.abs(f[pos] - ref) <= rtol_common * ref
        idx.append(np.where(ok)[0])
    common = np.intersect1d(np.intersect1d(idx[0], idx[1] if len(idx) > 1 else idx[0]),
                            idx[2] if len(idx) > 2 else idx[0])
    n_kept = int(common.size)
    if n_kept < keep_frac * min(n) or n_kept < 20:
        return dict(mode="blocked_grid", idx=None, dropped_hz=None, n_kept=n_kept,
                    why=f"common={n_kept} < {keep_frac:.0%} of {min(n)} or <20")
    per = []
    for f in fs:
        f = np.asarray(f, float)
        pos = np.abs(np.log(f)[None, :] - np.log(ref[common])[:, None]).argmin(axis=1)
        per.append(pos)
    dropped = [float(x) for x in ref if x not in set(ref[common])]
    return dict(mode="common_subset", idx=per, dropped_hz=dropped, n_kept=n_kept)


# ============================ 指标 ===========================================
def acf_pair(res: np.ndarray) -> dict:
    return dict(acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
                acf_re=float(E.acf_max(res.real)),
                acf_im=float(E.acf_max(res.imag)))


def band_acf(f: np.ndarray, res: np.ndarray) -> dict:
    out = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out[f"{lo:g}-{hi:g}"] = dict(
            n=int(m.sum()),
            re=float(E.acf_max(res.real[m])) if m.sum() >= 8 else None,
            im=float(E.acf_max(res.imag[m])) if m.sum() >= 8 else None)
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ============================ 单组分析 =======================================
def analyse_state(state: str, do_fit: bool = True,
                  in_root: str = IN_ROOT, out_root: str = OUT_ROOT) -> dict:
    d = os.path.join(in_root, state)
    if not os.path.isdir(d):
        return dict(state=state, status="missing_dir", dir=os.path.abspath(d))
    reps = sorted(x for x in os.listdir(d) if x.lower().endswith(".csv"))
    if len(reps) not in (N_REP_FROZEN, 5):
        return dict(state=state, status="bad_replica_count", found=reps,
                    expect="rep1..rep3（J3 后可为 rep1..rep5）")
    files = [os.path.join(d, x) for x in reps]
    loaded = [R.load_spectrum(p) for p in files]
    fs = [np.asarray(f, float) for f, _ in loaded]
    Zs = [np.asarray(Z, complex) for _, Z in loaded]

    g = check_grid(fs)
    if g["mode"] == "blocked_grid":
        return dict(state=state, status="blocked_grid", grid=g, files=reps)
    fs = [f[idx] for f, idx in zip(fs, g["idx"])]
    Zs = [Z[idx] for Z, idx in zip(Zs, g["idx"])]

    out = dict(state=state, status="ok", grid_mode=g["mode"],
               grid=dict(dropped_hz=g["dropped_hz"], n_kept=g["n_kept"]),
               replicas=[], pairs=[], files=[])
    if not do_fit:
        out["status"] = "layout_ok_not_fitted"
        out["files"] = [dict(rep=r, n=int(len(f))) for r, f in zip(reps, fs)]
        return out

    fit_dir = os.path.join(out_root, "posterior")
    res_dir = os.path.join(out_root, "residuals")
    os.makedirs(fit_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    res_re, res_im, res_re_n, res_im_n = [], [], [], []
    for rep, path, f, Z in zip(reps, files, fs, Zs):
        tag = f"{state}_{os.path.splitext(rep)[0]}"
        fs_res = H._fit_set(f, Z, [0], save_dir=fit_dir, tag=tag)   # 唯一拟合入口
        row = fs_res["rows"][0]
        pm = row["params"]
        Zm = T.z_nested(f, pm, branch=H.BRANCH, layers=H.LAYERS)
        res = Zm - Z
        a = acf_pair(res)
        rec = dict(rep=rep, sha256=sha256(path),
                   mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path))),
                   n=int(f.size), f_min=float(f.min()), f_max=float(f.max()),
                   rms_rel=float(row["rms_rel"]), **a,
                   band_acf=band_acf(f, res),
                   rhat_max=float(row["rhat_max"]), div=int(row["div"]),
                   ess_min=float(row["ess_min"]), ess_by_param=row["ess_by_param"],
                   ess_ok_supp=bool(row["ess_ok_supp"]),
                   degenerate=row["degenerate"],
                   params=pm, ci90=row["ci90"],
                   # 自检：残差是否就是 M-b 那条残差（口径一致性）
                   acf_matches_reported=bool(abs(a["acf"] - row["acf"]) <= 5e-3),
                   npz=os.path.join(fit_dir, f"{tag}_seed0.npz"))
        out["replicas"].append(rec)
        out["files"].append(dict(rep=rep, sha256=rec["sha256"], n=rec["n"]))
        res_re.append(res.real)
        res_im.append(res.imag)
        res_re_n.append(res.real / np.abs(Z))
        res_im_n.append(res.imag / np.abs(Z))
        csv = os.path.join(res_dir, f"{tag}_res.csv")
        np.savetxt(csv, np.column_stack([f, Z.real, Z.imag, Zm.real, Zm.imag,
                                         res.real, res.imag,
                                         res.real / np.abs(Z), res.imag / np.abs(Z)]),
                   delimiter=",", fmt="%.10g",
                   header="f,zre_obs,zim_obs,zre_fit,zim_fit,res_re,res_im,"
                          "res_re_norm,res_im_norm", comments="")

    # ---- 两两相关：两个口径 × 两个分量 × 全部组合 ----
    for form, (rr, ii) in dict(raw=(res_re, res_im), norm=(res_re_n, res_im_n)).items():
        for comp, series in (("re", rr), ("im", ii)):
            for p in pairwise_r(series):
                out["pairs"].append(dict(form=form, comp=comp, **p))

    def rmin(form: str) -> float:
        return aggregate_r([p for p in out["pairs"] if p["form"] == form])

    acf_all = [r["acf"] for r in out["replicas"]]
    acf_range = float(max(acf_all) - min(acf_all))
    null_p95 = surrogate_null_range(np.asarray(res_re[0]) + 1j * np.asarray(res_im[0]))
    v_raw, why_raw = classify(rmin("raw"), acf_range, null_p95)
    v_norm, why_norm = classify(rmin("norm"), acf_range, null_p95)
    if v_raw != v_norm:
        verdict, why = "J3", f"primary({v_raw})!=secondary({v_norm})"
    else:
        verdict, why = v_raw, why_raw
    if not all(r["acf_matches_reported"] for r in out["replicas"]):
        verdict, why = "J3", "residual_not_reproducing_reported_acf"
    # §3 原文口径并列输出（极差当门，门槛 0.05）
    orig_raw, _ = classify_orig(rmin("raw"), acf_range)
    orig_norm, _ = classify_orig(rmin("norm"), acf_range)
    verdict_orig = orig_raw if orig_raw == orig_norm else "J3"

    out.update(dict(r_min_primary=rmin("raw"), r_min_secondary=rmin("norm"),
                    acf_range=acf_range,
                    acf_range_re=float(max(r["acf_re"] for r in out["replicas"])
                                       - min(r["acf_re"] for r in out["replicas"])),
                    acf_range_im=float(max(r["acf_im"] for r in out["replicas"])
                                       - min(r["acf_im"] for r in out["replicas"])),
                    class_primary=v_raw, class_secondary=v_norm,
                    null_p95_range=null_p95, verdict=verdict, reason=why,
                    verdict_orig_sec3=verdict_orig))
    return out


# ============================ 自检 ===========================================
def selftest() -> int:
    ok = 0
    fail = []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # --- 判据边界 ---
    # §13 修正口径：极差是护栏，不是门
    chk("J1_at_threshold", classify(0.70, 0.05)[0] == "J1")
    chk("J1_typical", classify(0.93, 0.012)[0] == "J1")
    chk("J2_at_threshold", classify(0.30, 0.05)[0] == "J2")
    chk("J2_typical", classify(0.08, 0.02)[0] == "J2")
    chk("J3_mixed_band_lo", classify(0.31, 0.01)[0] == "J3")
    chk("J3_mixed_band_hi", classify(0.69, 0.01)[0] == "J3")
    chk("J1_range_no_longer_gates", classify(0.90, 0.06)[0] == "J1")
    chk("J2_range_no_longer_gates", classify(0.10, 0.14)[0] == "J2")
    chk("guard_kills_J1", classify(0.99, 0.20, null_p95=0.15)[0] == "J3")
    chk("guard_kills_J2", classify(0.05, 0.20, null_p95=0.15)[0] == "J3")
    chk("guard_not_triggered", classify(0.95, 0.10, null_p95=0.15)[0] == "J1")
    chk("J3_nan_r", classify(float("nan"), 0.01)[0] == "J3")
    chk("J3_nan_range", classify(0.95, float("nan"))[0] == "J3")
    # §3 原文口径仍可复现（并列输出用）
    chk("orig_range_gates", classify_orig(0.90, 0.06)[0] == "J3")
    chk("orig_range_gates2", classify_orig(0.10, 0.14)[0] == "J3")
    chk("orig_J1", classify_orig(0.93, 0.012)[0] == "J1")

    # --- 合并规则 ---
    chk("merge_J1", merge_states("J1", "J1")[0] == "J1")
    chk("merge_J2", merge_states("J2", "J2")[0] == "J2")
    chk("merge_mixed", merge_states("J1", "J2")[0] == "J3")
    chk("merge_with_J3", merge_states("J1", "J3")[0] == "J3")

    # --- 相关系数聚合：最小 |r|，不是平均 ---
    reps = [np.array([1., 2., 3., 4., 5., 6.]),
            np.array([1., 2., 3., 4., 5., 7.]),      # 与 0 高度相关
            np.array([6., 1., 5., 2., 4., 3.])]      # 与 0 差
    ps = pairwise_r(reps)
    chk("pairs_count", len(ps) == 3)
    chk("min_not_mean", aggregate_r(ps) < float(np.mean([abs(p["r"]) for p in ps])))

    # --- 退化序列 -> nan -> J3 ---
    ps2 = pairwise_r([np.zeros(6), np.ones(6), np.arange(6.)])
    chk("degenerate_r_nan", np.isnan(aggregate_r(ps2)))

    # --- 栅格规则 ---
    f = np.logspace(-2, 5, 50)
    chk("grid_exact", check_grid([f, f.copy(), f.copy()])["mode"] == "exact")
    f2 = f.copy()
    f2[-1] *= 1 + 1e-7
    chk("grid_common_subset", check_grid([f, f2, f.copy()])["mode"] == "common_subset")
    f3 = np.logspace(-2, 5, 30)
    chk("grid_blocked", check_grid([f, f3, f.copy()])["mode"] == "blocked_grid")
    no_interp = all(np.all(np.diff(i) >= 0) for i in
                    check_grid([f, f2, f.copy()])["idx"])
    chk("grid_no_interp_indices", no_interp)

    print(f"自检：{ok}/{ok + len(fail)} 通过")
    if fail:
        print("  失败：" + ", ".join(fail))
        return 1
    return 0


# ============================ 主流程 =========================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="只跑判据/栅格自检，不需要数据")
    ap.add_argument("--dry", action="store_true", help="只检查输入布局，不拟合")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "verdict.json"))
    # 只用于管线冒烟测试（合成副本）；正式运行必须用默认路径，输出也不得覆盖 out/e1
    ap.add_argument("--in-root", default=IN_ROOT)
    ap.add_argument("--out-dir", default=OUT_ROOT)
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    states = [analyse_state(s, do_fit=not a.dry, in_root=a.in_root,
                            out_root=a.out_dir) for s in STATES]
    if a.dry:
        print(json.dumps(states, ensure_ascii=False, indent=1)[:4000])
        bad = [s["state"] for s in states if s.get("status") != "layout_ok_not_fitted"]
        if bad:
            print(f"\n未就绪：{bad}")
            print(f"期望布局：{a.in_root}/<state>/rep1.csv .. rep3.csv（state ∈ {STATES}）")
            print("判据与聚合方式见 " + PREREG)
            return 2
        print("\n布局检查通过（未拟合）。去掉 --dry 即可正式跑。")
        return 0

    meta = dict(
        prereg=PREREG,
        states=list(STATES), n_rep_frozen=N_REP_FROZEN,
        fit=dict(entry="m4_holdout2._fit_set", budget=H.BUDGET, sigma_pct=H.SIGMA_PCT,
                 branch=H.BRANCH, layers=H.LAYERS, seed=0, m1_cut="none(E1 不剔点)"),
        thresholds=dict(r_hi=R_HI, r_lo=R_LO, acf_range_max=ACF_RANGE_MAX),
        aggregation="min |r| over {Re,Im} x all pairs; primary=raw residual, "
                    "secondary=|Z|-normalized; disagreement -> J3",
        grid_rule="exact(rtol 1e-9) -> common_subset(rtol 1e-6, >=90%) -> blocked; no interpolation",
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")),
        python=platform.python_version(),
        numpy=np.__version__,
        code_sha256={p: sha256(os.path.join(REPO, p)) for p in
                     ("evidence/nested_topology/m4_holdout2.py", "bm_top.py",
                      "bm_real.py", "bm_experiments.py")},
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if len(states) == 2 and all(s.get("status") == "ok" for s in states):
        overall, why = merge_states(states[0]["verdict"], states[1]["verdict"])
    else:
        overall, why = "J3", "state_incomplete_or_blocked"

    res = dict(meta=meta, states=states, overall=dict(verdict=overall, reason=why))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, allow_nan=False)

    for s in states:
        if s.get("status") != "ok":
            print(f"{s['state']:<10} {s.get('status')}")
            continue
        print(f"{s['state']:<10} 栅格={s['grid_mode']:<14} "
              f"acf={[round(r['acf'], 3) for r in s['replicas']]} "
              f"极差={s['acf_range']:.3f}  r_min(主)={s['r_min_primary']:.3f} "
              f"r_min(副)={s['r_min_secondary']:.3f}  → {s['verdict']}（{s['reason']}）")
    print(f"\nE1 总判定：{overall}（{why}）")
    print("允许的结论见 " + PREREG + " §3/§5；J1≠测量链已证明，J2≠发现样品动态。")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
