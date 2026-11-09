# -*- coding: utf-8 -*-
"""N1 影子实现：把 L1 的幅度判据从 `log|Z|` 换成 `log(-Im)`，**只改这一处**。

为什么不直接改 `bm_kk.py`
--------------------------
`evidence/m6/run_e2.py` 是**逐 job `subprocess.run`** 起新进程的，新进程会重新
import `bm_kk`。E2 还在跑（30 个 job），此时改生产文件会直接改变剩余 job 的
行为——那是"影响正在跑的"。所以本脚本：

  1. 在内存里复制一份 `diffusion_tail_candidate`，只换幅度判据的量；
  2. 在**已知真值**的合成谱与样例谱上对照旧/新，量化改动方向；
  3. 把改动写成 `patches/n1_l1_amplitude.patch`（对 `bm_kk.py` 副本做 diff），
     **不应用**。等 E2 跑完、用户批准后再打。

改动内容（三处，机械性）
------------------------
    1. 幅度判据的量：`log10|Z| ~ log10 f` → `log10(-Im Z) ~ log10 f`
    2. `-Im ≤ 0` 时早退（感抗/贴实轴，取对数无意义），`f_cut` 保持 None
    3. `slope_z` **仍存 |Z| 口径**（`kk_check` 用它填 `tail_slope`），
       新增 `slope_im` 存新口径

`slope_lo/slope_hi`、`ny_tol`、`frac`、`min_keep`、相位判据全部不动。

**这条改动有真实的下游行为影响，不只是改注释**：`kk_check` 用
`tail["f_cut"]` 决定要不要跑 `add_cap` 复验；`amp_ok` 由 False 变 True 会让
`f_cut` 从 None 变成有值，从而打开那条路径。所以"改了就等价于没改"是错的。

跑出来的结论要分三句说，不能合成一句
------------------------------------
  · **修掉 Ws 漏报**：`caseE` / `caseE_warburg` 的 `is_candidate` 由 False 变
    True（`slope_im` −0.49/−0.45，`|Z|` 只有 −0.03/−0.02）；
  · **代价**：`rq05_low`（窗外 `R||CPE(0.5)`）触发数 7/20 → 18/20。这是
    §5.1 的同形，不是实现 bug；
  · **真 Wo 的两条 `amp_ok` 反而翻成 False**（`caseF_wo`/`caseG`：Wo 低频
    `-Im ∝ f⁻¹`，斜率 −0.97 落在为 `Ws` 设的 `[-0.85,-0.20]` 之外）。它们的
    `is_candidate` 本来就是 False（相位判据挡住），所以**最终标志没变**，
    但 `why` 的解释文字会变。若将来要让 L1 管 Wo，带宽必须按端点走向分设。
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:/eis训练数据/bmdrt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evidence" / "ws_crit"))

import bm_kk
import bench

OUT = ROOT / "evidence" / "ws_crit" / "n1_shadow.json"
PATCH = ROOT / "evidence" / "ws_crit" / "patches"

#: 只有这三条在 `DIFFUSION_PLAN.md` §9 有登记真值；其余是 `unknown`
TRUTH = {
    "example_caseE_warburg.csv": "Ws(A_W=2)",
    "example_caseF_wo.csv": "Wo(R_W=30, τ_D=0.3s)",
    "example_caseG_wo_only.csv": "纯 Wo(R_W=30, τ_D=0.3s)",
}
#: 未登记，用 `n1_fitcasee.py` 的拟合测到的拓扑（不写进真值表）
DERIVED = {
    "example_caseE.csv": "拟合 A_W=2.03；与 caseE_warburg 同拓扑，窗口 0.01–1e5",
    "example_caseF.csv": "拟合无 ω^-1/2 项（低频 -Im→0）",
}

HUNK1_OLD = """    idx = np.argsort(f)[:n_low]
    lf, lz = np.log10(np.maximum(f[idx], EPS)), np.log10(np.maximum(np.abs(Z[idx]), EPS))
    try:
        slope = float(np.polyfit(lf, lz, 1)[0])
    except Exception:                          # noqa: BLE001
        return out"""

HUNK1_NEW = """    idx = np.argsort(f)[:n_low]
    lf = np.log10(np.maximum(f[idx], EPS))
    try:
        # slope_z 保持 |Z| 口径：`kk_check` 用它填 `tail_slope`，即使幅度判据
        # 不成立也要照着报（旧实现就是这么做的，别让它变成 None）。
        slope_z = float(np.polyfit(lf, np.log10(np.maximum(np.abs(Z[idx]), EPS)), 1)[0])
    except Exception:                          # noqa: BLE001
        return out
    # --- N1：幅度判据改用 -Im。R_inf 只加在实部上，log|Z| 会被它压平。
    # 实测 caseE / caseE_warburg：|Z| 斜率 -0.03/-0.02（判不成），
    # -Im 斜率 -0.49/-0.45。
    im_low = -Z[idx].imag
    if im_low.min() <= 0:
        ny, span, stab = nyquist_slope_low(f, Z, frac)
        out.update(slope_z=slope_z, slope_ny=ny, ny_span=span,
                   ny_stability=stab,
                   phase_ok=bool(ny is not None and abs(ny - 1.0) <= ny_tol))
        out["why"].append("幅度判据不可用：低频段 -Im ≤ 0（感抗或贴实轴），"
                          "取对数无意义")
        return out
    try:
        slope = float(np.polyfit(lf, np.log10(im_low), 1)[0])
    except Exception:                          # noqa: BLE001
        return out"""

HUNK2_OLD = "    out.update(slope_z=slope, slope_ny=ny, ny_span=span, ny_stability=stab)"
HUNK2_NEW = ("    out.update(slope_z=slope_z, slope_im=slope, slope_ny=ny,\n"
             "               ny_span=span, ny_stability=stab)")

HUNK3_OLD = "    out = dict(f_cut=None, slope_z=None, slope_ny=None, ny_span=None,"
HUNK3_NEW = ("    out = dict(f_cut=None, slope_z=None, slope_im=None,\n"
             "               slope_ny=None, ny_span=None,")


def candidate_new(f, Z, frac: float = 0.25, min_keep: int = 16,
                  slope_lo: float = -0.85, slope_hi: float = -0.20,
                  ny_tol: float = 0.35):
    """与 `bm_kk.diffusion_tail_candidate` 逐行相同，只换幅度判据的量。"""
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    out = dict(f_cut=None, slope_z=None, slope_im=None, slope_ny=None,
               ny_span=None, ny_stability=None, amp_ok=False, phase_ok=False,
               is_candidate=False, why=[])
    n = f.size
    n_low = max(4, int(round(n * frac)))
    if n - n_low < min_keep:
        out["why"].append(f"点数不足（{n} 点，最低段 {n_low} 点），不做扩散尾判断")
        return out

    idx = np.argsort(f)[:n_low]
    lf = np.log10(np.maximum(f[idx], 1e-300))
    try:
        out["slope_z"] = float(np.polyfit(
            lf, np.log10(np.maximum(np.abs(Z[idx]), 1e-300)), 1)[0])
    except Exception:                                    # noqa: BLE001
        return out

    # ---- 唯一改动：幅度判据改用 -Im ----
    im = -Z[idx].imag
    ny, span, stab = bm_kk.nyquist_slope_low(f, Z, frac)
    if im.min() <= 0:
        out.update(slope_ny=ny, ny_span=span, ny_stability=stab,
                   phase_ok=bool(ny is not None and abs(ny - 1.0) <= ny_tol))
        out["why"].append("幅度判据不可用：低频段 -Im ≤ 0（感抗或贴实轴）")
        return out

    slope = float(np.polyfit(lf, np.log10(im), 1)[0])
    out.update(slope_im=slope, slope_ny=ny, ny_span=span, ny_stability=stab)
    amp_ok = bool(slope_lo <= slope <= slope_hi)
    pha_ok = bool(ny is not None and abs(ny - 1.0) <= ny_tol)
    out.update(amp_ok=amp_ok, phase_ok=pha_ok)
    if amp_ok:
        out["f_cut"] = float(f[idx][-1])
    ny_txt = f"{ny:+.2f}" if ny is not None else "无定义"
    if amp_ok and pha_ok:
        out.update(is_candidate=True,
                   why=[f"log(-Im) 斜率 {slope:+.2f} 且 Nyquist 斜率 {ny_txt}"])
    elif amp_ok:
        out["why"].append(f"log(-Im) 斜率 {slope:+.2f} 在区间内，"
                          f"但 Nyquist 斜率 {ny_txt} 偏离 45°")
    elif pha_ok:
        out["why"].append(f"Nyquist 斜率 {ny_txt} 像 45°，但 log(-Im) 斜率 "
                          f"{slope:+.2f} 不在 [{slope_lo}, {slope_hi}]")
    else:
        out["why"].append(f"log(-Im) 斜率 {slope:+.2f}、两判据都不成立")
    return out


def load(fn):
    d = np.genfromtxt(ROOT / "examples" / fn, delimiter=",", names=True)
    nm = d.dtype.names
    return np.asarray(d[nm[0]], float), np.asarray(d[nm[1]], float) \
        + 1j * np.asarray(d[nm[2]], float)


def write_patch():
    """对副本做 diff 生成补丁，**不碰 bm_kk.py**。"""
    PATCH.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "bm_kk.py").read_text(encoding="utf-8")
    for tag, o in (("1", HUNK1_OLD), ("2", HUNK2_OLD), ("3", HUNK3_OLD)):
        if src.count(o) != 1:
            print(f"\n[警告] hunk {tag} 在 bm_kk.py 里出现 {src.count(o)} 次"
                  "（应为 1），未生成补丁。请人工核对。")
            return
    new = (src.replace(HUNK1_OLD, HUNK1_NEW)
              .replace(HUNK2_OLD, HUNK2_NEW)
              .replace(HUNK3_OLD, HUNK3_NEW))
    tmp = PATCH / "bm_kk_n1_tmp.py"
    tmp.write_text(new, encoding="utf-8", newline="")
    pf = PATCH / "n1_l1_amplitude.patch"
    # 必须 newline=""：否则 Windows 上 write_text 会把 `\n` 翻成 `\r\n`，
    # 补丁的每一行都会带 `\r`，`git apply` 报 "patch does not apply"。
    with open(pf, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(difflib.unified_diff(
            src.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/bm_kk.py", tofile="b/bm_kk.py", n=3)))
    print(f"\n补丁已生成（**未应用**）：{pf}")
    print(f"  副本可用于回归：{tmp}")
    print(f"  E2 跑完、用户批准后再打： git apply \"{pf}\"")


def main():
    rows = {"examples": [], "real": [], "synthetic": []}

    print("===== A. examples/（登记真值 + 拟合派生）=====")
    print(f"{'file':30s} {'旧amp':>6s} {'旧cand':>7s} {'新amp':>6s} {'新cand':>7s} | "
          f"{'slope_z':>8s} {'slope_im':>9s}  真值")
    for fn in ["example_caseE.csv", "example_caseE_warburg.csv",
               "example_caseF_wo.csv", "example_caseF.csv",
               "example_caseG_wo_only.csv"]:
        f, Z = load(fn)
        old = bm_kk.diffusion_tail_candidate(f, Z)
        new = candidate_new(f, Z)
        tag = TRUTH.get(fn) or f"[未登记] {DERIVED.get(fn, '')}"
        rows["examples"].append(dict(
            file=fn, truth_registered=TRUTH.get(fn), truth_derived=DERIVED.get(fn),
            old_amp=old["amp_ok"], old_candidate=old["is_candidate"],
            old_f_cut=old["f_cut"], new_amp=new["amp_ok"],
            new_candidate=new["is_candidate"], new_f_cut=new["f_cut"],
            slope_z=new["slope_z"], slope_im=new["slope_im"],
            phase_ok=new["phase_ok"]))
        sz = "—" if new["slope_z"] is None else f"{new['slope_z']:+.2f}"
        si = "—" if new["slope_im"] is None else f"{new['slope_im']:+.2f}"
        print(f"{fn:30s} {str(old['amp_ok']):>6s} {str(old['is_candidate']):>7s} "
              f"{str(new['amp_ok']):>6s} {str(new['is_candidate']):>7s} | "
              f"{sz:>8s} {si:>9s}  {tag}")

    print("\n===== B. 18 条真实谱（truth=unknown，只登记触发变化）=====")
    src = ROOT / "out" / "_m6_samples" / "z_csv"
    n_old = n_new = n_fcut = 0
    for p in sorted(src.glob("*.csv")):
        d = np.genfromtxt(p, delimiter=",", names=True)
        nm = d.dtype.names
        f = np.asarray(d[nm[0]], float)
        Z = np.asarray(d[nm[1]], float) + 1j * np.asarray(d[nm[2]], float)
        old = bm_kk.diffusion_tail_candidate(f, Z)
        new = candidate_new(f, Z)
        n_old += bool(old["is_candidate"])
        n_new += bool(new["is_candidate"])
        n_fcut += bool(old["f_cut"] is None and new["f_cut"] is not None)
        rows["real"].append(dict(name=p.stem, old=bool(old["is_candidate"]),
                                 new=bool(new["is_candidate"]),
                                 old_amp=bool(old["amp_ok"]),
                                 new_amp=bool(new["amp_ok"]),
                                 slope_z=new["slope_z"], slope_im=new["slope_im"]))
    print(f"  旧 candidate {n_old}/18，新 candidate {n_new}/18；"
          f"`f_cut` 由 None 变有值 {n_fcut}/18（会打开 kk_check 的 add_cap 复验）")

    print("\n===== C. 合成谱（构造真值，看触发方向）=====")
    f = np.logspace(np.log10(0.1), np.log10(1e3), 31)
    rng = np.random.default_rng(0)
    print(f"{'class':11s} {'旧cand':>8s} {'新cand':>8s}   含义")
    for cls in bench.CLASSES:
        o = n = 0
        for _ in range(20):
            Z = bench.make(f, cls, rng)
            sd = 0.01 * np.abs(Z)
            Zn = Z + sd * (rng.standard_normal(f.size)
                           + 1j * rng.standard_normal(f.size))
            o += bool(bm_kk.diffusion_tail_candidate(f, Zn)["is_candidate"])
            n += bool(candidate_new(f, Zn)["is_candidate"])
        rows["synthetic"].append(dict(cls=cls, old=o, new=n))
        pos = cls in ("ws_vis", "ws_buried", "wo_in", "wo_below", "wshort_in")
        print(f"{cls:11s} {o:8d} {n:8d}   {'正例：越高越好' if pos else '阴例：越低越好'}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    write_patch()
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
