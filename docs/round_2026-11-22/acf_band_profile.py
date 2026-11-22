"""acf_band_profile.py —— M-b 的 `acf` 门槛是被频率轴的哪一段顶住的（只读，零 MCMC）。

问题
----
§14 全量 14 条（F1 口径）：M-a 9/14、M-b 4/14、M-c 12/14，全过 4/14。
`acf = max(acf_max(Re res), acf_max(Im res))` 是一个**标量**，它不告诉你结构
落在频率轴的哪一段。如果 4/14 那条线其实是被端部少数几个点顶住的，那这个判据
测的就不是"模型表达力"，而是"端部数据质量"——两者要采取的行动完全不同。

做法
----
只读 `evidence/m6_f1_2026-11-22/m4_holdout2_f1.json` 里已落盘的 `z_fit_f1`
（F1 口径，长度 = `n_kept`），配 `cut_top` 重建的 `Z_obs`，做三类扫描：

1. **端部截断**：去掉最低频 k 点 / 最高频 k 点（k = 5/10/15），重算 acf；
2. **连续子窗**：窗宽 w = 40/30/20 滑动，记录窗内最大 acf 及其频段；
3. **同号游程**：每个通道最长连续同号的长度与位置（"成片同号"的量化）。

全部用**连续真实点**，不做"删中间段再拼接"——拼接会人为造出一个 lag 跳变，
把结果变成算法伪迹。

口径边界（必须写明）
--------------------
`acf_max` 是去均值后 lag 1–5 的最大 |ρ|，`dd` 用全段归一化。所以子窗结果是
**重算出来的新统计量**，不是原标量的可加分解。正确表述是"若只看这一段，
acf 是多少"，不能说"某段贡献了多少 acf"。

用法
----
    python docs/round_2026-11-22/acf_band_profile.py
    python docs/round_2026-11-22/acf_band_profile.py --json out/acf_band_profile.json
"""

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))

import bm_experiments as E          # noqa: E402
import bm_real as BR                # noqa: E402
import m4_holdout2 as M14           # noqa: E402  只用 SPEC_DIR / cut_top

SRC = os.path.join("evidence", "m6_f1_2026-11-22", "m4_holdout2_f1.json")
PASSED = ("AZ31-8Li_10", "AZ31-8Li_50", "AZ31-12Li_10", "AZ31-12Li_50")
ACF_MAX = 0.50


def acf_chan(x):
    """(acf, 主导 lag)。lag 与 `E.acf_max` 同口径（1–5）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    dd = float(np.dot(x, x))
    if dd <= 0:
        return 0.0, 0
    vals = [(abs(float(np.dot(x[:-L], x[L:])) / dd), L)
            for L in (1, 2, 3, 4, 5) if L < x.size]
    return max(vals)


def longest_run(x):
    """最长连续同号：返回 (长度, 起点下标, 终点下标, 符号)。"""
    s = np.sign(np.asarray(x, float))
    s[s == 0] = 1.0
    best, bi, bj = 1, 0, 0
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j - i + 1 > best:
            best, bi, bj = j - i + 1, i, j
        i = j + 1
    return best, bi, bj, float(s[bi])


def load_case(spec):
    """按 §14 的剔点规则重建 (f, Z_obs)。加载器会打印逐文件信息，这里静音。"""
    with contextlib.redirect_stdout(io.StringIO()):
        f_all, Z_all = BR.load_spectrum(os.path.join(REPO, M14.SPEC_DIR,
                                                     spec + ".csv"))
        keep = M14.cut_top(f_all, Z_all)["keep"]
    return f_all[keep], Z_all[keep]


def profile(f, Z, zre, zim):
    """单条谱的三类扫描。"""
    Zm = np.asarray(zre, float) + 1j * np.asarray(zim, float)
    res = Zm - Z
    n = f.size
    out = {"n": n, "f_lo": float(f[0]), "f_hi": float(f[-1]),
           "chan": {}, "trunc": {}, "window": {}}

    for name, x in (("re", res.real), ("im", res.imag)):
        a, lag = acf_chan(x)
        run, r0, r1, sg = longest_run(x)
        out["chan"][name] = {
            "acf": a, "lag": lag,
            "run_len": int(run), "run_f": [float(f[r0]), float(f[r1])],
            "run_sign": sg,
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "first5_mean": float(np.mean(x[:5])),
            "last5_mean": float(np.mean(x[-5:])),
        }
        # 端部截断
        tr = {}
        for k in (5, 10, 15):
            if n - k < 8:
                continue
            tr[f"drop_low{k}"] = acf_chan(x[k:])[0]
            tr[f"drop_high{k}"] = acf_chan(x[:-k])[0]
        out["trunc"][name] = tr
        # 连续子窗
        wd = {}
        for w in (40, 30, 20):
            if n < w:
                continue
            best = (-1.0, 0)
            for s0 in range(0, n - w + 1):
                a2, _ = acf_chan(x[s0:s0 + w])
                if a2 > best[0]:
                    best = (a2, s0)
            wd[f"w{w}"] = {"acf": best[0], "f": [float(f[best[1]]),
                                                 float(f[best[1] + w - 1])]}
        out["window"][name] = wd

    lead = "re" if out["chan"]["re"]["acf"] >= out["chan"]["im"]["acf"] else "im"
    out["lead"] = lead
    out["acf"] = out["chan"][lead]["acf"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    d = json.load(open(os.path.join(REPO, SRC), encoding="utf-8"))
    rows = []
    bad = 0
    for r in d["rows"]:
        spec = r["spec"]
        row = r["rows"][0]
        f, Z = load_case(spec)
        zf1 = row["z_fit_f1"]
        if len(zf1["re"]) != len(Z):
            print(f"  ! {spec}: z_fit {len(zf1['re'])} ≠ Z_obs {len(Z)}，跳过")
            bad += 1
            continue
        p = profile(f, Z, zf1["re"], zf1["im"])
        # 强校验：必须重现 JSON 里的 acf
        want = row["metrics_f1"]["acf"]
        if abs(p["acf"] - want) > 1e-6:
            print(f"  ! {spec}: 重算 acf {p['acf']:.6f} ≠ 产物 {want:.6f}，跳过")
            bad += 1
            continue
        p.update(spec=spec, acf_json=want, passed=spec in PASSED,
                 rms_f1=row["metrics_f1"]["rms_rel"])
        rows.append(p)

    print(f"校验：{len(rows)}/{len(rows) + bad} 条谱的 acf 由 z_fit_f1 精确复现"
          f"（容差 1e-6）")
    if bad:
        print(f"  {bad} 条未通过，已排除")
    print()

    # ---------- 表 1：主导通道 + 端部截断 ----------
    print("表 1  端部截断：去掉最低频 / 最高频 k 点后的 acf 与变化量 Δ")
    print("      Δ>0 = 去掉这段后**更差**（说明这段原本在稀释 lag 相关）")
    print("      Δ<0 = 去掉这段后**更好**（说明这段原本在支撑结构）")
    print(f"{'谱':<13}{'acf':>6}{'道':>3}{'lag':>4}"
          f"{'去低5':>8}{'Δ5':>7}{'去低15':>8}{'Δ15':>7}"
          f"{'去高15':>8}{'Δ':>7}")
    for r in sorted(rows, key=lambda x: x["acf"]):
        c = r["chan"][r["lead"]]
        t = r["trunc"][r["lead"]]
        base = r["acf"]

        def cell(key):
            v = t.get(key, float("nan"))
            dv = v - base
            s = f"{v:.3f}" + ("↓" if v <= ACF_MAX < base else "")
            return f"{s:>8}", f"{dv:+.3f}"

        a5, d5 = cell("drop_low5")
        a15, d15 = cell("drop_low15")
        h15, dh = cell("drop_high15")
        mark = "*" if r["passed"] else " "
        print(f"{r['spec'].replace('AZ31-',''):<13}{base:>6.3f}{r['lead']:>3}"
              f"{c['lag']:>4}{a5}{d5:>7}{a15}{d15:>7}{h15}{dh:>7}  {mark}")
    print("  （↓ = 该截断后已降到 0.50 以下；* = §14 判定通过）")
    print()

    # ---------- 表 2：连续子窗 ----------
    print("表 2  连续子窗最大 acf（窗宽 40/30/20，起点滑动）")
    print("      警告：窗口变短会系统性抬高 acf（局部均值≠全局均值），")
    print("      通过谱同样升到 0.8–0.9 ⇒ **此表不可用于比较谱与谱的优劣**，")
    print("      只用来确认结构不是集中在某个短窗里。")
    print(f"{'谱':<13}{'全窗':>7}{'w40':>7}{'w30':>7}{'w20':>7}   w20 频段 Hz")
    for r in sorted(rows, key=lambda x: x["acf"]):
        w = r["window"][r["lead"]]
        g = lambda k: w.get(k, {}).get("acf", float("nan"))
        fr = w.get("w20", {}).get("f")
        fs = f"{fr[0]:.2g}–{fr[1]:.2g}" if fr else "-"
        mark = "*" if r["passed"] else " "
        print(f"{r['spec'].replace('AZ31-',''):<13}{r['acf']:>7.3f}"
              f"{g('w40'):>7.3f}{g('w30'):>7.3f}{g('w20'):>7.3f}   {fs:<16}{mark}")
    print()

    # ---------- 表 3：同号游程 ----------
    print("表 3  最长连续同号段（成片同号的直接量化）")
    print(f"{'谱':<13}{'通道':>5}{'游程':>6}{'占 n':>7}   {'频段 Hz':<20}符号")
    for r in sorted(rows, key=lambda x: x["acf"]):
        for nm in ("re", "im"):
            c = r["chan"][nm]
            if c["run_len"] < 5:
                continue
            pct = 100 * c["run_len"] / r["n"]
            fr = c["run_f"]
            mark = "*" if r["passed"] else " "
            print(f"{r['spec'].replace('AZ31-',''):<13}{nm:>5}{c['run_len']:>6}"
                  f"{pct:>6.0f}%   {fr[0]:>8.2g}–{fr[1]:<9.2g} "
                  f"{'+' if c['run_sign'] > 0 else '-'}  {mark}")
    print()

    # ---------- 汇总 ----------
    print("=" * 78)
    fail = [r for r in rows if not r["passed"]]
    ok = [r for r in rows if r["passed"]]

    def dl(r, k):
        return r["trunc"][r["lead"]].get(k, float("nan")) - r["acf"]

    print("① 低频端截断的方向（主观察）")
    for k, lab in (("drop_low5", "5"), ("drop_low15", "15")):
        o = [dl(r, k) for r in ok]
        f = [dl(r, k) for r in fail]
        print(f"   去最低频 {lab:>2} 点：通过 {sum(v > 0 for v in o)}/{len(o)} 为**正**"
              f"（{'/'.join(f'{v:+.3f}' for v in o)}）")
        print(f"   {' ' * 12}失败 {sum(v > 0 for v in f)}/{len(f)} 为正，"
              f"中位 {np.median(f):+.3f}")
    n5 = sum(1 for r in fail if r["trunc"][r["lead"]].get("drop_low5", 9) <= ACF_MAX)
    n15 = sum(1 for r in fail if r["trunc"][r["lead"]].get("drop_low15", 9) <= ACF_MAX)
    print(f"   ⇒ 失败谱去 5 点达标 {n5}/{len(fail)}、去 15 点（30% 数据）达标 "
          f"{n15}/{len(fail)}")
    print()

    print("② 低频残差的形状：Δ(去低15) − Δ(去低5)　正=先回摆后延伸，负=单调")
    sh = {r["spec"]: dl(r, "drop_low15") - dl(r, "drop_low5") for r in rows}
    o = [sh[r["spec"]] for r in ok]
    f = [sh[r["spec"]] for r in fail]
    print(f"   通过 {len(ok)} 条中位 {np.median(o):+.3f}"
          f"（{'/'.join(f'{v:+.2f}' for v in sorted(o))}）")
    print(f"   失败 {len(fail)} 条中位 {np.median(f):+.3f}"
          f"（{sum(v > 0 for v in f)}/{len(f)} 为正）")
    print()

    n_hi = sum(1 for r in fail
               if min(r["trunc"][r["lead"]].get(f"drop_high{k}", 9)
                      for k in (5, 10, 15)) <= ACF_MAX)
    hi = [dl(r, "drop_high15") for r in rows]
    print(f"③ 高频端：全部 14 条的 Δ(去高15) 落在 "
          f"{min(hi):+.3f}~{max(hi):+.3f}，去高频达标 {n_hi}/{len(fail)}"
          f" ⇒ 结构**不在高频**")
    print()
    w20_ok = [r["window"][r["lead"]].get("w20", {}).get("acf", 0) for r in ok]
    w20_f = [r["window"][r["lead"]].get("w20", {}).get("acf", 0) for r in fail]
    print(f"④ 子窗不可判别：w20 中位 通过 {np.median(w20_ok):.3f} / "
          f"失败 {np.median(w20_f):.3f}"
          f"（通过谱也 {min(w20_ok):.2f}–{max(w20_ok):.2f}）")
    runs = [max(r["chan"]["re"]["run_len"], r["chan"]["im"]["run_len"]) / r["n"]
            for r in rows]
    print(f"⑤ 最长同号游程占全窗：中位 {100 * np.median(runs):.0f}%"
          f"、最大 {100 * max(runs):.0f}%（n=50 白噪声期望约 12%）")
    print("=" * 78)

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"src": SRC, "acf_max": ACF_MAX, "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\n已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
