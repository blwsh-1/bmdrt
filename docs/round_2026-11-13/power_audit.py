# -*- coding: utf-8 -*-
"""检出力审计（只读）：零分布 / 检出力 / 独立样本 / 可辨识性标签 / D_ab。

口径：`docs/round_2026-11-13/POWER_AUDIT_SCOPE.md`（先冻结）。
输入：`out/truth_recovery/results.json`、`out/truth_dataset/`。
输出：`out/power_audit/{audit.json,summary.md}`。

**不重拟合、不改判据、不改本轮判定、不给 grade/verdict/pass 键。**

    python docs/round_2026-11-13/power_audit.py
    python docs/round_2026-11-13/power_audit.py --boot 500 --quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "out" / "truth_dataset"
REC = ROOT / "out" / "truth_recovery"
OUT = ROOT / "out" / "power_audit"

sys.path.insert(0, str(ROOT / "docs" / "round_2026-11-12"))
import truth_recovery as TR  # noqa: E402  复用同一正演与 acf 口径

CAP = ("nested_L1_capacitive", "nested_L2_capacitive")
MAIN_KINDS = ("none", "heavy")
REF_PCT = 0.05


# ---------------------------------------------------------------- 小工具

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def acf_abs(d: np.ndarray) -> float:
    return float(max(TR.acf_max(d.real), TR.acf_max(d.imag)))


def acf_rel(d: np.ndarray, az: np.ndarray) -> float:
    return float(max(TR.acf_max(d.real / az), TR.acf_max(d.imag / az)))


def acf_many(x: np.ndarray, lags=(1, 2, 3, 4, 5)) -> np.ndarray:
    """批量 acf_max：x 形状 (B, n)，返回 (B,)。"""
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    dd = np.sum(x * x, axis=1)
    out = np.zeros(x.shape[0])
    for L in lags:
        num = np.sum(x[:, :-L] * x[:, L:], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dd > 0, np.abs(num / np.where(dd > 0, dd, 1.0)), 0.0)
        out = np.maximum(out, r)
    return out


def finite(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return finite(float(o)) if isinstance(o, np.floating) else int(o)
    if isinstance(o, float):
        return finite(o)
    return o


def load_dataset():
    man = json.loads((DATA / "truth_manifest.json").read_text(encoding="utf-8"))
    spec = {s["id"]: s for s in man["spectra"]}
    data = {}
    for sid, s in spec.items():
        a = np.genfromtxt(DATA / f"{sid}.csv", delimiter=",", names=True)
        data[sid] = (np.asarray(a["freq"], float),
                     np.asarray(a["Zreal"], float) + 1j * np.asarray(a["Zimag"], float))
    return man, spec, data


# ---------------------------------------------------------------- A0 独立样本

def audit_units(spec, results):
    shas = defaultdict(list)
    for sid, s in spec.items():
        shas[s.get("csv_sha256", "")].append(sid)
    dup = {k: v for k, v in shas.items() if len(v) > 1}

    def cfg(s):
        return (s["family"], s["difficulty"],
                hashlib.md5(json.dumps(s["truth_params"], sort_keys=True).encode()).hexdigest()[:8])

    units = defaultdict(list)
    for sid, s in spec.items():
        units[cfg(s)].append(sid)

    # 去重后按唯一谱重算 S 集比例
    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            sh = spec[r["id"]].get("csv_sha256")
            if sh in seen:
                continue
            seen.add(sh)
            out.append(r)
        return out

    S = [r for r in results if r["family"] in CAP and r["noise_kind"] in MAIN_KINDS]
    Sd = dedupe(S)
    by_kind = {}
    for kind in MAIN_KINDS:
        rows = [r for r in S if r["noise_kind"] == kind]
        rw = [r for r in Sd if r["noise_kind"] == kind]
        h1 = sum(1 for r in rows if _pair(r)[0]["acf"] > 0.50)
        h2 = sum(1 for r in rows if _pair(r)[0]["acf"] - _pair(r)[1]["acf"] >= 0.20)
        d1 = sum(1 for r in rw if _pair(r)[0]["acf"] > 0.50)
        d2 = sum(1 for r in rw if _pair(r)[0]["acf"] - _pair(r)[1]["acf"] >= 0.20)
        by_kind[kind] = dict(
            n_raw=len(rows), n_unique=len(rw),
            hit1_raw=h1, hit2_raw=h2, hit1_unique=d1, hit2_unique=d2,
        )

    tot1 = sum(1 for r in S if _pair(r)[0]["acf"] > 0.50)
    tot2 = sum(1 for r in S if _pair(r)[0]["acf"] - _pair(r)[1]["acf"] >= 0.20)
    u1 = sum(1 for r in Sd if _pair(r)[0]["acf"] > 0.50)
    u2 = sum(1 for r in Sd if _pair(r)[0]["acf"] - _pair(r)[1]["acf"] >= 0.20)
    return dict(
        n_files=len(spec), n_unique_sha=len(shas), n_dup_groups=len(dup),
        dup_kinds=dict(Counter(v[0].split("__")[-2] for v in dup.values())),
        n_units=len(units), unit_sizes=dict(Counter(len(v) for v in units.values())),
        S=dict(n=len(S), n_unique=len(Sd),
               hit1=tot1, hit2=tot2,
               hit1_unique=u1, hit2_unique=u2,
               rate1=wilson(tot1, len(S)), rate2=wilson(tot2, len(S)),
               rate1_unique=wilson(u1, len(Sd)), rate2_unique=wilson(u2, len(Sd))),
        by_kind=by_kind,
    )


def _pair(r):
    L = "L1" if "L1" in r["family"] else "L2"
    return r["candidates"][f"nested_{L}_inductive"], r["candidates"][r["family"]]


# ---------------------------------------------------------------- A1 零分布

def audit_null(spec, data, results, boot=2000, seed=20261113):
    rng = np.random.default_rng(seed)
    per = []
    for r in results:
        sid = r["id"]
        f, Z = data[sid]
        az = np.abs(Z)
        true_c = r["candidates"].get(r["family"])
        if true_c is None or not true_c.get("params"):
            continue
        d = TR.forward(r["family"], f, true_c["params"]) - Z
        obs = acf_abs(d)
        # 符号随机化：逐点保留 |d|，实部/虚部独立变号
        sr = rng.choice([-1.0, 1.0], size=(boot, d.size))
        si = rng.choice([-1.0, 1.0], size=(boot, d.size))
        a = acf_many(sr * d.real[None, :])
        b = acf_many(si * d.imag[None, :])
        null = np.maximum(a, b)
        p = float((np.sum(null >= obs) + 1) / (boot + 1))
        per.append(dict(id=sid, family=r["family"], difficulty=r["difficulty"],
                        noise=r["noise_kind"], obs=obs, p=p,
                        null_q50=float(np.quantile(null, 0.50)),
                        null_q95=float(np.quantile(null, 0.95)),
                        null_q99=float(np.quantile(null, 0.99)),
                        null_max=float(np.max(null)),
                        rwp=true_c["rwp_pct"]))
    buckets = defaultdict(list)
    for x in per:
        buckets[(x["noise"], "all")].append(x)
        buckets[(x["noise"], x["family"])].append(x)
    summary = {}
    for (kind, lv), rows in buckets.items():
        key = f"{kind}|{lv}"
        has_noise = kind != "none"     # none 档无噪声：替代数据零分布无效
        summary[key] = dict(
            n=len(rows), null_valid=has_noise,
            obs_q50=float(np.median([x["obs"] for x in rows])),
            obs_max=float(np.max([x["obs"] for x in rows])),
            null_q95_med=float(np.median([x["null_q95"] for x in rows])) if has_noise else None,
            null_q99_med=float(np.median([x["null_q99"] for x in rows])) if has_noise else None,
            null_max_med=float(np.median([x["null_max"] for x in rows])) if has_noise else None,
            frac_gt_050=float(np.mean([x["obs"] > 0.50 for x in rows])),
            frac_obs_gt_null95=(float(np.mean([x["obs"] > x["null_q95"] for x in rows]))
                                if has_noise else None),
            frac_p_lt_005=(float(np.mean([x["p"] < 0.05 for x in rows])) if has_noise else None),
        )
    return per, summary


# ---------------------------------------------------------------- A2 检出力

def audit_power(results):
    S = [r for r in results if r["family"] in CAP and r["noise_kind"] in MAIN_KINDS]
    rows = []
    for r in S:
        ind, cap = _pair(r)
        pct = 0.05 if r["noise_kind"] == "none" else float(r["noise_pct"])
        drwp = ind["rwp_pct"] - cap["rwp_pct"]
        rows.append(dict(id=r["id"], noise=r["noise_kind"], difficulty=r["difficulty"],
                         drwp=drwp, drwp_norm=drwp / (pct * 100.0),
                         acf_ind=ind["acf"], acf_cap=cap["acf"],
                         hit=int(ind["acf"] > 0.50),
                         hit2=int(ind["acf"] - cap["acf"] >= 0.20)))
    # 按噪声档分层（混合噪声档会让「偏置—检出」关系不可解释）
    tab = []
    for kind in MAIN_KINDS:
        sel = [x for x in rows if x["noise"] == kind]
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)):
            g = [x for x in sel if lo <= x["drwp_norm"] < hi]
            if not g:
                continue
            tab.append(dict(noise=kind, lo=lo, hi=(None if hi > 1e8 else hi), n=len(g),
                            hit1=sum(x["hit"] for x in g), hit2=sum(x["hit2"] for x in g),
                            ci1=wilson(sum(x["hit"] for x in g), len(g)),
                            med_acf=float(np.median([x["acf_ind"] for x in g]))))
    # 检出边界（不用 logistic：n=16 且 0.17–0.36 之间有间隙，logistic 不可识别）
    boundary = {}
    for kind in MAIN_KINDS:
        sel = [x for x in rows if x["noise"] == kind]
        miss = [x["drwp_norm"] for x in sel if not x["hit"]]
        hit = [x["drwp_norm"] for x in sel if x["hit"]]
        if miss and hit:
            boundary[kind] = dict(max_miss=max(miss), min_hit=min(hit),
                                  overlap=bool(max(miss) >= min(hit)))
    return rows, tab, boundary


# ---------------------------------------------------------------- A4 标签与 D_ab

def jacobian_log(name, f, p, h=1e-4):
    keys = TR.keys_for(name)
    z0 = TR.forward(name, f, p)
    cols = {}
    for k in keys:
        q = dict(p)
        if k.startswith("n"):
            q[k] = p[k] + h
            d = (TR.forward(name, f, q) - z0) / h
        else:
            q[k] = p[k] * math.exp(h)
            d = (TR.forward(name, f, q) - z0) / h
        cols[k] = np.concatenate([d.real, d.imag])
    return cols


def compensate_mismatch(name, f, p, scale=1.5, iters=4):
    """有限幅度退化检验：把 Q_b 乘 `scale`，看其余参数能否在线性化迭代下
    把阻抗**匹配回去**（profile 式补偿）。返回不可补偿的相对失配 rms。
    比 Jacobian 共线度可靠：共线度只看真值点切空间，不看好坏是否真能互抵。"""
    keys = [k for k in TR.keys_for(name) if k != "Q_b"]
    p_t = dict(p)
    p_t["Q_b"] = p["Q_b"] * scale
    Zt = TR.forward(name, f, p_t)
    az = np.abs(Zt)
    w = np.concatenate([1.0 / az, 1.0 / az])
    q = dict(p)
    for _ in range(iters):
        Zq = TR.forward(name, f, q)
        cols = jacobian_log(name, f, q)
        J = np.stack([cols[k] for k in keys], axis=1) * w[:, None]
        r = np.concatenate([(Zt - Zq).real, (Zt - Zq).imag]) * w
        d, *_ = np.linalg.lstsq(J, r, rcond=None)
        for k, dk in zip(keys, d):
            if k.startswith("n"):
                q[k] = min(0.999, max(0.30, q[k] + float(dk)))
            else:
                q[k] = q[k] * math.exp(float(dk))
    Zq = TR.forward(name, f, q)
    return float(np.sqrt(np.mean(np.abs(Zq - Zt) ** 2 / np.abs(Zt) ** 2))), q


def audit_labels(spec, data, results):
    """设计层标签 + 对照分解（几何代理，不是可辨识性证明）。"""
    out = []
    for r in results:
        if r["family"] not in CAP:
            continue
        sid = r["id"]
        s = spec[sid]
        p = dict(s["truth_params"])
        f, Z = data[sid]
        az = np.abs(Z)
        L = 1 if "L1" in r["family"] else 2
        RL = p[f"R{L}"]
        Qb, nb = p["Q_b"], p["n_b"]
        # 项目约定 Z_CPE = Q/(iω)^n → |Z_b| = Q/ω^n；交叉点 |Z_b| = R_L
        f0 = (Qb / RL) ** (1.0 / nb) / (2 * math.pi)
        pct = s["noise"]["pct"] or REF_PCT
        # 「去掉支路」反事实：Q_b→0 ⇒ Z_b→0（不是换成 R_ad∥L）
        z_full = TR.forward(r["family"], f, p)
        p_short = dict(p)
        p_short["Q_b"] = max(Qb * 1e-12, 1e-30)
        z_short = TR.forward(r["family"], f, p_short)
        eff = np.abs(z_full - z_short) / az
        effect_max = float(np.max(eff))
        cols = jacobian_log(r["family"], f, p)
        cq = cols["Q_b"]
        sens = float(np.sqrt(np.mean((np.abs(cq.reshape(2, -1)) / (pct * az)) ** 2)))
        others = np.stack([v for k, v in cols.items() if k != "Q_b"], axis=1)
        beta, *_ = np.linalg.lstsq(others, cq, rcond=None)
        resid = cq - others @ beta
        degen_lin = float(1.0 - (resid @ resid) / max(cq @ cq, 1e-300))
        mism, _ = compensate_mismatch(r["family"], f, p)
        window_ok = bool(f.min() <= f0 <= f.max())
        if (not window_ok) and effect_max < pct:
            lab = "non_identifiable_by_window"
        elif effect_max < pct:
            lab = "weakly_identifiable"
        elif mism < 0.5 * pct:
            lab = "non_identifiable_by_degeneracy"
        elif mism < pct:
            lab = "weakly_identifiable"
        else:
            lab = "identifiable_by_design"
        ind, cap = _pair(r)
        drwp = ind["rwp_pct"] - cap["rwp_pct"]
        out.append(dict(id=sid, family=r["family"], difficulty=r["difficulty"],
                        noise=r["noise_kind"], f0=f0, window_ok=window_ok,
                        effect_max=effect_max, effect_over_noise=effect_max / pct,
                        sens_snr=sens, degen_lin=degen_lin,
                        degen_mismatch=mism, degen_over_noise=mism / pct, label=lab,
                        drwp=drwp, control_ratio=(drwp / (effect_max * 100.0)
                                                  if effect_max > 0 else None),
                        acf_ind=ind["acf"], acf_cap=cap["acf"],
                        truth_selected=bool(r["truth_selected"]),
                        best=r["best_bic"]))
    for x in out:
        x["control_adds_element"] = bool(x["control_ratio"] and x["control_ratio"] > 2.0)
    return out


def audit_dab(spec, data, results):
    rows = []
    for r in results:
        f, Z = data[r["id"]]
        az = np.abs(Z)
        sig = REF_PCT * az
        conv = {k: v for k, v in r["candidates"].items()
                if v.get("params") and v.get("bic") is not None}
        zs = {k: TR.forward(k, f, v["params"]) for k, v in conv.items()}
        order = sorted(conv, key=lambda k: conv[k]["bic"])
        keys = sorted(zs)
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = zs[keys[i]] - zs[keys[j]]
                D = float(np.sum((np.abs(d) / sig) ** 2))
                pairs.append((keys[i], keys[j], D))

        def D_of(a, b):
            for x, y, D in pairs:
                if {x, y} == {a, b}:
                    return D
            return None

        chosen = r["best_bic"]
        tc = r["truth_candidate"]
        d_best_second = D_of(order[0], order[1]) if len(order) >= 2 else None
        rows.append(dict(id=r["id"], noise=r["noise_kind"], n_pairs=len(pairs),
                         frac_D_lt1=float(np.mean([D < 1 for _, _, D in pairs])),
                         min_D=float(np.min([D for _, _, D in pairs])) if pairs else None,
                         D_best_vs_second=d_best_second,
                         D_true_vs_chosen=(D_of(tc, chosen)
                                           if chosen != tc and D_of(tc, chosen) is not None
                                           else None),
                         truth_selected=bool(r["truth_selected"]),
                         chosen=chosen, truth=tc))
    return rows


# ---------------------------------------------------------------- 汇总输出

def fmt_ci(ci):
    if not ci or ci[0] is None:
        return "—"
    return f"{100*ci[0]:.1f}–{100*ci[1]:.1f}%"


def build_md(a0, a1s, a2tab, boundary, labels, dab):
    L = []
    w = L.append
    w("# 检出力审计（只读，不改本轮判定）\n")
    w("口径：`docs/round_2026-11-13/POWER_AUDIT_SCOPE.md`。只读 `results.json` 与数据集，"
      "**未重拟合、未改判据、未改 ①②③④ 判定**。\n")
    w("## A0 独立样本核算\n")
    w(f"- 谱文件 **{a0['n_files']}**，唯一 sha256 **{a0['n_unique_sha']}**，"
      f"重复组 **{a0['n_dup_groups']}**（按噪声档 {a0['dup_kinds']}）")
    w(f"- 独立真值配置 `(family, difficulty)` = **{a0['n_units']}** 个，"
      f"每个配置的重复实现条数分布 {a0['unit_sizes']}（3 噪声 × 2 seed = 6）")
    S = a0["S"]
    w(f"- 主集 S：按谱 n={S['n']}，去重唯一谱 n={S['n_unique']}")
    w(f"  - ① 按谱 **{S['hit1']}/{S['n']} = {100*S['hit1']/S['n']:.1f}%** "
      f"(95%CI {fmt_ci(S['rate1'])})｜去重 **{S['hit1_unique']}/{S['n_unique']} = "
      f"{100*S['hit1_unique']/S['n_unique']:.1f}%** (95%CI {fmt_ci(S['rate1_unique'])})")
    w(f"  - ② 按谱 **{S['hit2']}/{S['n']} = {100*S['hit2']/S['n']:.1f}%** "
      f"(95%CI {fmt_ci(S['rate2'])})｜去重 **{S['hit2_unique']}/{S['n_unique']} = "
      f"{100*S['hit2_unique']/S['n_unique']:.1f}%** (95%CI {fmt_ci(S['rate2_unique'])})")
    w("\n| 噪声档 | 谱 n | 唯一谱 n | ① 按谱 | ② 按谱 | ① 去重 | ② 去重 |")
    w("|---|---|---|---|---|---|---|")
    for k, v in a0["by_kind"].items():
        w(f"| {k} | {v['n_raw']} | {v['n_unique']} | {v['hit1_raw']}/{v['n_raw']} | "
          f"{v['hit2_raw']}/{v['n_raw']} | {v['hit1_unique']}/{v['n_unique']} | "
          f"{v['hit2_unique']}/{v['n_unique']} |")
    w("\n## A1 真族残差零分布（符号随机化替代数据）\n")
    w("| 噪声\\族 | n | 零分布有效 | 观测 acf 中位 | 观测上界 | 零 q95 中位 | 零 q99 中位 | obs>0.50 | obs>零 q95 | p<0.05 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(a1s.items()):
        q95 = "—" if v["null_q95_med"] is None else f"{v['null_q95_med']:.3f}"
        q99 = "—" if v["null_q99_med"] is None else f"{v['null_q99_med']:.3f}"
        f95 = "—" if v["frac_obs_gt_null95"] is None else f"{100*v['frac_obs_gt_null95']:.1f}%"
        fp = "—" if v["frac_p_lt_005"] is None else f"{100*v['frac_p_lt_005']:.1f}%"
        w(f"| {k} | {v['n']} | {'是' if v['null_valid'] else '**否（无噪声）**'} | "
          f"{v['obs_q50']:.3f} | {v['obs_max']:.3f} | {q95} | {q99} | "
          f"{100*v['frac_gt_050']:.1f}% | {f95} | {fp} |")
    w("\n> `none` 档没有噪声，残差是最小二乘投影残差（确定性），符号随机化零分布无意义，"
      "其 `obs>0.50` 不能读成假阳性。仅 `correlated` / `heavy` 两档的零分布可解释。")
    w("\n## A2 检出率—偏置关系（按噪声档分层，exploratory）\n")
    w("偏置归一化：`Δrwp / 噪声底`，噪声底 = 5%·100 = 5pp（`none` 按参考 5% 算）。\n")
    w("| 噪声 | Δrwp/噪声底 | n | ① 检出 | ① 率 (95%CI) | ② 检出 | 中位 acf_ind |")
    w("|---|---|---|---|---|---|---|")
    for t in a2tab:
        hi = "∞" if t["hi"] is None else f"{t['hi']:g}"
        w(f"| {t['noise']} | [{t['lo']:g}, {hi}) | {t['n']} | {t['hit1']} | "
          f"{100*t['hit1']/t['n']:.0f}% ({fmt_ci(t['ci1'])}) | {t['hit2']} | {t['med_acf']:.3f} |")
    if boundary:
        w("\n**检出边界（经验，不做外推）**：")
        for k, v in boundary.items():
            w(f"- `{k}`：未检出样本的 `Δrwp/噪声底` 最大 **{v['max_miss']:.2f}**，"
              f"检出样本最小 **{v['min_hit']:.2f}**"
              + ("（**区间重叠**，不能划单一边界）" if v["overlap"] else "（两类不重叠）"))
        w("  - 读数：这个边界是**观测到的事实**，不是建议阈值；下一轮要用受控扫描"
          "（系统改变支路幅度）才能把它变成可预注册的检出力曲线。")
    w("\n## A4 设计层标签 × 实测（几何代理）\n")
    lab_cnt = Counter(x["label"] for x in labels)
    w(f"标签分布：{dict(lab_cnt)}\n")
    w("| 标签 | n | 中位 f0 (Hz) | 窗口内 | 中位 effect/噪声 | 中位 degen 失配/噪声 | 中位 sens_snr | ① 命中 | 真族 BIC 选中 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lab in sorted(lab_cnt):
        sel = [x for x in labels if x["label"] == lab]
        w(f"| {lab} | {len(sel)} | {np.median([x['f0'] for x in sel]):.3g} | "
          f"{sum(1 for x in sel if x['window_ok'])}/{len(sel)} | "
          f"{np.median([x['effect_over_noise'] for x in sel]):.2f} | "
          f"{np.median([x['degen_over_noise'] for x in sel]):.2f} | "
          f"{np.median([x['sens_snr'] for x in sel]):.2f} | "
          f"{sum(1 for x in sel if x['acf_ind'] > 0.50)}/{len(sel)} | "
          f"{sum(1 for x in sel if x['truth_selected'])}/{len(sel)} |")
    w("\n### 标签能否预测失败（S 集 = none/heavy）\n")
    w("| 标签 | n | ① 未命中 | 真族 BIC 未中 |")
    w("|---|---|---|---|")
    for lab in sorted({x["label"] for x in labels}):
        sel = [x for x in labels if x["label"] == lab and x["noise"] in MAIN_KINDS]
        if not sel:
            continue
        w(f"| {lab} | {len(sel)} | {sum(1 for x in sel if x['acf_ind'] <= 0.50)} | "
          f"{sum(1 for x in sel if not x['truth_selected'])} |")
    w("\n### 对照分解：① 的对照换的是「另一种支路」不是「没有支路」\n")
    ca = [x for x in labels if x["control_adds_element"]]
    w(f"- `control_ratio = Δrwp/(effect_max×100) > 2` 的谱：**{len(ca)}/{len(labels)}**"
      f"（其中 S 集 {sum(1 for x in ca if x['noise'] in MAIN_KINDS)} 条）")
    if ca:
        w(f"  - 这些谱里 ① 命中 {sum(1 for x in ca if x['acf_ind'] > 0.50)}/{len(ca)}，"
          f"中位 Δrwp {np.median([x['drwp'] for x in ca]):.2f}pp、"
          f"中位 effect_max {100*np.median([x['effect_max'] for x in ca]):.2f}%")
        w("  - 读数：这类谱的残差结构大部分来自对照模型**多加的 `R_ad`**，"
          "不能当作「缺少容性支路」的证据。")
    else:
        w("  - 读数：本轮**没有**出现「对照失配远大于支路真实可见度」的谱，"
          "即 ① 的对照没有被 `R_ad` 主导 —— 这一条反事实混淆在本数据上不成立（如实记录，不夸大）。")
    w("\n## A4b 候选间可区分度 D_ab（σ=5%·|Z|）\n")
    mis = [x for x in dab if not x["truth_selected"] and x["D_true_vs_chosen"] is not None]
    ok = [x for x in dab if x["truth_selected"] and x["D_best_vs_second"] is not None]
    sep = [x for x in dab if x["D_best_vs_second"] is not None]
    w(f"- 全部谱：候选对中 `D<1`（该参考噪声下不可区分）占比中位 "
      f"**{100*np.median([x['frac_D_lt1'] for x in dab]):.1f}%**")
    w(f"- 真族被选中 n={len(ok)}，其 BIC 第一名 vs 第二名 `D` 中位 "
      f"**{np.median([x['D_best_vs_second'] for x in ok]):.3g}**")
    w(f"- 误选 n={len(mis)}，真族 vs 被选候选 `D` 中位 "
      f"**{np.median([x['D_true_vs_chosen'] for x in mis]):.3g}**，"
      f"其中 `D<1` 占 **{sum(1 for x in mis if x['D_true_vs_chosen'] < 1)}/{len(mis)}**")
    w("\n**BIC 第一名与第二名的可区分度**（同口径 `D`，看「选对了」是不是真有证据）：\n")
    w("| 门槛 | 全谱占比 | 其中选对 | 选对率 |")
    w("|---|---|---|---|")
    for thr in (1.0, 4.0, 25.0):
        g = [x for x in sep if x["D_best_vs_second"] > thr]
        c = sum(1 for x in g if x["truth_selected"])
        w(f"| `D>{thr:g}` | {len(g)}/{len(sep)} ({100*len(g)/len(sep):.1f}%) | {c}/{len(g) if g else 0} | "
          f"{(100*c/len(g)) if g else 0:.1f}% |")
    w("\n> `D` 由两侧**各自的最优拟合**算出：两个候选都能在噪声内解释数据时 `D≈0`。"
      "所以「BIC 选对」不等于「数据支持这个拓扑」——多数选对发生在 `D≪1`，"
      "即第一名与第二名在该噪声下不可区分。这是本轮最需要写进下一轮判据的事实。")
    w("\n## 诚实边界\n")
    w("1. 零分布用**真族在本谱上的残差**做符号随机化，保留逐点幅度（含异方差与重尾），"
      "只打乱频率顺序结构；标定的是「无序号结构」的零，不是「模型错」的备择。")
    w("2. `obs>零分布 q95` 的比例**不是新判据**，不得读作 ①② 满足；新判据只能写进下一轮预注册。")
    w("3. 各档 n 很小（heavy 16，窗内 12），比例为粗粒度；Wilson CI 已给，"
      "32 条无法分辨 75% 与 72%。")
    w("4. 设计标签是几何代理（交叉频率/可见度/灵敏度/线性退化），不是可辨识性证明；"
      "若标签预测不了失败，说明标签定义要改，不得反过来当结论。")
    w("5. `D_ab` 用固定 5% 参考噪声，跨档可比，但不是该谱实际噪声；不作门禁。")
    w("6. 估计器敏感性（Gaussian LS vs Student-t 似然）**本轮未做**，需重拟合，另立一轮。")
    w("7. 首版脚本把 CPE 约定写成 `1/(Qω^n)`，导致 `f0` 全部落在 1e-13 Hz、标签退化；"
      "已按 `Z = Q/(iω)^n` 订正并重跑。修正记录见本节末尾。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    man, spec, data = load_dataset()
    rec = json.loads((REC / "results.json").read_text(encoding="utf-8"))
    results = rec["results"]

    # ---- repro 自检：存盘参数必须能重建残差指标
    bad = 0
    for r in results:
        f, Z = data[r["id"]]
        for cname, c in r["candidates"].items():
            if not c.get("params"):
                continue
            d = TR.forward(cname, f, c["params"]) - Z
            ac = acf_abs(d)
            rwp = float(np.sqrt(np.mean(np.abs(d) ** 2 / np.abs(Z) ** 2)) * 100)
            if abs(ac - c["acf"]) > 1e-6 or abs(rwp - c["rwp_pct"]) > 1e-6:
                bad += 1
    print(f"[repro] 不一致 {bad} 条")
    if bad:
        raise SystemExit("残差无法从存盘参数精确重建，停止（不得用重建残差做零分布）")

    a0 = audit_units(spec, results)
    per, a1s = audit_null(spec, data, results, boot=args.boot)
    a2rows, a2tab, boundary = audit_power(results)
    labels = audit_labels(spec, data, results)
    dab = audit_dab(spec, data, results)

    payload = jsonable(dict(
        generated_by="docs/round_2026-11-13/power_audit.py",
        repro_mismatch=bad, boot=args.boot,
        a0_units=a0, a1_null=dict(summary=a1s, per_spectrum=per),
        a2_power=dict(rows=a2rows, bins=a2tab, boundary=boundary),
        a4_labels=labels, a4_dab=dab,
    ))
    # 显式 newline='\n'：write_text 默认走 os.linesep，Windows 上会把 md 写成 CRLF
    with open(outdir / "audit.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    with open(outdir / "summary.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_md(a0, a1s, a2tab, boundary, labels, dab))
    print(f"[ok] -> {outdir/'summary.md'}")


if __name__ == "__main__":
    main()
