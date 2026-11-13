# -*- coding: utf-8 -*-
"""检出力审计（只读）：零分布 / 检出力 / 独立样本 / 可辨识性标签 / D_ab。

口径：`docs/round_2026-11-13/POWER_AUDIT_SCOPE.md`（先冻结）。
输入：`out/truth_recovery/results.json`、`out/truth_dataset/`。
输出：`out/power_audit/{audit.json,summary.md}`。

**不重拟合、不改判据、不改本轮判定、不给 grade/verdict/pass 键。**

    python docs/round_2026-11-13/power_audit.py                     # 全量，B=2000
    python docs/round_2026-11-13/power_audit.py --boot 500 --out out/power_audit_try
    python docs/round_2026-11-13/power_audit.py --selftest          # 只跑内置对拍

依赖清单见 `docs/round_2026-11-13/REPRO_MANIFEST.md`（上游 commit / 文件 sha256 / seed）。
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


def cluster_bootstrap(per_unit, boot=2000, seed=20261113):
    """按 unit（真值配置）重采样的 clustered bootstrap。

    `per_unit` = [(hits, n), ...]，每个 unit 一项。重采样**整个 unit**（把同一配置下
    的噪声实现一起带走），因此保留配置内相关性 —— 与「按谱独立」的 Wilson 区间不是一回事。
    返回 pooled 比例的分位区间；n 很小时区间会很宽，这是分布本身宽，不是算错。
    """
    if not per_unit:
        return None
    h = np.array([float(a) for a, _ in per_unit])
    n = np.array([float(b) for _, b in per_unit])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(h), size=(boot, len(h)))
    hb = h[idx].sum(axis=1)
    nb = n[idx].sum(axis=1)
    r = np.where(nb > 0, hb / np.maximum(nb, 1e-12), np.nan)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return None
    return dict(lo=float(np.quantile(r, 0.025)), hi=float(np.quantile(r, 0.975)),
                med=float(np.median(r)), n_units=len(h), boot=int(r.size))



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

    # ---- S 集的 unit 层汇总：24 张唯一谱其实只来自 8 个真值配置
    uS = defaultdict(list)
    for r in S:
        uS[(r["family"], r["difficulty"])].append(r)
    unit_rows = []
    for key in sorted(uS):
        rows = uS[key]
        h1 = [int(_pair(r)[0]["acf"] > 0.50) for r in rows]
        h2 = [int(_pair(r)[0]["acf"] - _pair(r)[1]["acf"] >= 0.20) for r in rows]
        unit_rows.append(dict(
            family=key[0], difficulty=key[1], n=len(rows),
            hits1=h1, hits2=h2,
            hit1_any=int(any(h1)), hit1_all=int(all(h1)),
            hit2_any=int(any(h2)), hit2_all=int(all(h2)),
            hit1=int(sum(h1)), hit2=int(sum(h2)),
            n_unique=len({spec[r["id"]].get("csv_sha256") for r in rows}),
        ))
    boot1 = cluster_bootstrap([(x["hit1"], x["n"]) for x in unit_rows])
    boot2 = cluster_bootstrap([(x["hit2"], x["n"]) for x in unit_rows])
    return dict(
        n_files=len(spec), n_unique_sha=len(shas), n_dup_groups=len(dup),
        dup_kinds=dict(Counter(v[0].split("__")[-2] for v in dup.values())),
        n_units=len(units), unit_sizes=dict(Counter(len(v) for v in units.values())),
        S=dict(n=len(S), n_unique=len(Sd),
               hit1=tot1, hit2=tot2,
               hit1_unique=u1, hit2_unique=u2,
               rate1=wilson(tot1, len(S)), rate2=wilson(tot2, len(S)),
               rate1_unique=wilson(u1, len(Sd)), rate2_unique=wilson(u2, len(Sd)),
               n_units=len(unit_rows),
               units_all_hit1=sum(x["hit1_all"] for x in unit_rows),
               units_any_hit1=sum(x["hit1_any"] for x in unit_rows),
               units_all_hit2=sum(x["hit2_all"] for x in unit_rows),
               units_any_hit2=sum(x["hit2_any"] for x in unit_rows),
               cluster_boot1=boot1, cluster_boot2=boot2),
        S_units=unit_rows,
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
    """`Δrwp` 的归一化分母 = 纯噪声下 Rwp 的期望量级 `√2·pct·100`。

    推导：残差实/虚部独立、各 sd = pct·|Z| ⇒ `E|d|²/|Z|² = 2·pct²` ⇒
    `Rwp ≈ √2·pct·100`（pct=0.05 → 7.07pp）。**首版用 `pct·100 = 5pp` 作分母，
    少乘 √2**，故旧口径的比值系统性偏大 41%；两套都保留，便于与本轮已发布的数字对照。
    """
    S = [r for r in results if r["family"] in CAP and r["noise_kind"] in MAIN_KINDS]
    rows = []
    for r in S:
        ind, cap = _pair(r)
        pct = 0.05 if r["noise_kind"] == "none" else float(r["noise_pct"])
        drwp = ind["rwp_pct"] - cap["rwp_pct"]
        floor = math.sqrt(2.0) * pct * 100.0
        rows.append(dict(id=r["id"], noise=r["noise_kind"], difficulty=r["difficulty"],
                         drwp=drwp, floor_pp=floor,
                         drwp_norm=drwp / floor,
                         drwp_norm_5pp=drwp / (pct * 100.0),
                         acf_ind=ind["acf"], acf_cap=cap["acf"],
                         hit=int(ind["acf"] > 0.50),
                         hit2=int(ind["acf"] - cap["acf"] >= 0.20)))
    # 按噪声档分层（混合噪声档会让「偏置—检出」关系不可解释）；分箱只作粗览
    tab = []
    for kind in MAIN_KINDS:
        sel = [x for x in rows if x["noise"] == kind]
        for lo, hi in ((0.0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1e9)):
            g = [x for x in sel if lo <= x["drwp_norm"] < hi]
            if not g:
                continue
            tab.append(dict(noise=kind, lo=lo, hi=(None if hi > 1e8 else hi), n=len(g),
                            hit1=sum(x["hit"] for x in g), hit2=sum(x["hit2"] for x in g),
                            ci1=wilson(sum(x["hit"] for x in g), len(g)),
                            med_acf=float(np.median([x["acf_ind"] for x in g]))))
    # 检出边界（不用 logistic：n=16 且两类之间有间隙，logistic 不可识别）
    boundary = {}
    for kind in MAIN_KINDS:
        sel = [x for x in rows if x["noise"] == kind]
        miss = [x for x in sel if not x["hit"]]
        hit = [x for x in sel if x["hit"]]
        if miss and hit:
            boundary[kind] = dict(
                max_miss=max(x["drwp_norm"] for x in miss),
                min_hit=min(x["drwp_norm"] for x in hit),
                max_miss_5pp=max(x["drwp_norm_5pp"] for x in miss),
                min_hit_5pp=min(x["drwp_norm_5pp"] for x in hit),
                overlap=bool(max(x["drwp_norm"] for x in miss)
                             >= min(x["drwp_norm"] for x in hit)))
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
        # 选对时：top2 一定 ≠ 真族 ⇒ 这个 D 就是「真族 vs 最强对手」；
        # 选错时 top2 与真族的关系不定 ⇒ 改报「真族 vs 被选候选」。
        d_truth_runner = None
        if r["truth_selected"]:
            others = [k for k in order if k != tc]
            if others:
                d_truth_runner = D_of(tc, others[0])
        rows.append(dict(id=r["id"], noise=r["noise_kind"], n_candidates=len(keys),
                         n_points=int(f.size),
                         frac_D_lt1=float(np.mean([D < 1 for _, _, D in pairs])),
                         min_D=float(np.min([D for _, _, D in pairs])) if pairs else None,
                         D_best_vs_second=d_best_second,
                         D_truth_vs_runnerup=d_truth_runner,
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


def build_md(a0, a1s, a2tab, a2rows, boundary, labels, dab):
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
    # ---- A0b unit 层（真值配置）
    w(f"\n### A0b 按真值配置（unit）汇总 —— **有效独立样本是 {S['n_units']}，不是 {S['n_unique']}**\n")
    w(f"S 集 {S['n']} 张谱 = {S['n_unique']} 张唯一谱 = **{S['n_units']} 个真值配置**"
      f"（每个配置 {S['n'] // S['n_units']} 次噪声实现 = 2 噪声 × 2 seed）。"
      f"按谱的 Wilson 区间把 {S['n_unique']} 张谱当独立 Bernoulli，"
      "**没有考虑同一配置内的相关性**，只能当描述性区间。\n")
    w("| family | difficulty | 谱 n | 唯一谱 | ① 命中 | ① 全中 | ② 命中 | ② 全中 |")
    w("|---|---|---|---|---|---|---|---|")
    for x in a0["S_units"]:
        w(f"| {x['family']} | {x['difficulty']} | {x['n']} | {x['n_unique']} | "
          f"{''.join(str(h) for h in x['hits1'])} | {'是' if x['hit1_all'] else '否'} | "
          f"{''.join(str(h) for h in x['hits2'])} | {'是' if x['hit2_all'] else '否'} |")
    b1, b2 = S.get("cluster_boot1"), S.get("cluster_boot2")
    w(f"\n- ① 全部谱都命中的配置：**{S['units_all_hit1']}/{S['n_units']}**；"
      f"至少一张命中的配置：**{S['units_any_hit1']}/{S['n_units']}**")
    w(f"- ② 全部谱都命中的配置：**{S['units_all_hit2']}/{S['n_units']}**；"
      f"至少一张命中的配置：**{S['units_any_hit2']}/{S['n_units']}**")
    if b1 and b2:
        w(f"- **clustered bootstrap（按 unit 整体重采样，B={b1['boot']}）**："
          f"① **{100*b1['med']:.1f}%**（95%CI {100*b1['lo']:.1f}–{100*b1['hi']:.1f}%）；"
          f"② **{100*b2['med']:.1f}%**（95%CI {100*b2['lo']:.1f}–{100*b2['hi']:.1f}%）")
        w("  - 读数：区间宽是因为**只有 8 个配置**；它回答的是「换一批真值配置会怎样」，"
          "Wilson 区间回答的是「同一批配置换噪声实现会怎样」，两个问题不能混。"
          "本例两者数值接近（各配置谱数相同、且命中高度集中在少数配置上），但换设计规模后不会保持。")
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
    w("偏置：`Δrwp = rwp_ind − rwp_true`（pp，同谱配对）。归一化分母 = **纯噪声下 Rwp 的期望量级**"
      "`√2·pct·100`：残差实/虚部独立、各 sd = `pct·|Z|` ⇒ 期望 `Rwp ≈ √2·pct·100`"
      "（pct=0.05 → **7.07pp**）。\n")
    w("> 首版分母写 `pct·100 = 5pp`，**少乘 √2**，比值系统性偏大 41%；下表用修正口径，"
      "括号内给旧口径数值便于对照。这不改任何判定，只改归一化的定义。\n")
    w("| 噪声 | Δrwp/噪声底 | n | ① 检出 | ① 率 (95%CI) | ② 检出 | 中位 acf_ind |")
    w("|---|---|---|---|---|---|---|")
    for t in a2tab:
        hi = "∞" if t["hi"] is None else f"{t['hi']:g}"
        w(f"| {t['noise']} | [{t['lo']:g}, {hi}) | {t['n']} | {t['hit1']} | "
          f"{100*t['hit1']/t['n']:.0f}% ({fmt_ci(t['ci1'])}) | {t['hit2']} | {t['med_acf']:.3f} |")
    # heavy 档逐条明细（n 小，不靠分箱，直接摊开）
    hrows = [x for x in a2rows if x["noise"] == "heavy"]
    if hrows:
        w("\n**heavy 档逐条明细**（不靠分箱，直接摊开；`噪声底 = 7.07pp`）：\n")
        w("| 族 | id | 难度 | Δrwp (pp) | Δrwp/噪声底 | 旧口径(/5pp) | acf_ind | ① |")
        w("|---|---|---|---|---|---|---|---|")
        for x in sorted(hrows, key=lambda r: r["drwp_norm"]):
            fam = "L1" if "L1" in x["id"] else "L2"
            w(f"| {fam} | `{x['id'].split('__', 1)[1]}` | {x['difficulty']} | {x['drwp']:.2f} | "
              f"{x['drwp_norm']:.2f} | {x['drwp_norm_5pp']:.2f} | {x['acf_ind']:.3f} | "
              f"{'✓' if x['hit'] else '✗'} |")
    if boundary:
        w("\n**检出边界（经验，不做外推）**：")
        for k, v in boundary.items():
            w(f"- `{k}`：未检出样本的 `Δrwp/噪声底` 最大 **{v['max_miss']:.2f}**"
              f"（旧口径 {v['max_miss_5pp']:.2f}），检出样本最小 **{v['min_hit']:.2f}**"
              f"（旧口径 {v['min_hit_5pp']:.2f}）"
              + ("（**区间重叠**，不能划单一边界）" if v["overlap"] else "（在这批数据里两类不重叠）"))
        w("  - 读数：这只是**本批样本设计 + 本估计器 + 本 ACF 定义下的观测分界**，"
              "不能写成「低于此值原理上不可检出」。受控扫描（系统改变支路幅度）才能给检出力曲线。")
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
    w("\n## A4b 候选间预测距离 `D_ab`（**诊断量，不是可区分性判据**）\n")
    w("`D_ab = Σ_i |Z_a,i − Z_b,i|² / (0.05·|Z_obs,i|)²`，`Z_a/Z_b` 用两侧**各自的最优拟合**，"
      "固定 5% 参考噪声使其跨档可比。三点限制必须先说清：\n")
    w("1. **未校准**：`D` 是噪声归一化平方距离，不是 `P(可区分)`。它的分布取决于频率点数、"
      "实虚部噪声协方差、参数不确定性，以及两个模型各自的选择偏差；`D=1` 没有统计含义。")
    w("2. **本数据集内可比、跨数据集不可比**：所有谱都是同一 60 点频率网格，"
      "所以本表内部横向可比；换点数或换频段后 `D` 的量级会变。")
    w("3. **按「是否选对」分组时，两组 `D` 指的不是同一对模型**：选对时 top2 ≠ 真族，"
      "`D_best_vs_second` 就是「真族 vs 最强对手」；选错时 top2 与真族关系不定。"
      "⇒ 这一条件内生于结果，**不能按 `D` 分箱算选对率当校准曲线**。\n")
    sel_d = [x["D_truth_vs_runnerup"] for x in dab if x["D_truth_vs_runnerup"] is not None]
    mis = [x for x in dab if not x["truth_selected"] and x["D_true_vs_chosen"] is not None]
    w(f"- 真族被选中 n={len(sel_d)}，真族 vs 最强对手 `D`："
      f"中位 **{np.median(sel_d):.3g}**、p25/p75 "
      f"{np.quantile(sel_d, 0.25):.3g}/{np.quantile(sel_d, 0.75):.3g}、"
      f"`D>1` 占 **{sum(1 for v in sel_d if v > 1)}/{len(sel_d)}**")
    if mis:
        dc = [x["D_true_vs_chosen"] for x in mis]
        w(f"- 误选 n={len(mis)}，真族 vs 被选候选 `D`：中位 **{np.median(dc):.3g}**、"
          f"`D<1` 占 **{sum(1 for v in dc if v < 1)}/{len(dc)}**"
          "（`D<1` 只是说「在这套 5% 参考噪声下两条预测曲线几乎重合」，"
          "**不等于**这两个拓扑在数学上等价，也不等于数据不足）")
        w(f"  - 误选目标分布：{dict(Counter(x['chosen'] for x in mis).most_common(4))}")
    w("- 全部候选对里 `D<1` 的占比中位 "
      f"**{100*np.median([x['frac_D_lt1'] for x in dab]):.1f}%**"
      f"（每谱 {int(np.median([x['n_candidates'] for x in dab]))} 个收敛候选、"
      f"{int(np.median([x['n_points'] for x in dab]))} 个频点）")
    w("\n**已撤回的表述**：首版把 `D>1` 读成「存在可区分的赢家」并给出 31.5%，"
      "这个读法没有校准、且分箱条件内生，不能保留。`D` 现在的定位只有两条："
      "① 说明「两个候选的预测曲线在该参考噪声下差多少」；"
      "② 把「选错的谱」拆成「两条预测几乎重合」与「两条预测差得很远」两类，供下一轮分层。")
    w("\n> 下一轮要做的是**受控模拟**：在已知真值下扫 `D`、噪声、点数、参数自由度，"
      "估 `P(正确选择 | D, n, 噪声, 复杂度)`，再谈有没有可用阈值。本轮不给阈值。")
    w("\n## 诚实边界\n")
    w("1. 零分布用**真族在本谱上的残差**做符号随机化，保留逐点幅度（含异方差与重尾），"
      "只打乱频率顺序结构；标定的是「无序号结构」的零，不是「模型错」的备择。")
    w("2. `obs>零分布 q95` 的比例**不是新判据**，不得读作 ①② 满足；新判据只能写进下一轮预注册。")
    w(f"3. **有效独立样本是 {S['n_units']} 个真值配置**，不是 {S['n_unique']} 张唯一谱、"
      f"更不是 {S['n']} 条记录。按谱的 Wilson 区间忽略了配置内相关，只能当描述性；"
      "跨配置的推断看 clustered bootstrap，而它只有 8 个 cluster，区间必然宽。")
    w("4. 设计标签是几何代理（交叉频率/可见度/灵敏度/幅度补偿失配），不是可辨识性证明；"
      "`f0_char` 是「CPE 与串联电阻幅值交叉频率」，**不是整个嵌套网络的实际弛豫频率**。"
      "若标签预测不了失败，说明标签定义要改，不得反过来当结论。")
    w("5. `D_ab` 用固定 5% 参考噪声、同一 60 点网格，只在本数据集内可比；"
      "**未校准、不作门禁、不按它分箱算选对率**（分箱条件内生）。")
    w("6. 估计器敏感性（Gaussian LS vs Student-t 似然）**本轮未做**，需重拟合，另立一轮。")
    w("7. 检出边界是**本批样本 + 本估计器 + 本 ACF 定义**下的观测分界，"
      "不能写成「低于此值原理上不可检出」；要功率曲线必须做受控幅度扫描。")
    w("\n## 修订记录（第二版，审查后）\n")
    w("| # | 修订 | 来源 |")
    w("|---|---|---|")
    w("| 1 | 新增 unit 层汇总 + clustered bootstrap；明确 24 张谱只来自 8 个配置 | 审查 P0-1 |")
    w("| 2 | `D_ab` 降级为预测几何距离，撤回「31.5% 存在可区分赢家」；"
      "说明分组条件内生、不按 D 分箱算选对率 | 审查 P0-2 |")
    w("| 3 | 噪声底由 `pct·100`（5pp）改为 `√2·pct·100`（7.07pp），旧口径并列保留 | 审查 §二.1 |")
    w("| 4 | 检出边界表述由「原理上看不见」改为「本批观测分界」 | 审查 §二.1 |")
    w("| 5 | 规格文档 §1/§2/§4/§5/§6 与代码对齐（repro 覆盖 1503 对、分箱口径、"
      "标签判定顺序、`MES80` 已废） | 审查 P0-3 |")
    w("| 6 | 零分布经验 size 由 11.4% 更正为 **5/48 = 10.4%**；并补逐族异质性 | 审查 P0-4 |")
    w("\n> 以上都不改本轮 ①②③④ 判定，也不改任何生产代码/阈值/先验。")
    return "\n".join(L) + "\n"


def selftest():
    """内置对拍：交叉频率 / 噪声底系数 / clustered bootstrap / acf 定义。"""
    bad = 0

    def chk(name, got, want, tol):
        nonlocal bad
        ok = abs(got - want) <= tol * max(abs(want), 1e-30)
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {got:.12g} vs {want:.12g}")
        bad += 0 if ok else 1

    # 1) f0_char 解析式 vs 数值求根：|Z_CPE(f0)| 应等于 R_L
    f = np.logspace(-14, 12, 2600001)
    for Q, n, R in ((3.9e-5, 0.75, 1e3), (1e-6, 0.90, 10.0), (1e-3, 0.60, 1e3)):
        f0 = (Q / R) ** (1.0 / n) / (2 * math.pi)
        chk(f"CPE 交叉频率 Q={Q:g} n={n} R={R:g}", float(np.abs(TR.z_cpe(f0, Q, n))), R, 1e-9)
        z = np.abs(TR.z_cpe(f, Q, n))
        in_grid = f[0] < f0 < f[-1]
        i = int(np.argmin(np.abs(np.log(z) - math.log(R))))
        rel = abs(math.log(f[i] / f0)) / math.log(f[1] / f[0])
        print(f"    f0={f0:.4g} Hz 在网格内={in_grid}，数值根 f/f0={f[i]/f0:.9f}"
              f"（网格步长 {math.log(f[1]/f[0]):.2e} dex ⇒ {rel:.2f} 格）")
        bad += 0 if (in_grid and rel <= 1.0) else 1

    # 2) 噪声底系数：sd=pct·|Z| 的两路独立噪声 ⇒ Rwp 的 RMS = √2·pct·100
    rng = np.random.default_rng(20261113)
    Z = 10 + 1j * np.linspace(-8, 8, 60)
    pct = 0.05

    def rwp_of(d):
        return float(np.sqrt(np.mean(np.abs(d) ** 2 / np.abs(Z) ** 2)) * 100)

    g = [rwp_of(rng.normal(size=Z.size) * pct * np.abs(Z)
                + 1j * rng.normal(size=Z.size) * pct * np.abs(Z)) for _ in range(4000)]
    chk("高斯噪声 Rwp 的 RMS（应 = √2·5%）", float(np.sqrt(np.mean(np.square(g)))),
        math.sqrt(2) * pct * 100, 0.02)
    h = [rwp_of(rng.standard_t(3, Z.size) / math.sqrt(3.0) * pct * np.abs(Z)
                + 1j * rng.standard_t(3, Z.size) / math.sqrt(3.0) * pct * np.abs(Z))
         for _ in range(4000)]
    rms_h, med_h = float(np.sqrt(np.mean(np.square(h)))), float(np.median(h))
    print(f"    重尾档：RMS={rms_h:.2f} 中位={med_h:.2f}（t(3) 下中位 < RMS 属正常，"
          f"四阶矩发散使 RMS 波动大，故只作描述不作断言）")
    bad += 0 if med_h < rms_h else 1

    # 3) clustered bootstrap：全部 unit 命中 ⇒ 区间退化为 1
    cb = cluster_bootstrap([(4, 4)] * 8, boot=200)
    chk("cluster bootstrap 全命中", cb["lo"], 1.0, 1e-12)
    chk("cluster bootstrap 全命中(上界)", cb["hi"], 1.0, 1e-12)

    # 4) acf：交替符号序列应接近 1，随机符号应远小于 1
    alt = np.array([1.0, -1.0] * 30)
    rnd = np.random.default_rng(1).normal(size=2000)
    a1 = acf_abs(alt)
    a0 = acf_abs(rnd)
    print(f"[{'ok' if (a1 > 0.9 and a0 < 0.2) else 'FAIL'}] acf 定义: 交替={a1:.3f} 随机={a0:.3f}")
    bad += 0 if (a1 > 0.9 and a0 < 0.2) else 1

    print(f"\n[selftest] 失败 {bad} 项")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true", help="只跑内置对拍")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(1 if selftest() else 0)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    man, spec, data = load_dataset()
    rec = json.loads((REC / "results.json").read_text(encoding="utf-8"))
    results = rec["results"]

    # ---- repro 自检：存盘参数必须能重建残差指标（覆盖全部谱 × 全部有参数的候选）
    bad = 0
    n_pairs = 0
    for r in results:
        f, Z = data[r["id"]]
        for cname, c in r["candidates"].items():
            if not c.get("params"):
                continue
            n_pairs += 1
            d = TR.forward(cname, f, c["params"]) - Z
            ac = acf_abs(d)
            rwp = float(np.sqrt(np.mean(np.abs(d) ** 2 / np.abs(Z) ** 2)) * 100)
            if abs(ac - c["acf"]) > 1e-6 or abs(rwp - c["rwp_pct"]) > 1e-6:
                bad += 1
    print(f"[repro] 覆盖 {len(results)} 谱 / {n_pairs} 个 (谱, 候选) 对，不一致 {bad} 条")
    if bad:
        raise SystemExit("残差无法从存盘参数精确重建，停止（不得用重建残差做零分布）")

    a0 = audit_units(spec, results)
    per, a1s = audit_null(spec, data, results, boot=args.boot)
    a2rows, a2tab, boundary = audit_power(results)
    labels = audit_labels(spec, data, results)
    dab = audit_dab(spec, data, results)

    payload = jsonable(dict(
        generated_by="docs/round_2026-11-13/power_audit.py",
        repro=dict(n_spectra=len(results), n_pairs=n_pairs, mismatch=bad),
        boot=args.boot,
        a0_units=a0, a1_null=dict(summary=a1s, per_spectrum=per),
        a2_power=dict(rows=a2rows, bins=a2tab, boundary=boundary),
        a4_labels=labels, a4_dab=dab,
    ))
    # 显式 newline='\n'：write_text 默认走 os.linesep，Windows 上会把 md 写成 CRLF
    with open(outdir / "audit.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    with open(outdir / "summary.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_md(a0, a1s, a2tab, a2rows, boundary, labels, dab))
    print(f"[ok] -> {outdir/'summary.md'}")


if __name__ == "__main__":
    main()
