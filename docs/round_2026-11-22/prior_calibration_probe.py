"""prior_calibration_probe.py —— 先验定标审计 + 「只改串联电阻参照量」的隔离验证。

为什么要有这个脚本
------------------
`RINF_PRIOR_DIAGNOSIS.md` §4 证明 `R_inf` 的先验中心 `0.2·max|Z|` 用错了参照量，
但那次证据是用 `zmax_override = 5·|Z(f_max)|` 拿到的 —— 那会**同时**缩放
`W`/`Q`/`R1` 的先验（诊断 §6 自己说"那是运气，不是设计"）。
所以诊断 §6 提出的最小修法

    R_inf: LogNormal(log(0.2·zmax), 1.0)  →  LogNormal(log(0.5·|Z(f_max)|), 1.0)

**从未被真正测过**。本脚本把这件事拆成两个可回答的问题：

A. 审计：串联电阻先验偏离 `|Z(f_max)|` 多少 σ（`R_inf`/`Rs` 同一把尺，σ=1.0）。
   数据来源优先用已有产物（`out/batch21/results.json`），不重跑拟合。

B. 隔离验证：**只改这一条先验**（其余先验、采样、结构不动），走生产代码路径
   （harness 侧 monkey-patch `bm_top._prior_table`，`bm_*.py` 零改动），看：
   · 真实高动态范围谱（腐蚀/钝化）的 `rms_rel` 能不能从三位数掉到个位数；
   · B2 `nested_L2_L` 把 `R_ad`/`tau_ad` 的先验 σ 从 2 收到 0.5 后，
     `standard` 的 R̂=1.100 能不能自己掉到 1.05 以下（即"限住游走"是否真有效）。

钉住的口径
----------
· 只 patch `_prior_table`（nested/adr 的单一来源）；**voigt 的 `R_inf` 在
  `bm_core.make_model` 内部，没有表**，故本文的 B 部分只覆盖 nested/adr。
· patch 只替换 2~4 个 `LogNormal` 的中心/σ，别的一律照原表；`init_to_uniform` 不变。
· 每档都同时报 `R̂`/`div`/`rms_rel`，不允许只看 rms。

用法
----
    python docs/round_2026-11-22/prior_calibration_probe.py audit
    python docs/round_2026-11-22/prior_calibration_probe.py b2
    python docs/round_2026-11-22/prior_calibration_probe.py real
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(REPO)              # D:/eis训练数据
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs", "round_2026-11-21"))
sys.path.insert(0, os.path.join(REPO, "evidence", "nested_topology"))

import bm_real as BR                       # noqa: E402
import bm_top as T                         # noqa: E402

#: 串联电阻（voigt 的 `R_inf` / nested/adr 的 `Rs`）先验的当前规则：中心 0.2·zmax、σ=1.0
RS_CENTER_OF_ZMAX = 0.2
RS_SIGMA = 1.0

#: 拟议的参照量：0.5·|Z(f_max)|
RS_CENTER_OF_ZHF = 0.5

#: 当前 `R_ad`/`tau_ad` 的 σ（log 尺度）
AD_SIGMA_NOW = 2.0


# ---------------------------------------------------------------------------
# harness 侧 patch：只动 `_prior_table` 里的中心/σ
# ---------------------------------------------------------------------------
_ORIG_PRIOR_TABLE = T._prior_table
OVR = {"rs_zhf": None, "ad_sigma": None}


def _patched_prior_table(zmax, w_ref, branch, layers,
                         diffusion=None, include_L=False):
    t = _ORIG_PRIOR_TABLE(zmax, w_ref, branch, layers, diffusion, include_L)
    import jax.numpy as jnp
    import numpyro.distributions as dist

    if OVR["rs_zhf"]:
        # 只换参照量，σ 仍 1.0
        t["Rs"] = dist.LogNormal(jnp.log(RS_CENTER_OF_ZHF * OVR["rs_zhf"]),
                                 RS_SIGMA)
    if OVR["ad_sigma"]:
        # 只收宽度，中心仍 0.1·zmax / 1/w_ref
        s = float(OVR["ad_sigma"])
        if "R_ad" in t:
            t["R_ad"] = dist.LogNormal(jnp.log(0.1 * zmax), s)
        if "tau_ad" in t:
            t["tau_ad"] = dist.LogNormal(jnp.log(1.0 / w_ref), s)
    return t


def install_patch():
    T._prior_table = _patched_prior_table


def uninstall_patch():
    T._prior_table = _ORIG_PRIOR_TABLE


# ---------------------------------------------------------------------------
# A. 审计
# ---------------------------------------------------------------------------
def sigma_dist(zmax, z_hf):
    """先验中心离 `|Z(f_max)|` 多少 σ（σ=1.0，log 尺度）。"""
    return float(np.log(RS_CENTER_OF_ZMAX * zmax / z_hf))


def collect_paths():
    """所有能便宜加载的谱：batch21 的 21 条 + AZ31 z_csv + 2E/3E + naoh。"""
    out = []
    p = os.path.join(REPO, "out", "batch21", "results.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        rows = d["rows"] if isinstance(d, dict) else d
        for r in rows:
            out.append(("21条/" + r.get("class", "?"),
                        os.path.join(REPO, r["path"])))
    for pat, tag in (("bmdrt/out/_m6_samples/z_csv/*.csv", "AZ31"),
                     ("bmdrt/EISmeasurements/csv_3e/*.csv", "3E"),
                     ("bmdrt/EISmeasurements/csv_2e/*.csv", "2E")):
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            out.append((tag, f))
    return out


def cmd_audit(a):
    import contextlib
    import io

    rows = []
    bad = []
    for tag, path in collect_paths():
        if not os.path.exists(path):
            continue
        try:
            # 加载器很吵（每条都打一行"读入 …"），审计表不能被它冲掉
            with contextlib.redirect_stdout(io.StringIO()):
                f, Z = BR.load_spectrum(path)
        except (Exception, SystemExit) as ex:   # 加载器失败时走 SystemExit，不是 Exception
            bad.append((tag, os.path.basename(path), type(ex).__name__))
            continue
        zmax = float(np.abs(Z).max())
        z_hf = float(np.abs(Z[np.argmax(f)]))
        rows.append((tag, os.path.basename(path), zmax, z_hf,
                     zmax / z_hf, sigma_dist(zmax, z_hf)))
    if not rows:
        print("没有可加载的谱")
        for b in bad[:10]:
            print("  失败：", b)
        return 1
    print(f"共 {len(rows)} 条谱（加载失败 {len(bad)} 条）")
    groups = {}
    for tag, name, zmax, z_hf, rat, sd in rows:
        groups.setdefault(tag, []).append((name, zmax, z_hf, rat, sd))
    print(f"{'类':<12}{'n':>4}{'|σ距| 中位':>12}{'最大':>9}{'超2.5σ':>9}")
    for tag in sorted(groups):
        v = np.array([abs(g[4]) for g in groups[tag]])
        print(f"{tag:<12}{len(v):>4}{np.median(v):>12.2f}{v.max():>9.2f}"
              f"{int((v > 2.5).sum()):>6}/{len(v)}")
    sd_all = np.array([r[5] for r in rows])
    n25 = int((np.abs(sd_all) > 2.5).sum())
    print()
    print("最偏的 10 条：")
    print(f"{'类':<10}{'谱':<36}{'zmax':>12}{'|Z(fmax)|':>12}"
          f"{'zmax/zhf':>10}{'σ距':>8}")
    for tag, name, zmax, z_hf, rat, sd in sorted(rows,
                                                 key=lambda r: -abs(r[5]))[:10]:
        print(f"{tag:<10}{name[:35]:<36}{zmax:>12.4g}{z_hf:>12.4g}"
              f"{rat:>10.1f}{sd:>8.2f}")
    print()
    print(f"|σ距| > 2.5：**{n25}/{len(rows)}** = {100*n25/len(rows):.0f}%")
    print("（σ距 = ln(0.2·zmax / |Z(f_max)|)，σ=1.0；>2.5 表示先验中心离高频极限太远）")
    return 0


# ---------------------------------------------------------------------------
# B. B2 nested_L2_L：把 R_ad/tau_ad 的 σ 收到 0.5
# ---------------------------------------------------------------------------
def cmd_b2(a):
    """B2 `nested_L2_L`：先验的两种动法分开测，看 R̂ 能不能自己掉到 1.05 以下。"""
    import b2_baseline as B2
    import bm_select as SEL

    rec = [r for r in B2.RECIPES if r["name"] == "nested_L2_L"][0]
    f, zt, zo, sg, _ = B2.generate(rec)
    zmax = float(np.abs(zo).max())
    z_hf = float(np.abs(zo[np.argmax(f)]))
    kw = dict(B2.BUDGETS_LOCAL["standard"], num_chains=2)
    print(f"nested_L2_L：zmax={zmax:.1f}  |Z(fmax)|={z_hf:.1f}"
          f"  σ距={sigma_dist(zmax, z_hf):+.2f}"
          f"  （真值 Rs=50 / R_ad=0.5）")
    print(f"{'配置':<20}{'R̂_max':>9}{'div':>5}{'ESS_min':>9}"
          f"{'R2':>9}{'R2偏差':>8}{'R_ad':>9}{'tau_ad':>9}")
    install_patch()
    try:
        for label, ovr in (("基线（原样）", dict(rs_zhf=None, ad_sigma=None)),
                           ("只收 ad σ→0.5", dict(rs_zhf=None, ad_sigma=0.5)),
                           ("只改 Rs 参照量", dict(rs_zhf=z_hf, ad_sigma=None)),
                           ("两者都改", dict(rs_zhf=z_hf, ad_sigma=0.5))):
            OVR.update(ovr)
            c = SEL.fit_nested(f, zo, 2, branch="parallel",
                               seed=int(rec["seed"]), sigma_floor=0.0, kw=kw,
                               sigma=sg, sigma_from_data=True, include_L=True)
            raw, p = c["raw"], c["raw"]["s"]
            r2 = float(np.median(p["R2"]))
            # 拖拽强度：log-log 相关（与几何同坐标）
            try:
                po = raw["idata"].posterior
                def _f(v):
                    v = np.asarray(v, float)
                    return v[..., 0] if v.ndim == 3 else v
                rho = float(np.corrcoef(np.log(_f(po["R2"])).ravel(),
                                        np.log(_f(po["tau_ad"])).ravel())[0, 1])
            except Exception:                # noqa: BLE001
                rho = float("nan")
            print(f"{label:<20}{raw['rhat_max']:>9.4f}"
                  f"{int(raw['divergences'] or 0):>5}"
                  f"{raw['ess_min']:>9.0f}"
                  f"{r2:>9.2f}{(r2 / 200 - 1) * 100:>7.1f}%"
                  f"{float(np.median(p['R_ad'])):>9.3g}"
                  f"{float(np.median(p['tau_ad'])):>9.3g}"
                  f"   ρ(logR2,logτad)={rho:+.2f}", flush=True)
    finally:
        OVR.update(rs_zhf=None, ad_sigma=None)
        uninstall_patch()
    return 0


# ---------------------------------------------------------------------------
# C. 真实高动态范围谱：只改 Rs 参照量
# ---------------------------------------------------------------------------
def _pick_real():
    """从 batch21 产物里取 腐蚀/钝化 各 1 条（σ距最大的那条）。"""
    p = os.path.join(REPO, "out", "batch21", "results.json")
    d = json.load(open(p, encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) else d
    out = {}
    for want in ("腐蚀", "钝化", "涂层"):
        cand = [r for r in rows if r.get("class") == want]
        cand = [r for r in cand if r.get("zmax") and r.get("z_hf")]
        if not cand:
            continue
        best = max(cand, key=lambda r: sigma_dist(r["zmax"], r["z_hf"]))
        out[want] = best
    return out


def cmd_az31(a):
    """主线验收集里 `Rs` 先验偏离最大的几条：基线 vs 只改参照量。

    为什么要单独做这一组：`§14` 的 14 条里，σ距 ≥2.33 的 3 条全部失败，
    4 条通过的全部 ≤1.85；`§13` 的 `4Li_150`（3.19）与 `8Li_100`（2.26）也都失败。
    所以"先验是否卡住了主线验收"必须直测，不能靠推测。
    预算与 `§14` 一致（standard，2 链），口径取 F1（逐样本 Z 的逐点中位）。
    """
    import bm_select as SEL
    import circuit_runner as RU
    import spectrum_metrics as SM

    from m4_holdout2 import BUDGET as B14          # §14 的预算，直接用
    kw14 = dict(B14)
    kw14.setdefault("max_tree_depth", 10)          # §14 的 dict 里没写，fit_top 要求有

    cfg = dict(layers=2, branch="parallel", include_L=True, diffusion=None)
    specs = ["AZ31-4Li_50", "AZ31-4Li_150", "AZ31-8Li_150", "AZ31-4Li_10"]
    print(f"预算 = {kw14}（= §14 档）；判据线 rms≤5.0 / acf≤0.50 / R̂≤1.05")
    print(f"{'谱':<15}{'σ距':>6}  {'配置':<20}{'rms%':>8}{'acf':>7}"
          f"{'R̂':>7}{'div':>5}{'Rs':>8}")
    install_patch()
    try:
        for sp in specs:
            path = os.path.join(REPO, "out", "_m6_samples", "z_csv", sp + ".csv")
            f, Z = BR.load_spectrum(path)
            zmax = float(np.abs(Z).max())
            z_hf = float(np.abs(Z[np.argmax(f)]))
            sd = sigma_dist(zmax, z_hf)
            sg = (0.02 * np.abs(Z), 0.02 * np.abs(Z))
            for label, ovr in (("基线 0.2·zmax", dict(rs_zhf=None, ad_sigma=None)),
                               ("Rs→0.5·|Z(fmax)|",
                                dict(rs_zhf=z_hf, ad_sigma=None))):
                OVR.update(ovr)
                c = SEL.fit_nested(f, Z, 2, branch="parallel", seed=0,
                                   sigma_floor=0.0, kw=kw14,
                                   sigma=sg, sigma_from_data=True,
                                   include_L=True)
                raw = c["raw"]
                zm = RU._pointwise_median_z(raw, "nested", cfg, f)
                met = SM.metrics(f, Z, zm)
                print(f"{sp:<15}{sd:>+6.2f}  {label:<20}"
                      f"{met['rms_rel']:>8.2f}{met['acf']:>7.3f}"
                      f"{raw['rhat_max']:>7.3f}"
                      f"{int(raw['divergences'] or 0):>5}"
                      f"{float(np.median(raw['s']['Rs'])):>8.4g}", flush=True)
            OVR.update(rs_zhf=None)
    finally:
        OVR.update(rs_zhf=None, ad_sigma=None)
        uninstall_patch()
    print()
    print("注：acf 是 M-b 的卡点；rms 达标但 acf>0.50 仍不算过。")
    return 0


def cmd_real(a):
    """真实高动态范围谱：**只改 Rs 参照量**，走生产 quick 档。"""
    import bm_select as SEL
    import circuit_runner as RU
    import spectrum_metrics as SM

    cfg = dict(layers=2, branch="parallel", include_L=True, diffusion=None)
    picks = _pick_real()
    print(f"{'谱':<8}{'σ距':>7}  {'配置':<24}{'rms%':>9}{'acf':>7}"
          f"{'R̂':>8}{'div':>5}{'Rs':>9}")
    install_patch()
    try:
        for cls, r in sorted(picks.items()):
            path = os.path.join(REPO, r["path"])
            f, Z = BR.load_spectrum(path)
            zmax = float(np.abs(Z).max())
            z_hf = float(np.abs(Z[np.argmax(f)]))
            sd = sigma_dist(zmax, z_hf)
            sg = (0.02 * np.abs(Z), 0.02 * np.abs(Z))
            for label, ovr in (("基线", dict(rs_zhf=None, ad_sigma=None)),
                               ("Rs→0.5·|Z(fmax)|",
                                dict(rs_zhf=z_hf, ad_sigma=None))):
                OVR.update(ovr)
                c = SEL.fit_nested(f, Z, 2, branch="parallel", seed=0,
                                   sigma_floor=0.0,
                                   kw=dict(RU.BUDGETS["quick"]),
                                   sigma=sg, sigma_from_data=True,
                                   include_L=True)
                raw = c["raw"]
                zm = RU._pointwise_median_z(raw, "nested", cfg, f)
                met = SM.metrics(f, Z, zm)
                rs = float(np.median(raw["s"]["Rs"]))
                print(f"{cls:<8}{sd:>+7.2f}  {label:<24}"
                      f"{met['rms_rel']:>9.2f}{met['acf']:>7.3f}"
                      f"{raw['rhat_max']:>8.3f}"
                      f"{int(raw['divergences'] or 0):>5}{rs:>9.4g}", flush=True)
            OVR.update(rs_zhf=None)
    finally:
        OVR.update(rs_zhf=None, ad_sigma=None)
        uninstall_patch()
    print()
    print("注：quick 档只用于看方向，不作判定；acf/R̂ 一起读（R27）。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("audit", "b2", "real", "az31"))
    a = ap.parse_args(argv)
    return {"audit": cmd_audit, "b2": cmd_b2, "real": cmd_real,
            "az31": cmd_az31}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
