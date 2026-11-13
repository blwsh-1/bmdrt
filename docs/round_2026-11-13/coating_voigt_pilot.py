"""涂层族 Voigt 失效率 pilot（60 条，12 格 × 5）。

判据与口径见 COATING_VOIGT_PILOT_PREREG.md（跑前冻结，不改）。
逐条追加 results.jsonl，支持断点续跑。

用法：
    python docs/round_2026-11-13/coating_voigt_pilot.py            # 跑（可反复调用续跑）
    python docs/round_2026-11-13/coating_voigt_pilot.py --summary  # 只出汇总
"""
import sys, json, re, time, hashlib, pathlib, argparse, subprocess
import numpy as np

ROOT = pathlib.Path("D:/eis训练数据/bmdrt")
sys.path.insert(0, str(ROOT))
import bm_core as C
import bm_top as T
import bm_real as R
import bm_experiments as E
from bm_select import BUDGETS

SUB = pathlib.Path("D:/eis训练数据/测试数据/涂层/"
                   "Coating_Generated EIS data and Nyquist plots/Generated EIS data and Nyquist plots")
OUT = ROOT / "out" / "coating_voigt_pilot"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 判据常量（与 PREREG §3 一一对应，改这里必须改文档并登记） ----
RHAT_MAX = 1.05
ESS_WARN = 100
RATIO_STRUCT = 1.5
RATIO_RUNAWAY = 5.0
PARAM_TOL = 0.50
N_PER_CELL = 5
VOIGT_K = 2

TKEYS = ["Rs", "Q_P2", "n_P2", "R3", "Q_P4", "n_P4", "R5", "Q_P6", "n_P6"]


# --------------------------------------------------------------- 抽样
def parse_truth(path):
    s = path.stem
    vals = [float(x) for x in re.findall(
        r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s[s.rfind("[") + 1:s.rfind("]")])]
    if len(vals) != 9:
        return None
    return dict(zip(TKEYS, vals))


def build_manifest():
    files = sorted(SUB.glob("*.csv"))
    rows = []
    for p in files:
        t = parse_truth(p)
        if t is not None:
            rows.append((p, t))
    q2 = np.array([np.log10(t["Q_P2"]) for _, t in rows])
    r3 = np.array([np.log10(t["R3"]) for _, t in rows])
    q4 = np.array([np.log10(t["Q_P4"]) for _, t in rows])
    b_q2 = np.quantile(q2, [1 / 3, 2 / 3])
    b_r3 = np.quantile(r3, [0.5])
    b_q4 = np.quantile(q4, [0.5])

    def bin3(v, b):
        return 0 if v < b[0] else (1 if v < b[1] else 2)

    cells = {}
    for p, t in rows:
        k = (bin3(np.log10(t["Q_P2"]), b_q2),
             int(np.log10(t["R3"]) >= b_r3[0]),
             int(np.log10(t["Q_P4"]) >= b_q4[0]))
        cells.setdefault(k, []).append((p, t))

    picked = []
    for k in sorted(cells):
        members = sorted(cells[k], key=lambda pt: hashlib.md5(pt[0].name.encode()).hexdigest())
        for p, t in members[:N_PER_CELL]:
            picked.append(dict(path=str(p), name=p.name, cell=list(k), truth=t,
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest = dict(total=len(rows), cells={str(k): len(v) for k, v in sorted(cells.items())},
                    bounds=dict(q2=list(b_q2), r3=list(b_r3), q4=list(b_q4)),
                    picked=picked)
    return manifest


# --------------------------------------------------------------- 单条
def acf1(x):
    x = np.asarray(x, float) - np.mean(x)
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else 0.0


def rmsrel(Z, Zm):
    d = (Zm - Z) / np.abs(Z)
    return float(np.sqrt(np.mean(np.concatenate([d.real, d.imag]) ** 2))) * 100


def truth_z(f, t, w_ref):
    p = dict(Rs=t["Rs"], R1=t["R3"], Q1=t["Q_P2"] / w_ref ** t["n_P2"], n1=t["n_P2"],
             R2=t["R5"], Q2=t["Q_P4"] / w_ref ** t["n_P4"], n2=t["n_P4"],
             Q_b=t["Q_P6"] / w_ref ** t["n_P6"], n_b=t["n_P6"])
    return T.z_nested(f, p, branch="capacitive", layers=2)


def data_init(f, Z):
    zmax = float(np.abs(Z).max())
    fr = np.sqrt(f.min() * f.max())
    zfr = float(np.interp(np.log10(fr), np.log10(f)[::-1], np.abs(Z)[::-1]))
    return dict(Rs=float(np.abs(Z).min()), R1=0.5 * zmax, Q1=zfr, n1=0.9,
                R2=0.1 * zmax, Q2=zfr * 0.1, n2=0.9, Q_b=zfr * 0.01, n_b=0.6)


KW = {k: v for k, v in BUDGETS["standard"].items()
      if k in ("n_warmup", "n_samples", "num_chains", "max_tree_depth")}
KW["target_accept"] = 0.98

PAD_DEC = 1.5          # 与 bm_core.make_model 一致（tau 窗口外扩 1.5 decade）


def voigt_init(f, Z, K=VOIGT_K):
    """**数据驱动**初值（只看 f、|Z|，不读真值），返回**约束空间（物理值）**字典。

    空间约定必须分清（踩过）：
    · `bm_core.fit` → `init_strategy=init_to_value(values=...)`，
      `initialize_model` 内部先做一次 `transform_fn(invert=True)`，
      故传入的必须是**物理值**（`R_inf=Ω`、`ln_tau=ln(s)`、`phi∈[0.5,1]`）；
    · `bm_top.fit_top` → `init_params=...`，那条路径才是**无约束**值。
    两者不可混用：按无约束传物理值会被静默 `exp()`，按物理传无约束值会直接
    "Cannot find valid initial parameters"（本函数首版即如此）。
    """
    zmax = float(np.abs(Z).max())
    zmin = float(np.abs(Z).min())
    fmax, fmin = float(np.max(f)), float(np.min(f))
    tau_lo = 1.0 / (2 * np.pi * fmax) / 10 ** PAD_DEC
    tau_hi = 1.0 / (2 * np.pi * fmin) * 10 ** PAD_DEC
    l0, l1 = np.log(tau_lo), np.log(tau_hi)

    # 分量时标：窗口内按 (k+0.5)/K 均匀铺（只看频率窗口，不看数据形状）
    lntau = l0 + (np.arange(K) + 0.5) / K * (l1 - l0)
    s0 = 0.02 * float(np.mean(np.abs(Z)))

    init = {
        "R_inf": float(max(zmin, 1e-12)),                # Ω，正向平台 ≈ min|Z|
        "W": float(max(0.5 * (zmax - zmin), 0.05 * zmax)),   # Ω，极化幅度的一半
        "ln_tau": lntau.astype(float),                   # ln(s)，落在窗口内
        "phi": np.full(K, 0.8, float),                   # Uniform(0.5,1) 支撑内
        "s_re": float(s0), "s_im": float(s0),            # Ω，2% 均值幅值
    }
    if K > 1:                                            # v ~ Beta(1,α) ⊂ (0,1)
        init["v"] = np.full(K - 1, 1.0 / K, float)
    return init


def voigt_fit(f, Z, sigma, seed=0, warmup=None, init=None):
    """sigma=None → 生产口径（自由采样 σ）；否则逐点固定。init=None → 生产默认初始化。"""
    kw = dict(KW)
    if warmup:
        kw["n_warmup"] = warmup
    sg = None if sigma is None else (sigma[0], sigma[1])
    sv, mc = C.fit(f, Z, K=VOIGT_K, alpha=0.5, sigma=sg,
                   zeta_fixed=1.0, seed=seed, progress=False,
                   init_values=init, **kw)
    from numpyro.diagnostics import summary as nsum
    st = nsum(mc.get_samples(group_by_chain=True), prob=0.9)
    # W/pi/tau/phi/zeta 是向量参数（K 维），summary 的 r_hat/n_eff 也是数组
    rhat = max(float(np.max(np.asarray(v["r_hat"]))) for v in st.values())
    ess = min(float(np.min(np.asarray(v["n_eff"]))) for v in st.values())
    div = int(np.sum(np.asarray(mc.get_extra_fields()["diverging"])))
    pv = dict(R_inf=float(np.median(sv["R_inf"])), W=float(np.median(sv["W"])),
              pi=np.median(sv["pi"], axis=0), tau=np.median(sv["tau"], axis=0),
              phi=np.median(sv["phi"], axis=0), zeta=np.median(sv["zeta"], axis=0))
    Zm = C.model_z(f, pv, include_L=False)
    d = dict(rhat=rhat, ess=ess, div=div)
    if "sigma_re" in sv:
        d["sigma_re"] = float(np.median(sv["sigma_re"]))
        d["sigma_im"] = float(np.median(sv["sigma_im"]))
        d["sigma_re_at"] = float(np.median(sv["sigma_re"]) / np.median(np.abs(Z)) * 100)
    return Zm, d


def run_one(item):
    f, Z = R.load_spectrum(item["path"])
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    w_ref = 2 * np.pi * np.sqrt(f.min() * f.max())
    s = 0.02 * np.abs(Z); sigma = np.stack([s, s])

    Zt = truth_z(f, item["truth"], w_ref)
    floor = rmsrel(Z, Zt)

    rec = dict(name=item["name"], cell=item["cell"], truth=item["truth"],
               floor=round(floor, 4), f=list(f), Zr=Z.real.tolist(), Zi=Z.imag.tolist())

    # --- 主分析：voigt 生产口径（自由 σ + 默认初始化） ---
    def _pack(dv, Zm, t0):
        d = dict(rhat=round(dv["rhat"], 4), ess=round(dv["ess"], 1), div=dv["div"],
                 sec=round(time.time() - t0, 1), rms=round(rmsrel(Z, Zm), 4),
                 acf=round(max(E.acf_max((Zm - Z).real), E.acf_max((Zm - Z).imag)), 4),
                 curve=[Zm.real.tolist(), Zm.imag.tolist()])
        for k in ("sigma_re", "sigma_im", "sigma_re_at"):
            if k in dv:
                d[k] = round(dv[k], 4)
        return d

    t0 = time.time()
    try:
        Zv, dv = voigt_fit(f, Z, None)
        rec["voigt_prod"] = _pack(dv, Zv, t0)
    except Exception as ex:                      # 采样崩了也要留档
        rec["voigt_prod"] = dict(error=f"{type(ex).__name__}: {ex}", sec=round(time.time() - t0, 1))

    # --- 结构对照：同一模型、σ 固定 2%（与 nested 同口径） ---
    t0 = time.time()
    try:
        Zv2, dv2 = voigt_fit(f, Z, sigma)
        rec["voigt_fix2"] = _pack(dv2, Zv2, t0)
    except Exception as ex:
        rec["voigt_fix2"] = dict(error=f"{type(ex).__name__}: {ex}", sec=round(time.time() - t0, 1))

    # --- 参照：nested 数据驱动初值 ---
    t0 = time.time()
    init = T.physical_to_unconstrained(data_init(f, Z), zmax=float(np.abs(Z).max()),
                                       w_ref=w_ref, branch="capacitive", layers=2,
                                       sigma=(1.0, 1.0), num_chains=KW["num_chains"])
    o = T.fit_top(f, Z, branch="capacitive", layers=2, sigma=sigma, sigma_from_data=True,
                  init_values=init, progress=False, seed=0, **KW)
    med = {k: float(np.median(o["s"][k])) for k in T.sampled_keys("capacitive", 2)}
    Zm = T.z_nested(f, med, branch="capacitive", layers=2)
    tref = {k: item["truth"][v] for k, v in
            dict(Rs="Rs", R1="R3", Q1="Q_P2", n1="n_P2", R2="R5",
                 Q2="Q_P4", n2="n_P4", Q_b="Q_P6", n_b="n_P6").items()}
    for k in ("Q1", "Q2", "Q_b"):
        tref[k] = tref[k] / w_ref ** tref["n" + k[1:]]
    perr = {k: abs(med[k] / tref[k] - 1) for k in med}
    rec["nested"] = dict(rhat=round(float(o["rhat_max"]), 4), ess=round(float(o["ess_min"]), 1),
                         div=int(o["divergences"]), sec=round(time.time() - t0, 1),
                         rms=round(rmsrel(Z, Zm), 4),
                         acf=round(max(E.acf_max((Zm - Z).real), E.acf_max((Zm - Z).imag)), 4),
                         param_err={k: round(v, 4) for k, v in perr.items()},
                         param_max_err=round(max(perr.values()), 4),
                         RsR1=round(med["Rs"] + med["R1"], 1),
                         RsR1_truth=round(tref["Rs"] + tref["R1"], 1),
                         median={k: float(v) for k, v in med.items()},
                         curve=[Zm.real.tolist(), Zm.imag.tolist()])

    # --- 判定（口径：voigt_prod，见 PREREG §7 M1） ---
    v = rec["voigt_prod"]
    if "error" in v:
        rec["verdict"] = "unusable_sampling"
        rec["ratio"] = None
    else:
        rec["ratio"] = round(v["rms"] / floor, 4)
        if v["rhat"] > RHAT_MAX or v["div"] > 0:
            rec["verdict"] = "unusable_sampling"
        elif rec["ratio"] > RATIO_STRUCT:
            rec["verdict"] = "runaway" if rec["ratio"] > RATIO_RUNAWAY else "structural_mismatch"
        elif rec["nested"]["param_max_err"] > PARAM_TOL:
            rec["verdict"] = "param_unidentifiable"
        else:
            rec["verdict"] = "ok"
    rec["nested_ratio"] = round(rec["nested"]["rms"] / floor, 4)
    f2 = rec["voigt_fix2"]
    rec["fix2_ratio"] = (round(f2["rms"] / floor, 4)
                         if "error" not in f2 else None)
    rec["fix2_usable"] = ("error" not in f2 and f2["rhat"] <= RHAT_MAX and f2["div"] == 0)

    # --- 敏感性：仅对 prod 不可用的样本用**数据驱动初值**再跑（不进主比例） ---
    if rec["verdict"] == "unusable_sampling":
        t0 = time.time()
        try:
            Zv3, dv3 = voigt_fit(f, Z, None, seed=0, init=voigt_init(f, Z))
            s3 = dict(rhat=round(dv3["rhat"], 4), div=dv3["div"], ess=round(dv3["ess"], 1),
                      rms=round(rmsrel(Z, Zv3), 4),
                      ratio=round(rmsrel(Z, Zv3) / floor, 4),
                      method="data_init", seed=0,
                      note="仅换初始化（数据驱动），σ 与采样预算同主分析",
                      sec=round(time.time() - t0, 1))
            s3["recovered"] = bool(s3["rhat"] <= RHAT_MAX and s3["div"] == 0)
            rec["sensitivity"] = s3
        except Exception as ex:
            rec["sensitivity"] = dict(error=f"{type(ex).__name__}: {ex}", method="data_init")
    return rec


# --------------------------------------------------------------- 汇总
def summarize():
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: (r["cell"], r["name"]))
    n = len(rows)
    from collections import Counter
    vc = Counter(r["verdict"] for r in rows)
    print(f"完成 {n} 条")
    print("判定分布:", dict(vc))

    def wilson(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n; d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0, c - h), min(1, c + h))

    ratios = np.array([r["ratio"] for r in rows if r.get("ratio") is not None])
    print(f"\nratio: n={len(ratios)}  min {ratios.min():.2f}  p25 {np.percentile(ratios,25):.2f}  "
          f"中位 {np.median(ratios):.2f}  p75 {np.percentile(ratios,75):.2f}  max {ratios.max():.2f}")
    k_struct = sum(vc[x] for x in ("structural_mismatch", "runaway"))
    lo, hi = wilson(k_struct, n)
    print(f"结构性失配 {k_struct}/{n} = {k_struct/n:.1%}  Wilson95 [{lo:.1%}, {hi:.1%}]")
    lo2, hi2 = wilson(vc["unusable_sampling"], n)
    print(f"采样不可用 {vc['unusable_sampling']}/{n} = {vc['unusable_sampling']/n:.1%}  "
          f"Wilson95 [{lo2:.1%}, {hi2:.1%}]")
    lo3, hi3 = wilson(vc["param_unidentifiable"], n)
    print(f"参数不可辨识 {vc['param_unidentifiable']}/{n} = {vc['param_unidentifiable']/n:.1%}  "
          f"Wilson95 [{lo3:.1%}, {hi3:.1%}]")

    print("\n--- 按格（Q_P2 层, R3 层, Q_P4 层）---")
    byc = {}
    for r in rows:
        byc.setdefault(tuple(r["cell"]), []).append(r)
    print(f"{'格':10} {'n':>3} {'ratio 中位':>10} {'max':>7} {'结构失配':>8} {'采样失败':>8} {'nested中位':>10}")
    for k in sorted(byc):
        g = byc[k]
        rr = [x["ratio"] for x in g if x.get("ratio") is not None]
        ns = [x["nested_ratio"] for x in g]
        print(f"{str(k):10} {len(g):3d} {np.median(rr) if rr else float('nan'):10.2f} "
              f"{max(rr) if rr else float('nan'):7.2f} "
              f"{sum(1 for x in g if x['verdict'] in ('structural_mismatch','runaway')):8d} "
              f"{sum(1 for x in g if x['verdict']=='unusable_sampling'):8d} "
              f"{np.median(ns):10.2f}")

    print("\n--- 敏感性栏：unusable 样本换数据驱动初值后能否恢复（不进主比例）---")
    sens = [r for r in rows if (r.get("sensitivity") or {}).get("method") == "data_init"]
    if sens:
        rec = [r for r in sens if r["sensitivity"].get("recovered")]
        print(f"应补算 {sum(1 for r in rows if r['verdict']=='unusable_sampling')} 条，"
              f"已补 {len(sens)} 条；恢复 {len(rec)}/{len(sens)}")
        if rec:
            rr = np.array([r["sensitivity"]["ratio"] for r in rec])
            print(f"  恢复后的 ratio: min {rr.min():.2f}  中位 {np.median(rr):.2f}  max {rr.max():.2f}")
            n_struct = int((rr > RATIO_STRUCT).sum())
            print(f"  其中 ratio > {RATIO_STRUCT}（结构性失配）: {n_struct}/{len(rr)}"
                  f"  → 说明「初始化失败」掩盖了真实结构失配")
    else:
        print("  尚无补算记录（跑 --backfill-sens）")

    print("\n--- nested 参照 ---")
    nr = np.array([r["nested_ratio"] for r in rows])
    print(f"nested/floor: 中位 {np.median(nr):.2f}  p90 {np.percentile(nr,90):.2f}  max {nr.max():.2f}  "
          f">1.5 的 {int((nr>1.5).sum())}/{n}")

    print("\n--- σ 口径的影响（同一模型、同一默认初始化，只换 σ）---")
    ok_p = [r for r in rows if r["verdict"] != "unusable_sampling"]
    ok_f = [r for r in rows if r.get("fix2_usable")]
    print(f"prod(自由σ) 可用 {len(ok_p)}/{n}   fix2(固定2%) 可用 {len(ok_f)}/{n}")
    only_p = [r for r in rows if r["verdict"] != "unusable_sampling" and not r.get("fix2_usable")]
    only_f = [r for r in rows if r["verdict"] == "unusable_sampling" and r.get("fix2_usable")]
    print(f"仅 prod 可用: {len(only_p)}    仅 fix2 可用: {len(only_f)}")
    bothr = [(r["ratio"], r["fix2_ratio"]) for r in rows
             if r["verdict"] != "unusable_sampling" and r.get("fix2_usable")
             and r.get("fix2_ratio") is not None]
    if bothr:
        a = np.array(bothr)
        print(f"两者都可用 {len(a)} 条：ratio 差(fix2−prod) 中位 {np.median(a[:,1]-a[:,0]):+.3f}  "
              f"max |差| {np.max(np.abs(a[:,1]-a[:,0])):.3f}")

    json.dump(dict(n=n, verdicts=dict(vc), ratios=ratios.tolist(),
                   struct_ci=list(wilson(k_struct, n)),
                   unusable_ci=list(wilson(vc["unusable_sampling"], n)),
                   prod_usable=len(ok_p), fix2_usable=len(ok_f)),
              open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n写入", OUT / "summary.json")


def backfill_sens(limit=0):
    """对已完成的 `unusable_sampling` 记录，用**数据驱动初值**补算敏感性栏。

    只重算 `sensitivity`，`voigt_prod`/`voigt_fix2`/`nested`/`verdict` 一律不动。
    原记录先备份为 `results.jsonl.bak_<ts>`，再整体重写。
    """
    rpath = OUT / "results.jsonl"
    rows = [json.loads(l) for l in rpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    bak = rpath.with_suffix(f".jsonl.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    bak.write_text(rpath.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"备份 → {bak.name}", flush=True)

    need = [r for r in rows if r.get("verdict") == "unusable_sampling"
            and (r.get("sensitivity") or {}).get("method") != "data_init"]
    if limit:
        need = need[:limit]
    print(f"需补算 {len(need)} 条", flush=True)

    path_of = {}
    man = json.load(open(OUT / "sample_manifest.json", encoding="utf-8"))
    for x in man["picked"]:
        path_of[x["name"]] = x["path"]

    for i, r in enumerate(need, 1):
        f, Z = R.load_spectrum(path_of[r["name"]])
        f = np.asarray(f, float); Z = np.asarray(Z, complex)
        t0 = time.time()
        try:
            Zv3, dv3 = voigt_fit(f, Z, None, seed=0, init=voigt_init(f, Z))
            s3 = dict(rhat=round(dv3["rhat"], 4), div=dv3["div"], ess=round(dv3["ess"], 1),
                      rms=round(rmsrel(Z, Zv3), 4),
                      ratio=round(rmsrel(Z, Zv3) / r["floor"], 4),
                      method="data_init", seed=0,
                      note="仅换初始化（数据驱动），σ 与采样预算同主分析",
                      sec=round(time.time() - t0, 1))
            s3["recovered"] = bool(s3["rhat"] <= RHAT_MAX and s3["div"] == 0)
            r["sensitivity_old_warmup"] = r.get("sensitivity")
            r["sensitivity"] = s3
            print(f"[{i}/{len(need)}] {r['name'][:40]:40} R̂ {r['voigt_prod'].get('rhat')} → "
                  f"{s3['rhat']}  recovered={s3['recovered']}  ({s3['sec']}s)", flush=True)
        except Exception as ex:
            r["sensitivity"] = dict(error=f"{type(ex).__name__}: {ex}", method="data_init")
            print(f"[{i}/{len(need)}] {r['name'][:40]:40} ERROR {ex}", flush=True)
        # 每条都落盘，避免再次死机丢结果
        with open(rpath, "w", encoding="utf-8") as fh:
            for x in rows:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("补算完成", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worker", action="store_true",
                    help="内部使用：跑至多 --limit 条后退出（进程隔离）")
    ap.add_argument("--backfill-sens", action="store_true",
                    help="只对已完成记录补算敏感性栏（数据驱动初值），不重跑主分析")
    a = ap.parse_args()
    if a.summary:
        summarize(); return 0

    if a.backfill_sens:
        return backfill_sens(a.limit)

    mpath = OUT / "sample_manifest.json"
    if mpath.exists():
        man = json.load(open(mpath, encoding="utf-8"))
    else:
        man = build_manifest()
        json.dump(man, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"抽样 {len(man['picked'])} 条，写入 {mpath}")

    rpath = OUT / "results.jsonl"

    def done_set():
        if not rpath.exists():
            return set()
        return {json.loads(l)["name"]
                for l in rpath.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [x for x in man["picked"] if x["name"] not in done_set()]
    if a.limit:
        todo = todo[:a.limit]

    # ---- worker 模式：跑完自己这一批就退出，把 JAX 线程句柄交还系统 ----
    if a.worker:
        print(f"[worker] 待跑 {len(todo)} 条", flush=True)
        for i, item in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = run_one(item)
            except Exception as ex:
                rec = dict(name=item["name"], cell=item["cell"],
                           error=f"{type(ex).__name__}: {ex}")
            with open(rpath, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            v = rec.get("verdict", "ERROR")
            print(f"[{i}/{len(todo)}] {item['name'][:42]:42} cell={tuple(item['cell'])} "
                  f"ratio={rec.get('ratio')} floor={rec.get('floor')} → {v}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return 0

    # ---- 调度模式：每批起一个新进程，规避 `_beginthreadex failed` ----
    # 实测：同一进程累积约 5 次 C.fit 后 LLVM 线程创建失败（Windows 线程句柄耗尽）。
    # 每批 4 条（≈8 次 fit）重启一次，重启成本约 15 s（JAX import）。
    BATCH = 4
    total = len(todo)
    print(f"待跑 {total} 条（已完成 {len(done_set())}），每批 {BATCH} 条重启进程", flush=True)
    while True:
        left = [x for x in man["picked"] if x["name"] not in done_set()]
        if a.limit:
            left = left[:a.limit]
        if not left:
            break
        before = len(done_set())
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               "--worker", "--limit", str(BATCH)]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        after = len(done_set())
        if after == before:                 # 一条都没写进 → 不空转
            print(f"批次无进展（rc={rc}），中止；已完成 {after} 条", flush=True)
            break
        print(f"--- 进度 {after}/{len(man['picked'])} ---", flush=True)
    print(f"完成 {len(done_set())}/{len(man['picked'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
