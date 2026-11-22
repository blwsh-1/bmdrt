"""21 条谱的跨类拟合测试（7 类 × 3 条）。

类目与取样
----------
1. 合成（`out/ecm_test/csv/`，文件名即真值）：RCPE 纯弛豫 / Rs+Ws 扩散 / L_R_RCPE 带感抗；
2. 腐蚀（`测试数据/腐蚀/`，Uniform_Generated，1000 条）；
3. 钝化（`测试数据/钝化/`，Passivating_Generated，1000 条）；
4. 涂层（`测试数据/涂层/`，Coating_Generated，1000 条）；
5. 2E（`EISmeasurements/csv_2e/`，118 条）；
6. 3E（`EISmeasurements/csv_3e/`，139 条）；
7. AZ31（`out/_m6_samples/z_csv/`，18 条）。

2–7 类按文件名排序后**等距取 3 条**（索引 1/4、1/2、3/4 处），确定性、可复现，
不用随机种子。

候选集（与生产默认同口径）
--------------------------
`voigt K=1,2` / `nested L=1,2 (branch=parallel)` / `adr` —— 共 5 个。
`include_L` 由 `cmd_compare.need_L(f, Z)` **全局一次决定**（上一轮已把 L 设为
全局开关：同批不得混用），并记录到每条结果里。
扩散（`include_diffusion`）本轮**关闭**：21 × 12 候选会把预算摊得太薄，
留作下一轮单独测。

预算 `quick`（800/400/2）+ `likelihood=free`，与 `l_recovery.py` /
`ad_recovery.py` 同口径，便于横向比较。数字只用于**管线与稳定性**判断，
不作为准入证据。

用法
----
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/batch_fit_21.py
    # 只重算汇总，不重跑拟合：
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-21/batch_fit_21.py --report
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import bm_real as BR                                         # noqa: E402
import circuit_runner as RU                                  # noqa: E402

SEED = 0
OUT = "out/batch21"
JSON = os.path.join(OUT, "results.json")


def _pick_even(paths, n=3):
    """排序后等距取 n 条（确定性）。"""
    paths = sorted(paths)
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    idx = [int(round((i + 0.5) * len(paths) / n)) for i in range(n)]
    idx = [min(max(i, 0), len(paths) - 1) for i in idx]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(paths[i])
    return out


def build_cases():
    """返回 [(类名, [文件绝对/相对路径...]), ...]，共 7 类 × 3 条。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    root = os.path.dirname(root)          # -> D:/eis训练数据

    def _g(pat):
        return glob.glob(os.path.join(root, pat), recursive=True)

    cases = []

    # 1. 合成（4 条里取 3：纯弛豫 / 扩散 / 带感抗，覆盖三种结构）
    syn = [os.path.join("out", "ecm_test", "csv", n) for n in (
        "ecm_04375_RCPE_RCPE_RCPE.csv",
        "ecm_04382_Rs_Ws.csv",
        "ecm_05687_L_R_RCPE.csv",
    )]
    cases.append(("合成", [p for p in syn if os.path.exists(p)]))

    for name, sub in (("腐蚀", "测试数据/腐蚀"), ("钝化", "测试数据/钝化"),
                      ("涂层", "测试数据/涂层")):
        cases.append((name, _pick_even(_g(os.path.join(sub, "**", "*.csv")))))

    cases.append(("2E", _pick_even(_g("bmdrt/EISmeasurements/csv_2e/*.csv"))))
    cases.append(("3E", _pick_even(_g("bmdrt/EISmeasurements/csv_3e/*.csv"))))
    # AZ31：按字母等距取会全落在 `_150` 后缀（同一状态、不同 Li 含量被排到一起），
    # 代表性不够。显式取三个**不同 Li 含量**的样品（4Li/8Li/12Li），
    # 覆盖"低 Li → 高 Li"这一条主变轴。
    az = [os.path.join("out", "_m6_samples", "z_csv", n) for n in (
        "AZ31-4Li_150.csv", "AZ31-8Li_50.csv", "AZ31-12Li_150.csv")]
    have = [p for p in az if os.path.exists(p)]
    cases.append(("AZ31", have or _pick_even(_g(
        "bmdrt/out/_m6_samples/z_csv/*.csv"))))
    return cases


def fit_one(path):
    """一条谱 → 每候选的指标。返回 dict（含失败信息时不抛）。

    每组谱跑**两个口径**：
      · `prod`   —— 生产默认（`zmax = max|Z|`），只跑 `voigt K=1` 作对照；
      · `fixed`  —— `zmax_override = 5·|Z(f_max)|`，跑全部 5 个候选。
    两套数值不可混比（口径不同），分开记。
    """
    import cmd_compare as C
    rec = dict(path=path, status="ok", n_points=None, f_min=None, f_max=None,
               include_L=None, zmax=None, z_hf=None, rinf_prior_over_hf=None,
               cands=[], best=None, best_prod=None, error=None)
    try:
        f, Z = BR.load_spectrum(path)
        zmax = float(np.abs(Z).max())
        z_hf = float(np.abs(Z[np.argmax(f)]))
        rec.update(n_points=int(f.size), f_min=float(f.min()),
                   f_max=float(f.max()), zmax=zmax, z_hf=z_hf,
                   rinf_prior_over_hf=float(0.2 * zmax / z_hf))
        inc_l = bool(C.need_L(f, Z))
        rec["include_L"] = inc_l
        # 2026-11-22：adr 移出候选集（ADR_FAMILY_VERDICT：同参数数 12/13 条输给
        # 串联式，多出的自由度正好是不可辨那一维）。含 adr 的旧结果仍有效——
        # adr 在 21 条里从未当选 best（best 按 rms 定），所以数字不变。
        cands = RU.build_candidates(ks=(1, 2), layers=(1, 2), include_adr=False,
                                    include_L=inc_l, budget_profile="quick")
        plan = [(c, "prod", None) for c in cands
                if c.topology_id == "voigt" and int(dict(c.config)["K"]) == 1]
        plan += [(c, "fixed", 5.0 * z_hf) for c in cands]
        for cand, mode, zov in plan:
            cfg = dict(cand.config)
            tag = cand.topology_id
            if cand.topology_id == "voigt":
                tag += f"/K={cfg.get('K')}"
            elif cand.topology_id == "nested":
                tag += f"/L={cfg.get('layers')}"
            row = dict(label=tag, topology=cand.topology_id, mode=mode,
                       config=cfg)
            try:
                o = RU.fit_candidate(cand, f, Z, seed=SEED,
                                     budget_profile="quick", likelihood="free",
                                     zmax_override=zov)
                row.update(rms=float(o["metrics"]["rms_rel"]),
                           acf=float(o["metrics"]["acf"]),
                           bands=[None if b is None else float(b)
                                  for b in (o["metrics"].get("bands") or [])],
                           rhat=float(o["rhat_max"]),
                           div=int(o["divergences"] or 0),
                           ok=True)
            except Exception as e:                    # noqa: BLE001
                row.update(rms=None, acf=None, bands=[], rhat=None, div=None,
                           ok=False, error=repr(e)[:200])
            rec["cands"].append(row)
        good = [r for r in rec["cands"]
                if r.get("ok") and r.get("rms") is not None and r["mode"] == "fixed"]
        if good:
            b = min(good, key=lambda r: r["rms"])
            rec["best"] = dict(label=b["label"], rms=b["rms"], acf=b["acf"],
                               rhat=b["rhat"], div=b["div"], bands=b["bands"])
        prod = [r for r in rec["cands"]
                if r.get("ok") and r["mode"] == "prod" and r.get("rms") is not None]
        if prod:
            rec["best_prod"] = dict(label=prod[0]["label"], rms=prod[0]["rms"],
                                    bands=prod[0]["bands"])
    except Exception as e:                            # noqa: BLE001
        rec.update(status="fail", error=repr(e)[:300])
        rec["trace"] = traceback.format_exc()[-600:]
    return rec


def report(rows):
    print("\n" + "=" * 118)
    print(f"{'类':<5}{'文件':<30}{'点':>4}{'σ':>6}{'生产rms%':>10}"
          f"{'修正rms%':>10}{'hi%':>8} {'best':<11}{'Rhat':>7}{'div':>4}"
          f"{' L0':>6}")
    print("-" * 118)
    for r in rows:
        fn = os.path.basename(r["path"])
        if len(fn) > 28:
            fn = fn[:25] + "..."
        sig = r.get("rinf_prior_over_hf")
        sig_txt = "—" if sig is None else f"{np.log(max(sig, 1e-9)):.1f}"
        pr = r.get("best_prod") or {}
        b = r.get("best") or {}
        if r["status"] != "ok" or not b:
            print(f"{r['class']:<5}{fn:<30}{str(r.get('n_points')):>4}"
                  f"{sig_txt:>6}{'FAIL':>10}{'':>10}{'':>9}{'':<12}{'':>7}{'':>5}"
                  f"  {(r.get('error') or '')[:30]}")
            continue
        bd = b.get("bands") or []
        hi = bd[2] if len(bd) > 2 and bd[2] is not None else float("nan")
        # 层 0 标注（只显示，不改选择）：best 是按 rms 选的，**没有**采样门。
        # `LAYER0_GATE_AUDIT.md` 实测 21 条里 14 条 best 采样不过，其中 7 条是
        # budget_limited（standard 下能过）⇒ 这里只能标"待复核"，不能直接判死。
        rh = b["rhat"] if b["rhat"] is not None else float("nan")
        lay = "ok" if (rh <= 1.05 and (b["div"] or 0) <= 5) else "RECHK"
        print(f"{r['class']:<5}{fn:<30}{r['n_points']:>4}{sig_txt:>6}"
              f"{(pr.get('rms') if pr else float('nan')):>10.2f}{b['rms']:>10.2f}"
              f"{hi:>8.1f} {b['label']:<11}{rh:>7.3f}{b['div']:>4} {lay:>5}")

    classes = []
    for r in rows:
        if r["class"] not in classes:
            classes.append(r["class"])
    _b = [r["best"] for r in rows if r.get("best")]
    _pend = [x for x in _b if (x["rhat"] or 9e9) > 1.05 or (x["div"] or 0) > 5]
    print(f"\n层 0：best 采样待复核 {len(_pend)}/{len(_b)} 条"
          f"（rms 口径的 best 无采样门；逐条复核见 STD_RECHECK.md / "
          f"LAYER0_SUMMARY.md）")
    print("\n" + "=" * 92)
    print(f"{'类':<5}{'n':>3}{'生产rms中位':>13}{'修正rms中位':>13}"
          f"{'修正≤5%':>9}{'修正acf≤.50':>12}{'R̂≤1.05':>9}")
    print("-" * 92)
    for c in classes:
        sub = [r for r in rows if r["class"] == c and r.get("best")]
        n_all = len([r for r in rows if r["class"] == c])
        if not sub:
            print(f"{c:<5}{n_all:>3}{'—':>13}{'—':>13}{'0':>9}{'0':>12}{'0':>9}")
            continue
        pr = [r["best_prod"]["rms"] for r in sub if r.get("best_prod")]
        fx = [r["best"]["rms"] for r in sub]
        acf_ok = sum(1 for r in sub if (r["best"]["acf"] or 0) <= 0.50)
        rhat_ok = sum(1 for r in sub if r["best"]["rhat"] is not None
                      and r["best"]["rhat"] <= 1.05)
        print(f"{c:<5}{n_all:>3}"
              f"{(np.median(pr) if pr else float('nan')):>13.1f}"
              f"{np.median(fx):>13.2f}"
              f"{sum(1 for x in fx if x <= 5.0):>9}{acf_ok:>12}{rhat_ok:>9}")
    print("=" * 92)
    print("口径：quick/free，L 由 need_L 全局决定，扩散关闭；rms = 逐点相对（SM.metrics）。")
    print("生产 = 默认 zmax=max|Z|（只 voigt K=1）；修正 = zmax_override=5·|Z(fmax)|。")
    print("只作管线与稳定性判断，不作准入证据。")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    if "--report" in sys.argv:
        with open(JSON, encoding="utf-8") as fh:
            rows = json.load(fh)
        report(rows)
        return 0

    cases = build_cases()
    total = sum(len(v) for _, v in cases)
    print(f"共 {len(cases)} 类 / {total} 条")
    for name, paths in cases:
        print(f"  {name}: {len(paths)} 条")

    # 断点续跑：读已有结果，跳过已完成的 path（中断后不重跑已花的采样）
    rows = []
    if os.path.exists(JSON):
        try:
            with open(JSON, encoding="utf-8") as fh:
                rows = json.load(fh)
            done = {r["path"] for r in rows}
            print(f"续跑：已有 {len(done)} 条结果")
        except Exception:                             # noqa: BLE001
            rows = []
    else:
        done = set()
    done = {r["path"] for r in rows}

    t0 = time.time()
    for name, paths in cases:
        for i, p in enumerate(paths, 1):
            if p in done:
                print(f"[skip] {name} {i} {os.path.basename(p)[:40]}", flush=True)
                continue
            t = time.time()
            r = fit_one(p)
            r["class"] = name
            rows.append(r)
            done.add(p)
            dt = time.time() - t
            b = r.get("best") or {}
            rms_txt = "—" if not b else f"{b['rms']:.2f}%"
            print(f"[{len(rows)}/{total}] {name} {i} {os.path.basename(p)[:40]}"
                  f"  rms={rms_txt}  ({dt:.0f}s)", flush=True)
            with open(JSON, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, indent=1,
                          allow_nan=False)

    print(f"\n总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
