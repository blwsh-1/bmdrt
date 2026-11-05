#!/usr/bin/env python
"""σ 噪声地板敏感性扫描。

为什么要有这个脚本
------------------
`bm_core.make_model` 里 `s_re/s_im ~ LogNormal(log(0.01·zmax), 1)` **没有下界**。
遇到无噪声或近无噪声数据（ECM 正演谱、仿真、极高信噪比测量），只要模型族
能精确表示实部，似然就把 σ 一路推向 0 —— 实测落在先验中心下方 12σ，形成
σ→0 的漏斗，NUTS 树深饱和（触顶 53%），采样分级掉到 0，K 选择被全判
"不可用"。`sigma_floor` 把漏斗截断。

**但地板值不能拍脑袋定，也不能以"R̂ 变漂亮"为标准。** 这个脚本按下面五个
指标逐档对比，由人来判断：

  1. 参数后验是否稳定      —— R_inf / f0 / w / φ 随地板变化
  2. K 选择是否稳定        —— K 与 selection_status 随地板变化
  3. 树深饱和是否消失      —— tree_cap
  4. 参数相关性是否异常    —— |ρ|>0.9 的强相关对数与最强值
  5. 预测阻抗是否仍准确    —— rms/σ、PPC 覆盖率

用法
----
    python sweep_sigma_floor.py                        # 797,1139 × 默认五档 × quick
    python sweep_sigma_floor.py --full                 # 正式档（慢，结论才可信）
    python sweep_sigma_floor.py --noise 0.005          # 加 0.5% 噪声再扫
    python sweep_sigma_floor.py --floors 0,1e-4,1e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import test_ecm as T                                     # noqa: E402

DEFAULT_FLOORS = "0,1e-6,1e-5,1e-4,1e-3"


def _parse_floats(s: str) -> list[float]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _metrics(res: dict | None) -> dict:
    """从一条结果 json 里抽五个指标需要的量。"""
    if not res:
        return dict(ok=False)
    if res.get("fit_performed") is False:
        return dict(ok=False, kk=(res.get("kk") or {}).get("status"),
                    status=res.get("k_selection_status"))
    z = float(res.get("zmax_obs") or 1.0)
    sig_re, sig_im = res.get("sigma_re"), res.get("sigma_im")
    diag = res.get("diag") or {}
    rhat = max((v.get("rhat") or 1.0) for v in diag.values()) if diag else float("nan")
    tail = min((v.get("tail") or 0) for v in diag.values()) if diag else float("nan")
    pc = res.get("param_corr") or {}
    strong = pc.get("strong") or []
    rho_max = max((abs(p.get("rho", 0)) for p in strong), default=float("nan"))
    comps = res.get("components") or []
    return dict(
        ok=True,
        K=res.get("K"),
        status=res.get("k_selection_status"),
        grade=res.get("sample_grade"),
        sig_re_rel=(sig_re / z if sig_re else float("nan")),
        sig_pos=(float(np.log((sig_re / z) / 0.01)) if sig_re else float("nan")),
        tree_cap=float(res.get("tree_cap") or 0.0),
        rhat=rhat, tail=tail, ndiv=int(res.get("divergences") or 0),
        rms_over_sig=(float(res["rms_im"]) / float(res["sigma_im"])
                      if res.get("rms_im") and res.get("sigma_im") else float("nan")),
        cover=min(float(res.get("cover_re") or 1.0),
                  float(res.get("cover_im") or 1.0)),
        R_inf=res.get("R_inf"),
        n_strong=len(strong), rho_max=rho_max,
        f0=[c.get("f0") for c in comps],
        w=[c.get("w") for c in comps],
        phi=[c.get("phi") for c in comps],
        elpd=res.get("elpd_loo"),
        deind=(res.get("drt_deinduce") or {}).get("used"),
    )


def _fmt(v, spec="{:.3g}", dash="—"):
    if v is None:
        return dash
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return dash
    except Exception:                                     # noqa: BLE001
        pass
    return spec.format(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="σ 噪声地板敏感性扫描")
    ap.add_argument("--only", default="797,1139", help="行号，逗号分隔")
    ap.add_argument("--floors", default=DEFAULT_FLOORS,
                    help="地板值（相对 zmax），逗号分隔")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="注入相对噪声，0 = 用数据集原样的无噪谱")
    ap.add_argument("--full", action="store_true", help="正式档（3000/1000/3 链）")
    ap.add_argument("--auto-k", action="store_true",
                    help="扫 K（默认固定为真值 K）。默认固定 K：只改 σ 一个旋钮，"
                         "参数漂移才归因得清；扫 K 会让每条谱多花数倍时间"
                         "（每多一个 K 就多一次 JAX 重编译）")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--tag", default="sweep")
    a = ap.parse_args(argv)

    floors = _parse_floats(a.floors)
    idx = [int(x) for x in a.only.split(",") if x.strip()]
    df = T.load_table(T.DEFAULT_SRC)
    base = ROOT / "out" / f"ecm_{a.tag}"
    quick = not a.full

    print(f"[sweep] 行号 {idx}　地板 {floors}　"
          f"档位 {'quick' if quick else 'full'}　噪声 {a.noise:.2%}　"
          f"K={'扫 ' + str(a.kmax) if a.auto_k else '固定为真值 K'}")
    print(f"[sweep] 输出 {base}\n")

    rows = []
    for i in idx:
        row = df.iloc[i]
        t = T.truth_of(row)
        t["row_Z"] = row["Z"]
        stem = f"ecm_{i:05d}_{t['circ'].replace('-', '_')}"
        csv = base / "csv" / f"{stem}.csv"
        csv, Z, Z0 = T.export(t, csv, noise=a.noise, seed=i)

        for fl in floors:
            sub = base / f"floor_{fl:.0e}"
            kf = None if a.auto_k else max(1, int(t["K"]))
            print(f"[sweep] {i} {t['circ']}  floor={fl:.0e}  K={kf or 'auto'} ...",
                  flush=True)
            t0 = time.time()
            try:
                rec = T.run_one(csv, sub / "res", quick, a.kmax,
                                sigma_floor=fl, k_force=kf, quiet=True)
            except Exception as e:                        # noqa: BLE001
                print(f"    !! {type(e).__name__}: {e}")
                rows.append(dict(idx=i, truth=t, floor=fl, wall=time.time() - t0,
                                 m=dict(ok=False)))
                continue
            m = _metrics(rec.get("result"))
            rows.append(dict(idx=i, truth=t, floor=fl,
                             wall=time.time() - t0, m=m))
            print(f"    K={m.get('K')} 状态={m.get('status')} "
                  f"σ/zmax={_fmt(m.get('sig_re_rel'), '{:.2e}')} "
                  f"触顶={_fmt(m.get('tree_cap'), '{:.0%}')} "
                  f"R̂={_fmt(m.get('rhat'), '{:.3f}')} "
                  f"tail={_fmt(m.get('tail'), '{:.0f}')} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    md = _write_report(rows, idx, floors, a, base / "SIGMA_SWEEP.md")
    print(f"\n[sweep] 报告：{md}")
    return 0


def _write_report(rows, idx, floors, a, path: Path) -> Path:
    L = ["# σ 噪声地板敏感性扫描", "",
         f"- 行号：{idx}",
         f"- 地板：{', '.join(f'{f:.0e}' for f in floors)}（相对 zmax）",
         f"- 档位：{'quick（800/400/2 链）' if not a.full else 'full（3000/1000/3 链）'}",
         f"- 噪声：{'数据集原样（无噪）' if a.noise <= 0 else f'注入 {a.noise:.2%}·|Z|'}",
         f"- K：{'扫描 1..' + str(a.kmax) if a.auto_k else '**固定为真值 K**（只改 σ 一个旋钮）'}",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         "> 判读顺序：先看「树深触顶」是否消失，再看「参数/K」是否稳定，",
         "> 最后才看「rms/σ 与 PPC」有没有变差。**不要以 R̂ 最漂亮为标准选地板。**", ""]

    for i in idx:
        sub = [r for r in rows if r["idx"] == i]
        if not sub:
            continue
        t = sub[0]["truth"]
        L += [f"## [{i}] {t['circ']}", "",
              f"真值：K={t['K']}，R_inf={t['R_inf']:.4g} Ω，L={t['L'] * 1e6:.3g} µH",
              "", "| 地板 | K | 状态 | 分级 | σ_re/zmax | σ 先验位置 | 触顶 | R̂max "
              "| tail ESS | div | rms/σ | PPC | R_inf | 强相关对数 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            m = r["m"]
            if not m.get("ok"):
                L.append(f"| {r['floor']:.0e} | — | {m.get('status') or '未拟合'} "
                         f"| — | — | — | — | — | — | — | — | — | — | — |")
                continue
            L.append(
                f"| {r['floor']:.0e} | {m['K']} | {m['status']} | {m['grade']} "
                f"| {_fmt(m['sig_re_rel'], '{:.2e}')} "
                f"| {_fmt(m['sig_pos'], '{:+.1f}σ')} "
                f"| {_fmt(m['tree_cap'], '{:.0%}')} "
                f"| {_fmt(m['rhat'], '{:.3f}')} "
                f"| {_fmt(m['tail'], '{:.0f}')} | {m['ndiv']} "
                f"| {_fmt(m['rms_over_sig'], '{:.2f}')} "
                f"| {_fmt(m['cover'], '{:.3f}')} "
                f"| {_fmt(m['R_inf'], '{:.4g}')} | {m['n_strong']} |")
        L.append("")

        # ---- 分量稳定性：以 floor=0 为基准，看各档的 f0/w 漂移 ----
        base_m = next((r["m"] for r in sub if r["floor"] == 0 and r["m"].get("ok")), None)
        if base_m:
            L += ["**分量漂移（相对地板=0）**", "",
                  "| 地板 | K | Δlog10 f0（逐分量） | Δw/w | Δφ |",
                  "|---|---|---|---|---|"]
            for r in sub:
                m = r["m"]
                if not m.get("ok"):
                    continue
                if m["K"] != base_m["K"]:
                    L.append(f"| {r['floor']:.0e} | {m['K']} | 分量数变了，不可逐项比 "
                             f"| — | — |")
                    continue
                d_f0 = [abs(np.log10(a_ / b_)) if (a_ and b_ and a_ > 0 and b_ > 0) else np.nan
                        for a_, b_ in zip(m["f0"], base_m["f0"])]
                d_w = [abs(a_ / b_ - 1) if b_ else np.nan
                       for a_, b_ in zip(m["w"], base_m["w"])]
                d_ph = [abs(a_ - b_) if None not in (a_, b_) else np.nan
                        for a_, b_ in zip(m["phi"], base_m["phi"])]
                L.append(f"| {r['floor']:.0e} | {m['K']} "
                         f"| {', '.join(_fmt(x, '{:.2f}') for x in d_f0)} "
                         f"| {', '.join(_fmt(x, '{:.0%}') for x in d_w)} "
                         f"| {', '.join(_fmt(x, '{:.3f}') for x in d_ph)} |")
            L.append("")

    # ---- 结论提示 ----
    L += ["## 自动提示", ""]
    for i in idx:
        sub = [r for r in rows if r["idx"] == i and r["m"].get("ok")]
        if not sub:
            L.append(f"- [{i}] 全部档位都没拟合成功。")
            continue
        caps = [r["m"]["tree_cap"] for r in sub]
        ks = sorted({r["m"]["K"] for r in sub})
        sts = sorted({str(r["m"]["status"]) for r in sub})
        cov = [r["m"]["cover"] for r in sub if np.isfinite(r["m"]["cover"])]
        L.append(f"- [{i}] 触顶 {min(caps):.0%}–{max(caps):.0%}；"
                 f"K 取值 {ks}；状态 {sts}"
                 + (f"；PPC {min(cov):.2f}–{max(cov):.2f}" if cov else ""))
        if max(caps) < 0.05:
            L.append(f"  - 本次跑的所有档位触顶都 <5%。**注意这不能推出「这条谱本来"
                     f"没有 σ 塌陷」**——只有跑过足够小的地板（≤1e-6）才能这么说；"
                     f"没跑的话，只能说这些档位都压住了漏斗。")
        elif min(caps) < 0.05:
            good = [r["floor"] for r in sub if r["m"]["tree_cap"] < 0.05]
            L.append(f"  - 触顶降到 <5% 的最小地板：{min(good):.0e}"
                     f"（再小就压不住漏斗）。")
        else:
            L.append(f"  - **所有档位触顶都 ≥5%**：地板没解决这条谱的几何问题，"
                     f"别把它当归一解，要回查噪声模型本身。")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
