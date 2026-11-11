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

    第一轮地板实验（审查意见定案：v6 冻结 detect_L/include_L，只动噪声模型）:
    python sweep_sigma_floor.py --floors 0,1e-5,3e-5,1e-4 --tag f0f3
    python sweep_sigma_floor.py --tag f0f3 --regen   # 只重出报告，不重跑采样

新增「σ 后验是否贴地板」与六项解锁条件判定，全部**只读**已落盘的后验样本
`real_*_samples.npz`，不改 bm_core / bm_real 的任何采样路径或先验。
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


def _sigma_posterior(npz_path: Path | None, floor: float, zmax: float) -> dict:
    """σ 后验是否堆积在**人为地板**上（审查意见解锁条件第 5 项）。

    `make_model` 开地板时采的是 `b ~ LogNormal(log(0.01·zmax), 1)`，
    有效 σ = `deterministic(b + floor)`；两个量都落盘在 `real_*_samples.npz`。
    若 `b << floor`，说明似然仍想把 σ 压到地板以下、被硬截断 —— 漏斗只是
    被截断，不能宣称噪声模型已得到验证，只能声明为数值正则化。
    """
    out = dict(base_re=None, base_im=None, stack_re=None, stack_im=None,
               stack_max=None, share_re=None, share_im=None, share_max=None,
               sig_re_post=None, sig_im_post=None, verdict="n/a")
    if floor <= 0:
        out["verdict"] = "F0 无地板（基准）"
        return out
    if npz_path is None or not npz_path.exists() or not np.isfinite(zmax) or zmax <= 0:
        out["verdict"] = "后验样本缺失"
        return out
    try:
        with np.load(npz_path) as d:
            fl = floor * zmax
            for tag, key in (("re", "s_re_base"), ("im", "s_im_base")):
                if key not in d.files:
                    continue
                b = np.asarray(d[key], float).ravel()
                out[f"base_{tag}"] = float(np.median(b) / fl)
                out[f"stack_{tag}"] = float(np.mean(b <= 0.1 * fl))
                # 地板在有效 σ 中占的比例：越接近 1 越说明 σ 由人为下限决定，
                # 而不是由数据约束。与 stack 互补，避免 0.1*floor 阈值过严时漏判。
                out[f"share_{tag}"] = float(np.median(fl / (fl + b)))
            for tag, key in (("re", "s_re"), ("im", "s_im")):
                if key in d.files:
                    s = np.asarray(d[key], float).ravel()
                    out[f"sig_{tag}_post"] = float(np.median(s) / zmax)
    except Exception as e:                                   # noqa: BLE001
        out["verdict"] = f"读取失败：{type(e).__name__}"
        return out
    vals = [v for v in (out["stack_re"], out["stack_im"]) if v is not None]
    if not vals:
        out["verdict"] = "样本里没有 σ 基量（可能未开地板）"
        return out
    st = max(vals)                     # 取更容易贴地板的那一路，判定更保守
    out["stack_max"] = float(st)
    shares = [v for v in (out["share_re"], out["share_im"]) if v is not None]
    if shares:
        out["share_max"] = float(max(shares))   # 更被地板主导的那一路
    if st >= 0.5:
        out["verdict"] = (f"**贴地板**（{st:.0%} 样本 σ ≤ 1.1×floor）"
                          f"→ 漏斗被硬截断，只能声明为数值正则化")
    elif st >= 0.10:
        out["verdict"] = f"部分贴地板（{st:.0%}）"
    else:
        out["verdict"] = f"未贴地板（{st:.0%}）→ σ 由数据约束"
    return out


# 解锁条件阈值：写死在这里，避免「凭感觉」判后验是否健康。
UNLOCK_RHAT = 1.01
UNLOCK_ESS = 400
UNLOCK_CAP = 0.01


def _unlock(m: dict, floor: float) -> dict:
    """按审查意见六项解锁条件逐项判定。

    PASS       达标
    COND       有条件：贴地板但已明确识别为数值正则化 / 存在需人工解释的强相关
    NOT YET    至少一项硬性不达标 —— 不允许进入第二轮 include_L 消融
    """
    if not m.get("ok"):
        return dict(rhat="—", ess="—", div="—", cap="—", sigma="—",
                    corr="—", total="NOT YET")
    rh = float(m.get("rhat") or np.nan)
    bk = float(m.get("bulk") or np.nan)
    tl = float(m.get("tail") or np.nan)
    out = dict(
        rhat="PASS" if np.isfinite(rh) and rh <= UNLOCK_RHAT else "FAIL",
        ess="PASS" if (np.isfinite(bk) and bk >= UNLOCK_ESS
                       and np.isfinite(tl) and tl >= UNLOCK_ESS) else "FAIL",
        div="PASS" if int(m.get("ndiv") or 0) == 0 else "FAIL",
        cap="PASS" if float(m.get("tree_cap") or 0.0) < UNLOCK_CAP else "FAIL",
    )
    if floor <= 0:
        out["sigma"] = "N/A"
    else:
        st = (m.get("sig_post") or {}).get("stack_max")
        out["sigma"] = ("N/A" if st is None
                        else ("COND" if st >= 0.10 else "PASS"))
    # 第 6 项：README 口径是「无严重且**无法解释**的简并」，不是「没有任何强相关」。
    # 有已知机制解释的强相关不该压成 COND（审查 P2-10）。旧 json 没有
    # n_unexplained 时退回 n_strong，保持原判定。
    _nu = m.get("n_unexplained")
    if _nu is None:
        _nu = int(m.get("n_strong") or 0)
    out["corr"] = "PASS" if int(_nu) == 0 else "COND"
    vals = list(out.values())
    out["total"] = ("NOT YET" if "FAIL" in vals
                    else ("CONDITIONAL" if "COND" in vals else "PASS"))
    return out


def _metrics(res: dict | None, npz: Path | None = None,
             zmax: float | None = None, floor: float = 0.0) -> dict:
    """从一条结果 json + 后验样本 npz 抽五个指标与解锁条件需要的量。"""
    if not res:
        return dict(ok=False)
    if res.get("fit_performed") is False:
        return dict(ok=False, kk=(res.get("kk") or {}).get("status"),
                    status=res.get("k_selection_status"))
    z = float(res.get("zmax_obs") or 1.0)
    sig_re, sig_im = res.get("sigma_re"), res.get("sigma_im")
    diag = res.get("diag") or {}
    rhat = max((v.get("rhat") or 1.0) for v in diag.values()) if diag else float("nan")
    bulk = min((v.get("bulk") or 0) for v in diag.values()) if diag else float("nan")
    tail = min((v.get("tail") or 0) for v in diag.values()) if diag else float("nan")
    pc = res.get("param_corr") or {}
    strong = pc.get("strong") or []
    rho_max = max((abs(p.get("rho", 0)) for p in strong), default=float("nan"))
    comps = res.get("components") or []
    return dict(
        ok=True,
        K=res.get("K"),
        include_L=res.get("include_L"),
        sigma_floor_cfg=res.get("sigma_floor"),
        sig_post=_sigma_posterior(npz, float(floor or 0.0), float(zmax or 0.0)),
        status=res.get("k_selection_status"),
        grade=res.get("sample_grade"),
        sig_re_rel=(sig_re / z if sig_re else float("nan")),
        sig_pos=(float(np.log((sig_re / z) / 0.01)) if sig_re else float("nan")),
        tree_cap=float(res.get("tree_cap") or 0.0),
        rhat=rhat, bulk=bulk, tail=tail, ndiv=int(res.get("divergences") or 0),
        rms_over_sig=(float(res["rms_im"]) / float(res["sigma_im"])
                      if res.get("rms_im") and res.get("sigma_im") else float("nan")),
        cover=min(float(res.get("cover_re") or 1.0),
                  float(res.get("cover_im") or 1.0)),
        R_inf=res.get("R_inf"),
        n_strong=len(strong), rho_max=rho_max,
        # 有已知机制解释（why 非空）的强相关不算「无法解释的简并」。
        n_unexplained=sum(1 for p in strong
                          if not str(p.get("why") or "").strip()),
        strong_pairs=[(p.get("a"), p.get("b"), float(p.get("rho") or 0.0))
                      for p in strong],
        f0=[c.get("f0") for c in comps],
        w=[c.get("w") for c in comps],
        phi=[c.get("phi") for c in comps],
        elpd=res.get("elpd_loo"),
        deind=(res.get("drt_deinduce") or {}).get("used"),
    )


def _load_existing(base: Path, idx: list[int], floors: list[float]) -> list[dict]:
    """从已有结果 JSON + 后验样本 NPZ 重建 rows，供 `--regen` 用。"""
    df = T.load_table(T.DEFAULT_SRC)
    rows = []
    for fl in floors:
        d = base / f"floor_{fl:.0e}" / "res"
        for p in sorted(d.glob("*/real_*.json")):
            try:
                i = int(p.parent.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx and i not in idx:
                continue
            js = json.loads(p.read_text(encoding="utf-8"))
            npz = p.with_name(p.stem + "_samples.npz")
            m = _metrics(js, npz=npz, zmax=js.get("zmax_obs"), floor=fl)
            rows.append(dict(idx=i, truth=T.truth_of(df.iloc[i]),
                             floor=fl, wall=0.0, m=m))
    return rows


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
    ap.add_argument("--regen", action="store_true",
                    help="只从已有 JSON/NPZ 重新生成报告，不重跑 NUTS（报告逻辑迭代时用）")
    a = ap.parse_args(argv)

    floors = _parse_floats(a.floors)
    idx = [int(x) for x in a.only.split(",") if x.strip()]
    df = T.load_table(T.DEFAULT_SRC)
    base = ROOT / "out" / f"ecm_{a.tag}"
    quick = not a.full

    print(f"[sweep] 行号 {idx}　地板 {floors}　"
          f"档位 {'quick' if quick else 'full'}　噪声 {a.noise:.2%}　"
          f"K={'DRT 局部扫描（上界 ' + str(a.kmax) + '）' if a.auto_k else '固定为真值 K'}")
    print(f"[sweep] 输出 {base}\n")

    if a.regen:
        rows = _load_existing(base, idx, floors)
        md = _write_report(rows, idx, floors, a, base / "SIGMA_SWEEP.md")
        print(f"[sweep] 报告已重新生成：{md}（{len(rows)} 条结果，未重跑采样）")
        return 0

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
            npz = sub / "res" / stem / f"real_{stem}_samples.npz"
            zmax = (rec.get("result") or {}).get("zmax_obs")
            m = _metrics(rec.get("result"), npz=npz, zmax=zmax, floor=fl)
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
    # P3（二轮审查）：全部拟合失败仍 return 0 —— 上层按退出码门控会绿灯。
    _n_ok = sum(1 for r in rows if (r.get("m") or {}).get("ok"))
    return 0 if _n_ok else 1


def _write_report(rows, idx, floors, a, path: Path) -> Path:
    labels = {0.0: "F0", 1e-5: "F1", 3e-5: "F2", 1e-4: "F3"}

    def lab(v):
        return labels.get(float(v), f"{float(v):.0e}")

    def flabel(v):
        tag = labels.get(float(v))
        return f"{tag} {v:.0e}" if tag else f"{v:.0e}"

    inc = sorted({str(r["m"].get("include_L"))
                  for r in rows if r["m"].get("ok")})
    inc_txt = (", ".join(inc) if inc else "未知")
    inc_warn = ("" if len(inc) <= 1 else
                "  ⚠️ **include_L 跨档不一致，本轮对照无效**")
    is_f0f3 = set(float(f) for f in floors) == {0.0, 1e-5, 3e-5, 1e-4}
    title = ("# σ 噪声地板敏感性扫描（第一轮 F0–F3）" if is_f0f3
             else "# σ 噪声地板敏感性扫描")
    L = [title, "",
         f"- 行号：{idx}",
         f"- 地板：{', '.join(flabel(f) for f in floors)}（相对 zmax）",
         f"- 档位：{'quick（800/400/2 链）' if not a.full else 'full（3000/1000/3 链）'}",
         f"- 噪声：{'数据集原样（无噪）' if a.noise <= 0 else f'注入 {a.noise:.2%}·|Z|'}",
         f"- K：{'DRT 局部扫描（K_drt±1，上界 ' + str(a.kmax) + '）' if a.auto_k else '**固定为真值 K**（只改 σ 一个旋钮）'}",
         f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         "## 本轮固定条件（每组实验只允许 σ_floor 一个变量变化）", "",
         "| 项 | 固定值 / 来源 |",
         "|---|---|",
         f"| 数据 | {idx} 行正演谱"
         + ("（无附加噪声，数据集原样）" if a.noise <= 0
            else f"（注入 {a.noise:.2%}·|Z| 噪声）")
         + "，同一份 CSV 复用所有档 |",
         f"| K | {'DRT 局部扫描 K_drt±1（上界 ' + str(a.kmax) + '）' if a.auto_k else '固定为真值 K，不做 K 扫描'} |",
         f"| `include_L` | **{inc_txt}**（来自 v6 的 `detect_L`，跨档校验）{inc_warn} |",
         "| NUTS 预算 | " + ("3000 warmup / 1000 采样 / 3 链" if a.full
                                else "800 warmup / 400 采样 / 2 链") + "（`bm_real.KW`） |",
         "| 随机种子 | `seed=0`（`bm_real.KW` → `jax.random.PRNGKey(0)`）；PPC/null 亦为 0 |",
         "| 先验 | 全部沿用 `config.json`：`zeta=1.0`、`alpha=0.5`、`diffusion=none`、"
         "`L~LogNormal(log(1e-6),2)`；**未更换噪声先验分布** |",
         "| 门禁/诊断 | `kk.mode=gate`、`drt.deinduce=true`、`diag` 阈值全部不变 |",
         "| 噪声参数化 | v6 原样 shifted LogNormal：`s = floor + LogNormal(log(0.01·zmax), 1)` |",
         "| 唯一变量 | `fit.sigma_floor` |", "",
         "> 判读顺序：先看「树深触顶」是否消失，再看「参数/K」是否稳定，",
         "> 最后才看「rms/σ 与 PPC」有没有变差。**不要以 R̂ 最漂亮为标准选地板。**", "",
         "> 解锁门槛（写死在 `sweep_sigma_floor.py`）：`R̂≤1.01`、`bulk/tail ESS≥400`、",
         "> `div=0`、`触顶<1%`、`无 |ρ|>0.9 的**不可解释**强相关`。",
         "> **COND**（与代码 `:135-159` 的判定逐字对应）：σ 项 = `stack_max ≥ 0.10`"
         "（floor 占有效 σ 的份额超过 10%，即「贴地板」，不是「已明确识别为数值正则化」）；",
         "> corr 项 = 存在**无法用已知机制解释**的强相关（`n_unexplained > 0`；"
         "有已知机制解释的强相关不压成 COND）。任一项 COND → 总判定 CONDITIONAL。",
         "> `floor 占 σ_eff` = median(floor/(floor+base))，取 re/im 两路较大者；",
         "> 越接近 1 说明有效 σ 越由人为地板决定，而非由数据约束（与严格 stack 阈值互补）。",
         "> 保守解读：即使严格 stack 阈值判 PASS，只要 `floor 占 σ_eff` 明显偏高（>50%），",
         "> 有效 σ 仍主要由人为下限决定，应声明为**数值正则化**，而不是“噪声被数据约束”。",
         "> quick 档总样本仅 800，`tail ESS≥400` 通常达不到 —— 这是**预算不足**，",
         "> 不是模型被否定；ESS 项必须以第 4 步正式档回归为准。", ""]

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

        # ---- 噪声后验是否贴地板 ----
        L += ["**噪声后验是否堆积在地板上**（解锁条件第 5 项）", "",
              "| 档 | floor | σ_re/zmax | σ_im/zmax | base/floor (re / im) "
              "| σ≤1.1×floor 占比 | floor 占 σ_eff | 判读 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sub:
            m = r["m"]
            if not m.get("ok"):
                continue
            sp = m.get("sig_post") or {}
            b_re, b_im = sp.get("base_re"), sp.get("base_im")
            b_txt = ("—" if b_re is None or b_im is None
                     else f"{b_re:.2f} / {b_im:.2f}")
            st = sp.get("stack_max")
            L.append(f"| {lab(r['floor'])} | {r['floor']:.0e} "
                     f"| {_fmt(sp.get('sig_re_post'), '{:.2e}')} "
                     f"| {_fmt(sp.get('sig_im_post'), '{:.2e}')} "
                     f"| {b_txt} | {_fmt(st, '{:.0%}')} "
                     f"| {_fmt(sp.get('share_max'), '{:.0%}')} "
                     f"| {sp.get('verdict', '—')} |")
        L.append("")

        # ---- 六项解锁条件 ----
        L += ["**六项解锁条件逐项判定**", "",
              "| 档 | R̂max | bulk ESS | tail ESS | div | 触顶 "
              "| σ 后验 | 强相关 | **总判定** |",
              "|---|---|---|---|---|---|---|---|---|"]
        bad_notes = []
        for r in sub:
            m = r["m"]
            u = _unlock(m, r["floor"])
            if not m.get("ok"):
                L.append(f"| {lab(r['floor'])} | — | — | — | — | — | — | — "
                         f"| {u['total']}（未拟合） |")
                continue
            L.append(f"| {lab(r['floor'])} | {_fmt(m.get('rhat'), '{:.3f}')} "
                     f"| {_fmt(m.get('bulk'), '{:.0f}')} "
                     f"| {_fmt(m.get('tail'), '{:.0f}')} "
                     f"| {m.get('ndiv')} | {_fmt(m.get('tree_cap'), '{:.1%}')} "
                     f"| {u['sigma']} | {u['corr']}（{m.get('n_strong')} 对） "
                     f"| **{u['total']}** |")
            bad = [k for k in ("rhat", "ess", "div", "cap")
                   if u[k] == "FAIL"]
            if bad:
                bad_notes.append(f"- {lab(r['floor'])} 硬性未达标："
                                 f"{', '.join(bad)}")
        L.append("")
        if bad_notes:
            L += bad_notes + [""]

        L += ["**强相关明细**（审查意见第 6 项要求「无可解释不了的简并」，",
              "需人工看具体参数对、并比较跨档是否新增）", ""]
        for r in sub:
            m = r["m"]
            if not m.get("ok"):
                continue
            ps = m.get("strong_pairs") or []
            txt = (", ".join(f"`{a} ~ {b}` (ρ={rho:+.2f})"
                            for a, b, rho in ps) or "无")
            L.append(f"- {lab(r['floor'])}：{txt}")
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
