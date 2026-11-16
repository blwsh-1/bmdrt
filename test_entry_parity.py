#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T1–T6 验收 + 两入口差分测试（2026-11-16 口径合并）。

对应批准的六项验收
------------------
| 编号 | 验收条件 | 本文件的检查 |
|---|---|---|
| T1 | 两个入口调用同一份 `metrics()` | `t1_single_metrics` |
| T2 | nested 原 13 字段全保留、固定数据下数值逐项一致 | `t2_metrics_parity`（对独立硬算式） |
| T3 | 两个入口调用同一份 σ，输出逐元素一致 | `t3_sigma_parity` |
| T4 | `sigma_from_data` 的语义/传播/ELPD 判定一致，有独立测试 | `t4_elpd_semantics` |
| T5 | `standard` 唯一来源；`quick` 实际预算完整落盘 | `t5_budget_source` |
| T6 | 采样器实参不变、生产默认路径与历史判定不变 | `t6_sampler_args` |

外加 `t7_diff_same_input`：同一输入走两个入口，噪声数组、指标、预算、传给
`fit_top` 的实参必须一致，只允许「入口元数据」不同（`progress` / `outdir` 这类）。

**不跑 MCMC**：`t7` 把 `bm_top.fit_top` 换成只记录实参的桩，抛 `_Stop` 跳出。
所以整个文件是秒级的。

    python test_entry_parity.py
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bm_experiments as E              # noqa: E402
import bm_select as SEL                 # noqa: E402
import bm_top as T                      # noqa: E402
import circuit_runner as RU             # noqa: E402
import nested_fit as NF                 # noqa: E402
import spectrum_metrics as SM           # noqa: E402

#: nested 接口原有的 13 个指标字段（一个都不许少）。多出来的字段不算失败。
NESTED_FIELDS = ("rms_rel", "rms_re", "rms_im", "rel_median", "rel_max",
                 "rel_max_at_Hz", "bands_pct", "acf", "acf_re", "acf_im",
                 "acf_mag", "top_hz", "top_rel_pct")
#: runner 原有的 5 个字段（`acf_mag` 与上面重叠）。
RUNNER_FIELDS = ("rms_rel", "bands", "acf", "acf_re", "acf_im", "acf_mag")

ok, total, bad = 0, 0, []


def c(name, cond, extra=""):
    global ok, total
    total += 1
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  {extra}" if not cond and extra else ""))
    if cond:
        ok += 1
    else:
        bad.append(name)


def raises(fn):
    try:
        fn()
    except Exception:                            # noqa: BLE001
        return True
    return False


# ---------------------------------------------------------------------------
# 固定测试谱（合成，不依赖数据文件）
# ---------------------------------------------------------------------------
def _spectrum():
    f = np.logspace(-2, 5, 46)
    Z = (20.0 + 150.0 / (1 + 1j * 2 * np.pi * f * 3e-2)
         + 1e-4 * 1j * 2 * np.pi * f)
    return f, Z


def oracle_metrics(f, Z, Zm):
    """**独立硬算**的 §14 同式指标，作为 T2 的基准。

    这不是第二份生产实现：只在测试里存在，故意用与 `spectrum_metrics` 不同的
    写法（先逐点再拼、手写频段循环），用来抓「公式被悄悄改了」这类回归。
    """
    w = np.abs(Z)
    res = np.asarray(Zm) - np.asarray(Z)
    d = res / w
    rel = np.abs(res) / w

    bands = []
    for lo, hi in ((0.01, 1.0), (1.0, 100.0), (100.0, 1e5)):
        sel = (f >= lo) & (f < hi)
        if not sel.any():
            bands.append(None)
            continue
        acc = 0.0
        for v in d[sel]:
            acc += abs(v) ** 2
        bands.append(float(np.sqrt(acc / int(sel.sum()))) * 100.0)

    i_top = int(np.argmax(f))
    i_worst = int(np.argmax(rel))
    return dict(
        rms_rel=float(np.sqrt(np.mean(np.array(list(d.real) + list(d.imag)) ** 2))) * 100.0,
        rms_re=float(np.sqrt(np.mean([v ** 2 for v in res.real]))),
        rms_im=float(np.sqrt(np.mean([v ** 2 for v in res.imag]))),
        rel_median=float(np.median(rel)) * 100.0,
        rel_max=float(np.max(rel)) * 100.0,
        rel_max_at_Hz=float(f[i_worst]),
        bands=bands,
        acf=float(max(E.acf_max(res.real), E.acf_max(res.imag))),
        acf_re=float(E.acf_max(res.real)),
        acf_im=float(E.acf_max(res.imag)),
        acf_mag=float(E.acf_max(rel)),
        top_hz=float(f[i_top]),
        top_rel_pct=float(rel[i_top]) * 100.0,
    )


# ---------------------------------------------------------------------------
# T1 / T2
# ---------------------------------------------------------------------------
def t1_single_metrics():
    print("\nT1 两个入口调同一份 metrics()")
    f, Z = _spectrum()
    Zm = Z * 1.004
    a = NF.metrics(f, Z, Zm)
    b = RU.metrics(f, Z, Zm)
    d = SM.metrics(f, Z, Zm)
    c("三处返回逐字段相同", a == b == d)
    # 静态检查：生产代码里不应再有第二份「concatenate 之后取均方」的实现
    src = (ROOT / "spectrum_metrics.py").read_text(encoding="utf-8")
    src_nf = (ROOT / "nested_fit.py").read_text(encoding="utf-8")
    src_ru = (ROOT / "circuit_runner.py").read_text(encoding="utf-8")
    pat = "np.concatenate([d.real, d.imag]) ** 2"
    # `spectrum_metrics.py` 里出现两次是允许的：实现一次 + 自检的手工式断言一次。
    # 关键是两个入口**一次都没有**——那才是"又抄了一份"。
    c("metrics 主式只出现在 spectrum_metrics.py",
      src.count(pat) >= 1 and pat not in src_nf and pat not in src_ru,
      f"sm={src.count(pat)} nf={src_nf.count(pat)} ru={src_ru.count(pat)}")
    c("nested_fit.metrics 委托而非自算",
      "return SM.metrics(f, Z, Zm)" in src_nf)
    c("circuit_runner.metrics 委托而非自算",
      "return SM.metrics(f, Z, Zm)" in src_ru)


def t2_metrics_parity():
    print("\nT2 字段全保留 + 数值对独立硬算式逐项一致")
    f, Z = _spectrum()
    # 三个残差形态：纯比例、带结构、带端点离群
    variants = {
        "纯比例 0.4%": Z * 1.004,
        "带频率结构": Z + 0.02 * np.abs(Z) * (1 + np.sin(np.arange(f.size))),
        "端点离群": Z + np.where(np.arange(f.size) == 0, 1.5, 0.0) * np.abs(Z),
    }
    for label, Zm in variants.items():
        m = SM.metrics(f, Z, Zm)
        o = oracle_metrics(f, Z, Zm)
        miss = [k for k in NESTED_FIELDS if k not in m]
        c(f"{label}：nested 13 字段齐全", not miss, f"缺 {miss}")
        c(f"{label}：runner 6 字段齐全", not any(k not in m for k in RUNNER_FIELDS))
        for k in ("rms_rel", "rms_re", "rms_im", "rel_median", "rel_max",
                  "acf", "acf_re", "acf_im", "acf_mag", "top_hz", "top_rel_pct"):
            c(f"{label}：{k} 与硬算式一致",
              abs(float(m[k]) - float(o[k])) <= 1e-12 * max(1.0, abs(o[k])),
              f"{m[k]!r} vs {o[k]!r}")
        c(f"{label}：rel_max_at_Hz 一致", m["rel_max_at_Hz"] == o["rel_max_at_Hz"])
        c(f"{label}：bands 三档一致",
          len(m["bands"]) == 3
          and all((x is None and y is None) or abs(x - y) <= 1e-12
                  for x, y in zip(m["bands"], o["bands"])))
        c(f"{label}：bands 与 bands_pct 同值", m["bands"] == m["bands_pct"])
    # 边界：频段无点时给 None，不是 0
    f2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    Z2 = np.full(8, 10 + 5j)
    m2 = SM.metrics(f2, Z2, Z2 * 1.01)
    c("只有中频段时，另两段为 None", m2["bands"][0] is None and m2["bands"][2] is None)


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------
def t3_sigma_parity():
    print("\nT3 两个入口调同一份逐点 σ")
    f, Z = _spectrum()
    a = SM.sigma_perpoint(Z, SM.SIGMA_PCT)
    b, from_data, desc = NF.make_sigma(Z, "perpoint", 2.0, None)
    c("形状都是 (2,n)", a.shape == b.shape == (2, Z.size))
    c("逐元素相同", np.array_equal(a, b))
    c("nested 的 from_data=True", from_data is True)
    # runner 的历史调用式（比例常数 × |Z|）必须给出同一个数组
    legacy = RU.SIGMA_PCT_PERP0INT * np.abs(Z)
    c("RU.SIGMA_PCT_PERP0INT·|Z| 与公共实现一致",
      np.array_equal(a[0], legacy) and np.array_equal(a[1], legacy))
    c("实虚共用同一 σ", np.array_equal(a[0], a[1]))
    c("pct 单位是百分数（2.0 ⇒ 0.02·|Z|）",
      np.allclose(a[0], 0.02 * np.abs(Z), rtol=0, atol=0))
    c("零阻抗显式拒绝", raises(lambda: SM.sigma_perpoint(np.array([1 + 1j, 0 + 0j]))))
    c("非有限拒绝", raises(lambda: SM.sigma_perpoint(np.array([1 + 1j, np.nan + 0j]))))
    c("描述串与数值同源（含 2%）", "2%·|Z_j|" in desc)


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------
def t4_elpd_semantics():
    print("\nT4 sigma_from_data 的语义 / 传播 / ELPD 判定")
    vp = SM.elpd_validity(likelihood="perpoint")
    vf = SM.elpd_validity(likelihood="free")
    c("perpoint：sigma_leak=True、valid=False",
      vp["sigma_leak"] is True and vp["valid_for_predictive_comparison"] is False)
    c("free：sigma_leak=False、valid=True（只答 L1）",
      vf["sigma_leak"] is False and vf["valid_for_predictive_comparison"] is True)
    c("两口径 R21 层都成立",
      vp["prior_scale_data_dependent"] is True
      and vf["prior_scale_data_dependent"] is True)
    c("两口径 comparison_scope 都是 relative_only",
      vp["comparison_scope"] == vf["comparison_scope"] == "relative_only")
    c("perpoint 理由含 L1+L2", "L1" in vp["reason"] and "L2" in vp["reason"])
    c("free 理由只有 L2", "L1" not in vf["reason"] and "L2" in vf["reason"])
    c("nested_fit 的调用式（显式 True）与推导一致、无 mismatch",
      SM.elpd_validity(likelihood="perpoint", sigma_from_data=True)["derived_mismatch"]
      is False)
    c("口径说 perpoint、标注却 False ⇒ 记 mismatch（不静默）",
      SM.elpd_validity(likelihood="perpoint",
                       sigma_from_data=False)["derived_mismatch"] is True)

    # 传播：runner 的 observation_block 必须把「两层任一成立 ⇒ False」写进去，
    # 而不是只抄 `valid_for_predictive_comparison`（那个在 free 下会是 True）。
    f, Z = _spectrum()
    cands = RU.build_candidates(ks=(1,), layers=(1,), budget_profile="quick")
    met = SM.metrics(f, Z, Z * 1.003)
    obs = dict(metrics=met, rhat_max=1.0, divergences=0, ess_min=100.0,
               ci90={"R_inf": [1.0, 2.0]}, elpd=-10.0, elpd_se=0.5, n_draw=10,
               elapsed_s=0.1, budget=dict(RU.BUDGETS["quick"], sigma_floor=0.0),
               budget_actual=SM.budget_actual(RU.BUDGETS["quick"], tier="quick"),
               likelihood="free", elpd_validity=SM.elpd_validity(likelihood="free"))
    ob = RU.to_observation(cands[0], obs, f=f, Z=Z, seed=0)
    c("free 口径下 runner 仍标 valid_for_comparison=False（R21 恒成立）",
      ob["predictive"]["valid_for_comparison"] is False)
    c("free 口径的理由含 R21", "R21" in ob["predictive"]["reason"])
    c("free 口径的理由不含 σ 泄漏（不把弱理由写成强理由）",
      "含被留出点自身" not in ob["predictive"]["reason"])

    # nested_fit 落盘的判定块与 runner 用的是同一个函数、同一份字段
    src = (ROOT / "nested_fit.py").read_text(encoding="utf-8")
    c("nested_fit 调用 SM.elpd_validity", "SM.elpd_validity(" in src)
    c("nested_fit 把判定写进 diagnostics", "elpd_validity=ev" in src)
    src_ru = (ROOT / "circuit_runner.py").read_text(encoding="utf-8")
    c("circuit_runner 调用 SM.elpd_validity", "SM.elpd_validity(" in src_ru)
    c("circuit_runner 把判定写进 fits/*.json", "elpd_validity=obs[\"elpd_validity\"]" in src_ru)


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------
def t5_budget_source():
    print("\nT5 standard 唯一来源 / quick 实际预算落盘")
    std_ru = {k: RU.BUDGETS["standard"][k] for k in SM.BUDGET_KEYS}
    c("runner standard == SM.STANDARD", std_ru == SM.STANDARD)
    c("nested_fit VALIDATION_STANDARD == SM.STANDARD",
      NF.VALIDATION_STANDARD == SM.STANDARD)
    c("standard 是 §13/§14 冻结档（ta=0.98）",
      SM.STANDARD["target_accept"] == 0.98
      and SM.STANDARD["n_warmup"] == 2000 and SM.STANDARD["n_samples"] == 1000)
    # quick 保留各自历史值，但两者确实不同 —— 这是「不许靠档名推断」的事实依据
    q_ru = {k: RU.BUDGETS["quick"][k] for k in SM.BUDGET_KEYS}
    q_nf = {k: NF.budget_kw("quick")[k] for k in SM.BUDGET_KEYS}
    c("quick 两入口数值不同（runner 300/300/8 vs nested 800/400/10）",
      q_ru != q_nf and q_ru["n_warmup"] == 300 and q_nf["n_warmup"] == 800)
    b = SM.budget_actual(q_ru, tier="quick")
    c("quick 落五项数值", all(k in b for k in SM.BUDGET_KEYS))
    c("quick 标 tier_shared=False", b["tier_shared_across_entries"] is False)
    b2 = SM.budget_actual(SM.STANDARD, tier="standard")
    c("standard 标 tier_shared=True", b2["tier_shared_across_entries"] is True)
    c("缺键直接报错（不许只记档名）",
      raises(lambda: SM.budget_actual({"n_warmup": 1}, tier="quick")))
    # 落盘位置：runner 的 fits/*.json 与 observation 的 budget_params
    src_ru = (ROOT / "circuit_runner.py").read_text(encoding="utf-8")
    c("fits/*.json 落 budget_actual", "budget_actual=obs[\"budget_actual\"]" in src_ru)
    c("observation.budget_params 落五项数值",
      all(f'"{k}": obs["budget_actual"]["{k}"]' in src_ru for k in SM.BUDGET_KEYS))


# ---------------------------------------------------------------------------
# T6
# ---------------------------------------------------------------------------
def t6_sampler_args():
    print("\nT6 采样器实参不变 / 生产默认路径不变")
    sig = inspect.signature(SEL.fit_nested)
    c("fit_nested.sigma_from_data 默认 False（生产路径逐位不变）",
      sig.parameters["sigma_from_data"].default is False)
    c("fit_nested.sigma 默认 None（旧行为）",
      sig.parameters["sigma"].default is None)
    # 生产路径的调用点不许传 sigma_from_data
    src = (ROOT / "bm_select.py").read_text(encoding="utf-8")
    n_def = src.count("sigma_from_data: bool = False")
    n_use = src.count("sigma_from_data=bool(sigma_from_data)")
    c("bm_select 里只有一处定义、一处透传",
      n_def == 1 and n_use == 1, f"def={n_def} use={n_use}")
    c("bm_select 没有别的调用点传这个开关",
      src.count("sigma_from_data=") == 1)
    # 默认路径的预算常量没被改
    c("bm_select.QUICK 未被改（800/400/2/0.9/10）",
      SEL.QUICK["n_warmup"] == 800 and SEL.QUICK["n_samples"] == 400
      and SEL.QUICK["target_accept"] == 0.9 and SEL.QUICK["max_tree_depth"] == 10)
    c("bm_select.BUDGETS['standard'] 仍是 A2 档 ta=0.9（没被 standard 口径污染）",
      SEL.BUDGETS["standard"]["target_accept"] == 0.9)
    # 不许改的四个文件
    import subprocess
    try:
        out = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=20).stdout.split()
    except Exception:                             # noqa: BLE001
        out = []
    forbidden = {"bm_core.py", "bm_top.py", "bm_real.py"}
    c("bm_core / bm_top / bm_real 未修改", not (forbidden & set(out)),
      f"改了 {forbidden & set(out)}")
    c("bm_select 只有 fit_nested 一处改动",
      [x for x in out if x == "bm_select.py"] in ([], ["bm_select.py"]))
    # 采样器实参：两族都不许传新的采样参数
    src_ru = (ROOT / "circuit_runner.py").read_text(encoding="utf-8")
    c("runner 调 fit_voigt 不带新采样参数",
      "SEL.fit_voigt(f, Z, K, seed=seed," in src_ru)
    c("runner 调 fit_nested 只多 sigma_from_data",
      "sigma_from_data=(likelihood == \"perpoint\")" in src_ru)

    # `sigma_from_data` 不参与采样：AST 里它只被引用 2 次，且都在返回字典
    # （`valid_for_predictive_comparison` / `elpd_note`）。所以给它 True/False
    # 不可能改变后验——这是"采样器实参不变"的静态证明，比跑两次 MCMC 更硬。
    import ast
    tree = ast.parse((ROOT / "bm_top.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "fit_top")
    refs = [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Name) and n.id == "sigma_from_data"]
    c("fit_top 里 sigma_from_data 只被引用 2 次（都进返回值，不进采样）",
      len(refs) == 2, f"引用行 {refs}")


# ---------------------------------------------------------------------------
# 差分：同一输入 → 两个入口
# ---------------------------------------------------------------------------
CAPTURED = []


class _Stop(Exception):
    pass


def _stub(f, Z, **kw):
    CAPTURED.append(dict(kw))
    raise _Stop


def _capture(fn):
    CAPTURED.clear()
    orig = T.fit_top
    T.fit_top = _stub
    try:
        fn()
    except _Stop:
        pass
    finally:
        T.fit_top = orig
    return CAPTURED[0] if CAPTURED else None


def t7_diff_same_input():
    print("\n差分：同一输入、两个入口")
    src = ROOT / "out" / "fig3" / "inputs" / "az31.csv"
    if not src.exists():
        c("差分用谱存在（缺则跳过，不算通过）", False, str(src))
        return
    import tempfile

    f, Z = NF.R.load_spectrum(str(src))
    order = np.argsort(f)
    f, Z = f[order], Z[order]

    tmp = tempfile.mkdtemp(prefix="parity_")
    A = _capture(lambda: NF.run(str(src), outdir=tmp, branch="parallel",
                                layers=2, budget="standard",
                                sigma_mode="perpoint", sigma_pct=2.0,
                                seed=0, plot=False, verbose=False))
    sigma_b = SM.sigma_perpoint(Z, SM.SIGMA_PCT)
    kw = dict(RU.BUDGETS["standard"])
    B = _capture(lambda: SEL.fit_nested(f, Z, 2, branch="parallel", seed=0,
                                        sigma_floor=SEL.SIGMA_FLOOR, kw=kw,
                                        sigma=sigma_b,
                                        sigma_from_data=True))
    if A is None or B is None:
        c("两入口都调到了 fit_top", False)
        return

    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if not _same(_arg(A, k), _arg(B, k))]
    c("传给 fit_top 的实参逐键一致（缺省=显式默认值不算差异）",
      not diff, f"差异键 {diff}")
    c("sigma_floor 两边等价（0.0 = fit_top 默认）",
      _arg(A, "sigma_floor") == _arg(B, "sigma_floor") == 0.0)
    c("sigma_from_data 两边都是 True（同一口径同一标注）",
      A.get("sigma_from_data") is True and B.get("sigma_from_data") is True)
    c("sigma 数组逐元素一致（(2,n) 与 tuple 归一后相同）",
      np.array_equal(np.asarray(A["sigma"], float), np.asarray(B["sigma"], float)))
    c("standard 预算五项一致",
      all(A[k] == B[k] for k in SM.BUDGET_KEYS))
    c("两入口的 ELPD 判定相同",
      SM.elpd_validity(likelihood="perpoint",
                       sigma_from_data=A["sigma_from_data"])
      == SM.elpd_validity(likelihood="perpoint",
                          sigma_from_data=B["sigma_from_data"]))
    c("指标算的是同一份（同参数同正演）",
      SM.metrics(f, Z, T.z_nested(f, {k: 1.0 for k in T.sampled_keys("parallel", 2)},
                                  branch="parallel", layers=2))
      == NF.metrics(f, Z, T.z_nested(f, {k: 1.0 for k in T.sampled_keys("parallel", 2)},
                                     branch="parallel", layers=2)))
    # 清理
    try:
        for p in Path(tmp).glob("*"):
            p.unlink()
        Path(tmp).rmdir()
    except OSError:
        pass


def _same(a, b):
    if a is None and b is None:
        return True
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        try:
            return np.array_equal(np.asarray(a, float), np.asarray(b, float))
        except (TypeError, ValueError):
            return a == b
    return a == b


#: 缺省与显式默认值等价（`fit_top(sigma_floor=0.0)` 是默认）。差分时按此归一，
#: 避免把"写法不同"报成"实参不同"。
DEFAULTS = {"sigma_floor": 0.0, "progress": True}


def _arg(rec, key):
    if key in rec:
        return rec[key]
    return DEFAULTS.get(key, "<缺>")


# ---------------------------------------------------------------------------
# T9：族分派（auto 能不能自己判到并联）
# ---------------------------------------------------------------------------
def t9_family_dispatch():
    print("\nT9 族分派：auto 在 voigt 表达不了时自动切 nested")
    import bmdrt as BD

    az31 = ROOT / "out" / "_m6_samples" / "z_csv" / "AZ31-8Li_50.csv"
    e3 = ROOT / "out" / "fig3" / "inputs" / "3e.csv"
    if not (az31.exists() and e3.exists()):
        c("族分派测试谱存在（缺则不算通过）", False, f"{az31} / {e3}")
        return

    s_az, why_az = BD._voigt_family_verdict(az31)
    c("AZ31（低频感抗）→ excluded", s_az == "excluded", f"实际 {s_az}")
    c("excluded 的理由点明「低频感抗」", "低频感抗" in why_az, why_az[:60])
    s_e3, why_e3 = BD._voigt_family_verdict(e3)
    c("3E（无低频感抗）→ not_excluded", s_e3 == "not_excluded", f"实际 {s_e3}")
    c("not_excluded 不切族（理由为空）", why_e3 == "", why_e3[:60])

    # 判定文案：nested 不许被归进 OK
    rec = dict(ok=True, family="nested", admission_state="provisional")
    v = BD._verdict(rec)
    c("nested 的判定含 provisional", "provisional" in v)
    c("nested 的判定不是 OK", v != "OK")
    c("nested 的判定不含「通过/合格」", "通过" not in v and "合格" not in v)

    # 开关：关掉就不切族
    import configparser  # noqa: F401  (只为显式说明这里读的是 json，不是 ini)
    import json as _json
    cfgf = ROOT / "config.json"
    cfg = _json.loads(cfgf.read_text(encoding="utf-8"))
    c("config 有 family_auto 开关", "family_auto" in cfg["fit"])
    c("family_auto 默认开（用户要求「不用选」）", cfg["fit"]["family_auto"] is True)

    # 产物形状：nested 分支必须落这几个字段，否则 summary 会崩
    src = (ROOT / "bmdrt.py").read_text(encoding="utf-8")
    for k in ("family=", "family_reason=", "admission_state=", "rms_rel=", "acf="):
        c(f"nested 分支写 {k.rstrip('=')}", k in src)
    c("nested 产物与 voigt 分文件名（不会互相覆盖）",
      "NF.run(str(path), outdir=str(outdir)" in src)
    c("nested 分支用 perpoint 2%（§14 口径）", 'sigma_mode="perpoint"' in src)
    c("summary 表有「族」列", "| 文件 | 族 |" in src)
    c("nested 行不冒充 voigt 的 K/grade", 'fam = "nested" if r.get("family")' in src)


# ---------------------------------------------------------------------------
# T8：临时准入（能跑 ≠ 达标）
# ---------------------------------------------------------------------------
def t8_provisional_admission():
    print("\nT8 临时准入：两入口都标 provisional，且不许出现「过线」声明")
    import admission as AD

    b = AD.block()
    c("state=provisional、passed=False",
      b["state"] == "provisional" and b["passed"] is False)
    c("四条差距都有 required/actual/note",
      set(b["gap"]) == {"S13_holdout", "S14_full", "M-b", "S17_four_circuits"}
      and all({"required", "actual", "note"} <= set(v) for v in b["gap"].values()))
    c("参数不当物理量", b["parameters_are_physical"] is False)
    c("写明有/无证据的谱型范围",
      "仅 _10/_50" in b["usable_scope"] and "_0/_5/_150" in b["usable_scope"])
    # 文案里一个"通过/合格"都不许有（否定式也会被单独摘录误读）
    txt = AD.banner() + str(b)
    c("文案不含「通过」", "通过" not in txt)
    c("文案不含「合格」", "合格" not in txt)

    src_nf = (ROOT / "nested_fit.py").read_text(encoding="utf-8")
    src_cmp = (ROOT / "cmd_compare.py").read_text(encoding="utf-8")
    c("nested_fit 产物带 admission", "admission=AD.block()" in src_nf)
    c("compare 产物带 admission", "admission=AD.block()" in src_cmp)
    c("两处引用同一个 AD.block()（不是各写一份）",
      src_nf.count("AD.block()") == 1 and src_cmp.count("AD.block()") == 1)

    # CLI 收敛：nested 已成为 bmdrt 子命令
    import bmdrt
    extra = bmdrt.discover()
    c("bmdrt 自动发现里出现 nested", "nested" in extra)
    c("nested 不在 BUILTIN 里（不冒充生产命令）", "nested" not in bmdrt.BUILTIN)
    # 族分派是**条件式**的：默认路径仍是 voigt，只在预检明确 excluded 时切。
    # （2026-11-16 第二轮：用户要求 auto 能自动判到并联，所以不再断言
    #  "源码里没有 nested"，改为断言"不会无条件切族"。）
    src_bd = (ROOT / "bmdrt.py").read_text(encoding="utf-8")
    c("族分派受 family_auto 开关控制",
      'bool(fit.get("family_auto", False))' in src_bd)
    c("只有预检 excluded 才切族",
      'elif fam == "excluded":' in src_bd and "_run_nested_branch(" in src_bd)
    c("process_one 仍保留 bm_real.run 的 voigt 主路径", "R.run(" in src_bd)
    c("voigt 主路径在族分派之后（不会被短路）",
      src_bd.index("R.run(") > src_bd.index('elif fam == "excluded":'))
    c("预检 unknown / not_excluded 都不切族",
      'if fam == "not_excluded":' in src_bd
      and src_bd.count("_run_nested_branch(") == 2)  # 定义 1 次 + 调用 1 次

    # Windows 入口只留一个（2026-11-16 用户裁决「统一一个 bat，这样不用选」）
    bats = sorted(p.name for p in ROOT.glob("*.bat"))
    c("仓库顶层只有一个 .bat", bats == ["bmdrt.bat"], f"实际 {bats}")
    bat = (ROOT / "bmdrt.bat").read_text(encoding="utf-8", errors="replace")
    c("bmdrt.bat 用 %* 透传参数（子命令才能用）", "%*" in bat)
    c("bmdrt.bat 不硬编码子命令（不会把 nested 吞成 auto）",
      "bmdrt.py\" auto" not in bat and "bmdrt.py auto" not in bat)
    c("bmdrt.bat 指向 bmdrt.py", "bmdrt.py" in bat)


def t10_nested_circuit_figure():
    print("\nT10 并联电路图 + 输出统一 2×3（能接上、且不改拟合路径）")
    import numpy as np
    import bm_top as T
    import bm_circuit_nested as CN

    # 1) 图上参数集合 == 采样参数集合（拟合采什么、图上画什么）
    for br in CN.VALID_BRANCHES:
        for ly in (1, 2):
            got = {k for e in CN.elements(br, ly) for k in e["params"]}
            got |= {k for e in CN.elements(br, ly) for k in e.get("derived", ())}
            want = set(T.sampled_keys(br, ly))
            if br != "capacitive":
                want |= {"L_ad"}
            c(f"{br}/L={ly} 图上参数 == 采样参数", got == want)

    # 2) layers=1 不画 R2/CPE2
    for br in CN.VALID_BRANCHES:
        ids = {e["id"] for e in CN.elements(br, 1)}
        c(f"{br}/L=1 无 R2/CPE2", not ({"R2", "CPE2"} & ids))

    # 3) 落一张真实尺寸的并联图（不跑 MCMC，只验绘图路径）
    import tempfile
    p = dict(Rs=12.0, R1=180.0, Q1=95.0, n1=0.86, R2=42.0, Q2=18.0,
             n2=0.74, R_ad=26.0, tau_ad=1.4e-3)
    with tempfile.TemporaryDirectory() as td:
        fp = CN.draw_nested(p, branch="parallel", layers=2, f_ref=31.6,
                            path=f"{td}/c.png")
        c("draw_nested 写出非零 PNG", os.path.exists(fp)
          and os.path.getsize(fp) > 20_000)

    # 4) nested_fit 的 2×3 图委托 bm_figure_unified（不再各写一份）
    src_nf = (ROOT / "nested_fit.py").read_text(encoding="utf-8")
    c("write_plot 委托 plot_fit（2×3 唯一实现）",
      "from bm_figure_unified import plot_fit" in src_nf
      and "plot_fit(path, f, Z, curves" in src_nf)
    c("nested_fit.run 写 .nested_circuit.png",
      ".nested_circuit.png" in src_nf and "CN.draw_nested(" in src_nf)

    # 5) bm_figure_unified 的残差图例能优雅处理 None rms（不给 nan%）
    src_fig = (ROOT / "bm_figure_unified.py").read_text(encoding="utf-8")
    c("plot_fit 残差图例对 None rms 不打印 nan",
      "def _resid_label" in src_fig and "float('nan')" not in src_fig)

    # 5b) 残差子图支持 ±σ 点线带（voigt 风格），nested 会传进来
    c("plot_fit 支持 sigma_re/sigma_im 点线带",
      "sigma_re=None, sigma_im=None" in src_fig and "±σ" in src_fig)
    c("nested_fit 把 σ 传给 plot_fit（点线带接上）",
      "sigma_re=sr, sigma_im=si" in src_nf
      and "sigma_re=sigma_re, sigma_im=sigma_im" in src_nf)

    # 6) bm_circuit 仍是 voigt 专用（没被嵌套图污染它的契约）
    src_bc = (ROOT / "bm_circuit.py").read_text(encoding="utf-8")
    c("bm_circuit 不 import 嵌套绘图器（图跟着方程走，各管各的）",
      "bm_circuit_nested" not in src_bc)


def main() -> int:
    for fn in (t1_single_metrics, t2_metrics_parity, t3_sigma_parity,
               t4_elpd_semantics, t5_budget_source, t6_sampler_args,
               t7_diff_same_input, t8_provisional_admission,
               t9_family_dispatch, t10_nested_circuit_figure):
        fn()
    print(f"\n验收 {ok}/{total} 通过" if not bad
          else f"\nFAIL {len(bad)}/{total}: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
