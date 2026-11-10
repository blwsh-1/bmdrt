#!/usr/bin/env python
"""`voigt_inadmissible` 终态的离线单测：**不采样**，只验分支逻辑。

背景：`bm_select._compare_group` 新增了这个终态，但它在四类合成用例上**一次都不触发**
（Δ 显著时仍报 `nested`，降级时走旧分支）。所以跑一遍自检并不能证明它是对的——
必须直接喂构造好的候选，逐个分支验。

覆盖：
  1. `voigt_can_represent` 的**分频段**判据——低频感抗需 `adsorption`、高频需
     `include_L`，且「开 `include_L` 但不能覆盖低频」时必须仍判不可用；
  2. `_compare_group` 四个分支的顺序：Δ 显著指向 nested / voigt / 不显著+可选 /
     不显著+不可选；
  3. `_verdict` 对新终态的处理：**既不算错切也不算命中**；
  4. `_health` 的计数；
  5. `VOIGT_CAPS` 与 `fit_voigt` 签名一致性（判据不能描述一支不存在的族）。

用法：`python evidence/nested_topology/m4_inadmissible_check.py`
"""
from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import bm_select as S            # noqa: E402


def _cand(label: str, family: str, elpd: float, pt, rhat=1.00, div=0) -> dict:
    """构造一个最小可用候选：只需 `_compare_group` 用到的字段。"""
    return dict(family=family, label=label, elpd=float(elpd),
                pointwise=np.asarray(pt, float), rhat=rhat, div=div,
                ok=True, dof=2, branch=None if family == "voigt" else "parallel",
                se=1.0, grade=None, raw=dict(s={}))


def _pair(base: float, delta: float, n: int = 40, spread: float = 0.3):
    """造一对逐点 elpd：nested − voigt ≡ delta，逐点差有方差（dse 才有意义）。

    `spread` 决定 `dse ≈ spread·sqrt(n)`，进而决定 `z = delta/dse`。
    第一版用 `spread=1.0`、`n=40`，得 `dse≈6.3`，于是 Δ=12 的 `z` 只有 1.9 < 2，
    仍判不显著——**测的是测试构造，不是被测代码**。取 0.3 后
    `dse≈1.9`：Δ=12 → z≈6.3（显著）、Δ=0.5 → z≈0.26（不显著），两档都到位。
    """
    rng = np.random.default_rng(0)
    v = base + rng.normal(0, spread, n)
    shifts = rng.normal(0, spread, n)
    shifts = shifts - shifts.mean() + delta / n
    return v, v + shifts


def case_represent():
    print("一、voigt_can_represent：分频段")
    f = np.logspace(-2, 5, 50)
    zmax = 1000.0
    base = np.full(50, 500.0, complex)

    # 低频感抗：最低 6 点给正虚部
    z_lo = base.copy()
    z_lo[:6] += 1j * 0.10 * zmax
    # 高频感抗：最高 6 点
    z_hi = base.copy()
    z_hi[-6:] += 1j * 0.10 * zmax
    # 无感抗
    z_ok = base - 1j * 0.30 * zmax

    rows = [
        ("无感抗", z_ok, dict(), True),
        ("低频感抗", z_lo, dict(), False),
        ("低频感抗 + include_L", z_lo, dict(include_L=True), False),
        ("低频感抗 + adsorption", z_lo, dict(adsorption="RL"), True),
        ("高频感抗", z_hi, dict(), False),
        ("高频感抗 + include_L", z_hi, dict(include_L=True), True),
        ("两段都有", z_lo + (z_hi - base), dict(), False),
        ("两段都有 + 只开 L", z_lo + (z_hi - base), dict(include_L=True), False),
        ("两段都有 + 全开", z_lo + (z_hi - base),
         dict(include_L=True, adsorption="RL"), True),
    ]
    ok_all = True
    for name, Z, kw, want in rows:
        got, why = S.voigt_can_represent(f, Z, **kw)
        good = (got == want)
        ok_all &= good
        print(f"  {'✓' if good else '✗'} {name:24s} 期望={want!s:5s} 实得={got!s:5s}"
              f"  {why[:46]}")
    print(f"  → {'全部符合' if ok_all else '有偏差'}")
    return ok_all


def case_branches():
    print("\n二、_compare_group：四个分支的顺序")
    ok_all = True
    for delta, voigt_ok, want in (
            (+12.0, False, "nested"),        # Δ 显著指向 nested：正向信息要保留
            (-12.0, False, "voigt"),         # Δ 显著指向 voigt：不静默改判
            (+0.5, True, "indistinguishable"),
            (+0.5, False, "voigt_inadmissible"),
            (-0.5, False, "voigt_inadmissible"),
    ):
        v, n = _pair(100.0, delta)
        V = [_cand("K=2", "voigt", 100.0, v)]
        N = [_cand("parallel/L=2", "nested", 100.0 + delta, n)]
        g = S._compare_group(V, N, 1600, False, None,
                             voigt_ok=voigt_ok,
                             voigt_why="" if voigt_ok else "低频感抗需把 R∥L 放进第二并联节点")
        good = (g["decision"] == want)
        ok_all &= good
        print(f"  {'✓' if good else '✗'} Δ={delta:+6.2f} voigt_ok={voigt_ok!s:5s} "
              f"→ {g['decision']:<20s} 期望 {want}")
        # reason / note 不能混：reason 专指「算不出」
        if g["decision"] == "voigt_inadmissible":
            bad = (g["reason"] != "" or not g["note"])
            ok_all &= not bad
            if bad:
                print(f"      ✗ reason/note 混了：reason={g['reason']!r} "
                      f"note={g['note']!r}")
    print(f"  → {'全部符合' if ok_all else '有偏差'}")
    return ok_all


def case_verdict():
    print("\n三、_verdict / _health：新终态的计数口径")
    ok_all = True
    for expect, decisions, want in (
            ("nested", ["voigt_inadmissible"], (True, 0, 0)),
            ("voigt", ["voigt_inadmissible"], (True, 0, 0)),
            ("nested", ["nested", "voigt_inadmissible"], (True, 1, 0)),
            ("voigt", ["nested"], (False, 0, 1)),      # 真正的错切仍要抓到
    ):
        got = S._verdict(expect, decisions)
        good = (got == want)
        ok_all &= good
        print(f"  {'✓' if good else '✗'} expect={expect:<7s} {decisions!s:34s}"
              f" → {got}  期望 {want}")

    h = S._health([dict(decision="voigt_inadmissible", decisions=["voigt_inadmissible"]),
                   dict(decision="nested", decisions=["nested"])])
    good = h["voigt_inadmissible"] == 1 and h["gate_ok"] is True
    ok_all &= good
    print(f"  {'✓' if good else '✗'} _health：voigt_inadmissible="
          f"{h['voigt_inadmissible']}（期望 1）、gate_ok={h['gate_ok']}")
    return ok_all


def case_caps_consistency():
    print("\n四、VOIGT_CAPS 与 fit_voigt 一致（2026-11-10 起改为**签名推导**）")
    import inspect
    sv = inspect.signature(S.fit_voigt).parameters
    src_fit = inspect.getsource(S.fit_voigt)
    src_ref = inspect.getsource(S._refit)
    # 硬编码一段谱，只验「默认取值 = 显式传默认值」这一条恒等式，不做拟合。
    f = np.logspace(-2, 5, 20)
    Z = np.full(20, 500.0, complex) + 1j * 0.0
    Z[:3] += 1j * 100.0                       # 低频感抗，让判据真的走进分支
    checks = [
        ("VOIGT_CAPS.include_L 由签名推导",
         sv["include_L"].default == S.VOIGT_CAPS["include_L"]),
        ("VOIGT_CAPS.adsorption 由签名推导",
         sv["adsorption"].default == S.VOIGT_CAPS["adsorption"]),
        ("adsorption 已转发给 R.fit_one（否则开关静默失效）",
         "adsorption=adsorption" in src_fit),
        ("_refit 复现候选的族能力（重跑不得换候选空间）",
         "include_L" in src_ref and "adsorption" in src_ref),
        ("voigt_can_represent 不传参 == 显式传模块默认",
         S.voigt_can_represent(f, Z) ==
         S.voigt_can_represent(f, Z, include_L=S.VOIGT_CAPS["include_L"],
                               adsorption=S.VOIGT_CAPS["adsorption"])),
    ]
    ok = True
    for name, good in checks:
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {name}")
    print(f"  VOIGT_CAPS={S.VOIGT_CAPS}  fit_voigt 默认 "
          f"include_L={sv['include_L'].default}, adsorption={sv['adsorption'].default!r}")
    return ok


def case_consistency_with_probe():
    """两处「有感抗」判据必须同口径。

    `bm_select._ind_freqs` 与 `m4_spectrum_health.spectrum` 各有一份实现——
    **有意分两份**：探针要能在没有 jax 的环境里跑（`bm_select` import `bm_core`，
    后者 import jax），所以不能简单合并。但两份判据漂移过一次（`|Im|` vs 正虚部），
    代价是全谱 25–46% 误判，所以这里做交叉核对。
    """
    print("\n五、与探针脚本的判据一致（同口径交叉核对）")
    try:
        sys.path.insert(0, os.path.join(_ROOT, "evidence", "nested_topology"))
        import m4_spectrum_health as H
    except Exception as e:                       # 探针不可用时明确记为跳过
        print(f"  ⚠ 跳过：{type(e).__name__}: {e}")
        return True

    f = np.logspace(-2, 5, 50)
    base = np.full(50, 500.0, complex)
    cases = [("无感抗", base - 1j * 300.0)]
    for k, name in ((3, "低 3 点"), (12, "低 12 点")):
        z = base.copy()
        z[:k] += 1j * 120.0
        cases.append((name, z))
    for k, name in ((4, "高 4 点"), (10, "高 10 点")):
        z = base.copy()
        z[-k:] += 1j * 120.0
        cases.append((name, z))

    ok_all = True
    for name, Z in cases:
        a = S._ind_freqs(f, Z).size
        b = H.stability_shape(f, Z)["has_ind"]
        good = (a > 0) == bool(b)
        ok_all &= good
        print(f"  {'✓' if good else '✗'} {name:10s} _ind_freqs={a:2d} 点  "
              f"探针 has_ind={b!s:5s}")
    print(f"  → {'一致' if ok_all else '不一致，两处阈值的口径已经漂了'}")
    return ok_all


def main():
    res = [case_represent(), case_branches(), case_verdict(), case_caps_consistency(),
           case_consistency_with_probe()]
    print("\n" + "=" * 62)
    print(f"全部通过：{'是' if all(res) else '否'}")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main())
