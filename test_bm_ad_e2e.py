# -*- coding: utf-8 -*-
"""test_bm_ad_e2e.py —— M4 端到端接线：fit_one / init_values / run / CLI。

判据层（detect_ad / select_adsorption）由 test_bm_ad_select.py 覆盖，
这里只测**接线**：adsorption="RL" 从 CLI 一直透到采样、派生量、落盘 json，
且默认路径逐位不变（回归红线）。

用法：python test_bm_ad_e2e.py   （含 3 次 quick 档 NUTS，约 2-4 分钟）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bm_ad as AD            # noqa: E402
import bm_real as R           # noqa: E402

F = np.logspace(0, 5, 31)
ARCS = [(30.0, 300.0, 0.85)]
R_INF, NOISE = 10.0, 0.10


def _synth(f, with_ad=True, seed=3):
    w = 2 * np.pi * f
    Z = np.full(f.shape, R_INF, complex)
    for Rv, fp, phi in ARCS:
        Z = Z + Rv / (1.0 + (1j * w / (2 * np.pi * fp)) ** phi)
    if with_ad:
        Z = Z + AD.z_ad(f, 5.0, 1 / (2 * np.pi * 10.0))
    rng = np.random.default_rng(seed)
    return Z + NOISE * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))


def case1_init_values_ad_keys():
    """init_values(ad_init=...) 把 Rad/tau_ad 加进播种 dict；不给则不加。"""
    pk = [dict(f_peak_Hz=300.0, R_ohm=30.0, tau_s=1 / (2 * np.pi * 300))]
    pres = dict(R_inf=9.0)
    d0 = R.init_values(pk, pres, 1, ad_init=None)
    assert "Rad" not in d0 and "tau_ad" not in d0, "默认播种混入吸附键"
    d1 = R.init_values(pk, pres, 1, ad_init={"Rad": 4.0, "tau_ad": 0.016,
                                             "init_reliable": True})
    assert d1.get("Rad") == 4.0 and abs(d1.get("tau_ad") - 0.016) < 1e-12, \
        f"吸附初值没进播种：{sorted(d1)}"
    return "播种 dict 条件扩展正确"


def case2_fit_one_derives_ad_identifiability():
    """fit_one(adsorption='RL') 必须产出 p_ad_small / f_c_out_prob / L_ad。"""
    Z = _synth(F)
    kw = dict(n_warmup=400, n_samples=200, num_chains=2, target_accept=0.95,
              max_tree_depth=11, progress=False, seed=0,
              extra_fields=("num_steps",))
    r = R.fit_one(F, Z, K=1, include_L=False, zeta="1.0", kw=kw,
                  adsorption="RL")
    assert r.get("adsorption") == "RL"
    for key in ("p_ad_small", "f_c_out_prob", "L_ad"):
        assert key in r, f"fit_one 结果缺吸附派生量 {key}"
        assert np.isfinite(r[key]), f"{key} 非有限：{r[key]}"
    assert "Rad" in r["s"], "后验样本缺 Rad"
    assert r["p_ad_small"] < 0.5, \
        f"真环谱上 P(Rad<3σ̂)={r['p_ad_small']:.2f} 偏大（采样或口径有问题）"
    return (f"p_ad_small={r['p_ad_small']:.2f} f_c_out={r['f_c_out_prob']:.2f} "
            f"L_ad={r['L_ad'] * 1e6:.2f} μH")


def case3_run_auto_adsorption_end_to_end():
    """run(auto_adsorption=True)：门控 → 补跑 ad → 选择 → 落盘 json 全链路。"""
    tmp = ROOT / "out" / "_ad_e2e"
    tmp.mkdir(parents=True, exist_ok=True)
    Z = _synth(F, with_ad=True, seed=22)
    csv = tmp / "ad_e2e.csv"
    lines = ["f_Hz,Zre,Zim"] + [f"{f:.6g},{z.real:.6g},{z.imag:.6g}"
                                for f, z in zip(F, Z)]
    csv.write_text("\n".join(lines), encoding="utf-8")

    out_dir = tmp / "res"
    out_dir.mkdir(exist_ok=True)
    R.OUT = out_dir
    R.NULL_CACHE = {}
    best = R.run(str(csv), K_force=1, quick=True, kk_mode="off",
                 auto_adsorption=True)
    assert best is not None, "run() 返回 None（KK 拒了？应为 off）"
    js = out_dir / "real_ad_e2e.json"
    assert js.exists(), "结果 json 未写出"
    d = json.loads(js.read_text(encoding="utf-8"))
    for key in ("ad_gate", "ad_selection", "adsorption"):
        assert key in d, f"json 缺字段 {key}（save_out 未接）"
    gate, sel = d["ad_gate"], d["ad_selection"]
    assert gate is not None and gate["status"] in (
        "strong_candidate", "weak_candidate", "monotonic_rise",
        "capacitive_tail", "noise_only"), f"gate 异常：{gate}"
    assert sel["choice"] in ("none", "ad"), f"choice 非法：{sel}"
    assert sel["status"] in (
        "switched", "guard_degraded", "uncertain", "ad_worse",
        "gate_not_strong", "comparison_incomplete", "fit_missing",
        "skipped_by_gate", "not_tested", "ws_taken"), f"status 非法：{sel}"
    return (f"gate={gate['status']}, sel={sel['status']}/{sel['choice']}")


def case4_cli_passthrough():
    """bmdrt.py：--adsorption / --auto-adsorption 解析、_apply_cli、透传。"""
    import bmdrt as E
    paths, opt = E.parse_opts(["--adsorption", "RL", "--auto-adsorption"])
    assert opt.get("adsorption") == "RL", f"opt.adsorption={opt.get('adsorption')}"
    assert opt.get("auto_adsorption") is True
    cfg = E._apply_cli(E._deep_merge(E.DEFAULTS, {}), opt)
    assert cfg["fit"].get("adsorption") == "RL"
    assert cfg["fit"].get("auto_adsorption") is True
    # DEFAULTS 里要有这两个键（config.json 缺键时的兜底）
    assert "adsorption" in E.DEFAULTS["fit"] and "auto_adsorption" in E.DEFAULTS["fit"]
    # process_one 透传：源码里必须有 adsorption=fit.get(...)
    src = (ROOT / "bmdrt.py").read_text(encoding="utf-8")
    assert "fit.get(\"adsorption\"" in src or 'fit.get("adsorption"' in src, \
        "process_one 没透传 adsorption"
    return "CLI 三层（解析/配置/透传）齐"


def case5_default_path_zero_touch():
    """回归红线：不传吸附参数时 fit_one 结果键与吸附无关字段全 None/缺省。"""
    Z = _synth(F, with_ad=False)
    kw = dict(n_warmup=300, n_samples=150, num_chains=2, target_accept=0.95,
              max_tree_depth=11, progress=False, seed=0,
              extra_fields=("num_steps",))
    r = R.fit_one(F, Z, K=1, include_L=False, zeta="1.0", kw=kw)
    assert "Rad" not in r["s"] and "tau_ad" not in r["s"]
    assert r.get("adsorption") is None
    assert not np.isfinite(r.get("p_ad_small", float("nan"))), \
        "默认路径 p_ad_small 应为 NaN（未采吸附）"
    return "默认路径无吸附键，派生量 NaN"


CASES = [case1_init_values_ad_keys, case2_fit_one_derives_ad_identifiability,
         case3_run_auto_adsorption_end_to_end, case4_cli_passthrough,
         case5_default_path_zero_touch]

if __name__ == "__main__":
    bad = []
    for fn in CASES:
        try:
            print(f"[PASS] {fn.__name__}  {fn()}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"[FAIL]  {fn.__name__}\n        {e}")
        except Exception as e:                      # noqa: BLE001
            bad.append(fn.__name__)
            print(f"[ERROR] {fn.__name__}  {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - len(bad)}/{len(CASES)} 通过；未通过：{bad}")
    raise SystemExit(1 if bad else 0)
