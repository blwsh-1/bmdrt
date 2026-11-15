# -*- coding: utf-8 -*-
"""③ 阶段 C 第一轮探针：捕获值 == 真实构造实参 + 真实对象属性？

做法
----
**独立取证**：探针自己在 `MCMC.__init__` / `MCMC.run` 上挂类级包装，记录**真正绑定**
到构造与 run 的实参；再拿 `sampler_capture` 产出的记录逐字段对照。生产代码里没有
钩子——包装只存在于本进程，`Hooks.close()` 还原。

四条微预算路径（各 8 warmup + 8 samples，不写产物）：
  A  bm_core.fit / 单链 / progress=False，显式 extra_fields
  B  bm_core.fit / 3 链 / 1 设备（chain_method 被改写）
  C  bm_top.fit_top / 单链（`extra_fields` 无声明入口）
  D  bm_core.fit / CI 在场（progress_bar 被静默压掉）

    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/capture_probe.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

warnings.filterwarnings("ignore")

import bm_core as BC                                    # noqa: E402
import bm_top as BT                                     # noqa: E402
import sampler_capture as C                             # noqa: E402
import sampler_provenance as S                          # noqa: E402

#: 捕获字段 → 构造形参名（对照用）
PARAM = {"n_warmup": "num_warmup", "n_samples": "num_samples",
         "num_chains": "num_chains", "target_accept": "target_accept_prob",
         "max_tree_depth": "max_tree_depth", "chain_method": "chain_method",
         "progress_bar": "progress_bar"}


def _data(n=24):
    f = np.logspace(-2, 4, n)
    w = 2 * np.pi * f
    Z = 10.0 / (1 + 1j * w * 0.01) + 1.0
    return f, Z.astype(complex)


class Hooks:
    """类级包装：只在本进程内取证，跑完 `close()` 还原。

    **必须 `functools.wraps`**：捕获层用 `inspect.signature(MCMC.__init__)` 读库默认值，
    裸包装会让签名变成 `(obj, sampler, **kw)`，于是"没传的字段"读不到默认值 ⇒ 落
    `capture_error`。这不是 bug（它拒绝猜），但会让探针测的东西被它自己扰动。
    """

    def __init__(self):
        import functools

        from numpyro.infer import HMC, MCMC, NUTS

        self.MCMC, self.NUTS, self.HMC = MCMC, NUTS, HMC
        self.orig = {MCMC.__init__: MCMC, MCMC.run: MCMC, NUTS.__init__: NUTS,
                     HMC.__init__: HMC}
        self.inits: list[dict] = []
        self.runs: list[dict] = []
        self.kinits: list[dict] = []
        outer = self
        keep = {}

        def mcmc_init(obj, sampler, **kw):
            outer.inits.append({"kernel": type(sampler).__name__, "bound": dict(kw)})
            keep["mcmc_init"](obj, sampler, **kw)

        def mcmc_run(obj, rng_key, *args, **kw):
            outer.runs.append({"bound": dict(kw)})
            return keep["mcmc_run"](obj, rng_key, *args, **kw)

        def make_kern_hook(orig, name):
            def hook(obj, model, **kw):
                outer.kinits.append({"kind": name, "bound": dict(kw)})
                return orig(obj, model, **kw)     # 直接调原函数：不经过被替换的属性，
            return functools.wraps(orig)(hook)    # 否则 NUTS 的 super() 会再次进钩子

        keep["mcmc_init"] = MCMC.__init__
        keep["mcmc_run"] = MCMC.run
        keep["NUTS"] = NUTS.__init__
        keep["HMC"] = HMC.__init__
        MCMC.__init__ = functools.wraps(MCMC.__init__)(mcmc_init)
        MCMC.run = functools.wraps(MCMC.run)(mcmc_run)
        NUTS.__init__ = make_kern_hook(NUTS.__init__, "NUTS")
        HMC.__init__ = make_kern_hook(HMC.__init__, "HMC")
        self._keep = keep

    def reset(self):
        self.inits.clear()
        self.runs.clear()
        self.kinits.clear()

    def last(self):
        """内核构造可能记两条（NUTS 记一次、它的 super() 再记一次 HMC），合并即可。"""
        bound = {}
        for k in self.kinits:
            bound.update(k["bound"])
        return (self.inits[-1] if self.inits else None,
                self.runs[-1] if self.runs else None,
                {"bound": bound} if self.kinits else None)

    def close(self):
        self.MCMC.__init__ = self._keep["mcmc_init"]
        self.MCMC.run = self._keep["mcmc_run"]
        self.NUTS.__init__ = self._keep["NUTS"]
        self.HMC.__init__ = self._keep["HMC"]


def _read_attr(mcmc, field):
    """探针自己读一遍，**不复用**捕获层的读取器。

    注意字段名和属性名**不一样**：`n_warmup` → `mc.num_warmup`。这正是"从 kw 读"
    与"从 mc 读"之间最容易错位的一处。
    """
    attr = {"n_warmup": "num_warmup", "n_samples": "num_samples",
            "num_chains": "num_chains", "chain_method": "chain_method",
            "progress_bar": "progress_bar",
            "target_accept": None, "max_tree_depth": None}[field]
    if field == "target_accept":
        return mcmc.sampler._target_accept_prob
    if field == "max_tree_depth":
        return mcmc.sampler._max_tree_depth
    return getattr(mcmc, attr)


def compare(label, ev, hook, *, extra_declared, mcmc=None):
    """逐字段对照。`mcmc` 为 None 时只核 `resolved` 与 `extra_fields`（bm_top 不返回对象）。"""
    init, run, kinit = hook.last()
    bound = dict(kinit["bound"]) if kinit else {}
    bound.update(init["bound"])
    ok = True
    print(f"\n— {label} —  site={ev['site']}  kernel={ev['kernel']}"
          f"（type(sampler)={init['kernel']}）")
    print(f"   adapt_step_size={ev['adapt_step_size']}/{ev['adapt_step_size_source']}"
          f"  chains={ev['num_chains']}  devices={ev['device_count']}"
          f"  env_forced={ev['progress_bar_env_forced']}")
    print(f"   {'字段':<16}{'真实构造实参':<18}{'记录 resolved':<18}"
          f"{'真实属性':<16}{'记录 observed':<18}{'status':<12}reason")
    for field in S.FIELDS:
        rec = ev["fields"][field]
        got = rec["resolved"]["value"] if rec["resolved"] else "∅"
        obs = rec["observed"]["value"] if rec["observed"] else "∅"

        if field == "extra_fields":
            real_arg = run["bound"].get("extra_fields", "∅")
            real_attr = real_arg
            same = (obs == real_arg) and ((got == "∅") if not extra_declared
                                          else (got == real_arg))
        else:
            param = PARAM[field]
            real_arg = bound[param] if param in bound else "（走库默认）"
            real_attr = _read_attr(mcmc, field) if mcmc is not None else "（无对象）"
            same = (got == real_arg) if param in bound else (
                rec["resolved"] is not None and rec["resolved"]["origin"] == "default")
            if mcmc is not None:
                same &= obs == real_attr

        ok &= bool(same)
        print(f"   {field:<16}{str(real_arg):<18}{str(got):<18}"
              f"{str(real_attr):<16}{str(obs):<18}{rec['status']:<12}"
              f"{rec.get('reason_code') or ''}{'' if same else '   <<< 对不上'}")

    agg = ev["aggregate"]
    print(f"   聚合 weakest={agg['weakest']}  n_override={agg['n_override']}"
          f"  override_fields={agg['override_fields']}"
          f"  n_capture_errors={ev['n_capture_errors']}  judged={ev['judged']}"
          f"  upstream_origin_captured={ev['upstream_origin_captured']}")
    return ok


def main() -> int:
    f, Z = _data()
    ok = True

    print("=" * 78)
    print("A / B：bm_core.fit（mcmc 随返回值带出，属性可独立复读）")
    print("=" * 78)
    h = Hooks()
    try:
        h.reset()
        _, mc = BC.fit(f, Z, K=4, n_warmup=8, n_samples=8, num_chains=1,
                       progress=False, seed=0, extra_fields=("num_steps",))
        assert getattr(mc, C.PROVENANCE_ATTR, None) is mc.provenance, "事件没挂在 mc 上"
        ok &= compare("A 单链 / 显式 extra_fields", mc.provenance, h,
                      extra_declared=True, mcmc=mc)

        h.reset()
        _, mc3 = BC.fit(f, Z, K=4, n_warmup=8, n_samples=8, num_chains=3,
                        progress=False, seed=0)
        ok &= compare("B 3 链 / 1 设备", mc3.provenance, h,
                      extra_declared=True, mcmc=mc3)
    finally:
        h.close()

    print("\n" + "=" * 78)
    print("C：bm_top.fit_top（不返回 mcmc：只核 resolved 与 extra_fields）")
    print("=" * 78)
    h2 = Hooks()
    try:
        h2.reset()
        o = BT.fit_top(f, Z, branch="parallel", layers=1, n_warmup=8, n_samples=8,
                       num_chains=1, progress=False, seed=0)
        ok &= compare("C nested 单链", o["provenance"], h2, extra_declared=False)
    finally:
        h2.close()

    print("\n" + "=" * 78)
    print("D：CI 在场，请求 progress=True（numpyro 静默压掉）")
    print("=" * 78)
    os.environ["CI"] = "1"
    h3 = Hooks()
    try:
        h3.reset()
        _, mc4 = BC.fit(f, Z, K=4, n_warmup=8, n_samples=8, num_chains=1,
                        progress=True, seed=0)
        ok &= compare("D 环境覆盖", mc4.provenance, h3, extra_declared=True, mcmc=mc4)
    finally:
        h3.close()
        os.environ.pop("CI", None)

    print("\n" + "=" * 78)
    print("构造点账目")
    print("=" * 78)
    for site, where in C.CALL_SITES.items():
        print(f"  已接  {site:<16} {where}")
    for site, why in C.NON_PRODUCTION_SITES.items():
        print(f"  未接  {site}\n        {why}")

    print(f"\n{'全部字段对得上' if ok else '有字段对不上，见 <<< 标记'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
