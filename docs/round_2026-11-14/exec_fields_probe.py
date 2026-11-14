"""③ 补充审计的观测位置探针（三执行行为字段 + 内核适用性）。

只做一件事：确认 `chain_method` / `progress_bar` / `extra_fields` 各自
**在哪里能被读到**，以及 `target_accept` / `max_tree_depth` 在 NUTS 与 HMC
上是否都成立。不拟合任何真实谱，不动生产代码，不产出产物文件。

用法（项目 venv）：
    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/exec_fields_probe.py
"""

from __future__ import annotations

import inspect
import json
import os
import warnings

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import HMC, MCMC, NUTS

BRANCH = ["parallel", "sequential", "vectorized"]


def _model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    numpyro.sample("obs", dist.Normal(x, 0.1), obs=jnp.array(0.3))


def _build(*, n_chains, progress, ta=0.97, depth=7):
    """按生产代码的写法构造（两族都是显式传 ta/depth，从不省略）。"""
    kern = NUTS(handlers.seed(_model, rng_seed=0),
                target_accept_prob=ta, max_tree_depth=depth)
    mc = MCMC(kern, num_warmup=4, num_samples=4, num_chains=n_chains,
              progress_bar=progress)
    return mc


def _run(mc, extra_fields=()):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mc.run(jax.random.PRNGKey(0), extra_fields=extra_fields)
    return [str(w.message)[:60] for w in caught if issubclass(w.category, UserWarning)]


def case(label, **kw):
    """构造 → 读属性 → 跑 → 读 extra_fields 键。"""
    ef = kw.pop("extra_fields", ())
    mc = _build(**kw)
    row = dict(
        label=label,
        requested=dict(num_chains=kw["n_chains"], progress_bar=kw["progress"],
                       chain_method="(未传 → 库默认 parallel)", extra_fields=ef),
        # 构造后、run 前：属性里已经是**生效值**
        attr=mc.__dict__ and dict(
            chain_method=mc.chain_method,
            progress_bar=bool(mc.progress_bar),
            num_warmup=mc.num_warmup, num_samples=mc.num_samples,
            num_chains=mc.num_chains,
            sampler_depth=int(mc.sampler._max_tree_depth),
            sampler_ta=float(mc.sampler._target_accept_prob),
        ),
        has_attr_extra_fields=hasattr(mc, "extra_fields"),
        has_attr_chain_method_requested=any(
            k in mc.__dict__ for k in ("_chain_method", "requested_chain_method")),
        has_attr_progress_requested=any(
            k in mc.__dict__ for k in ("_progress_bar", "requested_progress_bar")),
    )
    row["warnings"] = _run(mc, ef)
    row["extra_keys"] = sorted(mc.get_extra_fields().keys())
    return row


def kernels():
    out = {}
    m = handlers.seed(_model, rng_seed=0)
    out["NUTS.signature_has"] = dict(
        target_accept_prob="target_accept_prob" in inspect.signature(NUTS.__init__).parameters,
        max_tree_depth="max_tree_depth" in inspect.signature(NUTS.__init__).parameters,
        chain_method="chain_method" in inspect.signature(NUTS.__init__).parameters,
    )
    out["HMC.signature_has"] = dict(
        target_accept_prob="target_accept_prob" in inspect.signature(HMC.__init__).parameters,
        max_tree_depth="max_tree_depth" in inspect.signature(HMC.__init__).parameters,
    )
    n = NUTS(m, target_accept_prob=0.97, max_tree_depth=7)
    out["NUTS.explicit"] = dict(ta=float(n._target_accept_prob),
                                depth=int(n._max_tree_depth))
    n0 = NUTS(m)                       # 全默认
    out["NUTS.default"] = dict(ta=float(n0._target_accept_prob),
                               depth=int(n0._max_tree_depth))
    h = HMC(m, target_accept_prob=0.97)   # 只能给 ta
    out["HMC.explicit_ta"] = dict(ta=float(h._target_accept_prob),
                                  depth=int(h._max_tree_depth))
    h0 = HMC(m)
    out["HMC.default"] = dict(ta=float(h0._target_accept_prob),
                              depth=int(h0._max_tree_depth))
    try:
        HMC(m, max_tree_depth=7)
        out["HMC.max_tree_depth_kwarg"] = "accepted"
    except TypeError as exc:
        out["HMC.max_tree_depth_kwarg"] = f"TypeError: {exc}"
    return out


def env_case():
    """CI 环境变量只在构造时被读一次；run 时不复查。"""
    os.environ["CI"] = "1"
    try:
        mc = _build(n_chains=1, progress=True)
        before = bool(mc.progress_bar)
    finally:
        os.environ.pop("CI", None)
    mc2 = _build(n_chains=1, progress=True)
    after = bool(mc2.progress_bar)
    return dict(CI_on_progress_true=before, CI_off_progress_true=after)


def main():
    print(f"numpyro {numpyro.__version__}  jax devices={jax.local_device_count()}")
    rows = [
        case("1 链 / progress=True / ef=()", n_chains=1, progress=True),
        case("3 链 / progress=True / ef=()", n_chains=3, progress=True),
        case("3 链 / progress=False / ef=(num_steps,)", n_chains=3, progress=False,
             extra_fields=("num_steps",)),
    ]
    for r in rows:
        print("\n---", r["label"])
        print("  requested :", r["requested"])
        print("  attr      :", r["attr"])
        print("  has .extra_fields:", r["has_attr_extra_fields"],
              "| 记录请求值:", r["has_attr_chain_method_requested"],
              r["has_attr_progress_requested"])
        print("  extra keys:", r["extra_keys"])
        print("  warnings :", r["warnings"])
    print("\n=== kernels")
    print(json.dumps(kernels(), indent=2, ensure_ascii=False))
    print("\n=== CI env")
    print(env_case())


if __name__ == "__main__":
    main()
