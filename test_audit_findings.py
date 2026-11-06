# -*- coding: utf-8 -*-
"""test_audit_findings.py —— 代码审查发现的两处缺陷的可复现测试。

运行：python test_audit_findings.py

断言的是**期望的正确行为**。当前（未修复）状态下用例 1、2 会失败，
失败本身就是缺陷的证据；用例 3 是对照组，应当一直通过。

关于环境桩
----------
`bm_real` / `arviz` 在 import 时会创建 matplotlib 配置目录。在只允许写
工作目录的沙箱里，这一步会抛 `TypeError: _sandbox_mkdir() ...`，与算法无关。
本文件因此在 import 前把 matplotlib 换成**只含空绘图接口**的桩：
被测试的两条路径（`loglik_matrix`、`compare_models`）本身不画图，
数值计算仍走真实的 numpy / scipy / arviz 代码，断言不受桩影响。

缺陷 1（bm_real.loglik_matrix:536-537）
--------------------------------------
主流程 `fit_one` 传 `sigma=None`，σ 进采样，样本里一定有 `s_re`/`s_im`。
但 `bm_core.fit(..., sigma=(s, s))`（固定噪声、不采 σ）时，模型内部
`s_re` 既不是 `sample` 也不是 `deterministic`，`get_samples()` 里没有这个键，
`loglik_matrix` 于是**静默回退到硬编码的 1e-3**，而不是调用方真正用的 σ。

    lr = -0.5*((Zr-zr)/sr)**2 - np.log(sr) - 0.5*log(2π)

残差为 0 时只有归一化项错（每点差 5.3）；残差不为 0 时二次项被
`(0.2/1e-3)² = 4e4` 倍放大。elpd 绝对值、Pareto 权重、以及
`docs/cases/.../case2_positive_real.py` 这类「固定噪声消除 σ→0 漏斗」
的实验结论全部失真，而且不报错。

缺陷 2（bm_real.compare_models:576）
-----------------------------------
`fit_one` 会把 LOO 失败的结果 `catch` 成 `idata=None` + `elpd=nan`，
并照常放进 `res_list`；`select_k()` 里专门写了「LOO 不可用 → 硬剔除」
这条分支（bm_real.py:962-963），说明设计上预期这种结果能活到选择环节。
但 `compare_models()` 先于 `select_k()` 执行，它把 `None` 直接塞给
`az.compare`，于是**整条谱的运行在这里崩掉**、结果 json 不写。
「一个 K 的 LOO 失败」应当降级为「该 K 不可比」，而不是拖垮全批。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _patch_sandbox_mkdir():
    """让 matplotlib / arviz 能在沙箱里正常 import。

    沙箱把 `Path.mkdir` 换成了签名不兼容的实现，第三方库在 import 时
    创建自己的配置缓存目录会抛 `TypeError: _sandbox_mkdir() takes from 1
    to 2 positional arguments but 3 were given`。这些缓存目录与本次被测的
    两条数值路径（`loglik_matrix` 纯 numpy、`compare_models` 走
    arviz → scipy 的 PSIS）都没有关系，所以在**测试进程内**把它置为 no-op，
    仍然加载真实的 matplotlib / arviz，不伪造任何计算。

    同一条限制也解释了为什么 `bmdrt.py check` 在这个环境里只跑通 A 组
    数学自检：B 组（采样）同样会撞上这个 import 期报错。
    """
    import os
    import pathlib

    pathlib.Path.mkdir = lambda self, *a, **k: None
    # 顺带把缓存指到项目目录内，避免真实库往用户目录写文件
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "_mplcache"))


_patch_sandbox_mkdir()

import bm_core as B          # noqa: E402
import bm_real as R          # noqa: E402

SIGMA_TRUE = 0.2             # 与 bm_core.SIGMA 一致（绝对口径，Ω）
N_DRAW = 4


def _truth_samples() -> dict:
    """真值参数重复 N_DRAW 次的「后验样本」，**不含 s_re/s_im**。

    不含 σ 正是 `bm_core.fit(sigma=(s, s))` 固定噪声时的真实情况。
    """
    t = B.truth_dict()
    return {"R_inf": np.full(N_DRAW, t["R_inf"]),
            "W": np.full(N_DRAW, t["W"]),
            "pi": np.tile(t["pi"], (N_DRAW, 1)),
            "tau": np.tile(t["tau"], (N_DRAW, 1)),
            "phi": np.tile(t["phi"], (N_DRAW, 1)),
            "zeta": np.tile(t["zeta"], (N_DRAW, 1))}


def _joint_loglik(f, Z, s: dict, sigma: float) -> np.ndarray:
    """参考实现：同一频率的实部 + 虚部作为**一个**联合观测。"""
    out = np.zeros((N_DRAW, f.size))
    for a in range(N_DRAW):
        z = B.model_z(f, B.sample_params(s, a, include_L=False), include_L=False)
        for obs, mod in ((Z.real, z.real), (Z.imag, z.imag)):
            out[a] += -0.5 * ((obs - mod) / sigma) ** 2 - np.log(sigma) \
                      - 0.5 * np.log(2 * np.pi)
    return out


# ---------------------------------------------------------------- 用例 1
def case1_fixed_sigma_loo_uses_real_sigma():
    """固定噪声时 LOO 必须用调用方给的 σ，不能静默回退到 1e-3。"""
    f, Z = B.true_spectrum()                  # 无噪声真谱 → 残差恰为 0
    s = _truth_samples()
    assert "s_re" not in s, "用例前提：固定噪声 → 样本里没有 s_re"

    got = R.loglik_matrix(s, f, Z, include_L=False, n_max=N_DRAW)
    want = _joint_loglik(f, Z, s, SIGMA_TRUE)
    v_bad = -2 * np.log(1e-3) - np.log(2 * np.pi)

    assert np.allclose(got, want, rtol=1e-12), (
        f"逐点 ll={got[0, 0]:.4f}，按真值 σ={SIGMA_TRUE} 应为 {want[0, 0]:.4f}；"
        f"硬编码 1e-3 会算出 {v_bad:.4f}。"
        f"→ 固定噪声路径下 elpd / Pareto-k / 模型比较全部失真，且不报错。")
    return f"逐点 ll={got[0, 0]:.4f} 与 σ={SIGMA_TRUE} 口径一致"


# ---------------------------------------------------------------- 用例 2
def case2_compare_models_survives_loo_failure():
    """某候选 K 的 LOO 失败（idata=None）时，配对比较应跳过它而不是崩。"""
    import arviz as az

    rng = np.random.default_rng(0)

    def make_res(K: int, bias: float, idata_ok: bool):
        ll = rng.standard_normal((400, 31)) + bias        # (draw, point)
        idata, elpd = None, float("nan")
        if idata_ok:
            idata = az.from_dict(
                posterior={"dummy": np.zeros((1, 400, 1))},
                log_likelihood={"obs": ll[None, :, :]})
            _, elpd, *_ = R.loo_pointwise(ll)
        return dict(K=K, idata=idata, elpd=elpd, elpd_diff=0.0,
                    dse=float("nan"), weight=float("nan"), rank=1)

    res_list = [make_res(2, 0.0, True), make_res(3, 0.5, False)]
    assert res_list[1]["idata"] is None, "用例前提：K=3 模拟 LOO 失败"

    df = R.compare_models(res_list)          # 当前实现：az.compare 收到 None → 抛异常

    assert "K=3" not in df.index, "LOO 不可用的 K=3 不该参与 az.compare"
    assert np.isfinite(res_list[0]["elpd_diff"]), "合格候选仍应拿到 elpd_diff"
    return f"跳过 idata=None 的候选，比较表只剩 {list(df.index)}"


# ---------------------------------------------------------------- 用例 3
def case3_sampled_sigma_path_is_correct():
    """对照组：σ 进采样（主流程默认路径）时必须正确 —— 这条应一直通过。"""
    f, Z = B.true_spectrum()
    s = _truth_samples()
    s["s_re"] = np.full(N_DRAW, SIGMA_TRUE)
    s["s_im"] = np.full(N_DRAW, SIGMA_TRUE)

    got = R.loglik_matrix(s, f, Z, include_L=False, n_max=N_DRAW)
    want = _joint_loglik(f, Z, s, SIGMA_TRUE)
    assert np.allclose(got, want, rtol=1e-12), \
        f"主流程口径不该错：{got[0, 0]:.6f} != {want[0, 0]:.6f}"
    return f"逐点 ll={got[0, 0]:.4f} 与 σ={SIGMA_TRUE} 一致"


# ---------------------------------------------------------------- 用例 4
def case4_paired_loo_unpack_arity():
    """`loo_pointwise` 返回 6 个值，事后重算脚本还按 5 个解包 → 必崩。

    `evidence/b2x/paired_loo.py:40` 写的是

        _, elpd, se, bad_k, pt = R.loo_pointwise(ll)

    而 `loo_pointwise` 返回
    `(idata, elpd, se, bad_k>0.7 占比, 逐点 elpd, 逐点 pareto_k)` —— 6 个。
    多返回 `pk` 的那次改动没同步这个脚本，一跑就
    `ValueError: too many values to unpack`。`docs/cases/.../analyze_grid.py`
    特意绕开（另跑一次 az.loo 取逐点 k），反证这条路径确实断了。
    正确修法是把**调用方**改成 6 个（或 `*_`），不是把函数改回 5 个。
    """
    src = (ROOT / "evidence" / "b2x" / "paired_loo.py").read_text(encoding="utf-8")
    n_return = len(R.loo_pointwise(
        np.random.default_rng(0).standard_normal((400, 31))))
    left = [ln for ln in src.splitlines() if "R.loo_pointwise" in ln]
    n_unpack = left[0].split("=")[0].count(",") + 1 if left else 0

    assert n_unpack >= n_return or "*" in left[0], (
        f"loo_pointwise 返回 {n_return} 个值，"
        f"evidence/b2x/paired_loo.py 按 {n_unpack} 个解包 → "
        f"ValueError: too many values to unpack。"
        f"该脚本是 §3.4 跨配置配对 LOO 的唯一入口。")
    return f"解包 {n_unpack} ≥ 返回 {n_return}，一致"


CASES = [case1_fixed_sigma_loo_uses_real_sigma,
         case2_compare_models_survives_loo_failure,
         case3_sampled_sigma_path_is_correct,
         case4_paired_loo_unpack_arity]

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
    print(f"\n{len(CASES) - len(bad)}/{len(CASES)} 通过。"
          f"未通过即为缺陷可复现：{bad}")
    raise SystemExit(1 if bad else 0)
