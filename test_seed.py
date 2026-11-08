# -*- coding: utf-8 -*-
"""test_seed.py —— B3b：sampler seed 的可控传递（只测种子链路，不改采样策略）。

为什么单独一个文件：`--seed` 的历史坑是「CLI 收下了、没传到采样」——参数解析
是通用的（`--xxx value` 直接进 opt），所以 `--seed 1` 不会报错，只会被**静默丢掉**。
因此这里既测数值语义（谁覆盖谁），也测**接线**（CLI → run() → _fit_ks → fit_one →
结果 json），并且断言结果里记的是**实际生效值**。

三条边界（写死，免得结果被当成别的东西）：
1. 只换随机流：模型、**初始化策略**、K 范围、warmup/samples/ta/树深全不变。
   注意"初始化策略不变"不等于"初值不变"：`init_values` 存在时走确定性
   `init_to_value`，初值固定；为 None 时 NumPyro 的默认初始化也消耗这条随机流，
   初值可能随 seed 变。case8 会把这条状态打出来。
2. 同 seed 重跑必须**逐位相同**（这是"随机流被固定住"的判据，不是"结果正确"）。
3. 不同 seed 只要求**样本确实不同**，不要求 K/分级不变——变了本身是信息。

用法：python test_seed.py        （含 8 次极短 NUTS，约 30-60 s）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bm_real as R           # noqa: E402

F = np.logspace(0, 5, 31)
ARCS = [(30.0, 300.0, 0.85)]
R_INF, NOISE = 10.0, 0.10

# 极短档：只够判断"随机流是否被固定"，不能当采样质量结论。
# num_chains 必须 ≥2：R̂ / ESS 的诊断原语要求多链，1 链会在 gelman_rubin 里断言失败。
KW_FAST = dict(n_warmup=200, n_samples=100, num_chains=2, target_accept=0.9,
               max_tree_depth=8, progress=False, extra_fields=("num_steps",))


def _synth(f, seed=3):
    w = 2 * np.pi * f
    Z = np.full(f.shape, R_INF, complex)
    for Rv, fp, phi in ARCS:
        Z = Z + Rv / (1.0 + (1j * w / (2 * np.pi * fp)) ** phi)
    rng = np.random.default_rng(seed)
    return Z + NOISE * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))


def _max_draws_diff(r1, r2, stats=None):
    """两组后验样本的最大逐点差（比较两边**全部**样本数组，不只是 1-D）。

    早期版本只挑 `np.ndim(...) == 1` 的键，于是 K=1 时像 `v` 这种 (n_draws, 1)
    的二维参数样本被静默跳过——"逐位相同"其实只覆盖了部分数组。这里收紧为：
      * 键集合必须一致（同配置同 K 下缺键本身是异常，不静默跳过）；
      * shape 必须一致；
      * 完全相同（含 NaN 同位）→ 跳过；否则若含非有限值 → 返回 inf，
        不能因为 `max(0.0, nan) == 0.0` 报出一个假的 0（M3.5 踩过这个坑）。

    stats（dict，可选）：回填实际比较覆盖，供测试断言"没有漏掉任何数组"。
    """
    a, b = r1["s"], r2["s"]
    ka, kb = set(a), set(b)
    if ka != kb:
        raise AssertionError(
            "样本键集合不一致：只在左 %s，只在右 %s"
            % (sorted(ka - kb), sorted(kb - ka)))
    assert ka, "两份结果都没有样本数组"
    m = 0.0
    n_arr = n_elem = n_nd = 0
    shapes = set()
    worst, worst_shape = "", ()
    for k in sorted(ka):
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape:
            raise AssertionError(f"样本形状不一致：{k} {x.shape} vs {y.shape}")
        n_arr += 1
        n_elem += int(x.size)
        shapes.add(int(x.ndim))
        if x.ndim > 1:
            n_nd += 1
        if x.size == 0:
            continue
        if np.array_equal(x, y, equal_nan=True):
            continue
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            m = float("inf")      # 有非有限值又不完全同位：不能报有限差
        else:
            dv = float(np.max(np.abs(x - y)))
            if dv >= m:
                m, worst, worst_shape = dv, k, tuple(x.shape)
    if stats is not None:
        stats.update(n_arrays=n_arr, n_elem=n_elem, n_ndim_gt1=n_nd,
                     ndims=sorted(shapes), keys=sorted(ka),
                     worst_key=worst, worst_shape=worst_shape)
    return m


# --------------------------------------------------------------------------
# 1. 数值语义：谁覆盖谁（不跑采样）
# --------------------------------------------------------------------------
def case1_resolve_seed_precedence():
    """显式 seed > kw["seed"] > KW["seed"]；显式 0 也是"显式"。"""
    assert R.KW["seed"] == 0, "默认 seed 必须仍是 0（旧命令逐位不变的底线）"
    assert R._resolve_seed(None, {}) == 0, "无入参、无 kw → 用 KW 默认 0"
    assert R._resolve_seed(None, None) == 0, "kw 为 None 也不能炸"
    assert R._resolve_seed(None, {"seed": 7}) == 7, "kw 里的 seed 必须被采纳"
    assert R._resolve_seed(5, {"seed": 7}) == 5, "显式 seed 必须压过 kw"
    assert R._resolve_seed(0, {"seed": 7}) == 0, "显式 0 不能被当成「没给」"
    assert R._resolve_seed("3", {}) == 3, "字符串数字要能转"
    return "显式 5>kw 7；kw 7>默认 0；显式 0 生效"


# --------------------------------------------------------------------------
# 2. CLI 三层：解析 → 配置 → 透传（不跑采样）
# --------------------------------------------------------------------------
def case2_cli_parsed_and_applied():
    """`--seed` / `--seed=` 都要进 opt，并落进 cfg["fit"]["seed"]。"""
    import bmdrt as CLI

    for argv, want in ((["a.csv", "--seed", "3"], 3),
                       (["--seed=4", "a.csv"], 4),
                       (["--seed", "0", "a.csv"], 0)):
        paths, opt = CLI.parse_opts(argv)
        assert paths == ["a.csv"], f"路径解析被 seed 干扰：{paths}"
        assert opt["seed"] == want and isinstance(opt["seed"], int), \
            f"{argv} → opt['seed']={opt['seed']!r}"

    cfg = CLI._apply_cli(CLI.resolve(CLI.load_cfg()),
                         CLI.parse_opts(["a.csv", "--seed", "3"])[1])
    assert cfg["fit"]["seed"] == 3, "opt 里的 seed 没有进入 fit 段（静默丢弃）"

    cfg0 = CLI._apply_cli(CLI.resolve(CLI.load_cfg()),
                          CLI.parse_opts(["a.csv"])[1])
    assert cfg0["fit"].get("seed") is None, "不传 --seed 时必须保持 None（= 默认 0）"

    try:
        CLI.parse_opts(["a.csv", "--seed", "abc"])
    except SystemExit as e:
        assert "整数" in str(e), f"错误信息不明确：{e}"
    else:
        raise AssertionError("--seed abc 应该直接退出，而不是静默丢弃")
    return "解析/配置/默认 None 三层齐；非整数直接退出"


def case3_passthrough_source():
    """接线是文本契约：任一层漏掉 seed，这里就该红。"""
    src_run = (ROOT / "bm_real.py").read_text(encoding="utf-8")
    src_cli = (ROOT / "bmdrt.py").read_text(encoding="utf-8")
    assert "def run(path, K_force=None" in src_run and "seed=None):" in src_run, \
        "run() 没有 seed 形参"
    assert 'kw["seed"] = int(seed)' in src_run, "run() 没把 seed 落进 kw"
    assert 'seed=kw.get("seed", KW["seed"])' in src_run, \
        "_fit_ks 调 fit_one 时没传 seed"
    assert "_resolve_seed(seed, kw)" in src_run, "fit_one 没走统一的 seed 解析"
    assert 'seed=int(seed_used)' in src_run, "fit_one 结果没记实际 seed"
    assert "sampling=dict(" in src_run, "fit_one 结果没记采样配置"
    assert 'seed=int(res.get("seed"' in src_run, "save_out 没把 seed 写进 json"
    assert 'sampling_performed=False' in src_run, "KK 拒绝记录没标「未采样」"
    assert 'seed=(None if fit.get("seed") is None' in src_cli, \
        "process_one 没把 seed 透传给 run()"
    return "run/_fit_ks/fit_one/save_out/process_one 五处齐"


# --------------------------------------------------------------------------
# 3. 真采样：同 seed 逐位相同、不同 seed 确实不同、记录的是生效值
# --------------------------------------------------------------------------
def _fit(seed_arg, kw_seed=None):
    kw = dict(KW_FAST)
    if kw_seed is not None:
        kw["seed"] = kw_seed
    return R.fit_one(F, _synth(F), K=1, include_L=False, zeta="1.0",
                     kw=kw, seed=seed_arg)


def case4_result_records_actual_seed():
    """kw 里的 seed 若被采纳，结果里必须写它——而不是外层默认的 0。"""
    r = _fit(None, kw_seed=11)
    assert r["seed"] == 11, f"kw seed 被吞了：r['seed']={r['seed']}"
    sp = r["sampling"]
    assert sp["n_warmup"] == KW_FAST["n_warmup"] and \
        sp["n_samples"] == KW_FAST["n_samples"] and \
        sp["num_chains"] == KW_FAST["num_chains"], f"采样配置记录不对：{sp}"
    r0 = _fit(0, kw_seed=11)
    assert r0["seed"] == 0, "显式 seed=0 应压过 kw 的 11"
    return f"kw seed=11 → 记录 11；显式 0 → 记录 0"


def case5_same_seed_bitwise_identical():
    """同 seed 重跑逐位相同：证明"随机流被固定"，不是"结果正确"。

    覆盖范围必须等于**全部**样本数组（审查 3.1）：旧实现只比 1-D，
    K=1 时 (n_draws, 1) 的参数会被漏掉，所以这里显式断言没漏。
    """
    a, b = _fit(0), _fit(0)
    st = {}
    d = _max_draws_diff(a, b, stats=st)
    assert st["n_arrays"] == len(a["s"]), \
        f"只比了 {st['n_arrays']}/{len(a['s'])} 个样本数组，有漏"
    assert d == 0.0, f"同 seed 两次采样不一致，max|Δ|={d:.3e}（不是逐位确定的）"
    assert a["grade"] == b["grade"] and a["ndiv"] == b["ndiv"]
    return (f"{st['n_arrays']} 个数组 / {st['n_elem']} 个样本值全部逐位相同"
            f"（ndim={st['ndims']}，二维数组 {st['n_ndim_gt1']} 个）；"
            f"grade={a['grade']}，div={a['ndiv']}")


def case6_different_seed_changes_draws():
    """不同 seed 必须真的换随机流；K/分级变不变不设判据。"""
    a, c = _fit(0), _fit(1)
    st = {}
    d = _max_draws_diff(a, c, stats=st)
    assert d > 0.0, "不同 seed 的样本完全相同——seed 没进到采样，或随机流被上层的常量覆盖"
    return (f"seed0 vs seed1 max|Δ|={d:.3e}（{st['worst_key']}{st['worst_shape']}，"
            f"覆盖 {st['n_arrays']} 个数组）；"
            f"grade {a['grade']}→{c['grade']}，div {a['ndiv']}→{c['ndiv']}"
            f"（变化与否只记录，不作判据）")


def case7_kk_reject_says_no_sampling():
    """KK 拒绝谱**没有采样**：记录必须是 sampling_performed=false + seed=null。

    否则 B3b 数「每个 seed 跑了几条谱」时，KK 拒绝谱会被当成"某个 seed 跑过了"，
    把失败混进分母（和 M6 §4.3 里 non_evaluable 不能算 TN 是同一个道理）。
    """
    import json
    import tempfile

    old = R.OUT
    try:
        with tempfile.TemporaryDirectory() as td:
            R.OUT = Path(td)
            js = R._write_kk_reject("x.csv", F, _synth(F),
                                    {"status": "fail", "reason": "测试用"},
                                    "gate", seed=None)
            d = json.loads(js.read_text(encoding="utf-8"))
    finally:
        R.OUT = old
    assert d["fit_performed"] is False, "KK 拒绝必须标 fit_performed=False"
    assert d["sampling_performed"] is False, "KK 拒绝记录缺 sampling_performed=False"
    assert d["seed"] is None, f"未采样时 seed 应为 null，实为 {d['seed']!r}"
    return "fit_performed=False + sampling_performed=False + seed=null"


def case8_runtime_args_reach_sampler():
    """文本契约只证明字符串在文件里；这里捕获**运行时实参**（审查 3.2）。

    `run() → _fit_ks() → fit_one() → B.fit()` 每跳都走全局名或关键字，
    所以包一层 `B.fit` 就能看到真正交给采样的 seed 和 init_values。
    顺带记录 init_values 是否为 None —— 这正是"换 seed 会不会连带换初值"
    的分岔点（§6.1.2）：True 则 init_to_value 固定初值，False 则 NumPyro
    默认初始化策略也吃这条随机流，初值可能随 seed 变。
    """
    import bm_core as B
    import tempfile

    csv = ROOT / "examples" / "example_caseE.csv"
    assert csv.exists(), "缺示例谱 examples/example_caseE.csv"

    fast = dict(KW_FAST)
    fast["seed"] = 0            # run() 里 _fit_ks 用 kw.get("seed", KW["seed"])
    seen = []
    orig_fit, orig_out, orig_kw = B.fit, R.OUT, R.KW

    def spy(*a, **kw):
        seen.append((kw.get("seed"), kw.get("init_values") is not None))
        return orig_fit(*a, **kw)

    try:
        B.fit = spy
        R.KW = fast
        with tempfile.TemporaryDirectory() as td:
            R.OUT = Path(td)
            R.run(csv, K_force=1, quick=False, kk_mode="off", seed=None)
            r0, seen[:] = list(seen), []
            R.run(csv, K_force=1, quick=False, kk_mode="off", seed=7)
            r7 = list(seen)
    finally:
        B.fit, R.OUT, R.KW = orig_fit, orig_out, orig_kw

    assert r0 and r7, "B.fit 一次都没被调用，测试本身失效"
    assert all(s == 0 for s, _ in r0), f"默认路径到达采样器的 seed 不是 0：{r0}"
    assert all(s == 7 for s, _ in r7), f"显式 seed 没到采样器：{r7}"
    seeded = sorted({q for _, q in r0 + r7})
    assert len(seeded) == 1, f"同一谱两次调用的播种状态不一致：{r0} / {r7}"
    return ("默认→0、显式→7 均到达 B.fit；"
            f"init_values={'有' if seeded[0] else '无（默认初始化也吃随机流）'}")


CASES = [case1_resolve_seed_precedence, case2_cli_parsed_and_applied,
         case3_passthrough_source, case4_result_records_actual_seed,
         case5_same_seed_bitwise_identical, case6_different_seed_changes_draws,
         case7_kk_reject_says_no_sampling, case8_runtime_args_reach_sampler]

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
