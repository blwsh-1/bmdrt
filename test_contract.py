# -*- coding: utf-8 -*-
"""test_contract.py —— 二轮代码审查 §5 的四组「契约」回归，防退化。

对应 `代码审查报告_2026-11-11.md` §5 的第 2–5 项。这四组都不是数值结论，
而是**行为契约**：以前修过的坑一旦被人改回去，这里就要红。

    A  CLI 白名单   —— 拼错/跨入口开关必须报错，且不许把输入文件吞成开关取值
    B  批量异常     —— 三个入口（bmdrt / nested_fit / batch_2e）「坏谱 + 好谱」
                       下，好谱仍被处理、汇总落盘、退出码如实（拒收 ≠ 失败）
    C  门禁三态     —— fail / unavailable(strict) / unavailable(fail-open) 三种输入，
                       gate 模式行为与 `kk_performed` 字段保持一致
    D  JSON 合法性  —— `_json_safe` + `allow_nan=False`；新产物能被严格解析器吃下
    E  参考 Rwp     —— 与生产解析式对拍、`Rwp = √2·rms_rel`、拓扑不一致必须 skipped
    F  结果/展示   —— 曲线≡逐样本分位、残差≡恒等式、CSV≡图、sha256、判定口径、
                      图形配置可复现、面板载荷与数据层逐值相等、
                      `least_squares.success=False` 的解不许当参考值
    F8 B 类 guard  —— `param_degenerate` / `param_at_bound` 在**两族**上都真的在判
                      （voigt 侧旧实现一个恒 triggered、一个恒 not_evaluable）

用法：
    python test_contract.py             # 全跑，约 60-90 s（含 2 次 quick 拟合）
    python test_contract.py --skip-fit  # 跳过要真拟合的用例，约 10 s

判据是「契约」，不是「结果正确」：比如 A 组只断言解析器行为，不评价任何谱的
拟合好坏；C 组用合成 kk 字典与桩函数，不依赖真实数据的门禁结论。
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bmdrt                      # noqa: E402
import bm_real as R               # noqa: E402

GOOD = ROOT / "out" / "_m6_samples" / "z_csv" / "AZ31-4Li_100.csv"
SKIP_FIT = "--skip-fit" in sys.argv[1:]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def strict_load(p):
    """严格 JSON 解析：裸 NaN / Infinity 直接判错（不是警告）。"""
    def _boom(x):
        raise ValueError(f"非法 JSON 常量：{x}")
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=_boom)


def _must_exit(fn, *a, **k):
    """断言 fn 以 SystemExit 结束，返回其消息（字符串或 code）。"""
    try:
        fn(*a, **k)
    except SystemExit as e:
        return e.args[0] if e.args else e.code
    raise AssertionError("应当 SystemExit，实际正常返回——静默吞参的旧病复发")


# ---------------------------------------------------------------------------
# A 组：CLI 白名单（审查 P1-2）
# ---------------------------------------------------------------------------

def case_a1_legal_forms():
    """合法开关：`--x v` 与 `--x=v` 等价；别名归一到规范键；枚举值大小写如实。"""
    spec = [
        (["--sigma-floor", "1e-4", "f.csv"], {"sigma_floor": "1e-4"}),
        (["--sigma-floor=1e-4", "f.csv"], {"sigma_floor": "1e-4"}),
        (["--kk", "warn", "f.csv"], {"kk": "warn"}),
        (["--kk=off", "f.csv"], {"kk": "off"}),
        (["--diffusion", "Ws", "f.csv"], {"diffusion": "Ws"}),
        (["--adsorption", "RL", "f.csv"], {"adsorption": "rl"}),
        (["--k", "3", "f.csv"], {"kmax": 3}),          # 别名 --k → kmax，且转 int
        (["--seed", "7", "f.csv"], {"seed": 7}),
        (["--quick", "--no-move", "--L", "f.csv"],
         {"quick": True, "no_move": True, "L": True}),
        (["--no-L", "f.csv"], {"L": False}),
        (["--ref-rwp", "f.csv"], {"ref_rwp": True}),
    ]
    for argv, want in spec:
        paths, opt = bmdrt.parse_opts(argv)
        assert paths == ["f.csv"], f"{argv} 把输入文件弄丢了：paths={paths}"
        for k, v in want.items():
            assert opt[k] == v, f"{argv} 的 {k}={opt[k]!r}，应为 {v!r}"
    return f"{len(spec)} 组合法用法路径不丢、取值正确"


def case_a2_unknown_flags():
    """未知/拼错开关必须报错，并点名那个开关（旧行为是静默吞掉后面的文件）。"""
    spec = [
        ["--multi-model", "myfile.csv"],       # 别的入口的合法开关
        ["--sigam-floor", "0.1", "myfile.csv"],  # 拼错
        ["--nope=1", "myfile.csv"],
    ]
    for argv in spec:
        msg = str(_must_exit(bmdrt.parse_opts, argv))
        flag = argv[0].split("=")[0]
        assert flag in msg, f"{argv} 报错没点名开关：{msg}"
        assert "未知开关" in msg, f"{argv} 报错文案不对：{msg}"
    return f"{len(spec)} 组未知开关全部报错并点名"


def case_a3_enum_values():
    """枚举值非法必须报错（`--kk foo` 以前会变成第四种状态静默通过）。"""
    spec = [
        (["--kk", "foo", "f.csv"], "--kk"),
        (["--diffusion", "ws", "f.csv"], "--diffusion"),   # 大小写敏感：合法值是 Ws
        (["--diffusion", "WS", "f.csv"], "--diffusion"),
        (["--adsorption", "xy", "f.csv"], "--adsorption"),
    ]
    for argv, flag in spec:
        msg = str(_must_exit(bmdrt.parse_opts, argv))
        assert flag in msg, f"{argv} 报错没点名 {flag}：{msg}"
    return f"{len(spec)} 组非法枚举全部拦下"


def case_a4_missing_or_bad_value():
    """缺取值 / 非整数必须报错（不许把下一个 token 当值、更不许静默丢掉）。"""
    spec = [
        (["--zeta"], "缺少取值"),
        (["--seed", "abc", "f.csv"], "需要整数"),
        (["--kmax", "x.y", "f.csv"], "需要整数"),
    ]
    for argv, want in spec:
        msg = str(_must_exit(bmdrt.parse_opts, argv))
        assert want in msg, f"{argv} 报错文案缺「{want}」：{msg}"
    return f"{len(spec)} 组缺参/坏值全部报错"


def case_a5_cli_reaches_config():
    """解析出来的值必须真的接进 config（σ 地板/门禁模式/吸附归一，P2-3 那类坑）。"""
    _, opt = bmdrt.parse_opts(["--sigma-floor", "1e-4", "--kk", "warn",
                               "--adsorption", "rl", "--seed", "5", "f.csv"])
    cfg = {"fit": {}, "paths": {}, "batch": {}}
    bmdrt._apply_cli(cfg, opt)
    assert cfg["fit"]["sigma_floor"] == 1e-4, cfg["fit"].get("sigma_floor")
    assert cfg["kk"]["mode"] == "warn", cfg.get("kk")
    # 下游判据是 `adsorption == "RL"`（大小写敏感），归一错了等于开关失效
    assert cfg["fit"]["adsorption"] == "RL", cfg["fit"].get("adsorption")
    assert cfg["fit"]["seed"] == 5
    return "σ地板 / 门禁模式 / 吸附归一 / seed 均落到 config"


# ---------------------------------------------------------------------------
# B 组：三入口批处理的异常策略（审查 P1-3 / P2-13）
# ---------------------------------------------------------------------------

def case_b1_nested_fit_batch(tmp):
    """nested_fit：坏谱记失败、好谱继续跑完、退出码 1（旧行为是整批死掉）。"""
    from nested_fit import main as nf_main

    bad = tmp / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp / "nf_out"
    rc = nf_main([str(bad), str(GOOD), "--budget", "quick", "--no-plot",
                  "--out", str(out)])
    assert rc == 1, f"部分失败必须返回 1，实际 {rc}"
    js = out / f"{GOOD.stem}.nested.json"
    assert js.exists(), "好谱没有被处理——坏谱又把整批打断了"
    d = strict_load(js)
    assert d["status"] == "exploratory", d["status"]
    return f"坏谱记失败、好谱出结果、rc=1（{js.name}）"


def case_b2_batch_2e_status():
    """batch_2e：rc≠0 且非 KK 拒绝 → fail(rc=)，不许凭残留产物记 ok（P2-13）。"""
    import batch_2e as B2

    tmp = Path(tempfile.mkdtemp(prefix="b2_"))
    old = B2.subprocess.run

    class _CP:
        def __init__(self, rc):
            self.returncode = rc

    try:
        # ① rc=1 且没有任何产物 → 必须 fail
        B2.subprocess.run = lambda cmd, **k: _CP(1)
        rec = B2._run_one((tmp, 1, {"stem": "s1", "csv": "s1.csv"},
                           "quick", 10, False, False))
        assert rec["status"] == "fail(rc=1)", rec

        # ② rc=1 但目录里有残留旧产物 → 也不许记 ok
        d = B2.result_dir(tmp, 1, "s1")
        d.mkdir(parents=True, exist_ok=True)
        (d / "real_s1.json").write_text('{"K": 2}', encoding="utf-8")
        (d / "real_s1.png").write_bytes(b"x")
        # 残留产物会让函数走「已处理完」的 skip 分支——这是设计行为
        rec2 = B2._run_one((tmp, 1, {"stem": "s1", "csv": "s1.csv"},
                            "quick", 10, False, False))
        assert rec2["status"] == "skip", rec2

        # ③ rc=0 且写出产物 → ok
        d2 = B2.result_dir(tmp, 2, "s2")

        def _ok(cmd, **k):
            d2.mkdir(parents=True, exist_ok=True)
            (d2 / "real_s2.json").write_text('{"K": 2}', encoding="utf-8")
            (d2 / "real_s2.png").write_bytes(b"x")
            return _CP(0)

        B2.subprocess.run = _ok
        rec3 = B2._run_one((tmp, 2, {"stem": "s2", "csv": "s2.csv"},
                            "quick", 10, False, False))
        assert rec3["status"] == "ok", rec3
    finally:
        B2.subprocess.run = old
        shutil.rmtree(tmp, ignore_errors=True)
    return "fail(rc=) / skip / ok 三态与退出码一致"


def case_b3_bmdrt_bad_and_good(tmp):
    """bmdrt：坏谱=拒收(skip)、好谱照跑、汇总落盘；退出码 0（拒收 ≠ 失败）。"""
    bad = tmp / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp / "bt_out"
    opt = {"out": str(out), "quick": True, "krange": "1,1", "no_move": True}
    rc = bmdrt.cmd_auto([str(bad), str(GOOD)], opt)
    assert rc == 0, f"只有拒收时退出码应为 0（拒收不是失败），实际 {rc}"

    meta_bad = json.loads((out / bad.stem / "meta.json").read_text(encoding="utf-8"))
    assert meta_bad["skip"] is True, meta_bad
    assert meta_bad["err"], "拒收必须写明理由，不能空着"

    js = out / GOOD.stem / f"real_{GOOD.stem}.json"
    assert js.exists(), "好谱没有出结果"
    assert (out / "summary.json").exists() or (out / "summary.md").exists(), \
        "汇总没有落盘"
    d = strict_load(js)                      # D 组：真实产物必须严格可解析
    # P2-1：单 K 没有比较，不许把「没比过」写成 elpd_diff=0 / rank=1
    assert d.get("comparison_performed") is False, d.get("comparison_performed")
    assert d.get("elpd_diff") is None and d.get("rank") is None, \
        f"单 K 仍落了比较结论：elpd_diff={d.get('elpd_diff')}, rank={d.get('rank')}"
    return f"坏谱拒收({meta_bad['err'][:18]}…)、好谱出结果、rc=0"


def case_b4_bmdrt_fit_failure(tmp):
    """bmdrt：run() 内部 SystemExit（如 Ws+ad 同开）记一条失败、批不死、退出码 1。"""
    out = tmp / "bt_fail"
    opt = {"out": str(out), "quick": True, "krange": "1,1", "no_move": True,
           "diffusion": "Ws", "adsorption": "rl"}
    rc = bmdrt.cmd_auto([str(GOOD)], opt)
    assert rc == 1, f"真失败必须返回 1，实际 {rc}"
    meta = json.loads((out / GOOD.stem / "meta.json").read_text(encoding="utf-8"))
    assert meta["ok"] is False and meta["err"], meta
    assert "中止" in meta["err"], meta["err"]
    return "run 内 SystemExit 被接住并记为失败、rc=1"


# ---------------------------------------------------------------------------
# C 组：门禁三态（审查 P2-2）
# ---------------------------------------------------------------------------

def _kk_stub(f, Z, status="fail", **kw):
    return {"status": status, "reason": f"桩：{status}", "attribution": [],
            "n": int(f.size), "f_min": float(f.min()), "f_max": float(f.max())}


def case_c_gate_tri_state(tmp):
    """gate 模式三态：fail 拒绝 / unavailable+strict 拒绝 / unavailable 不 strict 放行。

    两种拒绝都必须落 `real_<stem>.json`（`fit_performed=false`），且
    `kk_performed` 与「门禁到底跑成没跑成」一致——这正是 P2-2 的契约。
    """
    f, Z = R.load_spectrum(str(GOOD))
    stem = GOOD.stem
    old_kk, old_pre, old_out = R._kk_check, R.prescreen_stage, R.OUT

    class _ReachedFit(Exception):
        """桩：跑到「门禁之后」的第一站就停，不真的进 MCMC。"""

    def _boom(*a, **k):
        raise _ReachedFit()

    def _run(tag, kk_impl, strict):
        """跑一次 run()，返回 (返回值, 已打印文本, 是否放行到拟合前站)。"""
        R.OUT = tmp / tag
        R.OUT.mkdir(parents=True, exist_ok=True)
        R._kk_check = kk_impl
        R.prescreen_stage = _boom
        opts = {} if strict is None else {"strict": strict}
        buf = io.StringIO()
        ret, reached = None, False
        try:
            with contextlib.redirect_stdout(buf):
                ret = R.run(str(GOOD), K_force=None, krange="1,1",
                            include_L=True, kk_mode="gate", kk_opts=opts)
        except _ReachedFit:
            reached = True
        return ret, buf.getvalue(), reached

    try:
        # ① 门禁判 fail → 拒绝定 K，但门禁**确实跑过**（kk_performed=True）
        ret, txt, reached = _run("k1", _kk_stub, None)
        assert not reached and ret is None, "fail 必须拒绝定 K"
        d = strict_load(R.OUT / f"real_{stem}.json")
        assert d["fit_performed"] is False, d.get("fit_performed")
        assert d["kk"]["status"] == "fail" and d["kk"]["kk_performed"] is True, d["kk"]

        # ② 门禁抛异常 + strict → 同样拒绝，但 kk_performed=false（如实披露）
        def _raise(*a, **k):
            raise RuntimeError("桩：门禁内部崩了")

        ret, txt, reached = _run("k2", _raise, True)
        assert not reached and ret is None, "strict 下 unavailable 必须拒绝"
        d = strict_load(R.OUT / f"real_{stem}.json")
        assert d["kk"]["status"] == "unavailable", d["kk"]
        assert d["kk"]["kk_performed"] is False, d["kk"]

        # ③ 门禁抛异常 + 非 strict → fail-open，但必须显著披露
        ret, txt, reached = _run("k3", _raise, False)
        assert reached, "非 strict 下 unavailable 应当继续拟合（fail-open）"
        assert "kk_performed=false" in txt, \
            f"fail-open 必须在报告里如实披露，实际输出：{txt[-200:]}"

        # ④ 门禁依赖整体缺失（_kk_check is None）→ 直接跳过门禁
        ret, txt, reached = _run("k4", None, None)
        assert reached, "门禁模块缺失时应跳过门禁继续"
        assert "K-K 事前门禁" not in txt, "依赖缺失时不该打印门禁结论"
    finally:
        R._kk_check, R.prescreen_stage, R.OUT = old_kk, old_pre, old_out
    return "fail / unavailable+strict / fail-open / 依赖缺失 四态一致"


# ---------------------------------------------------------------------------
# D 组：JSON 合法性（审查 P2-4 / P2-1）
# ---------------------------------------------------------------------------

def case_d_json_safe():
    """`_json_safe` 把 NaN/±Inf 递归换成 None，配合 allow_nan=False 落盘。"""
    payload = {"a": float("nan"), "b": [1, float("inf"), {"c": float("nan")}],
               "d": np.float64("nan"), "e": np.array([np.nan, 2.0]),
               "f": True, "g": np.int64(3), "h": np.bool_(True)}
    safe = R._json_safe(payload)
    s = json.dumps(safe, allow_nan=False)      # 有漏网就 raise
    for token in ("NaN", "Infinity"):
        assert token not in s, f"落盘文本里还有 {token}：{s}"
    back = json.loads(s)
    assert back["a"] is None and back["d"] is None, back
    assert back["b"][1] is None and back["b"][2]["c"] is None, back
    assert back["e"] == [None, 2.0], back["e"]
    assert back["g"] == 3 and back["h"] is True
    return "嵌套 NaN/±Inf → null，严格 dumps 通过"


def case_d_guard_is_real():
    """反向断言：不经过 `_json_safe` 的 NaN 必须让 allow_nan=False 抛错。"""
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
    except ValueError:
        return "allow_nan=False 确实拦得住裸 NaN"
    raise AssertionError("allow_nan=False 没拦住 NaN——护栏形同虚设")


def case_d_legacy_inventory():
    """历史产物盘点（信息项，不判红）：修复前的 real_*.json 允许带 NaN。"""
    n_all = n_nan = 0
    for p in (ROOT / "out").rglob("real_*.json"):
        n_all += 1
        try:
            strict_load(p)
        except Exception:
            n_nan += 1
    assert n_all > 0, "out/ 下没有 real_*.json，检查是不是跑错目录了"
    return f"out/ 下 {n_all} 份历史 JSON，其中 {n_nan} 份带 NaN（修复前产物，不追溯）"


# ---------------------------------------------------------------------------
# E 组：参考性 Rwp（唯读对照，不进门禁/选K/参数推断）
# ---------------------------------------------------------------------------

def case_e1_ref_rwp_identity():
    """参考模块的两条恒等式：与生产解析式一致；Rwp = √2 · 项目 rms_rel。"""
    import bm_ref_rwp as REF
    r = REF.selftest()
    assert r["ok"], r
    return f"对拍 model_z {r['hn']:.2e}；Rwp/rms_rel 关系 {r['rms']:.2e}"


def case_e2_ref_rwp_skip_contract(tmp):
    """拓扑不一致必须 `skipped` 且 performed=False，但仍给出 rwp_primary。"""
    import bm_ref_rwp as REF

    f = np.logspace(-1, 5, 51)
    p = dict(R_inf=1.0, w=np.array([3.0, 2.0]),
             tau=np.array([1 / (2 * np.pi * 30.0), 1 / (2 * np.pi * 0.5)]),
             phi=np.array([0.9, 0.8]), L=1e-7)
    Z = REF.z_from_params(f, **p, include_L=True)
    base = dict(K=2, w=p["w"], f0=1 / (2 * np.pi * p["tau"]), phi=p["phi"],
                R_inf=p["R_inf"], L=p["L"], diffusion=None, adsorption=None)

    # ① 正常情形：应该真的重拟合，且 LS 不劣于 primary
    #    无噪合成谱上 primary 可能是 0，容差要带绝对下限（与模块护栏同口径）
    ok = REF.reference_block(f, Z, dict(base), True, 1.0)
    assert ok["ls_relative_refit_performed"] is True, ok
    assert ok["ls_relative_refit_status"] == "ok", ok
    assert ok["rwp_ls_relative"] <= max(ok["rwp_primary"], 1e-6) * 1.001, ok
    assert "sqrt(2)" in ok["ls_relative_weight_definition"], "口径串必须写明不除 √2"

    # ② ζ 自由 → 拓扑不同，跳过（但 primary 仍算）
    for patch, why in ((dict(zeta="free"), "ζ"),):
        blk = REF.reference_block(f, Z, dict(base), True, patch["zeta"])
        assert blk["ls_relative_refit_status"] == "skipped", blk
        assert blk["ls_relative_refit_performed"] is False, blk
        assert why in blk["reference_note"], blk["reference_note"]
        assert blk["rwp_primary"] is not None, "跳过重拟合也必须有 primary"

    # ③ 带扩散项 → 同样跳过
    blk = REF.reference_block(f, Z, dict(base, diffusion="Ws"), True, 1.0)
    assert blk["ls_relative_refit_status"] == "skipped", blk
    assert "扩散" in blk["reference_note"], blk["reference_note"]
    return "重拟合/ζ自由/带扩散 三态符合契约"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def case_e3_ref_rwp_requires_success(tmp):
    """`least_squares.success=False` 的解**不许**当参考值（外部审查 2026-11-11）。

    这是契约回归：`max_nfev` 耗尽时 `rr.fun` 照样有限，旧实现只看 `isfinite`，
    就会把「跑了 4000 次没收敛」的解写进 `rwp_ls_relative` 并标 `status="ok"`。
    这里用桩替换 `scipy.optimize.least_squares`，不依赖真数据、不跑真优化。
    """
    import scipy.optimize as SO

    import bm_ref_rwp as REF

    f = np.logspace(-1, 5, 51)
    p = dict(R_inf=1.0, w=np.array([3.0, 2.0]),
             tau=np.array([1 / (2 * np.pi * 30.0), 1 / (2 * np.pi * 0.5)]),
             phi=np.array([0.9, 0.8]), L=1e-7)
    Z = REF.z_from_params(f, **p, include_L=True)
    base = dict(K=2, w=p["w"], f0=1 / (2 * np.pi * p["tau"]), phi=p["phi"],
                R_inf=p["R_inf"], L=p["L"], diffusion=None, adsorption=None)
    kw = dict(R_inf=p["R_inf"], w=p["w"], tau=p["tau"], phi=p["phi"],
              L=p["L"], include_L=True)

    class _RR:
        def __init__(self, x0, fun, success, status, msg, nfev):
            self.x = np.asarray(x0, float)
            self.fun = fun
            self.success = success
            self.status = status
            self.message = msg
            self.nfev = nfev

    n = 2 * f.size

    def _all_max_nfev(resid, x0, **k):
        return _RR(x0, np.full(n, 0.5), False, 0,
                   "The maximum number of function evaluations is exceeded.",
                   int(k.get("max_nfev", 1)))

    calls = {"n": 0}

    def _mixed(resid, x0, **k):
        """第一个起点：未收敛但残差更低；其余：收敛但残差更高。"""
        calls["n"] += 1
        if calls["n"] == 1:
            return _RR(x0, np.full(n, 0.01), False, 0, "max_nfev", 4000)
        return _RR(x0, np.full(n, 0.5), True, 3, "gtol satisfied", 10)

    old = SO.least_squares
    try:
        # ① 全部起点 max_nfev 耗尽 → 不报数（旧实现会报一个"ok"的数）
        SO.least_squares = _all_max_nfev
        blk = REF.reference_block(f, Z, dict(base), True, 1.0, starts=3)
        assert blk["ls_relative_refit_performed"] is True, blk
        assert blk["ls_relative_refit_status"] == "not_converged", blk
        assert blk["ls_relative_refit_converged"] is False, blk
        assert blk["rwp_ls_relative"] is None, "未收敛解被当成参考值写进去了"
        assert blk["ls_relative_refit_not_converged_starts"] == 3, blk
        det = blk["ls_relative_refit_starts_detail"]
        assert len(det) == 3 and all(d["success"] is False for d in det), det
        assert all(d["cost"] is not None for d in det), det    # 残差有限，正是坑所在
        assert "success" in blk["reference_note"], blk["reference_note"]

        # ② 混合：只采纳收敛起点；同时如实标记"未收敛者代价更低"
        SO.least_squares = _mixed
        out = REF.refit(f, Z, starts=3, **kw)
        assert out["status"] == "ok", out
        assert out["not_converged_starts"] == 1, out
        assert out["unconverged_lower_cost"] is True, \
            "未收敛起点代价更低却没有标记，报值会显得是「最优」"
        assert out["starts_detail"][0]["success"] is False, out["starts_detail"][0]
        assert out["rwp_pct"] is not None
    finally:
        SO.least_squares = old
    return "全未收敛→不报数；混合→只采纳收敛解并标记代价更低的未收敛起点"


# ---------------------------------------------------------------------------
# F 组：结果层与展示层（bm_result / bm_dash / bm_figure）
# ---------------------------------------------------------------------------

def _fake_sample(tmp, grade=2, n_draw=240, seed=0, zero_sigma_im=False):
    """造一个**自包含**的样品目录：npz 后验 + 结果 JSON。

    不依赖 out/ 里的历史产物（那些不入库），也不跑 MCMC——
    这里验的是「数据层 → 图/CSV」的一致性，不是拟合正确性。
    """
    import bm_core as B

    rng = np.random.default_rng(seed)
    f = np.logspace(-1, 5, 40)
    tau0 = np.array([0.08, 8e-4])
    p0 = dict(R_inf=1.0, W=1.0, pi=np.array([4.0, 1.5]),
              tau=tau0, phi=np.array([0.9, 0.75]), zeta=np.ones(2),
              L=1.2e-7)
    Z0 = B.model_z(f, p0, include_L=True)
    sig0 = 0.01 * np.abs(Z0)

    draws = dict(R_inf=[], W=[], pi=[], tau=[], phi=[], zeta=[], L=[])
    for _ in range(n_draw):
        draws["R_inf"].append(1.0 * (1 + 0.002 * rng.normal()))
        draws["W"].append(1.0 * (1 + 0.002 * rng.normal()))
        draws["pi"].append(p0["pi"] * (1 + 0.01 * rng.normal(size=2)))
        draws["tau"].append(tau0 * np.exp(0.02 * rng.normal(size=2)))
        draws["phi"].append(p0["phi"] * (1 + 0.005 * rng.normal(size=2)))
        draws["zeta"].append(np.ones(2))
        draws["L"].append(1.2e-7 * (1 + 0.002 * rng.normal()))

    arr = {}
    for k, v in draws.items():
        arr[k] = np.asarray(v, float)
    n = arr["R_inf"].shape[0]
    zs = np.empty((n, f.size), complex)
    for i in range(n):
        zs[i] = B.model_z(f, B.sample_params({**arr, "ln_tau": np.log(arr["tau"])},
                                            i, True), True)

    Zobs = Z0 + rng.normal(0, 1, f.size) * sig0 \
        + 1j * rng.normal(0, 1, f.size) * sig0
    d = Path(tmp)
    s_im = np.zeros((n, f.size)) if zero_sigma_im else np.full((n, f.size), sig0)
    (d / "real_synthetic.json").write_text(json.dumps({}), encoding="utf-8")
    np.savez(d / "real_synthetic_samples.npz",
             f=f, Zre=Zobs.real, Zim=Zobs.imag,
             s_re=np.full((n, f.size), sig0), s_im=s_im,
             **arr)
    med = {k: np.median(v, axis=0) for k, v in arr.items()}
    f0 = 1.0 / (2 * np.pi * med["tau"])
    rec = dict(
        file="synthetic", n=int(f.size), f_min=float(f.min()),
        f_max=float(f.max()), K=int(tau0.size), include_L=True, zeta=1.0,
        fit_performed=True, sampling=dict(num_chains=2, num_warmup=1000,
                                          num_samples=1000),
        seed=seed, sample_grade=int(grade),
        grade_fail=[] if grade == 2 else ["tail ESS 244 < 400"],
        divergences=0, tree_cap=0.0,
        elpd_loo=123.4, elpd_se=3.1, elpd_diff=2.0, dse=1.5,
        comparison_performed=True,
        rms_re=float(np.sqrt(np.mean((Zobs.real - zs.real.mean(0)) ** 2))),
        rms_im=float(np.sqrt(np.mean((Zobs.imag - zs.imag.mean(0)) ** 2))),
        sigma_re=float(np.median(sig0)), sigma_im=float(np.median(sig0)),
        acf=0.31, trend_re=0.1, trend_im=0.1, cover_re=0.9, cover_im=0.92,
        R_inf=float(med["R_inf"]), W=float(med["W"]),
        L_uH=float(med["L"] * 1e6),
        components=[dict(f0=float(f0[i]), f0_lo=float(f0[i] * 0.95),
                         f0_hi=float(f0[i] * 1.05), w=float(med["pi"][i]),
                         w_lo=float(med["pi"][i] * 0.96),
                         w_hi=float(med["pi"][i] * 1.04),
                         phi=float(med["phi"][i]), p_outside=0.0,
                         p_tau_outside=0.0, p_peak_outside=0.0,
                         p_outside_caliber="peak") for i in range(tau0.size)],
        resid_bins=[], diag={"R_inf": dict(rhat=1.002, bulk=900, tail=800)},
        k_selection=["K=3 **剔除**：R̂=1.42；div=3"],
        warnings=["**测试用**合成谱，不是实验数据"],
        kk=dict(status="ok", reason="合成谱", mu=0.01, M=12),
        k_selection_status="converged" if grade == 2 else "degraded",
        selection_tier=2 if grade == 2 else 1,
        component_review=dict(parameter_identifiability=dict(
            strong_pairs=[], n_strong=0, max_at_bound=0.0, level="ok")),
        param_corr=dict(
            names=["R_inf", "tau_1", "tau_2", "phi_1", "phi_2"],
            corr=[[1.0, 0.1, 0.0, 0.2, 0.0],
                  [0.1, 1.0, -0.95, 0.3, 0.0],
                  [0.0, -0.95, 1.0, 0.0, 0.2],
                  [0.2, 0.3, 0.0, 1.0, 0.1],
                  [0.0, 0.0, 0.2, 0.1, 1.0]],
            summary=[dict(name="R_inf", q05=0.99, q50=1.0, q95=1.01,
                          sd=0.004, log=True),
                     dict(name="tau_1", q05=0.078, q50=0.08, q95=0.082,
                          sd=0.001, log=True),
                     dict(name="tau_2", q05=7.8e-4, q50=8e-4, q95=8.2e-4,
                          sd=1e-5, log=True),
                     dict(name="phi_1", q05=0.88, q50=0.9, q95=0.92,
                          sd=0.01, log=False),
                     dict(name="phi_2", q05=0.73, q50=0.75, q95=0.77,
                          sd=0.01, log=False)],
            fixed=[dict(name="zeta_1", value=1.0)],
            strong=[dict(a="tau_1", b="tau_2", rho=-0.95, why="测试")],
            medium=[], degenerate=[]),
        diffusion_resolution=dict(level="no_diffusion", resolvable=False),
        zmax_obs=float(np.abs(Z0).max()),
    )
    (d / "real_synthetic.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return d


def case_f1_data_layer_consistency(tmp):
    """数据层：曲线、残差、CSV、manifest 必须是同一份数（不许各算一遍）。"""
    import bm_core as B
    import bm_result as RES

    d = _fake_sample(tmp)
    res = RES.AnalysisResult.load(d)
    pay = res.payload()
    ref = np.load(d / "real_synthetic_samples.npz")
    s = {k: ref[k] for k in ref.files}
    f = s["f"]

    # ① 拟合曲线 = 逐样本算 Z 后的逐点中位（不是"中位参数算一条曲线"）
    idx = np.linspace(0, s["R_inf"].size - 1,
                      min(RES.MAX_DRAWS, s["R_inf"].size)).astype(int)
    zz = np.array([B.model_z(f, B.sample_params(s, int(i), True), True)
                   for i in idx])
    ref_fit = np.median(zz.real, 0) + 1j * np.median(zz.imag, 0)
    assert np.allclose(res.z_fit, ref_fit, rtol=1e-12, atol=0), \
        "拟合曲线与逐样本分位不一致——曲线被另一套参数算了"

    # ② 残差与标准化残差是恒等式，不是另一遍拟合
    assert np.allclose(res.resid["re"], res.z_obs.real - res.z_fit.real,
                       rtol=0, atol=1e-15)
    sig = np.median(np.asarray(s["s_re"]), 0)
    assert np.allclose(res.norm["re"], res.resid["re"] / sig, rtol=1e-12), \
        "标准化残差没有用拟合自己的 σ"

    # ③ 区间包含关系：预测区间（含噪声）必须包住响应区间
    band_w = res.band["re_hi"] - res.band["re_lo"]
    pred_w = res.pred["re_hi"] - res.pred["re_lo"]
    assert np.all(pred_w >= band_w * 0.999), "预测区间比响应区间还窄"

    # ④ 导出包：CSV 里的数必须能对回内存对象；manifest 的 sha256 必须对
    import csv
    root = res.bundle(Path(tmp) / "bundle")
    with open(root / "data" / "fitted_impedance.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == f.size
    got = np.array([float(r["Zre_fitted_median"]) for r in rows])
    assert np.allclose(got, res.z_fit.real, rtol=1e-9, atol=1e-12), \
        "CSV 的拟合值与图上的曲线不是同一份数"
    got_r = np.array([float(r["normalized_residual_im"]) for r in rows])
    assert np.allclose(got_r, res.norm["im"], rtol=1e-9, atol=1e-9)

    man = strict_load(root / "manifest.json")
    import hashlib
    for item in man["files"]:
        h = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        assert h == item["sha256"], f"{item['path']} 的 sha256 对不上"
    assert man["source"]["npz"]["sha256"] == hashlib.sha256(
        (d / "real_synthetic_samples.npz").read_bytes()).hexdigest()
    return f"n={f.size} 曲线/残差/CSV/sha256 一致"


def case_f2_verdict_is_honest(tmp):
    """口径：分级 1 必须显示为降级；区间类型必须分开标注。"""
    import bm_result as RES

    d = _fake_sample(tmp, grade=1)
    res = RES.AnalysisResult.load(d)
    v = res.verdict
    assert v["grade"] == 1
    assert "降级" in v["label"], f"分级 1 却写成 {v['label']!r}"
    assert v["level"] != "ok", "分级 1 不许给 ok 的总体等级"
    assert "结构判据" in v["label"], "总判定必须与结构判据分开陈述"

    pay = res.payload()
    assert "非联合" in pay["band_note"] or "边际" in pay["band_note"], \
        "区间口径说明缺失——会被读成联合可信域"
    assert "2%" in pay["sigma_note"], "标准化残差口径说明缺失"
    # 只有 1 个 K 时不许出现"比较过"的假象
    assert pay["comparison"] is True
    return v["label"]


def case_f3_spec_roundtrip(tmp):
    """图形配置：未知键必须报错；落盘再读回必须一致；尺寸按 mm 生效。"""
    import bm_figure as FIG
    import bm_result as RES

    d = _fake_sample(tmp)
    pay = RES.AnalysisResult.load(d).payload()

    spec = FIG.validate_spec({"theme": "acs", "layout": "pair",
                              "panels": ["nyquist", "drt"]})
    assert spec["canvas"]["width_mm"] == FIG.DEFAULT_SPEC["canvas"]["width_mm"]

    for bad in ({"themes": "acs"}, {"theme": "nope"}, {"layout": "2x3"},
                {"panels": ["nyquist", "bode", "drt"], "layout": "pair"},
                {"labels": "fr"}, {"canvas": {"width_mm": 5}}):
        try:
            FIG.validate_spec(bad)
        except FIG.SpecError:
            pass
        else:
            raise AssertionError(f"非法 spec 没报错：{bad}")

    # 落盘 → 读回 → 再校验，必须逐键相等（可复现的版式）
    p = Path(tmp) / "spec.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert FIG.validate_spec(back) == spec

    fig, spec2 = FIG.draw(pay, spec)
    w, h = fig.get_size_inches()
    assert abs(w - spec2["canvas"]["width_mm"] / 25.4) < 1e-9
    assert abs(h - spec2["canvas"]["height_mm"] / 25.4) < 1e-9, \
        "画布尺寸没有按 mm 精确生效（bbox_inches 会偷偷改尺寸）"
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f"{spec2['theme']}/{spec2['layout']} {spec2['canvas']}"


def case_f4_dashboard_payload_identity(tmp):
    """面板：内嵌 JSON 必须与数据层逐值相等，且不含 NaN/Infinity。"""
    import bm_dash
    import bm_result as RES

    d = _fake_sample(tmp)
    pay = RES.AnalysisResult.load(d).payload()
    html = bm_dash.build(pay, "synthetic", bm_dash.DEFAULT_LAYOUT, "minimal")
    assert "__DATA__" not in html and "__TITLE__" not in html, "占位符没替换"

    import re
    m = re.search(r"^const D = (\{.*\});$", html, re.M)
    assert m, "找不到内嵌载荷"
    back = json.loads(m.group(1))
    assert back["curves"]["f"] == pay["curves"]["f"]
    assert back["curves"]["fre"] == pay["curves"]["fre"]
    assert back["verdict"] == pay["verdict"]
    assert "NaN" not in m.group(1) and "Infinity" not in m.group(1), \
        "载荷里有非法 JSON 常量"
    # 三种面板都要能画：缺 DRT/缺后验时必须给文字而不是崩
    for pid in ("fit", "drt", "resid"):
        assert pid in html
    return f"{len(html) / 1024:.0f} KB"


def case_f5_bundle_is_finalized(tmp):
    """封包：bundle 先落 partial；写完 figures 再 finalize，图必须进清单。

    推荐流程就是「先 --bundle 再 --all」。如果 finalize 不给出版图补哈希，
    「每个文件都有 SHA256」这句承诺就是空话。
    """
    import bm_result as RES

    d = _fake_sample(tmp)
    res = RES.AnalysisResult.load(d)
    root = res.bundle(Path(tmp) / "bundle")
    man = strict_load(root / "manifest.json")
    assert man["status"] == "partial", "bundle 之后不该自称已完成"
    assert man["pending"].get("figures"), "没说清还缺什么"
    assert not any(f["path"].startswith("figures/") for f in man["files"])

    # 模拟 bm_figure 写图（不真跑 matplotlib，只验封包逻辑）
    (root / "figures").mkdir(exist_ok=True)
    (root / "figures" / "figure_nyquist.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (root / "figures" / "figure_spec.json").write_text("{}", encoding="utf-8")
    mp = RES.finalize_bundle(root)
    man = strict_load(mp)
    assert man["status"] == "finalized" and man["complete"] is True
    assert man["counts"]["figures"] == 2, man["counts"]
    paths = {f["path"]: f for f in man["files"]}
    assert "figures/figure_nyquist.png" in paths, "出版图没进 manifest"
    import hashlib
    for p, item in paths.items():
        assert item["sha256"] == hashlib.sha256((root / p).read_bytes()).hexdigest(), \
            f"{p} 的 sha256 对不上"
    RES.finalize_bundle(root)                       # 幂等
    assert len(strict_load(mp)["files"]) == len(paths)
    return f"{len(paths)} 个文件，figures {man['counts']['figures']}"


def case_f6_norm_sigma_halves(tmp):
    """口径：实部/虚部标准化残差**分别**判有效，不能一坏全丢、也不能一好全发。"""
    import bm_result as RES

    d = _fake_sample(tmp, zero_sigma_im=True)
    res = RES.AnalysisResult.load(d)
    assert res.norm["re"] is not None, "Re 的 σ 有效却把 Re 也置空了"
    assert res.norm["im"] is None, "Im 的 σ 全 0，标准化残差却是有效值"
    assert np.allclose(res.norm["re"],
                       res.resid["re"] / np.median(
                           np.load(d / "real_synthetic_samples.npz")["s_re"], 0),
                       rtol=1e-12)

    pay = res.payload()
    assert pay["norm_valid"] == {"re": True, "im": False}, pay["norm_valid"]
    assert "nre" in pay["curves"] and "nim" not in pay["curves"], \
        "载荷把无效的那半也发出去了"
    assert "Im" in " ".join(pay["layer_notes"]), "没有说明哪一半不可用"

    root = res.bundle(Path(tmp) / "bundle")
    import csv
    with open(root / "data" / "residuals.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["normalized_residual_im"] == "" for r in rows), \
        "无效列的单元格不是空的（写成 0 会被当成残差为零）"
    assert all(r["normalized_residual_re"] != "" for r in rows)
    return f"Re 有效 / Im 置空（{len(rows)} 行）"


def case_f7_spec_nested_keys(tmp):
    """图形配置：嵌套块也逐键校验；配色可改，语义不改。"""
    import bm_figure as FIG

    bad_specs = [
        {"font": {"famliy": "Arial"}},            # 拼错必须报错，不能静默忽略
        {"canvas": {"width": 180}},               # 少了 _mm
        {"style": {"fitt": "#ff0000"}},           # 拼错的配色键
        {"style": {"fit": "red"}},                # 不是 #hex
        {"style": {"lw_fit": 99}},                # 线宽越界
        {"titles": {"nyquisty": "x"}},            # 面板名拼错
    ]
    for s in bad_specs:
        try:
            FIG.validate_spec(s)
        except FIG.SpecError:
            pass
        else:
            raise AssertionError(f"嵌套非法键没报错：{s}")

    ok = FIG.validate_spec({"style": {"fit": "#FF0000", "lw_fit": 2.0}})
    assert ok["style"]["fit"] == "#FF0000" and ok["style"]["lw_fit"] == 2.0
    th = FIG._apply_style(FIG.THEMES["acs"], ok["style"])
    assert th["fit"] == "#FF0000"
    assert th["band"] == "#FF0000", "只改拟合色时区间必须跟着，否则读成两组数据"
    assert th["data"] == FIG.THEMES["acs"]["data"], "标记/编码不该被 style 改到"
    return "嵌套校验 6 例；配色只改颜色"


# ---------------------------------------------------------------------------
# F8 组：B 类 guard 的「族适配」（2026-11-11 修 `param_degenerate` /
# `param_at_bound`）。判据本身不在这里测数值，只测**该响的响、该静的静**：
# 两条正对照同时钉住旧实现的两个相反错误方向，防的是「把一套错标签换成
# 另一套不完整标签」——比如把 voigt 的 `param_degenerate` 一律改判 `ok`。
# ---------------------------------------------------------------------------

def _f8_sampler_s(s, k):
    """把合成后验喂给 `bm_corr.param_corr`（与生产同一条对齐路径）。"""
    import bm_core as B
    import bm_corr as CORR

    mm = B.match_components(s)
    return CORR.param_corr(s, include_L=False, mm=mm)


def _f8_cand(s, family, dof):
    import bm_select as SEL

    raw = dict(s=s)
    if family == "voigt":
        raw["param_corr"] = _f8_sampler_s(s, dof)
    c = dict(family=family, dof=dof, raw=raw, rhat=1.0, div=0, ok=True)
    n_draw = int(np.asarray(next(iter(raw["s"].values()))).shape[0])
    return c, SEL.guard_state(c, n_draw=n_draw)


def case_f8_guard_semantics():
    """F8-1…F8-5：voigt 的简并/贴边必须被判到，nested 与 K=1 的对照不许误报。"""
    import bm_select as SEL

    n = 600
    rng = np.random.default_rng(3)
    tot = 100.0
    r = 60.0 + 5.0 * rng.normal(size=n)
    w = tot - r                                   # K=1：w₁ ≡ W，总量补偿
    s_k1 = dict(R_inf=r, W=w, pi=np.ones((n, 1)), w=w[:, None],
                tau=np.exp(-1.0 + 0.05 * rng.normal(size=n))[:, None],
                phi=(0.85 + 0.01 * rng.normal(size=n))[:, None],
                zeta=np.ones((n, 1)))
    # 旧路径（只看标量列）在这条后验上必然触发——正是要修掉的假阳性
    old_k1 = SEL._corr_pairs(s_k1)
    _, g = _f8_cand(s_k1, "voigt", 1)
    assert old_k1, "F8-1 构造失效：旧路径本应看到 (R_inf, W)，否则对照不成立"
    assert g["param_degenerate"] == "ok", (
        f"F8-1 K=1 总量补偿被当成分量简并：{g['param_degenerate']}（旧路径 {old_k1[:1]}）")

    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    w2 = np.stack([np.exp(5.5 + 0.30 * x + 0.02 * rng.normal(size=n)),
                   np.exp(6.5 + 0.05 * rng.normal(size=n))], 1)
    s_k2 = dict(R_inf=60.0 + 2.0 * rng.normal(size=n), W=w2.sum(1),
                pi=w2 / w2.sum(1, keepdims=True),
                tau=np.stack([np.exp(-2.0 + 0.30 * x + 0.02 * rng.normal(size=n)),
                              np.exp(2.0 + 0.05 * rng.normal(size=n))], 1),
                w=w2,
                phi=np.stack([0.85 + 0.02 * rng.normal(size=n),
                              0.80 + 0.02 * rng.normal(size=n)], 1),
                zeta=np.ones((n, 2)))
    # 旧路径对这条**失明**（二维列被丢），所以旧实现是恒 ok
    assert SEL._corr_pairs(s_k2) == [], "F8-2 构造失效：旧路径本应对二维列失明"
    _, g = _f8_cand(s_k2, "voigt", 2)
    assert g["param_degenerate"] == "triggered", (
        f"F8-2 K=2 的 w₁–τ₁ 简并被漏报：{g['param_degenerate']}")

    def _one(name, s, fam, dof, key, want):
        _, gg = _f8_cand(s, fam, dof)
        assert gg[key] == want, f"{name}：{key}={gg[key]}，期望 {want}"
        return gg[key]

    rng = np.random.default_rng(5)
    W = 900.0 + 2.0 * rng.normal(size=n)
    s_phi = dict(R_inf=60.0 + 2.0 * rng.normal(size=n), W=W, pi=np.ones((n, 1)),
                 w=W[:, None],
                 tau=np.exp(-1.0 + 0.05 * rng.normal(size=n))[:, None],
                 phi=(0.501 + 0.0005 * np.abs(rng.normal(size=n)))[:, None],
                 zeta=np.ones((n, 1)))
    _one("F8-3 φ→0.5", s_phi, "voigt", 1, "param_at_bound", "triggered")

    rng = np.random.default_rng(6)
    W = 900.0 + 2.0 * rng.normal(size=n)
    s_zeta = dict(R_inf=60.0 + 2.0 * rng.normal(size=n), W=W, pi=np.ones((n, 1)),
                  w=W[:, None],
                  tau=np.exp(-1.0 + 0.05 * rng.normal(size=n))[:, None],
                  phi=(0.85 + 0.01 * rng.normal(size=n))[:, None],
                  zeta=(0.305 + 0.0005 * np.abs(rng.normal(size=n)))[:, None])
    _one("F8-4 ζ→0.3", s_zeta, "voigt", 1, "param_at_bound", "triggered")

    def _nested(n1):
        rr = np.random.default_rng(7)
        return dict(Rs=10.0 + 0.1 * rr.normal(size=n),
                    R1=55.0 + 1.0 * rr.normal(size=n),
                    Q1=200.0 + 2.0 * rr.normal(size=n),
                    n1=n1 + 0.0005 * rr.normal(size=n),
                    R2=600.0 + 2.0 * rr.normal(size=n),
                    Q2=900.0 + 3.0 * rr.normal(size=n),
                    n2=0.85 + 0.0005 * rr.normal(size=n),
                    R_ad=3000.0 + 5.0 * rr.normal(size=n),
                    tau_ad=1.6 + 0.01 * rr.normal(size=n))

    _one("F8-5 nested n1→0.5", _nested(0.502), "nested", 2,
         "param_at_bound", "triggered")
    # 对照：nested 原规则不许被这次改动改坏（该静默时静默）
    _one("F8-5b nested 对照", _nested(0.75), "nested", 2,
         "param_at_bound", "ok")
    # 固定参数（zeta_fixed）不参与统计：必须是 fixed，不是 ok 也不是 triggered
    bf, det = SEL.at_bound_frac(dict(phi=(0.85 + 0.01 * rng.normal(size=n)),
                                     zeta=np.ones(n)), "voigt")
    assert bf == 0.0 and det["fixed"] == ["zeta"], (
        f"F8-6 固定参数口径错：frac={bf} fixed={det['fixed']}")
    return "F8-1…F8-6 共 6 项：K=1 不误报、K=2 不漏报、φ/ζ/n1 贴边必报、nested 对照、固定参数"


def case_f9_nested_and_outwindow():
    """F9：nested 分量定位（τ 派生）+ voigt 的 `comp_out_window`。

    两件事都只测**该响的响、该静的静**，并守住「comp_out_window 与
    comp_unresolved 分开报、不压成一个布尔」这条口径。
    """
    import bm_select as SEL

    n = 500
    rng = np.random.default_rng(11)

    # F9-1 nested τ 派生公式：拐点处 |Z_CPE| = R1（物理自检，非循环）
    R1, Q1, n1, f_ref = 55.0, 200.0, 0.75, 1.0
    s = dict(R1=np.full(n, R1) + 1e-6 * rng.normal(size=n),
             Q1=np.full(n, Q1) + 1e-6 * rng.normal(size=n),
             n1=np.full(n, n1) + 1e-7 * rng.normal(size=n),
             R_ad=np.full(n, 3000.0), tau_ad=np.full(n, 1.6))
    d = SEL.nested_component_spans(s, f_ref, "parallel", 1)
    assert d is not None and "tau1" in d["tau_names"]
    import bm_top as TOP
    tau1 = d["tau_med"][d["tau_names"].index("tau1")]
    zc = TOP.z_cpe(np.array([1.0 / (2 * np.pi * tau1)]), Q1, n1, f_ref=f_ref)
    assert abs(abs(zc[0]) - R1) / R1 < 0.02, (
        f"F9-1 τ1 拐点不对：|Z_CPE|={abs(zc[0]):.3f} 应≈R1={R1}")

    # F9-2 nested：τ2 摊开 >2 dec → triggered；全紧 → ok
    s2 = dict(Rs=10.0 + 0.1 * rng.normal(size=n),
              R1=55.0 + 1.0 * rng.normal(size=n),
              Q1=200.0 + 2.0 * rng.normal(size=n),
              n1=0.75 + 0.01 * rng.normal(size=n),
              R2=np.exp(np.log(600.0) + 1.5 * rng.normal(size=n)),
              Q2=900.0 + 3.0 * rng.normal(size=n),
              n2=0.85 + 0.01 * rng.normal(size=n),
              R_ad=3000.0 + 5.0 * rng.normal(size=n),
              tau_ad=1.6 + 0.01 * rng.normal(size=n))
    c = dict(family="nested", dof=2, branch="parallel",
             raw=dict(s=s2, f_ref=f_ref), rhat=1.0, div=0, ok=True)
    assert SEL.guard_state(c, n_draw=n)["comp_unresolved"] == "triggered", \
        "F9-2 τ2 跨 >2 dec 没被判 unresolved"
    s2t = dict(s2, R2=600.0 + 3.0 * rng.normal(size=n))
    c = dict(family="nested", dof=2, branch="parallel",
             raw=dict(s=s2t, f_ref=f_ref), rhat=1.0, div=0, ok=True)
    assert SEL.guard_state(c, n_draw=n)["comp_unresolved"] == "ok", \
        "F9-2b 全紧的 nested 被误判 unresolved"

    # F9-3 voigt comp_out_window：跑出窗口 → triggered；窗内 → ok；缺 → not_evaluable
    def _voigt_out(p_out):
        c = dict(family="voigt", dof=2,
                 raw=dict(s=dict(tau=np.ones((n, 2)), w=np.ones((n, 2))),
                          p_tau_out=p_out), rhat=1.0, div=0, ok=True)
        return SEL.guard_state(c, n_draw=n)["comp_out_window"]
    assert _voigt_out([0.9, 0.1]) == "triggered"
    assert _voigt_out([0.1, 0.2]) == "ok"
    assert _voigt_out(None) == "not_evaluable"

    # F9-4 GUARD_B 现在含 comp_out_window 且与 comp_unresolved 并存
    assert "comp_out_window" in SEL.GUARD_B and "comp_unresolved" in SEL.GUARD_B
    return "F9-1…F9-4：nested τ 拐点物理自检、nested 定位正反、comp_out_window 三态、清单并存"


CASES = [
    ("A1 合法用法", case_a1_legal_forms, False),
    ("A2 未知开关", case_a2_unknown_flags, False),
    ("A3 非法枚举", case_a3_enum_values, False),
    ("A4 缺参/坏值", case_a4_missing_or_bad_value, False),
    ("A5 解析→config", case_a5_cli_reaches_config, False),
    ("B1 nested 批量", case_b1_nested_fit_batch, True),
    ("B2 batch_2e 三态", case_b2_batch_2e_status, False),
    ("B3 bmdrt 坏+好", case_b3_bmdrt_bad_and_good, True),
    ("B4 bmdrt 真失败", case_b4_bmdrt_fit_failure, True),
    ("C  门禁三态", case_c_gate_tri_state, False),
    ("D1 json_safe", case_d_json_safe, False),
    ("D2 护栏有效", case_d_guard_is_real, False),
    ("D3 历史盘点", case_d_legacy_inventory, False),
    ("E1 参考Rwp恒等", case_e1_ref_rwp_identity, False),
    ("E2 参考Rwp跳过", case_e2_ref_rwp_skip_contract, False),
    ("E3 参考Rwp收敛", case_e3_ref_rwp_requires_success, False),
    ("F1 数据层一致", case_f1_data_layer_consistency, False),
    ("F2 判定口径", case_f2_verdict_is_honest, False),
    ("F3 图形配置", case_f3_spec_roundtrip, False),
    ("F4 面板载荷", case_f4_dashboard_payload_identity, False),
    ("F5 结果包封口", case_f5_bundle_is_finalized, False),
    ("F6 标准化残差半失效", case_f6_norm_sigma_halves, False),
    ("F7 spec 嵌套校验", case_f7_spec_nested_keys, False),
    ("F8 guard 族适配", case_f8_guard_semantics, False),
    ("F9 nested/窗口", case_f9_nested_and_outwindow, False),
]


def main() -> int:
    assert GOOD.exists(), f"缺少测试用谱：{GOOD}"
    ok = 0
    bad = []
    skipped = 0
    for name, fn, needs_fit in CASES:
        if needs_fit and SKIP_FIT:
            print(f"[--] {name}：--skip-fit 跳过")
            skipped += 1
            continue
        tmp = Path(tempfile.mkdtemp(prefix="ct_"))
        try:
            msg = fn(tmp) if fn.__code__.co_argcount else fn()
            ok += 1
            print(f"[ok] {name}：{msg}")
        except AssertionError as exc:
            bad.append((name, f"{exc}"))
            print(f"[FAIL] {name}：{exc}")
        except BaseException as exc:                 # noqa: BLE001
            bad.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERR ] {name}：{type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    total = len(CASES) - skipped
    print(f"\n{ok}/{total} 通过" + (f"；跳过 {skipped}" if skipped else "")
          + ("；未通过：" + ", ".join(n for n, _ in bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
