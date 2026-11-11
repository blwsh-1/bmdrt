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
# 主流程
# ---------------------------------------------------------------------------

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
