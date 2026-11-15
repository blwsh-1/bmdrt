# -*- coding: utf-8 -*-
"""③ 第 3 步独立探针：五项预算「入口 → 最终构造实参」的完整传递链取证。

独立性（本轮第一约束）
----------------------
本探针 **不导入产出层、不调用它的任何构造函数 / 校验器 / 投影函数**。
所有数值只取自五个地方：

  1. 生产模块常量：`bm_real.KW`、`bm_select.BUDGETS`、`nested_fit.VALIDATION_STANDARD`；
  2. 生产函数真实输出：`nested_fit.budget_kw()`；
  3. 埋在**真实生产调用链**上的 wrapper 入参快照（`fit_voigt` / `fit_nested` /
     `fit_one` / `bm_core.fit` / `bm_top.fit_top`）；
  4. numpyro `MCMC` / `NUTS` / `HMC` 构造与 `MCMC.run` 的真实实参（类级 hook）；
  5. 构造后 `mcmc` 对象的真实属性 + `inspect.signature` 对生产签名的默认绑定。

`_assert_independent()` 用 AST 扫本文件 + `sys.modules` 前后差集双重锁定；
只要探针自己引用了产出层就拒跑。因此本探针采集值可作为**外部基准**，
不构成「调构造函数，再用它自己的输出证明自己正确」。

微预算注入（明账，不是篡改传递值）
----------------------------------
生产档位是 2000/1000/… 步，真跑几十分钟。探针在 **`MCMC.__init__` 类边界**
把 `num_warmup` / `num_samples` 降为 8，只影响实际采样步数：
  · `ctor_passed`   = 注入**前**的真实构造实参 ⇒ 这才是传递链终点、取证用这个；
  · `ctor_effective`= 注入后真正交给 numpyro 的值 ⇒ 与对象属性交叉核验。
不改任何生产文件、不改五项预算的取证值；注入差异被断言为**仅此两项**。
C8 的「人为篡改」是另一种注入：改在**转发层** `bm_core.fit` 的入参上，
那会真的改动传递链，与类边界注入层次不同。

不写产物、不拟合批量、不构造 `run_id` / `workflow_id`、不改三处旧挂账。

    D:/autoeis_env/Scripts/python.exe docs/round_2026-11-15/budget_chain_probe.py
"""
from __future__ import annotations

import ast
import functools
import inspect
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
import bm_real as R                                     # noqa: E402
import bm_select as BS                                  # noqa: E402
import bm_top as BT                                     # noqa: E402
import nested_fit as NF                                 # noqa: E402

#: 五项核心。探针自己写死，与产出层的常量**不共用**（共用就变成自证）。
KEYS = ("n_warmup", "n_samples", "num_chains", "target_accept", "max_tree_depth")
#: 五个预算键 → 真实构造形参名。
CTOR = {"n_warmup": "num_warmup", "n_samples": "num_samples",
        "num_chains": "num_chains", "target_accept": "target_accept_prob",
        "max_tree_depth": "max_tree_depth"}
MICRO = {"num_warmup": 8, "num_samples": 8}


def five(d: dict) -> dict:
    return {k: d[k] for k in KEYS if k in d}


def _data(n: int = 24):
    f = np.logspace(-2, 4, n)
    w = 2 * np.pi * f
    return f, (10.0 / (1 + 1j * w * 0.01) + 1.0).astype(complex)


def _assert_independent() -> None:
    """护栏 1：本文件 AST 里不得出现产出层的任何引用（文档字符串里的文字不算节点）。"""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    bad = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        elif isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names = [node.value.id]
        if any(n.split(".")[0] == "budget_params" for n in names):
            bad.append((type(node).__name__, getattr(node, "lineno", -1), names))
    assert not bad, f"探针引用了产出层，独立性失效：{bad}"
    assert "budget_params" not in globals(), "探针命名空间被污染"


def _assert_keys_aligned() -> None:
    """护栏 0：静态对齐（只解析文本，不 import / 不执行产出层）。

    探针核的五键必须等于产出层声明的五键，但取值绝不从产出层拿——
    常量逐位对齐是形状检查，不是「拿构造函数的输出证明自己」。
    """
    src = (_ROOT / "budget_params.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id == "BUDGET_KEYS" for x in node.targets):
            found = tuple(ast.literal_eval(node.value))
    assert found == KEYS, f"探针五键与产出层不对齐：{found} vs {KEYS}"


def _bound(orig, args: tuple, kwargs: dict) -> dict:
    """`apply_defaults()` 后的真实绑定：缺的键会显示成**函数签名默认值**。"""
    b = inspect.signature(orig).bind_partial(*args, **kwargs)
    b.apply_defaults()
    return dict(b.arguments)


def _read_obj(mc) -> dict:
    """探针自己读属性，字段名与属性名不同（`n_warmup` → `mc.num_warmup`）。"""
    return {"n_warmup": mc.num_warmup, "n_samples": mc.num_samples,
            "num_chains": mc.num_chains,
            "target_accept": mc.sampler._target_accept_prob,
            "max_tree_depth": mc.sampler._max_tree_depth}


def _fmt(vals) -> str:
    if not vals:
        return "—"
    return "  ".join(f"{vals.get(k, '—')!s:>7}" for k in KEYS)


class Probe:
    """一次进程内挂全部埋点；`begin()` / `mark()` / `finish()` 管一条 case 的账。"""

    WRAPS = (
        (BS, "fit_voigt", "④转发", "bm_select.fit_voigt"),
        (BS, "fit_nested", "④转发", "bm_select.fit_nested"),
        (R, "fit_one", "④转发", "bm_real.fit_one"),
        (BC, "fit", "⑤生产fit", "bm_core.fit"),
        (BT, "fit_top", "⑤生产fit", "bm_top.fit_top"),
    )

    def __init__(self):
        self.active = False
        self.micro = True
        self.tamper = None
        self.rows: list[tuple] = []
        self.label = ""
        self.expected: dict = {}
        self._install()

    # ------------------------------------------------------------------ 埋点
    def _install(self) -> None:
        from numpyro.infer import HMC, MCMC, NUTS

        self._cls = (MCMC, NUTS, HMC)
        keep = {"init": MCMC.__init__, "run": MCMC.run,
                "NUTS": NUTS.__init__, "HMC": HMC.__init__}
        self._keep_cls = keep
        outer = self

        def mcmc_init(obj, sampler, **kw):
            # 取证用注入前；真正交给 numpyro 的用注入后。
            outer.ctor_passed = {"kernel": type(sampler).__name__, **dict(kw)}
            eff = dict(kw)
            if outer.micro:
                eff.update(MICRO)
            outer.ctor_effective = dict(eff)
            keep["init"](obj, sampler, **eff)
            outer.mcmc_obj = obj

        def mcmc_run(obj, rng_key, *args, **kw):
            outer.run_passed = dict(kw)
            return keep["run"](obj, rng_key, *args, **kw)

        def kern_hook(name, orig):
            def hook(obj, model, **kw):
                outer.kern_passed[name] = dict(kw)
                return orig(obj, model, **kw)   # 直接调原函数，避免 NUTS.super() 递归
            return functools.wraps(orig)(hook)  # 保住 __wrapped__，库默认才读得到

        MCMC.__init__ = functools.wraps(MCMC.__init__)(mcmc_init)
        MCMC.run = functools.wraps(MCMC.run)(mcmc_run)
        NUTS.__init__ = kern_hook("NUTS", keep["NUTS"])
        HMC.__init__ = kern_hook("HMC", keep["HMC"])

        # 模块函数 wrapper：全部走模块属性（生产里是 `B.fit` / `R.fit_one` / `T.fit_top`）。
        mods = {}
        for mod, name, layer, where in self.WRAPS:
            orig = getattr(mod, name)
            mods[(mod.__name__, name)] = orig

            def make(orig, mod, name, layer, where):
                @functools.wraps(orig)
                def hook(*args, **kwargs):
                    if outer.tamper is not None and mod is BC and name == "fit":
                        before_v = five(kwargs)
                        kwargs = outer.tamper(kwargs)
                        if outer.active:
                            outer.rows.append((
                                "④'篡改", "Probe.tamper（转发层 kwargs）", five(kwargs),
                                f"n_warmup {before_v.get('n_warmup')} → "
                                f"{kwargs.get('n_warmup')}；探针注入，明账"))
                    if outer.active:
                        b = _bound(orig, args, kwargs)
                        direct = {k: b[k] for k in KEYS if k in b}
                        nested = (five(b["kw"]) if isinstance(b.get("kw"), dict) else {})
                        vals = nested or direct
                        defaulted = [k for k in KEYS
                                     if k in direct and k not in kwargs and not nested]
                        note = ("来自 kw 入参" if nested else
                                "顶层绑定" + (f"；签名补齐 {','.join(defaulted)}" if defaulted else ""))
                        outer.rows.append((layer, where, vals, note))
                    return orig(*args, **kwargs)
                return hook

            setattr(mod, name, make(orig, mod, name, layer, where))
        self._keep_mod = mods

    def close(self) -> None:
        MCMC, NUTS, HMC = self._cls
        MCMC.__init__ = self._keep_cls["init"]
        MCMC.run = self._keep_cls["run"]
        NUTS.__init__ = self._keep_cls["NUTS"]
        HMC.__init__ = self._keep_cls["HMC"]
        for (modname, name), orig in self._keep_mod.items():
            setattr(sys.modules[modname], name, orig)

    # ------------------------------------------------------------------ 账本
    def begin(self, label: str, expected: dict) -> None:
        self.label, self.expected = label, dict(expected)
        self.rows = []
        self.ctor_passed = {}
        self.ctor_effective = {}
        self.kern_passed = {}
        self.run_passed = {}
        self.mcmc_obj = None
        self.active = True

    def mark(self, layer: str, where: str, values: dict, note: str = "") -> None:
        self.rows.append((layer, where, values, note))

    def finish(self, *, note: str = "") -> bool:
        self.active = False
        print(f"\n{'=' * 96}\n{self.label}\n{'=' * 96}")
        print(f"   {'层':<10}{'位置':<44}{'n_warmup':>9}{'n_samples':>10}"
              f"{'num_chains':>12}{'target_accept':>15}{'max_tree_depth':>16}")
        for layer, where, vals, n in self.rows:
            print(f"   {layer:<10}{where:<44}{_fmt(vals)}   {n}")

        # ⑥ 真实构造实参 = NUTS(...) ∪ MCMC(...)；注入前的 ctor_passed 才是链终点。
        ctor = {}
        for k in ("HMC", "NUTS"):
            ctor.update(self.kern_passed.get(k, {}))
        ctor.update({k: v for k, v in self.ctor_passed.items() if k != "kernel"})
        real = {k: ctor.get(CTOR[k]) for k in KEYS}
        self.last_real = dict(real)          # 供末尾辅助核验用：探针自己采集的值
        print(f"   {'⑥构造实参':<10}{'MCMC/NUTS 真实入参（注入前）':<44}{_fmt(real)}")

        obj = _read_obj(self.mcmc_obj) if self.mcmc_obj is not None else {}
        print(f"   {'⑦对象属性':<10}{'mc.* / mc.sampler.*':<44}{_fmt(obj)}"
              f"   ← warmup/samples 为类边界微预算注入后的值")

        # 判定 1：完整传递链（真实构造实参 == 该 case 预期）
        chain_ok = (real == self.expected)
        # 判定 2：类边界注入只可能碰 MCMC 侧两项；NUTS 侧两项不经注入。
        # C6 显式传 8/8 ⇒ 注入不改值 ⇒ 差异是子集，不是「恰好两项」。
        mcmc_side = ("n_warmup", "n_samples", "num_chains")
        diff = {k for k in mcmc_side
                if self.ctor_effective.get(CTOR[k]) != self.ctor_passed.get(CTOR[k])}
        inject_ok = (
            diff <= {"n_warmup", "n_samples"}
            and all(obj.get(k) == self.ctor_effective.get(CTOR[k]) for k in mcmc_side)
            and all(obj.get(k) == real[k] for k in ("target_accept", "max_tree_depth"))
            and obj.get("n_warmup") == MICRO["num_warmup"]
            and obj.get("n_samples") == MICRO["num_samples"]
        )
        # 判定 3：未被注入的三项，对象属性必须等于链终点
        cross_ok = all(obj.get(k) == real[k] for k in ("num_chains", "target_accept", "max_tree_depth"))

        ext = []
        cm_default = inspect.signature(self._keep_cls["init"]).parameters["chain_method"].default
        ext.append(f"chain_method: 构造实参未传 ⇒ 库默认 {cm_default!r}"
                   f"（num_chains={real['num_chains']}"
                   f"{'，单链不适用' if real['num_chains'] == 1 else '，多链由设备数派生'}）")
        ext.append(f"progress_bar: 构造实参={self.ctor_passed.get('progress_bar', '—')!r}"
                   f" / 属性={getattr(self.mcmc_obj, 'progress_bar', '—')!r}")
        ext.append(f"extra_fields: MCMC.run 实参={self.run_passed.get('extra_fields', '—')!r}")
        for line in ext:
            print(f"   · {line}")
        if note:
            print(f"   · {note}")

        print(f"   · 类边界注入差异键 = {sorted(diff) or '无'}"
              f"（允许 ⊆ {{n_warmup, n_samples}}）")
        print(f"   ⇒ 传递链 {'一致 ✓' if chain_ok else '不一致 ✗（见上表）'}"
              f" | 注入边界 {'✓' if inject_ok else '✗'}"
              f" | 对象属性交叉 {'✓' if cross_ok else '✗'}")
        return bool(chain_ok and inject_ok and cross_ok)


# ---------------------------------------------------------------------------
# 八条 case
# ---------------------------------------------------------------------------
def case_C1(p: Probe, f, Z) -> bool:
    """默认值 + 单链不适用：直连生产 `bm_core.fit`，一个预算键都不传。"""
    exp = dict(n_warmup=600, n_samples=1200, num_chains=1,
               target_accept=0.9, max_tree_depth=10)
    p.begin("C1 默认值 / 单链不适用 —— bm_core.fit 不传任何预算键", exp)
    p.mark("①入口声明", "（探针未传预算键）", {})
    p.mark("②查表", "（直连生产 fit，无查表）", {})
    p.mark("③覆盖", "（无覆盖）", {})
    BC.fit(f, Z, K=2, seed=0, progress=False, extra_fields=("num_steps",))
    return p.finish(note="⑤ 层「签名补齐」列出五项 ⇒ 缺键落函数默认；"
                         "默认被显式传给 MCMC，不是落库默认")


def case_C2(p: Probe, f, Z) -> bool:
    """显式值：`bm_select.BUDGETS['standard']` 经 voigt 臂全链转发。"""
    std = dict(BS.BUDGETS["standard"])
    exp = five(std)
    p.begin("C2 显式值 —— bm_select.BUDGETS['standard'] → fit_voigt → fit_one → bm_core.fit", exp)
    p.mark("①入口声明", "budget='standard'（探针注入的档名）", {})
    p.mark("②查表", "bm_select.BUDGETS['standard']", exp)
    p.mark("③覆盖", "（无覆盖）", exp)
    cand = BS.fit_voigt(f, Z, 2, kw=std)
    claim = (cand.get("raw") or {}).get("sampling") or {}
    return p.finish(note=f"fit_one.sampling（声明记录）={five(claim)}；本例与构造实参一致")


def case_C3(p: Probe, f, Z) -> bool:
    """A 路径 quick 覆盖：复现 `bm_real.run` 的三行真实覆盖，depth 13 不被档名改写。"""
    kw = dict(R.KW)
    base = five(kw)
    exp = base | dict(n_warmup=800, n_samples=400, num_chains=2, target_accept=0.95)
    p.begin("C3 覆盖 / 同名档不同值 —— bm_real.KW + run:2771 quick + run:2774 --ta", exp)
    p.mark("①入口声明", "--quick --ta 0.95（探针注入）", {})
    p.mark("②查表", "bm_real.KW（模块常量）", base)
    kw.update(n_warmup=800, n_samples=400, num_chains=2)   # bm_real.py:2771 原文
    kw["target_accept"] = 0.95                             # bm_real.py:2774 原文
    covered = five(kw)
    assert covered == exp, "覆盖结果与预期不符，探针自己的预期不能是手写值"
    p.mark("③覆盖", "kw.update 三键 + kw['target_accept']", covered)
    R.fit_one(f, Z, 2, False, "1.0", kw=kw)
    ok = p.finish(note="quick 只覆盖三键 ⇒ max_tree_depth 保留 KW 的 13，"
                       "不因档名写成 BUDGETS 的 10（Q1 举例）")
    return ok and exp["max_tree_depth"] == R.KW["max_tree_depth"] == 13


def case_C4(p: Probe, f, Z) -> bool:
    """C 路径 standard：`nested_fit.budget_kw('standard')` 的 ta=0.98 → fit_top。"""
    kwc = NF.budget_kw("standard")
    kwb = BS.BUDGETS["standard"]
    exp = five(kwc)
    p.begin("C4 同名档不同值 —— nested_fit.budget_kw('standard') → bm_top.fit_top", exp)
    p.mark("①入口声明", "--budget standard（探针注入）", {})
    p.mark("②查表", "nested_fit.VALIDATION_STANDARD（经 budget_kw）", exp)
    p.mark("②′对照", "bm_select.BUDGETS['standard']", five(kwb))
    p.mark("③覆盖", "（无覆盖）", exp)
    p.mark("④转发", "nested_fit.fit_one: T.fit_top(f,Z,...,**kw)", exp,
           "等价展开；薄封装会写产物，未调用（明账）")
    BT.fit_top(f, Z, branch="parallel", layers=1, seed=0, progress=False, **kwc)
    return p.finish(note=f"同为 standard：nested_fit ta={kwc['target_accept']} "
                         f"vs bm_select ta={kwb['target_accept']}；"
                         "未调用会写产物的 nested_fit.fit_one（明账）")


def case_C5(p: Probe, f, Z) -> bool:
    """B 路径 nested 臂：fit_nested 逐项挑五键转发 + 硬编码 progress=False。"""
    std = dict(BS.BUDGETS["standard"])
    exp = five(std)
    p.begin("C5 挑键转发 —— bm_select.fit_nested 逐项 int()/float() → bm_top.fit_top", exp)
    p.mark("①入口声明", "kw=BUDGETS['standard']（探针注入）", {})
    p.mark("②查表", "bm_select.BUDGETS['standard']", exp)
    p.mark("③覆盖", "（无覆盖）", exp)
    BS.fit_nested(f, Z, 1, kw=std)
    return p.finish(note="fit_nested 只挑 FIT_KEYS 五键；progress 硬编码 False、"
                         "extra_fields 在 fit_top 内硬编码 ('diverging',)")


def case_C6(p: Probe, f, Z) -> bool:
    """环境覆盖：CI 在场时 progress_bar 被静默压掉，五项预算不受影响。"""
    exp = dict(n_warmup=8, n_samples=8, num_chains=1,
               target_accept=0.9, max_tree_depth=10)
    p.begin("C6 环境覆盖 —— CI=1 且请求 progress=True（display_only）", exp)
    p.mark("①入口声明", "progress=True + CI=1（探针注入）", {})
    p.mark("②查表", "（直连生产 fit，无查表）", {})
    p.mark("③覆盖", "（无覆盖）", exp)
    os.environ["CI"] = "1"
    try:
        BC.fit(f, Z, K=2, n_warmup=8, n_samples=8, num_chains=1,
               target_accept=0.9, max_tree_depth=10, progress=True, seed=0,
               extra_fields=("num_steps",))
    finally:
        os.environ.pop("CI", None)
    ok = p.finish(note="progress_bar 是构造实参/属性差异，五项预算逐项不变 ⇒ "
                       "按 Q6 不进预算身份")
    return ok and (p.ctor_passed.get("progress_bar") is not None
                   and getattr(p.mcmc_obj, "progress_bar", None) !=
                   p.ctor_passed.get("progress_bar"))


def case_C7(p: Probe, f, Z) -> bool:
    """缺键三态：① 转发层硬索引 KeyError；② 查表静默丢 + fit 默认补齐。"""
    print(f"\n{'=' * 96}\nC7 缺键行为三态（真实生产反应，不靠猜）\n{'=' * 96}")

    # (a) bm_select.fit_nested 里 `int(kw["max_tree_depth"])` 是硬索引 ⇒ KeyError
    partial = dict(BS.BUDGETS["standard"])
    del partial["max_tree_depth"]
    err = None
    try:
        BS.fit_nested(f, Z, 1, kw=partial)
    except KeyError as exc:
        err = exc.args[0]
    a_ok = err == "max_tree_depth"
    print(f"   (a) fit_nested 缺 max_tree_depth ⇒ KeyError({err!r})  "
          f"{'✓' if a_ok else '✗'}  【值在④转发层就丢，整个拟合没启动】")

    # (b) nested_fit.budget_kw 的 `{k: b[k] for k in FIT_KEYS if k in b}` ⇒ 静默丢键，
    #     再 `**kw` 到 fit_top ⇒ 该项落 fit_top 签名默认 10。
    NF.BUDGETS["_probe_partial"] = {k: v for k, v in NF.budget_kw("standard").items()
                                    if k != "max_tree_depth"}
    try:
        kwp = NF.budget_kw("_probe_partial")
        b_lost = "max_tree_depth" not in kwp
        exp = five(kwp) | {"max_tree_depth": inspect.signature(
            BT.fit_top).parameters["max_tree_depth"].default}
        p.begin("C7(b) 静默丢键 → fit_top 签名默认补齐", exp)
        p.mark("①入口声明", "临时档 _probe_partial（探针注入，跑完还原）", {})
        p.mark("②查表", "nested_fit.budget_kw：if k in b 静默丢键", five(kwp))
        p.mark("③覆盖", "（无覆盖）", five(kwp))
        p.mark("④转发", "nested_fit.fit_one: T.fit_top(..., **kw)", five(kwp),
               "等价展开；薄封装会写产物，未调用（明账）")
        BT.fit_top(f, Z, branch="parallel", layers=1, seed=0, progress=False, **kwp)
        b_ok = p.finish(note="缺键没报错，fit_top 用签名默认 10 ⇒ "
                             "同一『缺键』在两条路径上反应完全不同")
    finally:
        NF.BUDGETS.pop("_probe_partial", None)

    # (c) bm_core.fit 缺键落函数默认 —— C1 已真跑到 MCMC，此处引用不重跑。
    c_ok = all(inspect.signature(BC.fit).parameters[k].default ==
               dict(n_warmup=600, n_samples=1200, num_chains=1,
                    target_accept=0.9, max_tree_depth=10)[k] for k in KEYS)
    print(f"   (c) bm_core.fit 缺键 ⇒ 落函数签名默认 600/1200/1/0.9/10 "
          f"{'✓' if c_ok else '✗'}  【C1 已真跑到构造点】")
    return bool(a_ok and b_ok and b_lost and c_ok)


def case_C8(p: Probe, f, Z) -> bool:
    """人为篡改传递值：只改转发层 kwargs，看声明记录与真实构造实参是否分叉。"""
    std = dict(BS.BUDGETS["standard"])
    exp = five(std) | {"n_warmup": 16}
    p.begin("C8 人为篡改传递值 —— Probe.tamper 改 bm_core.fit 的 n_warmup 2000→16", exp)
    p.mark("①入口声明", "budget='standard'（探针注入）", {})
    p.mark("②查表", "bm_select.BUDGETS['standard']", five(std))
    p.mark("③覆盖", "（无覆盖）", five(std))
    p.tamper = lambda kw: {**kw, "n_warmup": 16}
    try:
        cand = BS.fit_voigt(f, Z, 2, kw=std)
    finally:
        p.tamper = None
    claim = (cand.get("raw") or {}).get("sampling") or {}
    ok = p.finish(note=f"fit_one.sampling（现有产物声明记录）={five(claim)}")
    real = p.ctor_passed.get("num_warmup")
    diverged = (claim.get("n_warmup") != real) and claim.get("n_warmup") == 2000 and real == 16
    print(f"   ⇒ 声明记录 sampling.n_warmup={claim.get('n_warmup')} "
          f"≠ 真实构造实参 num_warmup={real}  "
          f"{'独立探针抓到了分叉 ✓' if diverged else '未复现分叉 ✗'}")
    print("   ⇒ 这正是第 1 步审计的结论：`sampling` 记的是配置声明，不是实际执行条件")
    return bool(ok and diverged)


def projection_check(values: dict) -> bool:
    """**辅助核验，不是传递链证据**：拿探针自己采集的真实构造实参喂给既有 v1.4 投影器。

    这里 import 的是 `exec_identity`（既有身份层），不是第 2 步产出层。
    """
    import exec_identity as E

    block = {k: {"value": values[k], "source": "derived"} for k in KEYS}
    proj = E._budget_payload(block)
    flags = E._chk_budget(block)
    ok = proj == block and len(flags) == 5 and all("budget_not_recorded" in s for s in flags)
    print(f"\n{'=' * 96}\n辅助核验：v1.4 预算身份投影（用探针采集的真实构造实参）\n{'=' * 96}")
    print(f"   E._budget_payload(...) 逐位相等 = {proj == block}")
    print(f"   E._chk_budget(...) 全 5 项 budget_not_recorded = {len(flags) == 5}")
    print("   ⇒ 这**只**证明预算身份投影兼容。")
    print("     不证明上游来源已闭合；不证明完整 run_id / workflow_id 已可可靠生成；")
    print("     本探针不构造任何号，五项 source 只能是 derived。")
    return bool(ok)


def main() -> int:
    _assert_independent()
    _assert_keys_aligned()
    before = set(sys.modules)
    f, Z = _data()
    p = Probe()
    results = []
    try:
        results.append(("C1 默认值/单链", case_C1(p, f, Z)))
        results.append(("C2 显式值全链", case_C2(p, f, Z)))
        c2_real = dict(p.last_real)      # 探针采集的真实构造实参，不是查表常量
        results.append(("C3 quick 覆盖", case_C3(p, f, Z)))
        results.append(("C4 C 路径 standard", case_C4(p, f, Z)))
        results.append(("C5 挑键转发", case_C5(p, f, Z)))
        results.append(("C6 环境覆盖", case_C6(p, f, Z)))
        results.append(("C7 缺键三态", case_C7(p, f, Z)))
        results.append(("C8 人为篡改", case_C8(p, f, Z)))
        results.append(("v1.4 投影辅助核验", projection_check(c2_real)))
    finally:
        p.close()

    # 护栏 2：整个取证过程没有（直接或间接）加载产出层。
    loaded = set(sys.modules) - before
    tainted = [m for m in loaded if m == "budget_params" or m.startswith("budget_params.")]
    assert not tainted, f"取证过程加载了产出层：{tainted}"

    print(f"\n{'=' * 96}\n汇总\n{'=' * 96}")
    for name, ok in results:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    print(f"   独立性：AST 无产出层引用、五键静态对齐；sys.modules 新增 "
          f"{len(loaded)} 个模块，产出层加载数 = {len(tainted)}")
    print("   取证来源：生产常量 / 生产函数 / wrapper 入参 / 类级 hook 构造实参 / 对象属性")
    print("   未做：不改生产代码、不写产物、不构造 run_id/workflow_id、不动三处旧挂账、不启动 runner")
    ok = all(v for _, v in results)
    print(f"\n{'全部字段对得上' if ok else '有字段对不上，见上方 ✗ 标记'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
