# -*- coding: utf-8 -*-
"""采样构造点的**中立钩子**：数值内核到这里为止，不再向下 import 审计层。

依赖方向
--------
旧：`bm_core` / `bm_top` → `sampler_capture` → (`budget_capture_adapter`,
`sampler_provenance`)。数值内核因此拖着整条 L5 审计链，打包/独立 import 时
只要审计层缺失就失败（`CODE_MAP.md` §4-⑤，软件化第一硬阻塞）。

新：`bm_core` / `bm_top` → **本模块**（零项目依赖、默认 no-op）；捕获实现由
调用方显式 `install()`。默认未安装 ⇒ 本次拟合不产事件，与"不接 runner"等价，
拟合数值与样本逐位不变。需要审计的调用方（harness / 复现脚本 / 测试）显式
`sampling_hook.install(sampler_capture.capture_at_construction)` 即恢复原能力。

本模块**不 import 任何项目模块**，也不出现 `sampler_provenance` 字样：
`test_sampler_provenance.X1`（生产零引用）与 `test_sampler_capture.G6`
（生产只连中立钩子）都据此守卫。它不判定、不构造 run_id、不写盘。

接口
----
install(fn)                 安装捕获实现（可重复；返回旧实现）
uninstall()                 卸载，回到 no-op（返回旧实现）
installed()                 是否已安装
emit(mcmc, *, site, passed, run_call)  未安装返回 None；已安装原样转交
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

#: 当前捕获实现。None = no-op（默认）。
_IMPL: Callable[..., Any] | None = None


def install(fn: Callable[..., Any]) -> Callable[..., Any] | None:
    """安装捕获实现，返回被替换的旧实现（便于调用方成对恢复）。"""
    global _IMPL
    if not callable(fn):
        raise TypeError(f"捕获实现必须可调用，收到 {type(fn).__name__}")
    old, _IMPL = _IMPL, fn
    return old


def uninstall() -> Callable[..., Any] | None:
    """卸载捕获实现，回到 no-op；返回被卸载的实现（可能是 None）。"""
    global _IMPL
    old, _IMPL = _IMPL, None
    return old


def installed() -> bool:
    return _IMPL is not None


def emit(mcmc, *, site: str, passed: Mapping | None = None,
         run_call: Mapping | None = None) -> dict | None:
    """调用已安装的捕获实现；未安装时返回 None（不抛异常、不伪造事件）。

    **不做异常吞掉**：捕获层自己的编程错误（未知站点、非法字段）照原样抛出，
    否则会把"钩子装错了"伪装成"没有事件"。采不到的运行期异常由捕获层自己
    记 `capture_error`，与本层无关。
    """
    fn = _IMPL
    if fn is None:
        return None
    return fn(mcmc, site=site, passed=passed, run_call=run_call)
