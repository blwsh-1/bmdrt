# -*- coding: utf-8 -*-
"""exec_identity.py —— 执行身份契约 `v1.4` 的**纯函数**实现（不跑拟合、不写产物）。

本模块只做三件事，全部无副作用：

1. **规范化与摘要**：`canonical_json` / 三层源码摘要（declared / numeric / workflow）/
   `sigma` 实际数组摘要；
2. **算号**：由一条 v1 记录算 `run_id`（`R1-`/`D1-`）与 `workflow_id`（`W1-`）；
3. **判状态**：由记录判 `recorded`/`derived`/`unconfirmed` 与原因码。

它**不**做的事（属 runner，尚未授权）：捕获调用点实参、读 `sys.modules`、落补丁包、
写 `source_manifest.json`、跑拟合。所以这里没有任何 `open(..., "w")`、没有时间戳、
没有 `os.environ` 读取——同一份输入永远得到同一份输出，这才能被反例矩阵逐条钉住。

设计要点（与本文件同名的规则在 `docs/EXECUTION_IDENTITY_CONTRACT.md`）：

- **三层摘要不是一个划分**。同一个文件可以既进 numeric 层又进 workflow 层：
  入口脚本（`bm_real`/`nested_fit`/`bm_select`）既决定"算什么"，也含门禁与重跑逻辑。
  把三层当互斥集合，就会出现"改了 `bm_select` 的重跑逻辑，workflow 身份却没变"。
  故：`compute_role` 管 numeric 层，`workflow_scope`（默认 `True`）管 workflow 层，
  两个维度独立。
- **默认一律保守**。`compute_role` 省略 = `numeric`；`workflow_scope` 省略 = `True`。
  降级必须给 `why_not_numeric`（另加 `evidence`）或 `why_not_workflow`，否则拒收。
  理由：误降会**合并**两次本应分开的运行，且不报错；误升只会多换号，看得见。
- **`_UNSET` 一律拒收**（F12）。本模块不解析哨兵（那需要数据与档案），只保证哨兵
  到不了哈希里：传进来就抛 `SentinelNotResolved`，让"没解析"变成一次失败而不是一个值。
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import pathlib
import re
from typing import Any, Iterable, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# 版本与闭集
# ---------------------------------------------------------------------------

#: 进 `run_id` 哈希的契约世代标签（D-4）。语义一变就换号，载荷不变也换。
#: `1.3`（P0-1/P0-2 + 两个 P1）：σ 的来源子字段退出哈希、覆盖关系改为集合复算、
#: 配置类型进判定、`floor` 显式三分。载荷语义变了 ⇒ 必须换号——哪怕一条记录都还没有。
#: `1.4`（P0-1 收尾）：`sigma_floor` 也退出哈希（它与 `sigma.floor` 是同一个量，
#: 原来由两条通道分别进哈希）；持久化记录的 `mode`↔`kind` 自洽检查从构造函数移进
#: `sigma_run_id_value()`。键集少了 `sigma_floor` ⇒ 载荷变 ⇒ 换号。
#: `1.5`（身份链 P0 修复，第 4 步复裁 §6）：
#:   C1/G1 `source=recorded` 从字符串自检升为**结构化证据门槛**；
#:   C2/G5 `run_id_config_keys_version` 进记录 / 两个载荷 / `_require_version()`；
#:   C3/G3 `retry_kw` 退出 `workflow_id`，且在算号前强制拦截五项预算键；
#:   C3/G4 每次重跑由 `rerun.attempt` 生成独立子 `run_id`；
#:   C5/G2 捕获事件的 `origin` 字面量禁止升级为 `recorded`。
#: 载荷键集与语义都变了 ⇒ **所有 v1.4 号整体失效**（全换号是预期代价，
#: 不许用「兼容旧号」绕过）。
IDENTITY_CONTRACT_VERSION = "1.5"

#: `RUN_ID_CONFIG_KEYS` 的闭集版本。加键/改 `effect` 必须 +1，
#: 且**必须同时**升 `IDENTITY_CONTRACT_VERSION`。
#: G5：v1.4 里它既不进哈希也不被校验 ⇒ 单升它是空操作；v1.5 起
#: 进两个载荷、被 `_require_version()` 校验，并由记录显式携带。
RUN_ID_CONFIG_KEYS_VERSION = 2
RUN_ID_CONFIG_KEYS_VERSION_FIELD = "run_id_config_keys_version"

#: `effect` 六值闭集（D-6）。前两个进 `run_id`，后四个不进。
EFFECTS = (
    "model_definition",
    "execution_condition",
    "admission_condition",
    "decision_condition",
    "workflow_scope",
    "display_only",
)

#: 进 `run_id` 的 `effect` 子集。
RUN_ID_EFFECTS = ("model_definition", "execution_condition")

#: 五项采样预算（D-2）。五项数值 + 每项来源都进 `run_id`，档名不进。
BUDGET_KEYS = ("n_warmup", "n_samples", "num_chains", "target_accept",
               "max_tree_depth")

#: `rng.plan` 五项（§3.5）。v1.0 曾写"四项"，现统一五项。
RNG_PLAN_KEYS = ("root_seed", "prng", "derivation", "n_chains", "init_strategy")

#: 三档身份状态。`unconfirmed` 不发 `run_id`。
IDENTITY_STATES = ("recorded", "derived", "unconfirmed")

#: 三个快照时点（§3.1.2）。`recorded` 要求三个齐。
#: 只有两个时，必须在 `code.runtime_snapshot_limitation` 里显式写"缺哪个、为什么、
#: 降级处置"，**且照样降 `derived`**——两个时点的成功采集不能被说成三个都完成了。
SNAPSHOT_POINTS = ("before_fit", "sampler_entry", "after_fit")

#: 前缀只作辅助标识（D-4），权威是 `identity_contract_version` + 状态 + 世代。
#: v1.4 → v1.5 是**新世代**：`R1-/D1-` 留给 v1.4 历史号，不复用、不兼容。
RUN_ID_PREFIX = {"recorded": "R2-", "derived": "D2-"}

WORKFLOW_PREFIX = "W2-"

#: v1.5 权威世代表。`budget_params.IDENTITY_GENERATIONS` 是第 2 步的 v1.4 投影，
#: 其文件字节被 `8015721` 审查脚本按 blob 锁定，本轮**不改**；新世代判定以本表为准。
#: 三套号（registry `R-/D-` 10hex、v1.4 `R1-/D1-`、v1.5 `R2-/D2-`）互不跨世代比号。
IDENTITY_GENERATIONS = {
    "R-": "registry_0.7.0", "D-": "registry_0.7.0",
    "R1-": "exec_identity_1.4", "D1-": "exec_identity_1.4",
    "R2-": "exec_identity_1.5", "D2-": "exec_identity_1.5",
}

#: C1 结构化证据门槛：身份层本地复刻 `budget_params` 的闭集词表。
#: 本模块**零 import 产出层**（既有 AST 守护），逐项相等由
#: `test_budget_capture_adapter.py` / `test_exec_identity.py` 交叉核验。
BUDGET_CELL_KEYS = ("value", "source", "origin", "requested",
                    "transformed", "where")
BUDGET_SOURCE_STATES = ("recorded", "derived", "unavailable")
BUDGET_ORIGINS = (
    "cli_explicit", "module_constant", "programmatic_override",
    "function_default", "library_default", "runtime_derived", "unknown")
BUDGET_EVIDENCE_FIELD = "budget_evidence"
BUDGET_EVIDENCE_KEYS = ("upstream_origin_captured", "capture_version",
                        "capture_site", "evidence_sha256", "event_sha256")

#: G4 重跑身份：每次实际执行是一个独立子运行，用 `attempt` 区分。
RERUN_FIELD = "rerun"
RERUN_KEYS = ("attempt", "reason", "parent_workflow_ref")

#: Gate 2A F-03：五项预算的值域。整数预算必须是 ≥1 的整数值，
#: `target_accept` 必须是开区间 (0,1) 内的实数。`None` 只许配 `unavailable`。
BUDGET_INT_KEYS = ("n_warmup", "n_samples", "num_chains", "max_tree_depth")

#: Gate 2A F-05：捕获证据的版本与站点闭集。逐项与
#: `sampler_capture.CAPTURE_VERSION` / `CALL_SITES` 对齐，由测试交叉核验；
#: 身份层不 import 捕获层（既有 AST 守护）。
BUDGET_CAPTURE_VERSIONS = ("1.0",)
BUDGET_CAPTURE_SITES = ("bm_core.fit", "bm_top.fit_top")

#: Gate 2A F-06：预算数值**唯一**允许的落点是 `budget_params`。
#: 这些 workflow 整块字段走 `_plain` 进 `workflow_id`，若把五项键塞进去，
#: 就等于绕开 G3 二次承载预算 ⇒ 算号前必须拒。
WORKFLOW_PLAIN_FIELDS = ("search", "extras", "guards", "gate", "retry")

# ---------------------------------------------------------------------------
# Gate 2B / F-01：可信来源三分（内容完整性 / 来源真实性 / 执行关联性）
# ---------------------------------------------------------------------------
# 重要边界（不得夸大）：`evaluate()` 吃的是普通 dict，**纯函数层无法密码学地证明**
# 「证据来自真实 MCMC 构造过程」。这里实现的是工程上的「可信边界签发 + 执行实例
# 强绑定 + 落盘可审计」：证据必须由受控捕获接口签发、签名可复算、绑定本次执行的
# 数据身份与候选；跨运行复制 / 错关联会被拒。它不是抗抵赖的密码学证明——
# 能读源码并主动复刻签发逻辑的内部调用者仍可构造，这一点由审计日志与落盘约束覆盖。

ATTESTATION_FIELD = "capture_attestation"
ATTESTATION_VERSION = "1"
ATTESTATION_ISSUER = "capture_boundary/v1"
ATTESTATION_KEYS = ("version", "issued_by", "event_sha256",
                    "budget_sha256", "instance_token", "signature")

EXECUTION_INSTANCE_FIELD = "execution_instance"
EXECUTION_INSTANCE_KEYS = ("data_sha256", "candidate_id", "site", "nonce")

#: 冻结的唯一执行路径；`bm_top.fit_top` 仅作第二站点兼容验证对象。
#: nested 独立接口 / K3 `m35_cond_n2.py:116` / `bm_select.py` retry / 其余入口
#: **不纳入、不外推**。
FROZEN_PRIMARY_SITE = "bm_core.fit"
FROZEN_COMPAT_SITES = ("bm_top.fit_top",)
FROZEN_EXECUTION_SITES = (FROZEN_PRIMARY_SITE,) + FROZEN_COMPAT_SITES

#: 项目内受控签发密钥。写在这里意味着验证是**纯函数、可跨进程复现**（落盘后
#: 重读仍可验证），也意味着它不是私钥：签发能力属于受控捕获接口，不属于记录
#: 本身。明账为「内部可信边界」，不宣称抗抵赖。
_ATTESTATION_KEY = b"bmdrt/execution-attestation/v1/project-controlled-boundary"

#: `run_id` 认的是"进似然/进模型的东西"；键名原样，不改写不加前缀。
#: 值 = (类型标签, `effect`)。类型标签只用于校验，不参与哈希。
RUN_ID_CONFIG_KEYS: dict[str, tuple[str, str]] = {
    # --- 模型定义（改完模型空间变了） ---
    "branch": ("enum[parallel,series]", "model_definition"),
    "layers": ("int", "model_definition"),
    "K": ("int", "model_definition"),
    "include_L": ("bool", "model_definition"),
    "adsorption": ("enum[none,RL]|null", "model_definition"),
    "diffusion": ("enum[none,Ws,Wo,WsWo]", "model_definition"),
    "zeta": ("enum[free]|float", "model_definition"),
    # --- 执行条件（改完这次运行不再可复现） ---
    "sigma": ("composite", "execution_condition"),        # 噪声条件**唯一**载体
    # ↓ 四个都是 `sigma` 复合块的副本/来源：`effect` 归类是"执行条件"，
    #   但**不进哈希**（`config_keys_in_run_id` 会跳过，见 SIGMA_MEMBER_KEYS）。
    #   "属于哪一类条件"与"由谁承载"是两个问题，别混。
    #   `sigma_floor` ≡ `sigma.floor`（`bm_top._check_sigma` 的同一个参数），
    #   让它也进哈希就是同一个噪声地板有两个身份入口。
    "sigma_floor": ("float", "execution_condition"),
    "sigma_mode": ("enum[free,scalar,perpoint]", "execution_condition"),
    "sigma_pct": ("float", "execution_condition"),
    "sigma_abs": ("float|null", "execution_condition"),
    "zmax_override": ("float|null", "execution_condition"),
    # --- 门禁：只拦不改数 ---
    # 闭集来源：`bm_real.py:2663`「gate | warn | off」；不再用开放 `enum`，
    # 否则 `kk_mode="bypass"` 这种笔误能一路走到门禁里当"不拦"用。
    "kk_mode": ("enum[gate,warn,off]", "admission_condition"),
    "kk_opts": ("dict", "admission_condition"),
    # --- 事后诊断 / 选择阈值 ---
    "diag": ("dict", "decision_condition"),
    "drt_opts": ("dict", "decision_condition"),
    "gate_b": ("bool", "decision_condition"),
    "strict": ("bool", "decision_condition"),
    # --- 工作流范围：描述"跑几次"，不描述"哪一次" ---
    "krange": ("str|null", "workflow_scope"),
    "wide": ("bool", "workflow_scope"),
    "kmax": ("int", "workflow_scope"),
    "K_force": ("int|null", "workflow_scope"),
    "branches": ("list[str]", "workflow_scope"),
    "pair_mode": ("enum[pooled,per-branch]", "workflow_scope"),
    "n_seeds": ("int", "workflow_scope"),
    "retry": ("int", "workflow_scope"),
    "auto_mechanism": ("bool", "workflow_scope"),
    "auto_mechanism_full": ("bool", "workflow_scope"),
    "auto_adsorption": ("bool", "workflow_scope"),
    "with_voigt": ("bool", "workflow_scope"),
    "voigt_K": ("int|null", "workflow_scope"),   # 父=scope，子=model_definition
    "ref_rwp": ("bool", "workflow_scope"),
    # --- 展示 ---
    "progress": ("bool", "display_only"),
    "verbose": ("bool", "display_only"),
    "plot": ("bool", "display_only"),
    "dry": ("bool", "display_only"),
    "trace": ("bool", "display_only"),
}

#: `voigt_K` 是唯一一个"同一个键在不同 run 里 `effect` 不同"的键：
#: 父工作流开对照时它只决定"多跑一次"，子运行里它就是模型定义。
_EFFECT_BY_ROLE = {("voigt_K", "child"): "model_definition"}

#: `sigma` 复合块的**冗余副本 / 来源子字段**，一律不进哈希。
#:
#: · `sigma_mode`/`sigma_pct`/`sigma_abs` 是 `make_sigma(Z, mode, pct, abs)` 的三个
#:   入参（`nested_fit.py:321`），复合块里的 `mode`/`pct`/`abs` 是它们的副本；
#: · `sigma_floor` 是 `bm_top._check_sigma` 的噪声地板参数，**就是** `sigma.floor`
#:   （契约 §3.3.5 原话）。它漏在这里是 P0-1 的第二个入口：块里的 `floor` 已经
#:   进去哈希了，顶层再进一次，同一个地板就有了两个身份。
#:
#: 噪声条件由复合块**一次**承担（`array_sha256` + `floor`）。四个键只作记录：
#: 若再让它们各自进哈希，两次运行的实际 σ 数组逐位相同、仅"来源口径"不同就会换号，
#: 与 D-8「实际进入似然的数组才是噪声条件」直接冲突（T19a）。
#: 这是 P0-1：**复合键与独立键不能重复主张同一个条件**。
#:
#: 退出哈希**不等于**不校验：副本与本体必须一致（`_chk_config_sigma`），
#: 不一致的记录降级——不能让一条自相矛盾的噪声条件静默拿到号。
SIGMA_MEMBER_KEYS = ("sigma_mode", "sigma_pct", "sigma_abs", "sigma_floor")

#: `source_coverage` 的闭集（§3.1.2 四种结果）。
COVERAGE_STATES = ("complete", "undeclared_loaded", "external_path", "snapshot_missing")


class SentinelNotResolved(ValueError):
    """`_UNSET` 之类的哨兵逃到了哈希层（F12 的显式失败）。"""


class _Unset:
    """`bm_select._UNSET` 的替身：只用于识别"没解析就开始算身份"。"""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:                      # pragma: no cover
        return "_UNSET"

    def __bool__(self) -> bool:
        raise SentinelNotResolved(
            "`_UNSET` 被当布尔用：哨兵必须先解析成 VOIGT_CAPS 的值再算身份（F12）")


UNSET = _Unset()


# ---------------------------------------------------------------------------
# 基本校验
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canon_sha256(v: Any, *, what: str = "sha256") -> str:
    """64 位十六进制，大小写与首尾空白归一。**无效就抛，不退化**。

    退化成 `None` 会把调用方的 bug 藏成"证据不足"——空串在 `source.sha256` 里是
    "缺失"，在 `execution.data.sha256` 里是"错误"，两者的区别只能靠抛错守住。
    """
    if not isinstance(v, str):
        raise ValueError(f"{what} 必须是字符串，收到 {type(v).__name__}")
    s = v.strip().lower()
    if not _HEX64.match(s):
        raise ValueError(f"{what} 必须是 64 位十六进制（收到 {v!r}）")
    return s


def valid_sha256(v: Any) -> bool:
    """`canon_sha256` 的布尔版。与 `circuit_registry._valid_sha256` 语义一致。"""
    try:
        canon_sha256(v)
    except ValueError:
        return False
    return True


def canon_sha1_commit(v: Any) -> str:
    """`git_commit` 必须是 40 位。短哈希不许冒充代码身份（§3.1）。"""
    if not isinstance(v, str):
        raise ValueError(f"git_commit 必须是字符串，收到 {type(v).__name__}")
    s = v.strip().lower()
    if not _HEX40.match(s):
        raise ValueError(f"git_commit 必须是 40 位十六进制（收到 {v!r}）")
    return s


# ---------------------------------------------------------------------------
# 规范化（§3.3.6）
# ---------------------------------------------------------------------------

def _plain(obj: Any, *, path: str = "") -> Any:
    """把 numpy 标量/元组/映射转成可哈希的基本类型；数组与未知对象一律拒收。"""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not np.isfinite(f):
            raise ValueError(f"{path or '<value>'} 是 {f!r}：NaN/Inf 不能进身份哈希")
        return f
    if isinstance(obj, np.ndarray):
        raise ValueError(f"{path or '<value>'} 是 ndarray：数组必须走摘要，不许整块进 JSON")
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"{path or '<value>'} 的键必须是字符串，收到 {k!r}")
            out[k] = _plain(v, path=f"{path}.{k}" if path else k)
        return out
    if isinstance(obj, (list, tuple)):
        return [_plain(v, path=f"{path}[{i}]") for i, v in enumerate(obj)]
    raise ValueError(f"{path or '<value>'} 的类型 {type(obj).__name__} 不支持进身份哈希")


def canonical_json(obj: Any) -> str:
    """按键名码位排序、无空格、`allow_nan=False` 的规范 JSON。

    浮点用 Python 的 `repr`（最短往返、17 位有效数字），**不四舍五入**——
    `json.dumps` 对 float 走的就是 `float.__repr__`，所以只要不让 numpy 标量
    进来、不设 `float_format`，就不会出现"改了第 12 位却没换号"。
    """
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def budget_evidence_sha256(block: Mapping) -> str:
    """六键预算块的证据指纹（C1 / G1）。

    与 `budget_capture_adapter.budget_evidence_sha256` 同构：先规整成
    `BUDGET_KEYS × BUDGET_CELL_KEYS` 固定顺序再算摘要。身份层拿重算值比对
    记录里 `budget_evidence.evidence_sha256` ⇒ **验证完再篡改块**会被抓到。
    注意：这只是**结构与自洽**校验，不是「真的捕获过」的密码学证明；
    后者只能由生产捕获接口 + 显式 `upstream_origin_captured` 给出。
    """
    if not isinstance(block, Mapping):
        raise ValueError("budget_params 必须是映射")
    absent = [k for k in BUDGET_KEYS if k not in block]
    extra = [k for k in block if k not in BUDGET_KEYS]
    if absent or extra:
        raise ValueError(f"budget_params 键必须恰好五项：缺 {absent}、多 {extra}")
    norm = {}
    for k in BUDGET_KEYS:
        cell = block[k]
        if not isinstance(cell, Mapping):
            raise ValueError(f"budget_params[{k!r}] 必须是六键复合块")
        missing = [c for c in BUDGET_CELL_KEYS if c not in cell]
        over = [c for c in cell if c not in BUDGET_CELL_KEYS]
        if missing or over:
            raise ValueError(
                f"budget_params[{k!r}] 键必须恰好是 {list(BUDGET_CELL_KEYS)}；"
                f"缺 {missing}、多 {over}")
        norm[k] = {c: _plain(cell[c], path=f"budget_params.{k}.{c}")
                   for c in BUDGET_CELL_KEYS}
    return hashlib.sha256(canonical_json(norm).encode("utf-8")).hexdigest()


def id_generation(run_id: Any) -> str | None:
    """号的契约世代（v1.5 权威）。长前缀先匹配，避免 `R1-` 被 `R-` 吃掉。"""
    if not isinstance(run_id, str):
        return None
    for prefix in ("R2-", "D2-", "R1-", "D1-", "R-", "D-"):
        if run_id.startswith(prefix):
            return IDENTITY_GENERATIONS[prefix]
    return None


def can_compare_ids(a: Any, b: Any) -> dict:
    """跨世代不比号。同世代也**不声明等价**，只说可按各世代自己的规则比对。"""
    ga, gb = id_generation(a), id_generation(b)
    comparable = ga is not None and gb is not None and ga == gb
    return {
        "comparable": comparable,
        "generation_a": ga,
        "generation_b": gb,
        "reason": (
            "同世代，可按各世代自己的规则比对" if comparable else
            "不同契约世代（registry 0.7.0 的 R-/D- 10hex、v1.4 的 "
            "R1-/D1-、v1.5 的 R2-/D2-）：禁止直接拿前缀或摘要比对来判断历史记录漂移"),
        "declares_equivalence": False,
    }


def _norm_float(v: Any, *, key: str) -> float:
    if v is None:
        raise ValueError(f"配置 {key!r} 是 None：未知不能当成 0.0 或 1.0")
    try:
        f = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置 {key!r} 不是数值：{v!r}（{exc}）") from None
    if not np.isfinite(f):
        raise ValueError(f"配置 {key!r} 是 {f!r}：NaN/Inf 不能进身份哈希")
    return f


def norm_effective(key: str, value: Any) -> Any:
    """把一个**已解析**的生效配置值规范成可哈希形态（§3.3.6）。

    只做"同一个值只有一种写法"这件事，不做语义解析：哨兵到这里就该已经没了。
    """
    if isinstance(value, _Unset):
        raise SentinelNotResolved(
            f"配置 {key!r} 仍是 `_UNSET`：先用 VOIGT_CAPS/档位解析成实际值再算身份")
    # 这里**刻意没有**"bool 原样返回"的通用短路：`True` 是 `int` 的子类，
    # 一旦在键分支之前就返回，`layers=True` 会绕过"必须是整数（bool 不算）"，
    # 然后被当成第 1 层用。bool 只在 bool 类型的分支里被接受。
    if key == "zeta":
        # `zeta="1.0"` 与 `zeta=1.0` 是同一个固定值（T21）；`"free"` 保持字符串。
        if isinstance(value, str) and value.strip().lower() != "free":
            return _norm_float(value, key=key)
        return value
    if key in ("sigma_pct", "sigma_abs", "sigma_floor", "zmax_override"):
        return None if value is None else _norm_float(value, key=key)
    if key in ("layers", "K", "kmax", "K_force", "voigt_K", "n_seeds", "retry"):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"配置 {key!r} 必须是整数（bool 不算），收到 {value!r}")
        return int(value)
    if key in ("include_L", "wide", "auto_mechanism", "auto_mechanism_full",
               "auto_adsorption", "with_voigt", "ref_rwp", "progress", "verbose",
               "plot", "dry", "trace", "gate_b", "strict"):
        if value is None:
            raise SentinelNotResolved(
                f"配置 {key!r} 是 None：`None` 必须先解析成 bool（detect_L / VOIGT_CAPS）")
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"配置 {key!r} 必须是 bool，收到 {value!r}")
        return bool(value)
    if key == "sigma":
        if not isinstance(value, Mapping):
            raise ValueError("配置 'sigma' 必须是复合块（见 sigma_block）")
        return _plain(dict(value))
    if key in ("kk_opts", "diag", "drt_opts"):
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"配置 {key!r} 必须是 dict 或 None，收到 {value!r}")
        return _plain(dict(value))
    if key == "branches":
        if isinstance(value, str) or not isinstance(value, Iterable):
            raise ValueError(f"配置 {key!r} 必须是序列，收到 {value!r}")
        return [str(v) for v in value]
    return _plain(value)


def config_effect(key: str, *, role: str = "parent") -> str | None:
    """键的 `effect`；未登记返回 `None`（由调用方转成 `unclassified_keys`）。"""
    spec = RUN_ID_CONFIG_KEYS.get(key)
    if spec is None:
        return None
    return _EFFECT_BY_ROLE.get((key, role), spec[1])


def config_keys_in_run_id(effective: Mapping, *, role: str = "parent") -> list[str]:
    """`config.effective` 里**进 `run_id`** 的键（按码位排序）。

    两道门：`effect ∈ RUN_ID_EFFECTS`，**且**不是 `sigma` 复合块的冗余副本
    （`SIGMA_MEMBER_KEYS`，四个）。后一道是 P0-1 的修正：`sigma_mode`/`sigma_pct`/
    `sigma_abs`/`sigma_floor` 的语义已被复合键承载（`array_sha256` + `floor`），
    再让它们各自进哈希就等于允许"来源写法"冒充"噪声条件"：`abs=None` 物化成
    `0.01·max|Z|` 与显式 `abs=0.03` 写出同一个数组、`sigma_abs` 与块里 `abs` 却不同，
    这时换号就是 D-8 禁止的（T19a）。`sigma_floor` 同理——它就是块里的 `floor`，
    同一个地板不能有第二个身份入口（T19d）。

    它们仍留在 `RUN_ID_CONFIG_KEYS` 里（`effect=execution_condition` 不变）：
    这张表管的是"这个键属于哪一类条件"，进不进哈希是"由谁承载"的第二个问题。
    记录里保留它们，供人读"这个数组 / 这个地板是怎么来的"。
    """
    return sorted(k for k in effective
                  if k not in SIGMA_MEMBER_KEYS
                  and config_effect(k, role=role) in RUN_ID_EFFECTS)


# ---------------------------------------------------------------------------
# 源文件字节与内容摘要
# ---------------------------------------------------------------------------

#: 行尾归一的文件类型（§3.1.1）。其余文件按**原始字节**哈希：二进制资产里
#: `\r\n` 可能就是数据的一部分，替它做主会得到一份没有对应文件的摘要。
TEXT_SUFFIXES = (".py", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini")


def normalize_source_bytes(data: bytes, path: str) -> bytes:
    """文本源码把 CRLF 归一成 LF，其余原样。"""
    if path.lower().endswith(TEXT_SUFFIXES):
        return data.replace(b"\r\n", b"\n")
    return data


def source_file_sha256(path: Any) -> str:
    """一份声明文件的内容摘要（归一后）。runner 与清单预览必须共用这一处。"""
    p = path if isinstance(path, pathlib.Path) else pathlib.Path(str(path))
    return hashlib.sha256(normalize_source_bytes(p.read_bytes(), p.name)).hexdigest()


# ---------------------------------------------------------------------------
# sigma：复合键，认实际数组（D-8）
# ---------------------------------------------------------------------------

SIGMA_KINDS = ("none", "scalar_pair", "perpoint")


def sigma_array_sha256(arr: Any) -> str:
    """实际 σ 数组的摘要：`<f8` 小端、C 顺序、**含形状**、拒 NaN/Inf/空数组。

    形状必须进载荷：`(2, 3)` 与 `(3, 2)` 在 C 顺序下字节完全不同，但这是巧合而不是
    保证——显式写形状，才不会靠字节长度反推。`kind`/`mode` 不进这里（T19）：
    数组逐位相同 ⇒ 似然相同 ⇒ 同一组条件。
    """
    a = np.asarray(arr)
    if a.dtype == object or a.size == 0:
        raise ValueError(f"σ 数组为空或 dtype 非法（形状 {a.shape}）：空数组不等于 kind=none")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"σ 数组 dtype 必须是数值，收到 {a.dtype}")
    a = np.ascontiguousarray(a, dtype="<f8")
    if not np.all(np.isfinite(a)):
        raise ValueError("σ 数组含 NaN/Inf：先修数据，别让非法噪声进身份")
    shape = "x".join(str(n) for n in a.shape)
    head = f"v1\n{shape}\n".encode("utf-8")
    return hashlib.sha256(head + a.tobytes(order="C")).hexdigest()


#: `sigma` 进 `run_id` 的**投影**：只有实际数组摘要与噪声下界。
#: `kind`/`mode`/`pct`/`abs`/`from_data`/`elpd_valid` 全部留记录不进哈希——
#: 它们是"这个数组是怎么来的"，而似然只认数组本身（D-8 / T19）。
SIGMA_RUN_ID_FIELDS = ("array_sha256", "floor")

#: `make_sigma` 的模式与"实际进似然的形态"是**一一对应**的（`nested_fit.py:113`）：
#: `free` ⇒ 无固定数组；`scalar` ⇒ `(2,)` 广播；`perpoint` ⇒ `(2, n)`。
#: 所以 `kind` 与 `mode` 不能各写各的——形状由模式决定，配错就是造一个不存在的记录。
KIND_BY_MODE = {"free": "none", "scalar": "scalar_pair", "perpoint": "perpoint"}


def sigma_run_id_value(block: Mapping) -> dict:
    """把 `sigma` 复合块投影成进哈希的那两个子字段。

    `floor` 显式区分**缺失 / `None` / `0.0`**：`0.0` 是"无下界"这个真实取值，
    `None` 是"没记录"，缺字段是"没写"。原来的 `block.get("floor") or 0.0`
    把后两种**静默当成 `0.0`**——那等于替记录编一个值，而且编得看不出来。

    **`mode`↔`kind` 的自洽在这里校，不只在 `sigma_block()` 里校**：构造函数只能
    约束"我们自己造出来的块"；处理历史 / 外部记录时走的是这条路径，谁也没拦着
    有人手写一个 `{kind: "scalar_pair", mode: "free"}` 的块——它描述的是一个
    物理上不存在的形态（`free` 意味着没有固定数组），却带着 `array_sha256` 拿号。
    所以身份入口必须自己校一遍（T19e）。

    `sigma` 与 `sigma_floor` 在 `bm_top._check_sigma` 里是**互斥**的：固定 σ 时
    floor 无意义、会直接报错。因此 `kind≠none` 且 `floor≠0` 是一条不可能发生的
    记录，只可能来自写错的 runner——同样在这里拒掉，别让它带号。
    """
    if not isinstance(block, Mapping):
        raise ValueError("sigma 必须是复合块（见 sigma_block），收到 "
                         f"{type(block).__name__}")
    if "floor" not in block:
        raise ValueError("sigma.floor 缺失：噪声下界必须显式记录（无下界写 0.0）")
    if block.get("floor") is None:
        raise ValueError("sigma.floor 是 None：无下界请写 0.0，None 表示没记录")
    floor = _norm_float(block.get("floor"), key="sigma.floor")
    if floor < 0:
        raise ValueError("sigma.floor 必须非负（与 bm_top._check_sigma 同口径），"
                         f"收到 {floor!r}")
    kind = block.get("kind")
    if kind not in SIGMA_KINDS:
        raise ValueError(f"sigma.kind 必须是 {SIGMA_KINDS} 之一，收到 {kind!r}")
    mode = block.get("mode")
    if mode is None:
        raise ValueError("sigma.mode 缺失：来源口径必须记（契约 §3.3.5 标为必填），"
                         "缺了就无法校验它和 kind 是否自洽")
    if mode not in KIND_BY_MODE:
        raise ValueError(f"sigma.mode 必须是 {tuple(KIND_BY_MODE)} 之一，收到 {mode!r}")
    if KIND_BY_MODE[mode] != kind:
        raise ValueError(
            f"sigma.kind={kind!r} 与 mode={mode!r} 不自洽："
            f"mode={mode!r} 只能配 {KIND_BY_MODE[mode]!r}（形状由模式决定）")
    arr = block.get("array_sha256")
    if arr is not None:
        arr = canon_sha256(arr, what="sigma.array_sha256")
    if kind != "none" and arr is None:
        raise ValueError("sigma.kind≠none 却没有 array_sha256：实际噪声数组才是身份")
    if kind == "none" and arr is not None:
        raise ValueError("sigma.kind='none' 却带 array_sha256：自由 σ 没有固定数组")
    if kind != "none" and floor != 0.0:
        raise ValueError(
            f"sigma.kind={kind!r} 却带 floor={floor!r}：`_check_sigma` 里固定 σ 与 "
            "`sigma_floor` 互斥，这条记录不可能来自一次真实运行")
    return {"array_sha256": arr, "floor": floor}


def sigma_block(*, kind: str, mode: str, sigma: Any = None,
                pct: float | None = None, abs_: float | None = None,
                floor: float = 0.0, from_data: bool = False) -> dict:
    """构造 `config.effective["sigma"]` 的复合值（§3.3.5）。

    `kind` 是**实际进似然的形态**（`_check_sigma` 的三分支），`mode` 是**来源口径**
    （`make_sigma`）。两者必须自洽（`KIND_BY_MODE`），但 `pct`/`abs` 与块里的
    投影值**不必相等**：`abs=None` 会被物化成 `0.01·max|Z|`，块里记的是物化后的数，
    而 `sigma_abs` 记的是入参 `None`。**只有 `mode` 两边必须一致**——它决定数组形状。

    `mode` **没有默认值**：§3.3.5 把它标为必填，而它又是 `kind` 的唯一校验依据。
    给个默认值等于允许"没记来源"的块进入流通，到 `sigma_run_id_value()` 才炸——
    那时候已经分不清是"没记"还是"记错了"。
    """
    if kind not in SIGMA_KINDS:
        raise ValueError(f"sigma.kind 必须是 {SIGMA_KINDS} 之一，收到 {kind!r}")
    if mode is None:
        raise ValueError("sigma.mode 不能是 None：它决定数组形状，必须显式记录")
    if mode not in KIND_BY_MODE:
        raise ValueError(f"sigma.mode 必须是 {tuple(KIND_BY_MODE)} 之一，"
                         f"收到 {mode!r}")
    if KIND_BY_MODE[mode] != kind:
        raise ValueError(
            f"sigma.kind={kind!r} 与 mode={mode!r} 不自洽："
            f"mode={mode!r} 只能配 {KIND_BY_MODE[mode]!r}（形状由模式决定）")
    if floor is None:
        raise ValueError("sigma.floor 不能是 None：无下界写 0.0，None 表示没记录")
    if kind != "none" and floor != 0.0:
        raise ValueError(f"kind={kind!r} 与 floor={floor!r} 互斥：固定 σ 时噪声地板"
                         "无意义（bm_top._check_sigma 会直接报错）")
    if kind == "none":
        if sigma is not None:
            raise ValueError("kind='none' 表示噪声由后验采样，sigma 必须是 None")
        arr_sha = None
    else:
        if sigma is None:
            raise ValueError(f"kind={kind!r} 必须有实际 σ 数组；None 只属于 kind='none'")
        a = np.asarray(sigma, dtype=float)
        if kind == "scalar_pair" and a.shape != (2,):
            raise ValueError(f"kind='scalar_pair' 要求形状 (2,)，收到 {a.shape}")
        if kind == "perpoint":
            if a.ndim != 2 or a.shape[0] != 2:
                raise ValueError(f"kind='perpoint' 要求形状 (2, n)，收到 {a.shape}")
            if (a <= 0).any():
                raise ValueError("逐点 σ 必须严格为正（Normal.scale 不接受 0）")
        if kind == "scalar_pair" and (a <= 0).any():
            raise ValueError("标量 σ 必须严格为正")
        arr_sha = sigma_array_sha256(a)
    return {
        "kind": kind,
        "mode": mode,
        "pct": None if pct is None else _norm_float(pct, key="sigma.pct"),
        "abs": None if abs_ is None else _norm_float(abs_, key="sigma.abs"),
        "array_sha256": arr_sha,
        "floor": _norm_float(floor, key="sigma.floor"),
        "from_data": bool(from_data),
        "elpd_valid": bool(from_data),
    }


# ---------------------------------------------------------------------------
# 三层源码摘要（D-5）
# ---------------------------------------------------------------------------

SOURCE_LAYERS = ("declared", "numeric", "workflow")


def source_tree_sha256(items: Iterable[Mapping], *, layer: str) -> str:
    """对声明集的某一层求摘要。层名进域分隔前缀，三层永不撞值。

    `items` 的每项至少要有 `path` 与 `sha256`；`compute_role` 决定它进不进 numeric 层，
    `workflow_scope` 决定它进不进 workflow 层。两层**可以重叠**（入口脚本必须重叠）。
    """
    if layer not in SOURCE_LAYERS:
        raise ValueError(f"layer 必须是 {SOURCE_LAYERS} 之一，收到 {layer!r}")
    body: dict[str, str] = {}
    seen: set[str] = set()
    for it in items:
        path = it.get("path")
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
            raise ValueError(f"声明项 path 必须是 POSIX 相对路径，收到 {path!r}")
        sha = canon_sha256(it.get("sha256"), what=f"source_sha256[{path}]")
        if path in seen:
            raise ValueError(f"声明集有重复 path：{path}")   # 重复会让分母不可复算
        seen.add(path)
        if layer == "numeric" and it.get("compute_role", "numeric") != "numeric":
            continue
        if layer == "workflow" and it.get("workflow_scope", True) is False:
            continue
        body[path] = sha
    if not body:
        raise ValueError(f"{layer} 层为空：一层没有任何文件就不是身份，是遗漏")
    text = f"v1\n{layer}\n" + "".join(f"{p}\0{body[p]}\n" for p in sorted(body))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def declared_source_tree_sha256(items: Iterable[Mapping]) -> str:
    return source_tree_sha256(items, layer="declared")


def numeric_source_tree_sha256(items: Iterable[Mapping]) -> str:
    return source_tree_sha256(items, layer="numeric")


def workflow_source_tree_sha256(items: Iterable[Mapping]) -> str:
    return source_tree_sha256(items, layer="workflow")


def check_declared_set(items: Iterable[Mapping]) -> None:
    """声明集的静态合法性：降级必须有理由 + 证据；入口必须在 workflow 层。

    这条检查的作用是**堵住"为了让哈希小一点先降一档"**：没有 `why_not_numeric`
    与 `evidence`，非 numeric 一律拒收；入口脚本不许退出 workflow 层（门禁与重跑
    逻辑就在入口里）。
    """
    entries = 0
    for it in items:
        path = it.get("path")
        role = it.get("role", "module")
        cr = it.get("compute_role", "numeric")
        if role not in ("entry", "module", "config", "asset"):
            raise ValueError(f"{path}: role 非法 {role!r}")
        if cr not in ("numeric", "decision", "plot_only", "cli_only",
                      "reference_only", "budget_source"):
            raise ValueError(f"{path}: compute_role 非法 {cr!r}")
        if cr != "numeric":
            if not it.get("why_not_numeric"):
                raise ValueError(f"{path}: 降为 {cr!r} 必须写 why_not_numeric")
            if not it.get("evidence"):
                raise ValueError(f"{path}: 降为 {cr!r} 必须写 evidence（哪个反例/实测支撑）")
        if it.get("workflow_scope", True) is False and not it.get("why_not_workflow"):
            raise ValueError(f"{path}: workflow_scope=False 必须写 why_not_workflow")
        if role == "entry":
            entries += 1
            if it.get("workflow_scope", True) is False:
                raise ValueError(f"{path}: 入口不许退出 workflow 层（门禁/重跑逻辑在入口里）")
    if not entries:
        raise ValueError("声明集里没有任何 role=entry 的项：入口缺失，分母是错的")


# ---------------------------------------------------------------------------
# run_id / workflow_id
# ---------------------------------------------------------------------------

def _require_version(rec: Mapping) -> str:
    """C2/G5：版本从「注释性常量」升为**记录字段 + 校验项**。

    `run_id_config_keys_version` 现在必须由记录显式携带、是整数、与本常量相等；
    缺字段 / 字符串 / bool / 旧世代值一律拒 → `unhashable:` → 无号。
    单升 `RUN_ID_CONFIG_KEYS_VERSION` 而不同时升 `IDENTITY_CONTRACT_VERSION`
    仍然不够：两个都要变，且都要写进载荷。
    """
    v = (rec.get("identity_contract_version") or "")
    if v != IDENTITY_CONTRACT_VERSION:
        raise ValueError(
            f"identity_contract_version 必须是 {IDENTITY_CONTRACT_VERSION!r}，收到 {v!r}"
            "：v0 / v1.4 记录走各自的旧规则，不许套 v1.5 载荷")
    kv = rec.get(RUN_ID_CONFIG_KEYS_VERSION_FIELD)
    if (isinstance(kv, bool) or not isinstance(kv, int)
            or kv != RUN_ID_CONFIG_KEYS_VERSION):
        raise ValueError(
            f"{RUN_ID_CONFIG_KEYS_VERSION_FIELD} 必须是整数 "
            f"{RUN_ID_CONFIG_KEYS_VERSION}，收到 {kv!r}："
            "v1.4 空常量不算身份输入，缺字段不许用默认值补")
    if not rec.get("candidate_id"):
        raise ValueError("candidate_id 缺失：run_id 的模型定义层没有锚点")
    return v


def _rerun_attempt(rec: Mapping) -> int:
    """G4：父运行 attempt=0；每次重跑 attempt≥1 且必须显式写整数。"""
    rr = rec.get(RERUN_FIELD)
    if rr is None:
        return 0
    if not isinstance(rr, Mapping):
        raise ValueError("rerun 必须是映射或 None")
    unknown = [k for k in rr if k not in RERUN_KEYS]
    if unknown:
        raise ValueError(f"rerun 出现未登记键 {unknown}")
    a = rr.get("attempt")
    if isinstance(a, bool) or not isinstance(a, int) or a < 0:
        raise ValueError(f"rerun.attempt 必须是非负整数，收到 {a!r}")
    return a


# ---------------------------------------------------------------------------
# Gate 2A 最小修复集：F-03 / F-04 / F-05 / F-06 / F-07 / F-08 / F-09
# ---------------------------------------------------------------------------

_ID_REF_PREFIXES = tuple(IDENTITY_GENERATIONS) + (WORKFLOW_PREFIX,)


def _is_identity_ref(v: Any) -> bool:
    """F-07：`children` / `parent_workflow_ref` 只许是身份号引用。

    形状 = 已登记世代前缀 + 非空 hex 摘要。字符串里夹预算数值（如
    `"n_samples=1000"`）不是引用，必须拒；dict 展开子任务参数更不是引用。
    """
    if not isinstance(v, str):
        return False
    for p in _ID_REF_PREFIXES:
        if v.startswith(p):
            tail = v[len(p):]
            return bool(tail) and re.fullmatch(r"[0-9a-f]+", tail) is not None
    return False


def _budget_keys_in(obj: Any, *, _depth: int = 0) -> list[str]:
    """F-06：递归找出藏在 Mapping 键里的五项预算键（值里的字符串不看）。"""
    if _depth > 8:
        return []
    found: list[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k in BUDGET_KEYS:
                found.append(str(k))
            found += _budget_keys_in(v, _depth=_depth + 1)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            found += _budget_keys_in(x, _depth=_depth + 1)
    return found


def _budget_value_ok(key: str, v: Any) -> bool:
    """F-03：按参数语义校值域。类型门禁仍在这里，不外包给产出层。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    f = float(v)
    if not np.isfinite(f):
        return False
    if key == "target_accept":
        return 0.0 < f < 1.0
    if f != int(f):                       # 整数预算不许 2.5 之类
        return False
    return int(f) >= 1


def _rerun_lineage(rec: Mapping) -> dict | None:
    """F-08/F-09：重跑的「来自哪个父工作流」与「为什么重跑」是执行实例条件。

    父记录返回 None；子记录带 `reason` + `parent_workflow_ref`。两者进
    `run_id` 载荷 ⇒ 不同 lineage 的同 attempt 子运行不再撞号，失败/成功
    重试也不再同号。它们**不是**模型定义，不进 `config`。
    """
    rr = rec.get(RERUN_FIELD)
    if rr is None:
        return None
    return {"reason": rr.get("reason"),
            "parent_workflow_ref": rr.get("parent_workflow_ref")}


def _chk_rerun(rec: Mapping) -> list[str]:
    """F-08：子运行必须显式给出父工作流引用，否则不同 lineage 会同号。

    `reason` 也强制非空：失败重试与门禁重试是两种执行条件，不许静默合并。
    """
    rr = rec.get(RERUN_FIELD)
    if rr is None:
        return []
    if not isinstance(rr, Mapping):
        return ["rerun_not_mapping"]
    try:
        attempt = _rerun_attempt(rec)
    except ValueError:
        return ["rerun_invalid"]
    if attempt < 1:
        return ["rerun_attempt_zero"]
    r: list[str] = []
    reason = rr.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        r.append("rerun_reason_missing")
    if not _is_identity_ref(rr.get("parent_workflow_ref")):
        r.append("rerun_parent_ref_invalid")
    return r


def _chk_workflow(wf: Mapping) -> list[str]:
    """F-06/F-07：workflow 整块字段不得二次承载预算；children 只放引用。"""
    r: list[str] = []
    for f in WORKFLOW_PLAIN_FIELDS:
        if _budget_keys_in(wf.get(f)):
            r.append(f"workflow_budget_keys:{f}")
    pm = wf.get("pair_mode")
    if isinstance(pm, str) and any(k in pm for k in BUDGET_KEYS):
        r.append("workflow_budget_keys:pair_mode")
    ch = wf.get("children")
    if ch is None:
        ch = []
    if not isinstance(ch, (list, tuple)):
        r.append("workflow_children_not_list")
    elif any(not _is_identity_ref(c) for c in ch):
        r.append("workflow_children_not_reference")
    return r


# ---------------------------------------------------------------------------
# F-01：可信捕获凭证（受控签发 / 实例绑定 / 签名复算）
# ---------------------------------------------------------------------------

def _attestation_body(a: Mapping) -> str:
    """签名覆盖的五个字段；`signature` 自身不进消息。"""
    return canonical_json({k: a.get(k) for k in ATTESTATION_KEYS
                           if k != "signature"})


def attestation_signature(a: Mapping) -> str:
    """HMAC-SHA256。纯函数、无 I/O、无时间戳、无环境 ⇒ 落盘后重读可复算。"""
    return hmac.new(_ATTESTATION_KEY, _attestation_body(a).encode("utf-8"),
                    hashlib.sha256).hexdigest()


def execution_instance_token(inst: Mapping) -> str:
    """执行实例指纹：数据身份 + 候选 + 站点 + 本次 nonce。"""
    return hashlib.sha256(
        canonical_json({k: inst.get(k) for k in EXECUTION_INSTANCE_KEYS}
                       ).encode("utf-8")).hexdigest()


def issue_capture_attestation(*, event_sha256: str, budget_sha256: str,
                              instance: Mapping) -> dict:
    """受控签发接口：只能由 harness 在拿到**真实捕获事件**后调用。

    `event_sha256` / `budget_sha256` 必须取自事件的 `budget_evidence`，不许
    调用者另造；`instance` 绑定本次执行的数据身份、候选、站点与唯一 nonce。
    """
    for name, val in (("event_sha256", event_sha256),
                      ("budget_sha256", budget_sha256)):
        if not isinstance(val, str) or re.fullmatch(r"[0-9a-f]{64}", val) is None:
            raise ValueError(f"{name} 必须是 64 位小写 hex")
    if not isinstance(instance, Mapping):
        raise ValueError("instance 必须是映射")
    missing = [k for k in EXECUTION_INSTANCE_KEYS if k not in instance]
    if missing:
        raise ValueError(f"execution_instance 缺字段 {missing}")
    if instance.get("site") not in FROZEN_EXECUTION_SITES:
        raise ValueError(f"site 不在冻结路径内：{instance.get('site')!r}")
    a = {"version": ATTESTATION_VERSION, "issued_by": ATTESTATION_ISSUER,
         "event_sha256": event_sha256, "budget_sha256": budget_sha256,
         "instance_token": execution_instance_token(instance),
         "signature": ""}
    a["signature"] = attestation_signature(a)
    return a


def _chk_attestation(rec: Mapping) -> list[str]:
    """F-01 三分门槛，缺一即 `unconfirmed` 无号。

    · 内容完整性：`event_sha256` / `budget_sha256` 与 `budget_evidence` 逐位对齐；
    · 来源真实性：签名由受控接口用项目密钥重算比对（自洽字典签不出来）；
    · 执行关联性：`instance_token` 与 `execution_instance` 自洽，且实例的
      `data_sha256` / `candidate_id` / `site` 与**本条记录**一致 ⇒ 把 A 次执行
      的凭证交给 B 次执行会被拒。
    """
    a = rec.get(ATTESTATION_FIELD)
    if not isinstance(a, Mapping):
        return ["attestation_missing"]
    r: list[str] = []
    missing = [k for k in ATTESTATION_KEYS if k not in a]
    over = [k for k in a if k not in ATTESTATION_KEYS]
    if missing:
        r.append("attestation_missing_fields:" + ",".join(missing))
    if over:
        r.append("attestation_unknown_fields:"
                 + ",".join(sorted(str(k) for k in over)))
    if a.get("version") != ATTESTATION_VERSION:
        r.append("attestation_version_invalid")
    if a.get("issued_by") != ATTESTATION_ISSUER:
        r.append("attestation_issuer_invalid")

    ev = rec.get(BUDGET_EVIDENCE_FIELD) or {}
    if a.get("event_sha256") != ev.get("event_sha256"):
        r.append("attestation_event_sha_mismatch")
    if a.get("budget_sha256") != ev.get("evidence_sha256"):
        r.append("attestation_budget_sha_mismatch")

    inst = rec.get(EXECUTION_INSTANCE_FIELD)
    if not isinstance(inst, Mapping):
        r.append("execution_instance_missing")
    else:
        im = [k for k in EXECUTION_INSTANCE_KEYS if k not in inst]
        io = [k for k in inst if k not in EXECUTION_INSTANCE_KEYS]
        if im or io:
            r.append("execution_instance_shape")
        if inst.get("data_sha256") != (rec.get("data") or {}).get("sha256"):
            r.append("attestation_data_binding_mismatch")
        if inst.get("candidate_id") != rec.get("candidate_id"):
            r.append("attestation_candidate_binding_mismatch")
        if inst.get("site") not in FROZEN_EXECUTION_SITES:
            r.append("attestation_site_not_frozen")
        n = inst.get("nonce")
        if not isinstance(n, str) or not n.strip():
            r.append("execution_instance_nonce_invalid")
        if not im and not io and \
                a.get("instance_token") != execution_instance_token(inst):
            r.append("attestation_instance_token_mismatch")

    sig = a.get("signature")
    if not isinstance(sig, str) or not hmac.compare_digest(
            sig, attestation_signature(a)):
        r.append("attestation_signature_invalid")
    return r


def _budget_payload(bud: Mapping) -> dict:
    out = {}
    for k in BUDGET_KEYS:
        cell = bud.get(k)
        if not isinstance(cell, Mapping):
            raise ValueError(f"budget_params[{k!r}] 必须是 {{value, source}}")
        out[k] = {"value": _plain(cell.get("value")),
                  "source": cell.get("source")}
    return out


def _data_payload(dat: Mapping) -> dict:
    return {
        "sha256": canon_sha256(dat.get("sha256"), what="data.sha256"),
        "n_points": int(dat.get("n_points")),
        "subset_sha256": (None if dat.get("subset_sha256") is None
                          else canon_sha256(dat["subset_sha256"], what="data.subset_sha256")),
        "f_min": None if dat.get("f_min") is None else _norm_float(dat["f_min"], key="f_min"),
        "f_max": None if dat.get("f_max") is None else _norm_float(dat["f_max"], key="f_max"),
        "transform": _plain(dat.get("transform") or {"fscale": 1.0, "zscale": 1.0,
                                                     "mask": None}),
    }


def run_id_payload(rec: Mapping, *, role: str = "parent") -> dict:
    """`run_id` 的哈希载荷（§6 纳入口径的唯一实现处）。

    刻意**不含**：`git_describe`/`git_dirty`/`patch_bundle_*`（展示与可重建）、
    `runtime_loaded_modules`/`source_coverage`（覆盖性核对）、
    `declared_source_tree_sha256`/`compute_role`（完整性证据）、
    `workflow.*`（描述跑几次）、`rng.actual.*`/`instance.*`（实例）、
    `provenance.registry_version`（映射工具）。
    """
    ver = _require_version(rec)
    code = rec.get("code") or {}
    env = rec.get("env") or {}
    cfg = (rec.get("config") or {}).get("effective") or {}
    payload_cfg = {}
    for k in config_keys_in_run_id(cfg, role=role):
        cell = cfg[k]
        raw = cell.get("value") if isinstance(cell, Mapping) else cell
        payload_cfg[k] = (sigma_run_id_value(raw) if k == "sigma"
                          else norm_effective(k, raw))
    return {
        "v": ver,
        # C2/G5：键表版本进载荷（改键表 / 升版本 ⇒ 全换号，不再是空操作）。
        "config_keys_version": rec[RUN_ID_CONFIG_KEYS_VERSION_FIELD],
        # G4：第几次重跑是**条件**，不是结果；同一预算重跑两次也必须不同号。
        "rerun_attempt": _rerun_attempt(rec),
        # F-08/F-09：lineage 与重跑原因也是执行实例条件。缺了它们，
        # 两个不同父工作流的同 attempt 子运行会撞号、失败/成功重跑会同号。
        "rerun_lineage": _rerun_lineage(rec),
        "candidate_id": rec["candidate_id"],
        "data": _data_payload(rec.get("data") or {}),
        "rng_plan": {k: _plain((rec.get("rng") or {}).get("plan", {}).get(k))
                     for k in RNG_PLAN_KEYS},
        "budget": _budget_payload(rec.get("budget_params") or {}),
        "config": payload_cfg,
        "code": {
            "git_commit": canon_sha1_commit(code.get("git_commit")),
            "numeric_source_tree_sha256": canon_sha256(
                code.get("numeric_source_tree_sha256"), what="numeric_source_tree_sha256"),
        },
        "env": {
            "env_sha256": canon_sha256(env.get("env_sha256"), what="env_sha256"),
            "backend": env.get("backend"),
            "x64_enabled": bool(env.get("x64_enabled")),
        },
    }


def run_id_digest(rec: Mapping, *, role: str = "parent") -> str:
    """不带前缀的摘要（32 字节 hex）。状态决定前缀，不决定摘要。"""
    return hashlib.sha256(
        ("v1\nrun_id\n" + canonical_json(run_id_payload(rec, role=role))).encode("utf-8")
    ).hexdigest()


def workflow_payload(rec: Mapping) -> dict:
    """`workflow_id` 的载荷。

    只装**条件**，不装**结果**：`kk_status`/`kk_performed` 是观测，装进来就把
    工作流身份变成结果身份。门禁真的起作用时，它改变的是 `children` ——
    结果通过"跑没跑那个子运行"进来，而不是通过一个字符串。
    """
    wf = rec.get("workflow") or {}
    gate = wf.get("gate") or {}
    return {
        "v": _require_version(rec),
        "config_keys_version": rec[RUN_ID_CONFIG_KEYS_VERSION_FIELD],
        "workflow_source_tree_sha256": canon_sha256(
            (rec.get("code") or {}).get("workflow_source_tree_sha256"),
            what="workflow_source_tree_sha256"),
        "search": _plain(wf.get("search") or {}),
        "extras": _plain(wf.get("extras") or {}),
        "gate": {"kk_mode": gate.get("kk_mode"),
                 "kk_opts": _plain(gate.get("kk_opts") or {}),
                 "strict": gate.get("strict")},
        "guards": _plain(wf.get("guards") or {}),
        "pair_mode": wf.get("pair_mode"),
        "retry": wf.get("retry"),
        # C3/G3：`retry_kw` 整块退出 workflow_id。生产曾把 BUDGETS 五数值整包
        # 塞进来（`bm_select.py`），造成「改预算换父号」的二次承载。重跑策略
        # 由 `retry` / `children` 表达；每次实际预算由该次子 run_id 承担。
        # `evaluate()` 会在算号前强制拦截含五键的 retry_kw（错误记录不发号）。
        "children": sorted(wf.get("children") or []),
    }


def workflow_id_digest(rec: Mapping) -> str:
    return hashlib.sha256(
        ("v1\nworkflow_id\n" + canonical_json(workflow_payload(rec))).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# 状态判定（§5）
# ---------------------------------------------------------------------------

def _chk_data(dat: Mapping) -> list[str]:
    r = []
    if dat.get("id_kind") != "content_sha256":
        r.append("data_not_recorded")
    try:
        canon_sha256(dat.get("sha256"), what="data.sha256")
    except ValueError:
        r.append("data_sha256_invalid")
    n = dat.get("n_points")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        r.append("data_n_points_invalid")
    if dat.get("subset_sha256") is not None:
        try:
            canon_sha256(dat["subset_sha256"], what="data.subset_sha256")
        except ValueError:
            r.append("data_subset_sha256_invalid")
    return r


def _chk_rng(rng: Mapping, bud: Mapping) -> list[str]:
    plan = (rng.get("plan") or {})
    missing = [k for k in RNG_PLAN_KEYS if plan.get(k) in (None, "")]
    if missing:
        return [f"rng_plan_missing:{k}" for k in missing]
    if not isinstance(plan.get("root_seed"), int) or isinstance(plan.get("root_seed"), bool):
        return ["rng_root_seed_not_int"]
    ev = rng.get("seed_evidence") or {}
    if not str(ev.get("source", "")).startswith("artifact"):
        return ["seed_not_recorded"]
    n_chains = (bud.get("num_chains") or {}).get("value")
    if plan.get("n_chains") != n_chains:
        return ["rng_budget_mismatch"]
    return []


def _chk_budget_evidence(bud: Mapping, evidence: Any) -> list[str]:
    """C1：证据块必须**显式落盘**并与预算块自洽，不许靠 runner 自觉。

    五键恰好、`upstream_origin_captured is True`（bool）、捕获版本/站点非空、
    `evidence_sha256` 用预算块重算比对、`event_sha256` 格式合法；若记录带了
    `budget_observations`，连事件指纹也一并重算。
    """
    if evidence is None:
        return ["budget_evidence_missing"]
    if not isinstance(evidence, Mapping):
        return ["budget_evidence_invalid"]
    r: list[str] = []
    missing = [k for k in BUDGET_EVIDENCE_KEYS if k not in evidence]
    over = [k for k in evidence if k not in BUDGET_EVIDENCE_KEYS]
    if missing:
        r.append("budget_evidence_missing_fields:" + ",".join(missing))
    if over:
        r.append("budget_evidence_unknown_fields:"
                 + ",".join(sorted(str(k) for k in over)))
    flag = evidence.get("upstream_origin_captured")
    if flag is not True:                                  # 不许 truthy 蒙混
        r.append("budget_evidence_upstream_not_captured")
    # F-05：版本/站点必须落在闭集里，不是「任意非空字符串」。
    if evidence.get("capture_version") not in BUDGET_CAPTURE_VERSIONS:
        r.append("budget_evidence_capture_version_invalid")
    if evidence.get("capture_site") not in BUDGET_CAPTURE_SITES:
        r.append("budget_evidence_capture_site_invalid")
    try:
        want = budget_evidence_sha256(bud)
    except (ValueError, TypeError, KeyError):
        r.append("budget_evidence_sha_uncomputable")
        return r
    if evidence.get("evidence_sha256") != want:
        r.append("budget_evidence_sha_mismatch")
    # F-04：必须是 64 位小写 hex。只校长度会让 'z'*64 / 中文 / 空格蒙混过关。
    es = evidence.get("event_sha256")
    if not isinstance(es, str) or re.fullmatch(r"[0-9a-f]{64}", es) is None:
        r.append("budget_evidence_event_sha_invalid")
    any_recorded = any(isinstance(c, Mapping) and c.get("source") == "recorded"
                       for c in bud.values())
    if any_recorded and flag is not True:
        r.append("budget_recorded_without_upstream")
    return r


def _chk_budget(bud: Mapping, evidence: Any = None) -> list[str]:
    """C1/G1：`recorded` 的**结构化证据门槛**（第 4 步复裁 §6 C1）。

    v1.4 只校 `cell.get("source") == "recorded"` 这个字符串，裸
    `{value, source:"recorded"}` 就能拿 `R1-` 号；同一块喂
    `budget_params.validate_budget_params()` 却被拒 ⇒ 两层 `recorded` 两个门槛。
    v1.5 起身份层自己校验：五项恰好 / 每项恰好六键 / `source` 三值闭集 /
    `origin` 七值闭集 / `value` 类型与 finite / `where` 与 `transformed` 形状 /
    `recorded` 还要求 `origin != unknown` 且 `where` 非空；外加
    `_chk_budget_evidence()` 的显式证据块与指纹重算。

    **保留** `budget_not_recorded:{k}` 原因码与状态阶梯语义：非 `recorded`
    ⇒ `unconfirmed` 无号，不是降 `derived` 发号。
    """
    r: list[str] = []
    if not isinstance(bud, Mapping):
        return ["budget_not_mapping"]
    over_top = [k for k in bud if k not in BUDGET_KEYS]
    if over_top:
        r.append("budget_unknown_keys:"
                 + ",".join(sorted(str(k) for k in over_top)))
    n_recorded = 0
    for k in BUDGET_KEYS:
        cell = bud.get(k)
        if not isinstance(cell, Mapping):
            r.append(f"budget_missing:{k}")
            continue
        src = cell.get("source")
        # 状态阶梯的自然短路：`data`/`rng`/`budget` 任一不过就已经是
        # `unconfirmed` 无号。对**必然发不出号**的项，只精确点名一项原因，
        # 不再堆形状/证据细节——形状与证据门槛是为「可能发号」的 recorded
        # 项设的。这既保留 v1.4 的原因码语义，也不给错误记录任何可乘之机。
        if src != "recorded":
            r.append(f"budget_not_recorded:{k}")
            continue
        n_recorded += 1
        missing = [c for c in BUDGET_CELL_KEYS if c not in cell]
        over = [c for c in cell if c not in BUDGET_CELL_KEYS]
        if missing or over:
            r.append(f"budget_cell_shape:{k}")
            continue
        if cell["origin"] not in BUDGET_ORIGINS:
            r.append(f"budget_origin_invalid:{k}")
        v = cell["value"]
        # F-03：类型 + 值域一起校。只校「是有限数」会让 -1 / 0 / 2.5
        # 这种不可能是采样预算的值照样拿到 R2- 号。
        if not _budget_value_ok(k, v):
            r.append(f"budget_value_invalid:{k}")
        if not isinstance(cell["transformed"], (list, tuple)):
            r.append(f"budget_transformed_invalid:{k}")
        w = cell["where"]
        if w is not None and (not isinstance(w, str) or not w.strip()):
            r.append(f"budget_where_invalid:{k}")
        if cell["origin"] == "unknown":
            r.append(f"budget_recorded_origin_unknown:{k}")
        if not w:
            r.append(f"budget_recorded_where_missing:{k}")
    # 证据门槛：有 recorded 项（可能发号）时强制；显式带了证据块时也校验，
    # 免得「五项 unavailable 却自称上游已闭合」被静默放过（G2②）。
    if n_recorded or evidence is not None:
        r += _chk_budget_evidence(bud, evidence)
    return r


def _chk_retry_kw(wf: Mapping) -> list[str]:
    """C3/G3：算号前强制拦截 `retry_kw` 的五项预算键（Q5）。

    `budget_params.check_retry_kw()` 之前**不在算号路径**上：生产把 BUDGETS
    整包传进 `workflow.retry_kw`，v1.4 又把它整块塞进 `workflow_id`。
    v1.5 两处都做：载荷里移除整块 + 这里在发号前拦截 ⇒ 错误记录不发号。
    """
    rkw = wf.get("retry_kw")
    if rkw is None:
        return []
    if not isinstance(rkw, Mapping):
        return ["retry_kw_not_mapping"]
    found = [k for k in BUDGET_KEYS if k in rkw]
    if found:
        return ["retry_kw_budget_keys:" + ",".join(found)]
    # F-06：顶层干净不代表嵌套里没藏预算键（如 {"outer": {"n_samples": 1000}}）。
    nested = _budget_keys_in(rkw)
    if nested:
        return ["retry_kw_budget_keys:" + ",".join(sorted(set(nested)))]
    return []


def _cell_value(cell: Any) -> Any:
    """`{value, origin, effect}` 里取 `value`；裸值原样返回（兼容手写记录）。"""
    return cell.get("value") if isinstance(cell, Mapping) else cell


def _type_ok(tag: str, value: Any) -> bool:
    """按 `RUN_ID_CONFIG_KEYS` 的类型标签校验一个**已归一**的值（`|` 表示"或"）。

    存在的意义是把"标签只是注释"变成"标签会被执行"。没有它，表里写着 `int`
    而记录里放 `True` 也照样 recorded——`True` 在 Python 里是 `int` 的子类，
    这类错不报错、只会在下游被当成层数用。未知标签**报错**而不是放过：
    表里写错标签比记录写错值更难发现。
    """
    for alt in str(tag).split("|"):
        alt = alt.strip()
        if alt == "null":
            if value is None:
                return True
        elif alt == "bool":
            if isinstance(value, (bool, np.bool_)):
                return True
        elif alt == "int":
            if (isinstance(value, (int, np.integer))
                    and not isinstance(value, (bool, np.bool_))):
                return True
        elif alt == "float":
            if (isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))):
                return True
        elif alt == "str":
            if isinstance(value, str):
                return True
        elif alt in ("dict", "composite"):
            if isinstance(value, Mapping):
                return True
        elif alt == "list[str]":
            if (isinstance(value, (list, tuple))
                    and all(isinstance(v, str) for v in value)):
                return True
        elif alt == "enum":
            if isinstance(value, str) and value.strip():
                return True
        elif alt.startswith("enum[") and alt.endswith("]"):
            if value in [x for x in alt[5:-1].split(",") if x]:
                return True
        else:
            raise ValueError(f"未知类型标签 {tag!r}")
    return False


def _chk_config_sigma(eff: Mapping) -> list[str]:
    """`sigma` 复合块的有效性，以及它与**四个冗余副本**的一致性。

    **为什么必须要求 `sigma` 在场**：P0-1 之后噪声条件只由复合块承载
    （`array_sha256` + `floor`）。它要是不在，`sigma_mode`/`sigma_pct`/`sigma_abs`/
    `sigma_floor` 又都不进哈希，噪声条件就**完全从身份里消失了**——所以缺块必须降级。

    **为什么校 `sigma_floor`，不校 `pct`/`abs`**：`pct`/`abs` 是**入参**，块里记的是
    **物化后**的值（`scalar` + `abs=None` ⇒ `0.01·max|Z|`），入参 `None` 与物化值不
    相等是正常的，硬校会把一次正常运行判成矛盾。`sigma_floor` 与块里的 `floor` 却是
    **同一个数**（`bm_top._check_sigma` 的参数就是它），两处不一致只可能是记录坏了。
    不一致时**降级**，不静默选一个：选哪个都是替记录编条件。
    """
    if "sigma" not in eff:
        return ["config_incomplete:sigma"]
    block = _cell_value(eff.get("sigma"))
    if not isinstance(block, Mapping):
        return ["sigma_block_invalid:not_mapping"]
    try:
        proj = sigma_run_id_value(block)
    except (ValueError, TypeError) as exc:
        return [f"sigma_block_invalid:{exc}"]

    out: list[str] = []
    mode = block.get("mode")
    got = _cell_value(eff.get("sigma_mode"))
    if mode is not None and got is not None and got != mode:
        out.append(f"sigma_mode_mismatch:{got}≠{mode}")

    if "sigma_floor" not in eff:
        # **生效值总是存在**，只是来源不同：`bm_top.fit_top` 与 `bm_real` 的入口有这个
        # 参数；`nested_fit.run` 没有（`nested_fit.py:319` 的签名里没有它），它调
        # `fit_top` 时也不传，于是走 `fit_top` 的默认 `0.0`。两种情形在记录里的区别只是
        # `origin`（`cli` vs `default`），**不是"这一项可以不存在"**。
        # 所以缺它就是 capture 漏了。不许拿全局默认值补齐——补出来的 0.0 会让
        # "没记录"看起来像"记了 0"，而这正是身份契约最不能有的那种静默。
        out.append("config_incomplete:sigma_floor")
    else:
        try:
            fl = _norm_float(_cell_value(eff.get("sigma_floor")), key="sigma_floor")
        except (ValueError, TypeError) as exc:
            out.append(f"config_value_invalid:sigma_floor({exc})")
        else:
            if fl != proj["floor"]:
                out.append(f"sigma_floor_mismatch:{fl!r}≠{proj['floor']!r}")
    return out


def _chk_config(cfg: Mapping, rec: Mapping, *, role: str) -> list[str]:
    r: list[str] = []
    eff = cfg.get("effective") or {}
    if cfg.get("unknown_keys", []) != []:
        r.append("config_incomplete:unknown_keys")
    if not cfg.get("capture_interface"):
        r.append("config_incomplete:capture_interface")
    unclassified = []
    for k in sorted(eff):
        cell = eff[k]
        declared = cell.get("effect") if isinstance(cell, Mapping) else None
        expect = config_effect(k, role=role)
        if expect is None or declared is None:
            unclassified.append(k)
            continue
        if declared != expect:
            r.append(f"config_effect_mismatch:{k}")
            continue
        try:
            got = norm_effective(k, _cell_value(cell))
        except (ValueError, TypeError) as exc:
            r.append(f"config_value_invalid:{k}({exc})")
            continue
        try:
            ok = _type_ok(RUN_ID_CONFIG_KEYS[k][0], got)
        except ValueError as exc:
            r.append(f"config_key_table_invalid:{k}({exc})")
            continue
        if not ok:
            r.append(f"config_type_mismatch:{k}")
    if unclassified:
        r.append("config_unclassified:" + ",".join(sorted(unclassified)))
    r += _chk_config_sigma(eff)
    mismatch = rec.get("candidate_config_mismatch") or []
    if mismatch:
        r.append("candidate_config_mismatch:" + ",".join(sorted(mismatch)))
    return r


def _path_set(entries: Any, what: str) -> tuple[set[str], list[str]]:
    """从 `[{"path": …}]`（或裸字符串列表）抽 POSIX 相对路径集。

    非法项进原因而不是被静默丢掉：丢一项就等于把"加载了未声明模块"变成
    "加载了少一项"，覆盖关系会凭空变成 complete。
    """
    out: set[str] = set()
    bad: list[str] = []
    if entries is None:
        return out, bad
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Iterable):
        # 写成 `entries in (None, [])` 会在 numpy 数组上触发逐元素比较而抛
        # "truth value is ambiguous"——那是个与身份无关的崩溃，不该出现在这里。
        return out, [f"{what}_not_list"]
    for it in entries:
        p = it.get("path") if isinstance(it, Mapping) else it
        if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p:
            bad.append(f"{what}_path_invalid:{p!r}")
            continue
        out.add(p)
    return out, bad


def _chk_code(code: Mapping) -> list[str]:
    r: list[str] = []
    try:
        canon_sha1_commit(code.get("git_commit"))
    except ValueError:
        r.append("code_commit_missing")

    items = code.get("declared_source_set") or []
    declared_paths: set[str] = set()
    if not items:
        r.append("declared_set_missing")
    else:
        try:
            check_declared_set(items)
        except ValueError as exc:
            r.append(f"declared_set_invalid:{exc}")
        for it in items:
            p = it.get("path") if isinstance(it, Mapping) else None
            if isinstance(p, str) and p:
                declared_paths.add(p)
        for layer, rec_key, fn in (
            ("declared", "declared_source_tree_sha256", declared_source_tree_sha256),
            ("numeric", "numeric_source_tree_sha256", numeric_source_tree_sha256),
            ("workflow", "workflow_source_tree_sha256", workflow_source_tree_sha256),
        ):
            try:
                expect = fn(items)
            except ValueError as exc:
                r.append(f"{layer}_layer_invalid:{exc}")
                continue
            got = code.get(rec_key)
            if not valid_sha256(got):
                r.append(f"{layer}_digest_missing")
            elif got.strip().lower() != expect:
                r.append(f"{layer}_digest_mismatch")

    # --- 覆盖关系：**从两个集合重算**，不信 `source_coverage`（P0-2） ---
    runtime, bad = _path_set(code.get("runtime_loaded_modules"), "runtime")
    r += bad
    if not runtime:
        r.append("runtime_set_empty")
    undeclared = sorted(runtime - declared_paths)
    not_loaded = sorted(declared_paths - runtime)
    rec_und, bad = _path_set(code.get("undeclared_modules"), "undeclared")
    r += bad
    rec_nl, bad = _path_set(code.get("declared_not_loaded"), "declared_not_loaded")
    r += bad
    if rec_und != set(undeclared):
        # 自称的未声明集合与两个集合的差集不符：这一条比覆盖标记更可信
        r.append("undeclared_claim_mismatch")
    if rec_nl != set(not_loaded):
        r.append("declared_not_loaded_mismatch")
    if undeclared:
        r.append("undeclared_loaded:" + ",".join(undeclared))

    coverage = code.get("source_coverage")
    if coverage not in COVERAGE_STATES:
        r.append("snapshot_missing" if coverage is None
                 else f"source_coverage_invalid:{coverage}")
    else:
        expect_cov = "undeclared_loaded" if undeclared else "complete"
        if coverage in ("complete", "undeclared_loaded") and coverage != expect_cov:
            r.append("source_coverage_mismatch")
        elif coverage == "external_path" and not code.get("external_modules"):
            r.append("source_coverage_mismatch")
        elif coverage == "snapshot_missing":
            r.append("snapshot_missing")

    if code.get("runtime_snapshot_taken") is not True:
        r.append("runtime_snapshot_missing")
    pts = list(code.get("runtime_snapshot_points") or [])
    unknown = sorted(set(pts) - set(SNAPSHOT_POINTS))
    if unknown:
        r.append("runtime_snapshot_point_unknown:" + ",".join(unknown))
    missing = [p for p in SNAPSHOT_POINTS if p not in pts]
    if missing:
        lim = code.get("runtime_snapshot_limitation") or {}
        declared_missing = lim.get("missing_points")
        ok_decl = (isinstance(declared_missing, (list, tuple))
                   and sorted(str(x) for x in declared_missing) == sorted(missing)
                   and str(lim.get("reason") or "").strip()
                   and str(lim.get("fallback") or "").strip())
        # 声明了限制**也照样降级**：两个时点的采集成功不能冒充三个时点。
        # 区别只在原因码——读者能看出"这是结构上挂不上"，不是"忘了采"。
        r.append("snapshot_point_declared_missing:" + ",".join(missing)
                 if ok_decl else
                 "runtime_snapshot_points_incomplete:" + ",".join(missing))

    if code.get("external_modules"):
        r.append("external_path")
    if code.get("git_dirty") is None:
        r.append("dirty_unknown")
    scope = code.get("dirty_scope")
    if scope in ("stray", "unknown", None):
        r.append("dirty_unknown" if scope is None else f"dirty_scope:{scope}")
    if code.get("git_dirty") is True:
        if not valid_sha256(code.get("patch_bundle_sha256")):
            r.append("bundle_missing_when_dirty")
        leaked = [e.get("path") for e in (code.get("bundle_excluded") or [])
                  if e.get("path") in declared_paths]
        if leaked:
            r.append("bundle_excluded_declared:" + ",".join(sorted(leaked)))
    return r


def _env_sha256(python: str, impl: str, platform: str, packages: Mapping) -> str:
    body = (f"python=={python}\nimpl=={impl}\nplatform=={platform}\n"
            + "".join(f"pkg=={k}=={packages[k]}\n" for k in sorted(packages)))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _chk_env(env: Mapping) -> list[str]:
    r = []
    try:
        expect = _env_sha256(env.get("python"), env.get("python_impl"),
                             env.get("platform"), env.get("packages") or {})
    except Exception:                                # noqa: BLE001
        return ["env_missing"]
    got = env.get("env_sha256")
    if not valid_sha256(got):
        r.append("env_missing")
    elif got.strip().lower() != expect:
        r.append("env_digest_mismatch")
    if env.get("backend") not in ("cpu", "gpu", "tpu"):
        r.append("env_missing:backend")
    if not isinstance(env.get("x64_enabled"), bool):
        r.append("env_missing:x64")
    return r


def evaluate(rec: Mapping, *, role: str = "parent") -> dict:
    """一条 v1 记录的完整判定：状态、原因、`run_id`、`workflow_id`。

    优先级（§5）：`data`/`rng`/`budget` 任一不过 ⇒ `unconfirmed`（连"跑的是什么"
    都不知道，发号就是编造）；否则 `code`/`env`/`config` 任一不过 ⇒ `derived`；
    全过 ⇒ `recorded`。

    `unconfirmed` 时 `workflow_id` 也置 `None`：一个成员没有条件身份的工作流，
    给它发工作流号就是把"不知道"包装成"知道"。宁可不发，也不发一个空壳。

    另一条通往 `unconfirmed` 的路是**结构性非法**（`unhashable:`）：`sigma` 块坏掉、
    `floor` 缺失、类型不对时，载荷根本规范化不出来。这时三档里只有 `unconfirmed`
    允许没有号——"降 `derived` 发个号"在这里做不到，够不着的东西不许假装够得着。
    """
    dat = rec.get("data") or {}
    rng = rec.get("rng") or {}
    bud = rec.get("budget_params") or {}
    cfg = rec.get("config") or {}
    code = rec.get("code") or {}
    env = rec.get("env") or {}
    wf = rec.get("workflow") or {}

    reasons: list[str] = []
    reasons += _chk_data(dat)
    reasons += _chk_rng(rng, bud)
    # C1：证据块与预算块一起进门槛；缺证据 / 篡改指纹 ⇒ 不发号。
    reasons += _chk_budget(bud, rec.get(BUDGET_EVIDENCE_FIELD))
    # G3：`retry_kw` 含五项预算 ⇒ 错误记录，归入 fatal（unconfirmed 无号）。
    reasons += _chk_retry_kw(wf)
    # F-06/F-07：workflow 整块字段二次承载预算、children 不是引用 ⇒ 无号。
    reasons += _chk_workflow(wf)
    # F-08/F-09：子运行缺父引用 / 缺重跑原因 ⇒ 无号（防撞号）。
    reasons += _chk_rerun(rec)
    # F-01：内容完整性 / 来源真实性 / 执行关联性三分。没有受控签发的
    # 捕获凭证，自洽字典一律拿不到号（Gate 2A 的 F-01 由此关闭）。
    reasons += _chk_attestation(rec)
    fatal = bool(reasons)

    reasons += _chk_config(cfg, rec, role=role)
    reasons += _chk_code(code)
    reasons += _chk_env(env)

    status = "unconfirmed" if fatal else ("derived" if reasons else "recorded")
    reasons = list(dict.fromkeys(reasons))          # 去重但保留首次出现的顺序
    out = {"status": status, "reasons": reasons,
           "run_id": None, "workflow_id": None}
    if status == "unconfirmed":
        return out
    try:
        rid = run_id_digest(rec, role=role)
        wid = workflow_id_digest(rec)
    except (ValueError, TypeError, KeyError) as exc:
        # 记录结构性坏掉（`sigma` 块非法、缺 `floor`、字段类型不对…）时，
        # **任何号都是猜的**：不是"降一档"，而是"发不出号"。三档状态里
        # 只有 `unconfirmed` 允许没有号，所以落在这里，并在原因里点名。
        return {"status": "unconfirmed",
                "reasons": reasons + [f"unhashable:{exc}"],
                "run_id": None, "workflow_id": None}
    out["run_id"] = RUN_ID_PREFIX[status] + rid[:32]
    out["workflow_id"] = WORKFLOW_PREFIX + wid[:32]
    return out


# ---------------------------------------------------------------------------
# G4：重跑身份独立
# ---------------------------------------------------------------------------

def make_rerun_record(parent: Mapping, *, attempt: int,
                      budget_params: Mapping, budget_evidence: Mapping,
                      reason: str | None = None,
                      parent_workflow_ref: str | None = None,
                      rng: Mapping | None = None,
                      execution_instance: Mapping | None = None,
                      capture_attestation: Mapping | None = None) -> dict:
    """生成一次**独立**重跑的子记录（纯函数：不写盘、不改入参、不构造号）。

    父记录里保留的是**身份条件**：`candidate_id` / `data` / `rng` / `config` /
    `code` / `env` / 两个版本字段。换掉的是本次实际执行的
    `budget_params` + `budget_evidence`；`rerun.attempt` 进 `run_id` 载荷，
    所以**同一组预算重跑两次也必然不同号**（失败与重跑各有明确身份）。

    丢掉：`instance` / `provenance` / `kk_status` / `patch_bundle_*`（结果或实例）、
    父 `workflow.retry_kw`（G3：预算数值不许进工作流身份）、`children`（子运行
    不拥有子运行）。父 `workflow.children` 只放本次子号作引用，不放预算。

    换预算时 `rng.plan.n_chains` 必须与 `budget_params.num_chains.value` 一致，
    由调用方用 `rng=` 显式给；本函数不猜、不自动折算。

    Gate 2A F-08/F-09：调用方**必须**给 `parent_workflow_ref`（合法身份号引用）
    与非空 `reason`；`_chk_rerun()` 会在算号前拦下缺失者。否则两个不同父工作流
    的同 attempt 子运行会撞号、失败与成功重跑会同号。
    """
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt 必须是 ≥1 的整数，收到 {attempt!r}")
    if not isinstance(budget_params, Mapping) or not isinstance(budget_evidence, Mapping):
        raise ValueError("budget_params / budget_evidence 必须是映射")
    keep = ("candidate_id", "data", "rng", "config", "code", "env",
            "identity_contract_version", RUN_ID_CONFIG_KEYS_VERSION_FIELD)
    out = {k: copy.deepcopy(dict(parent[k]) if isinstance(parent.get(k), Mapping)
                            else parent[k])
           for k in keep if k in parent}
    if rng is not None:
        out["rng"] = copy.deepcopy(dict(rng))
    out["budget_params"] = copy.deepcopy(dict(budget_params))
    out["budget_evidence"] = copy.deepcopy(dict(budget_evidence))
    # F-01：子运行是**新的执行实例**，绝不继承父的 instance/attestation
    # （继承就等于跨运行复用凭证）。必须由调用方为新实例重新签发。
    if execution_instance is not None:
        out[EXECUTION_INSTANCE_FIELD] = copy.deepcopy(dict(execution_instance))
    if capture_attestation is not None:
        out[ATTESTATION_FIELD] = copy.deepcopy(dict(capture_attestation))
    out[RERUN_FIELD] = {"attempt": attempt, "reason": reason,
                        "parent_workflow_ref": parent_workflow_ref}
    out["workflow"] = {"role": "child", "retry": 0, "retry_kw": {}, "children": []}
    return out
