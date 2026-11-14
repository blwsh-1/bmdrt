# -*- coding: utf-8 -*-
"""exec_identity.py —— 执行身份契约 `v1.2` 的**纯函数**实现（不跑拟合、不写产物）。

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

import hashlib
import json
import pathlib
import re
from typing import Any, Iterable, Mapping

import numpy as np

# ---------------------------------------------------------------------------
# 版本与闭集
# ---------------------------------------------------------------------------

#: 进 `run_id` 哈希的契约世代标签（D-4）。语义一变就换号，载荷不变也换。
IDENTITY_CONTRACT_VERSION = "1.2"

#: `RUN_ID_CONFIG_KEYS` 的闭集版本。加键/改 `effect` 必须 +1。
RUN_ID_CONFIG_KEYS_VERSION = 1

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

#: 前缀只作辅助标识（D-4），权威是 `identity_contract_version` + 状态。
RUN_ID_PREFIX = {"recorded": "R1-", "derived": "D1-"}

WORKFLOW_PREFIX = "W1-"

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
    "sigma": ("composite", "execution_condition"),
    "sigma_floor": ("float", "execution_condition"),
    "sigma_mode": ("enum[free,scalar,perpoint]", "execution_condition"),
    "sigma_pct": ("float", "execution_condition"),
    "sigma_abs": ("float|null", "execution_condition"),
    "zmax_override": ("float|null", "execution_condition"),
    # --- 门禁：只拦不改数 ---
    "kk_mode": ("enum", "admission_condition"),
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


def _norm_float(v: Any, *, key: str) -> float:
    f = float(v)
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
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool):
        return value
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
    """`config.effective` 里**进 `run_id`** 的键（按码位排序）。"""
    return sorted(k for k in effective if config_effect(k, role=role) in RUN_ID_EFFECTS)


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


def sigma_run_id_value(block: Mapping) -> dict:
    """把 `sigma` 复合块投影成进哈希的那两个子字段。"""
    if not isinstance(block, Mapping):
        raise ValueError("sigma 必须是复合块（见 sigma_block），收到 "
                         f"{type(block).__name__}")
    arr = block.get("array_sha256")
    if arr is not None:
        arr = canon_sha256(arr, what="sigma.array_sha256")
    if block.get("kind") != "none" and arr is None:
        raise ValueError("sigma.kind≠none 却没有 array_sha256：实际噪声数组才是身份")
    return {"array_sha256": arr,
            "floor": _norm_float(block.get("floor") or 0.0, key="sigma.floor")}


def sigma_block(*, kind: str, mode: str | None = None, sigma: Any = None,
                pct: float | None = None, abs_: float | None = None,
                floor: float = 0.0, from_data: bool = False) -> dict:
    """构造 `config.effective["sigma"]` 的复合值（§3.3.5）。

    `kind` 是**实际进似然的形态**（`_check_sigma` 的三分支），`mode` 是**来源口径**
    （`make_sigma`）。两者不同不必换号，但都要记：前者决定 `array_sha256` 有没有值，
    后者供人读。
    """
    if kind not in SIGMA_KINDS:
        raise ValueError(f"sigma.kind 必须是 {SIGMA_KINDS} 之一，收到 {kind!r}")
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
    v = (rec.get("identity_contract_version") or "")
    if v != IDENTITY_CONTRACT_VERSION:
        raise ValueError(
            f"identity_contract_version 必须是 {IDENTITY_CONTRACT_VERSION!r}，收到 {v!r}"
            "：v0 记录走 circuit_registry 的旧规则，不许套 v1 载荷")
    if not rec.get("candidate_id"):
        raise ValueError("candidate_id 缺失：run_id 的模型定义层没有锚点")
    return v


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
        "retry_kw": _plain(wf.get("retry_kw") or {}),
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


def _chk_budget(bud: Mapping) -> list[str]:
    r = []
    for k in BUDGET_KEYS:
        cell = bud.get(k)
        if not isinstance(cell, Mapping):
            r.append(f"budget_missing:{k}")
        elif cell.get("source") != "recorded":
            r.append(f"budget_not_recorded:{k}")
    return r


def _chk_config(cfg: Mapping, rec: Mapping, *, role: str) -> list[str]:
    r = []
    eff = cfg.get("effective") or {}
    if cfg.get("unknown_keys", []) != []:
        r.append("config_incomplete:unknown_keys")
    if not cfg.get("capture_interface"):
        r.append("config_incomplete:capture_interface")
    unclassified = []
    for k, cell in eff.items():
        declared = cell.get("effect") if isinstance(cell, Mapping) else None
        expect = config_effect(k, role=role)
        if expect is None or declared is None:
            unclassified.append(k)
        elif declared != expect:
            r.append(f"config_effect_mismatch:{k}")
    if unclassified:
        r.append("config_unclassified:" + ",".join(sorted(unclassified)))
    mismatch = rec.get("candidate_config_mismatch") or []
    if mismatch:
        r.append("candidate_config_mismatch:" + ",".join(sorted(mismatch)))
    return r


def _chk_code(code: Mapping) -> list[str]:
    r = []
    try:
        canon_sha1_commit(code.get("git_commit"))
    except ValueError:
        r.append("code_commit_missing")
    items = code.get("declared_source_set") or []
    if not items:
        r.append("declared_set_missing")
    else:
        try:
            check_declared_set(items)
        except ValueError as exc:
            r.append(f"declared_set_invalid:{exc}")
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
    if code.get("runtime_snapshot_taken") is not True:
        r.append("runtime_snapshot_missing")
    pts = code.get("runtime_snapshot_points") or []
    if not {"before_fit", "after_fit"} <= set(pts):
        r.append("runtime_snapshot_points_incomplete")
    coverage = code.get("source_coverage")
    if coverage != "complete":
        r.append({None: "snapshot_missing"}.get(coverage, str(coverage)))
    if code.get("undeclared_modules"):
        r.append("undeclared_loaded")
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
        declared_paths = {it.get("path") for it in items}
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
    """
    dat = rec.get("data") or {}
    rng = rec.get("rng") or {}
    bud = rec.get("budget_params") or {}
    cfg = rec.get("config") or {}
    code = rec.get("code") or {}
    env = rec.get("env") or {}

    reasons: list[str] = []
    reasons += _chk_data(dat)
    reasons += _chk_rng(rng, bud)
    reasons += _chk_budget(bud)
    fatal = bool(reasons)

    reasons += _chk_config(cfg, rec, role=role)
    reasons += _chk_code(code)
    reasons += _chk_env(env)

    status = "unconfirmed" if fatal else ("derived" if reasons else "recorded")
    reasons = list(dict.fromkeys(reasons))          # 去重但保留首次出现的顺序
    out = {"status": status, "reasons": reasons,
           "run_id": None, "workflow_id": None}
    if status != "unconfirmed":
        out["run_id"] = RUN_ID_PREFIX[status] + run_id_digest(rec, role=role)[:32]
        out["workflow_id"] = WORKFLOW_PREFIX + workflow_id_digest(rec)[:32]
    return out
