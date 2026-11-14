# 三时点模块快照：最小实测（契约 §18.2 第 ① 项）

**日期**：2026-11-14　**基线**：`4c4f7d1`（身份契约 v1.4，68/68）
**脚本**：`docs/round_2026-11-14/snapshot_probe.py`　**原始数据**：`out/snapshot_probe/*.json`
**范围**：只验证**采集机制**。不改生产拟合代码、不启动批量拟合、不改身份契约、不动 runner。

---

## 0. 结论

| 问题 | 答案 | 依据 |
|---|---|---|
| `sampler_entry` 能挂上吗 | **能**。`numpyro.infer.MCMC.run` 类属性包装，每次采样恰好触发 1 次；`fit_top` 的调用链不变 | §3.1 |
| 三个时点采到的是不是同一个集合 | **是**。Profile B 的三个时点项目文件集合**逐位相同**（15/15/15），独立进程 3/3 一致 | §3.2 |
| 覆盖判定能不能到 `complete` | **能，但必须先解决"runner 自己算未声明"**。排除探针自身后 3/3 `complete`；不排除则 3/3 `undeclared_loaded` | §3.3 |
| 三时点设计在本路径上有信息增量吗 | **没有**。B 的项目模块导入是纯静态的，第三点与第一点同集；三点仍应采（契约要求），但别指望它能抓到"拟合中途的延迟导入"——它抓不到 | §3.2、§3.5 |
| 存在会让记录**静默**出错的坑吗 | **四个**：runner 自身未声明、路径写成绝对 Windows 路径、一次工作流多次采样、钩子状态不清零 | §4 |

一句话：**机制可用，但"能采到"离"采对"还有四个必须钉死的接口细节**。这四个都不需要改身份算法，属于 runner 的采集契约。

---

## 1. 要回答与不回答

**回答**：

1. `before_fit` / `sampler_entry` / `after_fit` 三个时点能否取到完整项目模块集合；
2. 取到的集合与 `source_manifest.json` Profile B 的覆盖关系（§3.1.2 四种状态哪一种）；
3. 同一代码路径重复运行时，快照是否稳定（进程内 vs 独立进程）。

**不回答**（留给后续项）：

- ③ `sampler_entry` 处的**参数捕获**是否可作为生效值证据。本轮顺带读到了 `num_warmup/num_samples/num_chains/_target_accept_prob/_max_tree_depth`（§3.1），但**没有**核对配置项与三入口委托表是否一致——那是 ③ 的事，本轮不据它下结论；
- ② 补丁包重建；④ runner 原型；
- Profile A / C 的同一问题（本轮只测 B）。

---

## 2. 方法

### 2.1 挂接点

`numpyro.infer.MCMC.run` 是 `fit_top` 里唯一的采样调用（`bm_top.py:1006`）。探针在import 之后、`fit_top` 调用之前把该类属性换成"记录快照 → 调原函数"，因此：

- 采样调用点仍是生产路径（`nested_fit.run` → `T.fit_top` → `mcmc.run`），探针只在前后读 `sys.modules`；
- 探针 import 了 `bm_experiments / bm_real / bm_top / bm_select / nested_fit`，与 `nested_fit.run` 的模块级导入面一致；
- 唯一偏离：不写 JSON/CSV/PNG（`nested_fit.run` 会写），其余实参同源（`make_sigma`、`budget_kw` 都从 `nested_fit` 取，避免两套口径）。

### 2.2 口径

- **只记录项目代码**：`sys.modules` 里 `__file__` 的 realpath 落在仓库根内且以 `.py` 结尾（契约 §3 表："只含项目代码；realpath 记录"）。
- 按**文件**归并（同一文件可有 `bm_top` 与 `__main__.bm_top` 两个名字），名字一并留证。
- 非 `.py` 声明条目（`config.json`，role=`config`）不参与模块覆盖比较，单独报。
- 覆盖判定按 §3.1.2 四态：`complete` / `undeclared_loaded` / `external_path` / `snapshot_missing`。

### 2.3 复现

```
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/snapshot_probe.py --fresh --repeat 3
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/snapshot_probe.py --repeat 3
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/snapshot_probe.py --two-fits --repeat 1
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-14/snapshot_probe.py --no-tiny --fresh --repeat 1
```

谱：`archive/2EMC1-1_0005mA.csv`（56 点，1–1e5 Hz）。`tiny` 档把 `quick` 的 800/400/2 覆盖为 8/8/1（冒烟用），`--no-tiny` 用原值。

---

## 3. 结果

### 3.1 挂接点可用，且生效参数一次读全

独立进程 3 次，每次 `n_calls = 1`，`resolved_attrs` 五项全部命中、`missing_attrs` 为空：

```
num_warmup=8  num_samples=8  num_chains=1  _target_accept_prob=0.9  _max_tree_depth=10
```

钩子在 `orig(...)` **之前**触发，所以这是"进入采样器时的生效值"，不是事后从参数表反推的意图值。对 ③ 是好消息（可捕获），但**能不能作为配置身份的来源**另有三个问题没查（值从哪来、默认值如何区分"未传"、多入口委托），本轮不下结论。

耗时（tiny 档）：到拟合返回 6.9 s，其后处理 0.3 s（独立进程三次同量级）。

### 3.2 三个时点采到同一个集合，跨进程逐位一致

| 运行 | `before_fit` | `sampler_entry` | `after_fit` | 三时点差集 |
|---|---|---|---|---|
| fresh #1 | 15 | 15 | 15 | 全空 |
| fresh #2 | 15 | 15 | 15 | 全空 |
| fresh #3 | 15 | 15 | 15 | 全空 |

`drift`：三时点各自 `n_distinct_sets = 1`（3 次运行只出现一种集合）。文件清单：

```
bm_ad bm_core bm_corr bm_diff bm_experiments bm_grade bm_kk bm_real
bm_select bm_top bm_top_gen bm_ws drt nested_fit        (+ 探针自身)
```

**这不是"三点都采到了"的漂亮结果，而是一个负结果**：Profile B 的项目模块全部在 `import` 阶段加载完（`bm_real` 顶层就 import 了 `bm_core/bm_experiments/bm_grade/drt`，`nested_fit` 顶层 import 了 `bm_select`），`_make_model` / `MCMC.run` / `_loglik_chained` 内部只 import 第三方（`numpyro` / `arviz`）。所以第三点与第一点同集，**"延迟导入发生在拟合中途"在 B 上没有可观测的实例**。

结论按用途分开写：

- 对**判定**：三点仍然必须采（契约要求，且 A/C 未必如此），但不能用"三点齐"证明"采集深入"；
- 对**证据**：本轮只能写"B 路径上三时点同集"，**不能**写"三时点机制能捕捉中途延迟导入"——后者需要 A 或 C 上出现真实差集才算证据。

### 3.3 覆盖判定：先 `undeclared_loaded`，排除探针后 `complete`

| 口径 | 3 次运行 | 未声明集 |
|---|---|---|
| 原始（含探针自身） | `undeclared_loaded` ×3 | `docs/round_2026-11-14/snapshot_probe.py` |
| 排除探针自身 | `complete` ×3 | 空 |

`n_union = 14`（18 个声明 `.py` 中加载了 14 个）。**"未声明"的唯一来源是探针自己**——它不是被拟合的代码，却是 `sys.modules` 里真实存在的项目模块。这不是探针的失误，而是**任何 runner 都会遇到的结构性问题**：见 §4.1。

### 3.4 声明了但没加载的 4 个文件

`bm_circuit.py`、`bm_ref_rwp.py`、`bmdrt.py`、`fit_store.py`。

- `bm_circuit` / `bm_ref_rwp`：与 F11 一致（`fit_top` 不调 `drt`/`bm_kk`/`bm_circuit`；`--ref-rwp` 默认关）；
- `bmdrt.py`：本轮绕过进程入口直接调 `fit_top`，真实 CLI 运行会加载它；
- `fit_store`：本路径不落库。

`not_loaded` 不降级（§3.1.2 第一行），所以这 4 个不影响判定；但它们说明 `declared_source_tree_sha256` 里有 4/18 的文件与本次数值无关——这正是 §3.1.4「降级双条件」要处理的账，别在这里重复记账。

另：`config.json`（role=`config`）**永远**出现在 `declared_not_loaded` 里（它不是模块，不会出现在 `sys.modules`）。实现层无害，但会让这个列表恒非空、把真信号淹掉。

### 3.5 总模块数：1255 → 1262 → 2134

三个时点的 `len(sys.modules)`：`before_fit` 1255、`sampler_entry` 1262、`after_fit` 2134。**采样期间新增 872 个模块，项目代码 0 个。**

这解释了为什么"只含项目代码"这条口径是必要的：不做过滤，三时点差集会给出 872 个 `jax._src.*` / `arviz` 模块，覆盖判定永远 `undeclared_loaded`，而且不可复现（jax 惰性导入顺序受缓存影响）。§3 表里那句"只含项目代码"不是优化，是**判定能不能成立的前提**。

### 3.6 完整 `quick` 档单次（排除"tiny 太短"的反驳）

`--no-tiny`（`n_warmup=800 / n_samples=400 / num_chains=2 / target_accept=0.9 / max_tree_depth=10`，即 `BUDGETS["quick"]` 原值）单次独立进程：

| 项 | 值 |
|---|---|
| 三时点项目文件数 | 15 / 15 / 15，`point_delta` 全空 |
| `len(sys.modules)` | 1255 → 1262 → 2137（采样期间新增 875，项目代码 0） |
| 覆盖（排除探针） | `complete`，`n_union=14`，`declared_not_loaded` 仍是那 4 个 |
| 采样质量 | `R̂_max=1.0023`、`divergences=0` |
| 耗时 | 到拟合返回 11.7 s，后处理 0.4 s |

即：**预算从 8/8/1 提到 800/400/2，项目模块集合一位没变**。"三时点同集"不是 tiny 档的假象。

（`R̂`/`div` 只用于说明这次采样是正常跑完的，不是一轮收敛证据。）

---

## 4. 四个会静默出错的接口细节

这四个都没有报错、都产出"看着合理"的记录。按危害排序。

### 4.1 runner 自身文件必然未声明 → 所有真实运行永久 `derived`

真实 runner 的入口脚本会出现在 `sys.modules` 里（`__main__`），它的 realpath 在仓库根内、以 `.py` 结尾，因此**必然**进入 `runtime_loaded_modules`。若它不在 `declared_source_set` 里，每次运行都是 `undeclared_loaded` → `derived` → `R-` 一条都不会有。

三条出路，**都需要裁决**，本轮不选：

1. runner 把自己写进声明集（`role=entry`、`compute_role` 待定）——最直白，但声明集是"参与计算的代码"，runner 只采集不算数，`compute_role=numeric` 会让"只改采集逻辑"也换号；
2. 采集时排除 `__main__` 及其文件——必须写清排除规则，否则是**为测试变绿而降低覆盖**；
3. 契约增加"harness 类模块"概念——代价最大，但能一劳永逸。

**注意**：本轮探针用"排除自身"得到 `complete`，**这只是为了让测量本身可读**，不是对 4.1 的裁决。

### 4.2 路径写成绝对 Windows 路径 → 被接受，但永不匹配

契约表格两侧的措辞不一样：`declared_source_set` 明写"path 为 POSIX 相对路径"，
而 `runtime_loaded_modules` 只写"realpath 记录"。实现的两处校验（`check_declared_set:557`、
`_path_set:940`）用的是同一条规则——`p.startswith("/")` 或含 `\\` 即拒。于是本机的
realpath 形状 `D:/eis训练数据/bmdrt/bm_top.py` **两侧都能通过**：

```python
E._path_set([{"path": "D:/eis训练数据/bmdrt/bm_top.py"}], "runtime")
# → ({'D:/eis训练数据/bmdrt/bm_top.py'}, [])      合法通过
```

但它与声明集里的 `bm_top.py` **是不相等的字符串** ⇒ 全部进 `undeclared_modules`
⇒ `undeclared_loaded` ⇒ 永久 `derived`。而这台机器上 `os.path.realpath` 天然返回
`D:/...`，所以"照契约写"的实现会直接掉进来。

**"realpath 记录"与"POSIX 相对路径比较"是两句话**，实现只接受后者。runner 必须
显式 `relpath` + 正斜杠；契约措辞需要澄清（本轮不改）。更麻烦的是：这里的失败
是**全量未声明**，不是个别文件——一眼看去像"代码库整个没声明"，而不是"路径格式错了"。

### 4.3 一次工作流里有多次采样 → 单槽位会张冠李戴

`--two-fits`（同进程连跑 `parallel` 与 `series` 两次采样）实测：`n_calls = 2`，两次都被记录（`idx 0/1`，各 15 个文件）。但**如果钩子只存一个槽位**（`HOOK["snapshot"] = ...` 覆盖式），最终留下的是**第二次**快照，而两条记录都会把它当成自己的 `sampler_entry`——静默错配，值看着完全正常。

这条不是假想：`nested_fit.run(with_voigt=True)` 会在 `voigt_reference` 里再跑一次 `bm_core.fit`（同样是 `MCMC.run`）；Profile C 的 `bm_select` 一次比较就是 `fit_voigt` + `fit_nested` 两次采样。

结论：采集必须是**带归属的列表**，且每个 `run_id` 认领自己那次采样，不能共用"最后一个"。

### 4.4 钩子状态不清零 → 进程内重复时报告上一次的快照

本轮实测踩到：第一版 `run_once` 没有清空 `HOOK`，进程内重复 3 次时后两次的 `sampler_entry` 全都在报**第一次**的值（1263），而 `before_fit` 已是 2135——数字自相矛盾才被发现。

已修（`run_once` 开头清空）。写进报告是因为它属于**采集层的系统性错误类型**：状态跨运行泄漏不报错、不崩溃，只在数字之间留下一个"说得通的差"。

### 4.5 进程内批量：`before_fit` 会变成"上次跑过的一切"

进程内重复 3 次的 `len(sys.modules)`：

```
run0: 1256 → 1263 → 2135
run1: 2135 → 2135 → 2135
run2: 2135 → 2135 → 2135
```

项目模块集合仍是 15（三次全同），所以**判定不受影响**；受影响的是记录里 `runtime_loaded_modules` 的语义：它变成"到此刻为止进程里出现过的全部项目模块"，而不是"这次运行用到的"。

后果：批量复用进程时，后段运行的 `undeclared_modules` / `declared_not_loaded` 与首段不同。若某天出现一个只在第二次才懒加载的模块，**同一代码路径的第 1 次和第 5 次会得到不同的覆盖判定**。而 `numeric_source_tree_sha256` 只由声明集算出（`_chk_code` 用 `items` 整块算，不与运行时集求交），所以**换号不会发生**——只有降级会，且是顺序相关的。

建议（不裁决）：runner 要么一拟合一进程，要么把 `runtime_loaded_modules` 记成"本次运行新增/触达"的集合并在 `runtime_snapshot_limitation` 里写明口径。

---

## 5. 对 runner 的最小接口结论

只写"实测支持"的部分：

1. **挂接点**：`numpyro.infer.MCMC.run` 的类属性包装可用，`n_calls` 与真实采样次数严格一致；不需要改 `bm_top`/`nested_fit`。第 ③ 项可以直接复用同一挂接点。
2. **快照内容**：项目文件路径（**相对 POSIX**）+ 模块名列表；不含第三方。三点并集。
3. **必须多槽位**：每条记录认领自己的采样调用（§4.3）。
4. **必须写清 runner 自身如何处置**（§4.1），否则覆盖判定没有意义。
5. **报告要带 `n_total_modules`**：否则读者无法判断"三点同集"是"确实没新导入"还是"过滤器把一切都滤掉了"（§3.5）。

---

## 6. 未验证 / 边界

- 只测 Profile B（`nested_fit` → `bm_top.fit_top`）。A（`bm_real.fit_one`）与 C（`bm_select`）未测；C 的多次采样只由 §4.3 的 `--two-fits` 间接说明，不是 C 的真实路径。
- 未测 `git_dirty` / 补丁包与快照的交互（②项）。
- 未测 `external_path`：本机没有 editable install 指向仓库外的项目副本。
- 未验证 `role` 字段：契约要求 `runtime_loaded_modules` 是 `list[{path, role}]`，但**未声明文件的 role 没有定义**（声明集里才有 role）。本轮只产出 path。
- `--no-tiny` 只跑 1 次，不是稳定性证据，只用于排除"预算太短导致没有延迟导入"。

---

## 7. 一手数据

| 文件 | 内容 |
|---|---|
| `out/snapshot_probe/fresh_tiny.json` | 独立进程 ×3（主结果） |
| `out/snapshot_probe/inproc_tiny.json` | 进程内 ×3（§4.5） |
| `out/snapshot_probe/two_fits.json` | 同进程两次采样（§4.3） |
| `out/snapshot_probe/fresh_quick.json` | 完整 quick 档 ×1（§3.6） |
| `out/snapshot_probe/smoke.json` | 首版冒烟（含 §4.4 的错配，留证） |

`out/` 不入库；重跑脚本即可复现（除 `smoke.json`，它是修复前的产物）。
