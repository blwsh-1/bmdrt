# 阶段 0：历史产物的结构化映射（机器可读，v4）

状态：**映射已按第三轮审计返工；第五、六轮（registry 0.6.0 → 0.7.0）复核零影响**。
v4 只做三件事——确认重复的门槛提到 `recorded`、数据身份改用内容哈希、`--check`
校验整个快照。不跑拟合、不改生产、不进入阶段 1。

> 0.7.0 复核（第六轮审计）：`sha256` 原来只查非空，两条 `"same"` 会凑出
> `same_artifact`。改成 64 位十六进制校验，格式错误直接报错（见 §3.1 末段）——
> **本批 239 条一条都不受影响**（实测 239/239 是 `hashlib` 算出的 64 位十六进制），
> `coverage.json` 重算字节与 0.6.0 冻结副本逐位相同（`d58a14a6d0398a00`），
> 快照已按 0.7.0 重冻结，`--check` rc=0。

> 0.6.0 复核（第五轮审计）：`same_artifact` 原来只要求 `path is not None`，
> 两条 `{"path": ..., "sha256": None}` 会被判成同一产物。修法与影响见 §3.1——
> **本批 239 条一条都不受影响**（没有缺 `sha256` 的 `source`），
> `--check` 在 0.6.0 下重跑仍 rc=0。

产物：
- `map_stage0_records.py`（v4，只读 `evidence/` 与原始谱，含身份完整性断言）
- `out/stage0_records/{records.jsonl, coverage.json}` → 字节级副本在 `evidence/stage0_records/`
- `evidence/stage0_records/MANIFEST.json`（含三份文件索引、注册表指纹、9 份原始证据与 63 份原始谱的 sha256）
- `test_stage0_freeze.py`（9 条用例：快照完整性 + 清单删字段，实跑 **9/9**）
- 记录数 **239**，**独立 239**（严格口径 **237**），`confirmed_duplicate` **0**、
  `possible_duplicate` **2**

> **口径变了，必须两个数一起报。** v3 说"239 条 → 237 独立"，那是把两对同条件重跑
> 直接扣掉了。v4 起：这两对**没有任何一份产物记代码版本**，所以扣掉是猜。
> 主口径 `n_independent=239`（上限），严格口径 `n_independent_strict=237`
> （按"同 seed 同预算同摘要即视为重跑"扣）。单独引用任何一个都会误导。

---

## 0. v4 改的三件事

### 0.1 确认重复必须要求 `run_id_status=recorded`（审计 P0，成立）

`execution_identity()` 的 `R-` 门槛写的是四条：seed 由产物记录 + 预算 `recorded`
+ **代码版本已知** + **数据有内容哈希**。而 v3 的 `duplicate_verdict()` 里
`observed` 只查了前两条：

```python
observed = all(e["seed_source"].startswith("artifact") and e["budget_source"] == "recorded" ...)
```

于是出现了自相矛盾的状态：同一份报告里说"239 条全是 `D-`，因为没有一份产物记代码版本"，
却又把两对判成 `confirmed_duplicate`，而那两对的 `software.code_version` 都是 `None`。
**同 seed、同预算、同摘要，仍然可能出自不同版本的拟合代码。** 用自己定义的门槛去
绕过自己定的门槛，等于把 `R-` 的定义降成摆设。

v4 的确认条件按 `recorded` 的**完整定义**整体判（`_conditions_recorded()`）：
`run_id_status == "recorded"` ∧ seed `artifact` ∧ 预算 `recorded` ∧ 代码版本已知
∧ `data_id_kind == "content_sha256"`。本批 **0 条**满足 ⇒ 两对降为 `possible_duplicate`，
`confirmed_duplicate` 归零（用例 D1 / D2 各守一侧）。

### 0.2 数据身份从"文件名"改成"内容哈希"（审计 P1，成立）

v3 的 `data.full` 是 `Path(...).name`，pilot 用 `r["name"]`。文件名不是身份：
两条不同实验的谱都可能叫 `spectrum.csv`、频率窗也一样，`run_id` 就会撞车——
这和 v3 刚修好的"60 条谱不能共享一个 `run_id`"是同一个错，只是从数量换成了命名。

v4 加 `DATA_ID_KINDS = ("content_sha256", "ref_only")`：

| 情形 | `data_id_kind` | 后果 |
|---|---|---|
| 有原始文件、算出 sha256 | `content_sha256` | 可参与 `recorded` |
| 只有文件名 | `ref_only` | **最高只到 `derived`**，不能参与确认重复 |

`ref` 仍是文件名（给人看），但**不进哈希**——完整路径是机器相关的，会让同一个实验
换台机器就换号。哈希只吃 `sha256`；没有 sha256 时才退到 `ref`，并由 `id_kind` 留痕
（哈希之后看不出来的东西必须在 `data_id_kind` 里写清楚）。

本批 239/239 都拿到了 `content_sha256`（63 份唯一原始谱：涂层 2 + pilot 60 + AZ31 1）。

### 0.3 `--check` 校验整个快照，不再只比两份数据文件（审计 P1，成立）

v3 的 `--check` 只重算 `records.jsonl` 与 `coverage.json` 的 sha256，**没碰** MANIFEST
里的 `registry_version` / `registry_digest` / `source_sha256`。漏洞很具体：注册表改了
但记录序列化恰好没变时，检查照样"通过"，而清单里的指纹已经不代表当前注册表。

v4 的 `verify_frozen()` 分四层，任意一层不一致就返回非零：

1. 重新生成的**三份文件**（含 MANIFEST 自身）vs 冻结副本磁盘字节；
2. 冻结清单的 `files` 段 vs 冻结副本磁盘字节——副本被单独改过也能抓到；
3. `registry_version` / `registry_digest` vs 当前注册表（审计 D4）；
4. `source_sha256`（9 份产物）与 `data_sha256_files`（63 份原始谱）逐份重算（审计 D5）。

另加 `--frozen <dir>`，测试可以指向一份改过的副本，而**不动仓库里的冻结件**。

### 0.4 v3 修对的部分（保留）

- 去重判定按来源三态，`independent` 是派生值（v3 §0.1）。
- `candidate_id` 不动，执行身份单独一层 `run_id`（v3 §0.2）。
- pilot 的 `sensitivity` 段按调用点判成 voigt（v3 §0.3）；voigt 默认初始化改回 `uniform`（v3 §0.4）。
- AZ31 的 voigt 行按 `label` 分流，不再与 nested 撞 `candidate_id`（v2 §0.5）。

---

## 1. 初始化判定表

| 产物 | 条数 | 生成者 / 调用点 | 判定 |
|---|---|---|---|
| `*.nested.json` | 6 | `nested_fit.py:339` 不传 `init_values` | `uniform` |
| AZ31 6 行 | 6 | `fit_nested:161` / `fit_voigt:105` 都不传 | `uniform` |
| pilot `voigt_prod` / `voigt_fix2` | 120 | `voigt_fit(init=None)` | `uniform` |
| pilot `nested` | 60 | `voigt_fit`→`data_init`（:219） | `data_driven` |
| pilot `sensitivity` | 46 | `voigt_fit(init=voigt_init)`（:268） | `data_driven` |
| oracle | 1 | 读文件名里的真值参数 | `value` |

核对：`uniform 132 + data_driven 106 + value 1 = 239`。全部 `identity_status=derived`。

---

## 2. 记录构成（239 条 / 15 个唯一候选）

| 来源 | 条数 | 说明 |
|---|---|---|
| `*.nested.json` | 6 | 涂层 4（parallel/capacitive/quick/1113）、钝化 2（default/1113） |
| oracle 诊断 | 1 | `truth_init`，初值用真值构造，`production_reachable=false` |
| pilot `voigt_prod` | 60 | voigt / 自由 σ / 默认初值 |
| pilot `voigt_fix2` | 60 | voigt / 固定 2% / 默认初值 |
| pilot `nested` | 60 | nested capacitive L2 / 逐点 2% / 数据驱动初值 |
| pilot `sensitivity` | 46 | voigt / 自由 σ / 数据驱动初值，仅 unusable 行 |
| AZ31 | 6 | 4 条 nested + 2 条 voigt，真实谱无真值 |

---

## 3. 重复判定：本批 0 确认、2 条可能

| 记录 | `duplicate_of` | 状态 | 依据 | 缺什么 |
|---|---|---|---|---|
| coating `standard_1113` | coating `standard` | `possible_duplicate` | 同候选、同 `run_id`（`D-f953bba8d3`）、`summary_equal=true`、`config.seed=0` 与 `budget_kw` 五项都由产物记录、数据同内容哈希 | **代码版本未记录** |
| passivation `default_init` | passivation `1113` | `possible_duplicate` | 同上（`D-21214e4025`）；20.1 vs 31.4 s | **代码版本未记录** |

这两对的人工核查结论没有变：`elapsed` 不同（57.9 vs 110.1）、指标逐位相同、输出目录不同、
`with_voigt` 只影响参照不影响本候选后验 ⇒ 很可能是同条件重跑。**"很可能"不是"确认"。**
按现在的执行身份规则，它们证明不了"同一软件条件下的同一次执行"，所以：

- `provenance.independent` 保持 `true`（不扣）；
- `coverage.n_independent_strict = 237` 给出扣掉后的对照；
- `duplicate_evidence` 里逐项列出 `run_id_status / code_version / data_id_kind /
  seed_source / budget_source / summary_equal / paths`，复核时不用翻代码。

规则在 `duplicate_verdict()` 里，不在文档里。

### 3.1 第五轮补的边界：`same_artifact` 的缺字段碰撞

`same_artifact` 是判重里**唯一不经过 `recorded` 门槛**的一支（它有自己的依据：
两条记录指向同一份产物）。0.5.0 给它写的条件是
`sa.get("path") is not None and 四个定位字段全等`，于是：

```python
{"path": "result.json", "sha256": None}  vs  {"path": "result.json", "sha256": None}
```

被判成同一产物。两个 `None` 相等**不是**来源证据，这跟 0.4.0 拿摘要哈希当证据是
同一个毛病。0.6.0 的 `_artifact_key()` 改成：

| 条件 | 键 | 结论 |
|---|---|---|
| `path` 与 `sha256` 均非空、无 `row` | `("file", path, sha256)` | 文件级定位 |
| `path` 与 `sha256` 均非空、有 `row` | `("row", path, sha256, row, section)` | 行级定位 |
| 任一为空 | `None` | 不可参与确认 |

前缀不同 ⇒ 文件级与行级永不相等；`section` 脱离 `row` 直接报错；不确认时
`duplicate_evidence["artifact_gap"]` 写明缺哪个字段/哪里不同。

**对判重结论的影响：零。** 239 条的 `source` 全部带 64 位 `sha256`，6 条 `*.nested.json`
与 oracle 用文件级键、pilot 60 行与 AZ31 6 行用行级键，两对重跑仍是
`possible_duplicate`（它们的 `path` 本来就不同，靠 `run_id` 配对）。
`n_independent` 239 / `n_independent_strict` 237 / 唯一 `run_id` 237 都不变。

但**239 行的 `run_id` 数值全部变了**：`software.registry_version` 在 `RUN_ID_BASIS` 里，
0.5.0 → 0.6.0 改变了执行指纹（本批唯一改动就是这一个字段 + 派生出的 `run_id`）。
这是设计后果：`run_id` 只在**同一 registry 版本内**可比，跨版本比号没有意义。
0.7.0 又重复了一次（`bf22f64b75ea8ea9` 取代 `ebd8f8a08f1ed071`），结论不变。

**0.7.0 补的一段：`sha256` 非空 ≠ 有效。** 上面那张表的前两行都要求"`sha256` 非空"，
但 `_nonempty_str` 只回答"填了没有"——两条 `{"path": "x", "sha256": "same"}` 照样会
造出相等的键、被判成 `same_artifact`。现在是 64 位十六进制校验（大小写/首尾空白归一），
格式错误**抛 `ValueError`** 而不是退化成 `None`：后者会把调用方的 bug 藏成"证据不足"。
本批 239 条全部合法，所以这张表和所有计数一个都不用改。

---

## 4. 三类事实必须分开写（审计要求）

同一句话可能来自三种强度完全不同的依据。本文件与 README 都按此标注：

| 类别 | 含义 | 本批例子 |
|---|---|---|
| **A 产物核对** | 直接读产物字段，可逐位复核 | `config.seed=0`、`budget_kw` 数值、`metrics`、`elapsed_s`、`files.json`、`input.path` |
| **B 代码路径推导** | 产物没写，但生成脚本唯一且调用点可读 | 初始化档（`init_values` 未传）、pilot/AZ31 的 seed（调用点写死 0）、预算数值（`BUDGETS` 档名 + 脚本覆盖）、代码版本（**没有**） |
| **C 脚本自动判定** | 映射脚本按 A/B 组合算出来的结论 | `candidate_id`、`run_id`、`duplicate_status`、`data_id_kind`、`ratio` 的 `denominator_scope` |

**C 类不能倒推 A 类。** `duplicate_status` 是 C 类结论，它的依据逐项写在
`provenance.duplicate_evidence`；`run_id` 是 C 类指纹，`run_id_status=derived` 说明它不是
产物记录的；`data_id_kind=content_sha256` 是 C 类判定，但它**基于**对原始谱文件重算 sha256
这件事（可复核：路径与哈希都在 `MANIFEST.data_sha256_files` 里）。

v2 那句"6 个 nested.json 只有 4 次唯一运行"当时只有 C 类依据（摘要指纹）却写成 A 类事实。
现在它的 A 类依据（`config.seed`）有了，但"同一次执行"仍是 C 类——所以 v4 只写
`possible`。

---

## 5. 六项字段的落地情况（v4 计数）

| 字段 | 回答的问题 | 本批 |
|---|---|---|
| `param_estimate` | 估到多少、离真值多远 | **67 条**（pilot nested 60 + nested.json 6 + oracle 1）；172 条 `null` |
| `param_stability` | 后验区间稳不稳 | **6 条**（只有 `nested.json` 存了 `ci90`） |
| `ratio` | 残差 / 噪声底 | **226 条**，全部带 `n_used`/`n_total`/`n_usable` |
| `noise_floor` | 底从哪来 | 226 `truth_forward`、13 `not_available` |
| `predictive` | ELPD 能否当样本外证据 | 12 条有 `elpd`（6 条标 `False` + 原因；6 条 AZ31 标 `True`） |
| `budget_params` | 实际用了多少算力 | 6 `recorded`、**233** `derived_from_tier` |

---

## 6. 执行身份落地情况

| 项 | 本批 |
|---|---|
| 唯一 `run_id` | **237**（239 条记录，2 条 `possible_duplicate` 与对方共享条件指纹） |
| `run_id_status` | `derived` **239**、`recorded` **0** |
| seed 来源 | `artifact` **6**（`nested.json` 的 `config.seed`）、`producer_code` **233** |
| 预算来源（身份内） | `recorded` **6**、`derived_from_tier` **233** |
| 数据身份 | `content_sha256` **239**、`ref_only` **0** |
| 代码版本已知 | **0** |

**`recorded` 目前不可达**，唯一卡点是代码版本：6 条 `nested.json` 的 seed 与预算都是
产物记录、数据也有内容哈希，只差一份产物记下 git sha / 包版本。这是待补字段
（下一阶段的 `runner` 必须往产物里写），不是把门槛降低的理由。
`R-` 前缀因此一条都没发出去。

---

## 7. 承载力判定

| 能力 | 结论 | 依据 |
|---|---|---|
| 残差 / 采样状态 / 解释限定 | ✅ | 70 条"跑完但采样未过"入库且标 `descriptive_only` |
| 候选身份（模型层） | ✅ 有条件 | 239 条全 `derived`；0 条 `recorded` |
| 执行身份（条件层） | ⚠️ 部分 | 239 条 `derived`；seed 产物记录 6 条；代码版本 0 条（缺口审计见 `HISTORICAL_EXEC_IDENTITY_GAP.md`） |
| 数据身份 | ✅ | 239 条内容哈希，含 63 份原始谱 |
| 参数估计 vs 真值 | ⚠️ 部分 | 67 条带 `rel_err` |
| 比值与噪声底 | ✅ | 226 条带完整分母与底来源 |
| ELPD 可用性 | ✅ | 12 条带 `valid_for_comparison` + 原因 |
| **独立观测计数** | ⚠️ 区间 | 239（上限）/ 237（严格）；0 条可确认重复 |
| 快照可校验 | ✅ | `--check` 四层校验，`test_stage0_freeze.py` 9/9 |

**阶段 1 仍不进。** 硬缺口两条：233/239 条的预算数值由档名/脚本推出，
**0 条记录代码版本**。做历史对照时"同档名"不等于"同预算"，更不等于"同代码"。

---

## 8. `MANIFEST.json` 与 `--check`

```bash
python docs/round_2026-11-13/map_stage0_records.py --freeze   # 写三份副本 + MANIFEST
python docs/round_2026-11-13/map_stage0_records.py --check    # 四层校验，不写文件
python docs/round_2026-11-13/test_stage0_freeze.py            # 8 条篡改用例
```

`MANIFEST.json` 含：三份文件的 `sha256`/字节数、`registry_version` 与 `registry_digest`、
记录数/独立数/严格数/重复态/`data_id_kind`、`run_id` 统计、9 份原始证据的 `path → sha256`、
63 份原始谱的 `path → sha256`（`data_sha256_files`，绝对路径，随机器而变属预期）。

`--check` 的**四层**（详见 §0.3）与对应失败模式：

| 层 | 抓到什么 | 用例 |
|---|---|---|
| 重算文本 vs 冻结字节 | 映射代码改了、原始产物改了 | D5c |
| 清单 `files` vs 磁盘字节 | 副本被单独改过 | D5c |
| `registry_version`/`digest` | 注册表改了但记录序列化没变 | D4a / D4b |
| `source_sha256` / `data_sha256_files` | 原始证据或原始谱被改 | D5a / D5e |

另外守住反向：未改动时必须通过（D0 / D5b）——一个只会喊"失败"的检查等于没有检查。

---

## 9. 复算

```bash
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --check
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/test_stage0_freeze.py
D:/autoeis_env/Scripts/python.exe test_circuit_registry.py
```

脚本内置断言（不通过直接退出，不产出半成品）：
`record_id` 唯一；同一 `candidate_id` 必须对应同一份六字段定义；
`identity_status ∈ IDENTITY_STATES`；`init_profile=unknown` ⇒ `U-` 且不可比较；
`duplicate_status ∈ DUPLICATE_STATES` 且 `independent` 与它一致；
`run_id` 与状态前缀一致，未发号必须有 `RUN_ID_MISSING` 里的原因。
