# registry 0.8.0 + adr 默认关 + 两脚本去 adr（2026-11-22）

> 昨天留了三项待裁决，本轮执行。**其中 A 项的代价比昨天估计的严重得多**，
> 见 §3 —— 它不只是"换个版本号"，它让 239 条历史记录的 `run_id` 全部重算成新值。

## 1. 三项改动

| 项 | 改动 | 文件 |
|---|---|---|
| **A** | `REGISTRY_VERSION` `0.7.0` → **`0.8.0`** | `circuit_registry.py:149` |
| **B** | `bmdrt compare` 的 adr **默认从"追加"改为"不追加"** | `cmd_compare.py:362` |
| **C** | 两个分析脚本 `include_adr=True` → `False` | `batch_fit_21.py:133`、`std_recheck.py:142` |

### A 的理由与代际划分

| 世代 | version | digest | 内容 |
|---|---|---|---|
| 冻结 | `0.7.0` | `b2675c8255bea716` | 2026-11-14，239 条记录所属 |
| 中间（**未登记**） | `0.7.0` | `496946a5f5ea1b7d` | 2026-11-20 加 5 个槽位，版本号没跟着升 |
| 现役 | **`0.8.0`** | **`aeb6aca1c13dd2ff`** | 2026-11-22 补上版本号 |

中间那一代是**版本号语义错误**：同一个 `0.7.0` 对应两套注册表内容。
升 `0.8.0` 把三个世代分开。

`evidence/stage0_records/MANIFEST.json` **不重冻结** —— 它记的 `0.7.0/b2675c82`
正是那 239 条记录所属的世代。

### B 的接口设计

```python
--adr      (store_true, default=False)   追加；研究用途
--no-adr   (store_false, default=SUPPRESS)  兼容旧命令，现已无效果
```

用 `SUPPRESS` 而不是给 `--no-adr` 写 `default=True`：同 `dest` 挂两个 action 时，
后者的 `default` 会覆盖前者，`SUPPRESS` 能让 `--adr` 的 `default=False` 生效。
实测三种情况：

| 命令 | `include_adr` | 横幅 |
|---|---|---|
| 默认 | `False` | 无 `+adr` |
| `--adr` | `True` | `、+adr` |
| `--no-adr` | `False` | 无 `+adr`（不报错） |

## 2. C 为什么不改变既有数字

21 条里 **adr 从未当选 best**（best 按 `rms` 定，adr 的 rms 中位 22.23%）。
`std_recheck` 只重跑 `results.json` 记下的那一个 best，所以候选列表里去掉 adr
不影响匹配。既跑结果原样有效。

## 3. A 的真实代价：`run_id` 239/239 全部漂移

这是昨天**没有预料到**的。`records.jsonl` 每条记录都嵌了
`observation.execution.software.registry_version`，而 **`run_id` 的哈希包含它**：

```
0.7.0:  D-4d61cf10ae
0.8.0:  D-9751ecfac8     ← 同一条观测，只有版本号变了
```

实测（`round_2026-11-22/runid_generation_probe.py`，内存里跑两次 `build()`）：

| 量 | 结果 |
|---|---|
| 记录数 | 239 / 239 |
| `run_id` 变化的记录 | **239 / 239** |
| 差异字段 | 仅 `software.registry_version` 与 `execution.run_id`（及其在 `provenance.duplicate_evidence` 里的引用） |

**这不违反"239 条保持原等级"**：冻结副本里存的 run_id 是当时用 0.7.0 算的，
那是历史事实，没被改；`records.jsonl` 的**内容**（rms、参数、sha256、判重结论）
逐行等价。变的是"用当前代码重算会得到新 run_id"。

但它有一个真后果：**`--check` 从此会永久报警**，而报警分不清"版本升级"和"记录被篡改"。

### 处置：给 `--check` 加第 5 层（跨代识别）

`map_stage0_records.py::_same_except_generation()`：把
`registry_version` / `registry_digest` / `run_id` 抹平后逐行比；
**其余任何字段的差异仍然返回 False**。

现在 `--check` 的输出：

```
records.jsonl: 重算 283bf0ee / 冻结 bf22f64b → 不一致
跨代差异 1 处（预期，不计为问题）：
  ~ records.jsonl: …但内容逐行等价（差异仅 registry_version/run_id）
发现 3 处不一致：
  - MANIFEST.json: 重算 11dcf721 ≠ 冻结副本 37a43874
  - MANIFEST.registry_version='0.7.0' ≠ 当前 '0.8.0'
  - MANIFEST.registry_digest='b2675c8255bea716' ≠ 当前 'aeb6aca1c13dd2ff'
  注：registry_version/digest 不匹配 = 注册表代际变更，属有意升级…
```

**~~剩下这 3 处是刻意保留的~~（已被下方订正推翻）**，不是没修完：第 3 层（注册表身份）的原始设计意图就是
"注册表改过、但记录序列化没变时，这一层是唯一能发现的"（审计 D4）。
把它降级成"预期"会让 `--check` 失去这个能力。

> **订正（2026-11-22 同日，见 `CHECK_ALARM_ROOT_CAUSE.md`）**：上面这段结论**已被推翻**。
> 那 3 处不是"刻意的代价"，而是**判据失效**——冻结快照按构造永远追不上现役代码，
> 于是永久红，红里分不出升级与篡改；且同期 `test_stage0_freeze.py` 已掉到 5/9
> （守卫本身坏了，没人发现）。已改为：第 3 层查**已登记代际表**（`_KNOWN_GENERATIONS`），
> 代际差归 `crossgen`、不阻断；未知代际仍是问题；**现役代际未登记也报错**（逼升版补一行）。
> 现状 `--check` = **rc 0**、`test_stage0_freeze.py` = **12/12**、
> `--recompute-runids` = **239/239 可复算**。
> 原句保留在此，因为"不要重冻结 MANIFEST"这一条**仍然成立**（重冻结=把新注册表追认给 239 条历史）。

## 4. 测试

| 套件 | 结果 |
|---|---|
| `test_entry_parity.py` | **207/207** |
| `circuit_runner.py --selftest` | **81/81** |
| `test_contract.py` | **27/27** |
| `test_circuit_registry.py` | **76/76** |
| `test_sampler_capture.py` | **29/29** |
| `bmdrt.py check` | **18/18** |
| `topology_limit_check.py` | **9/9**（退出码 0） |
| `test_stage0_freeze.py` | 改前 **5/9** → 改后 **10/10**（`CHECK_ALARM_ROOT_CAUSE.md`） |

## 5. 边界与未做的事

| 项 | 说明 |
|---|---|
| 未重跑任何拟合 | 21 条、§14、B2 的既有数字全部原样 |
| `--check` 全绿了 | 同日改判据后 rc=0（3 项代际差异单列为 `crossgen`）；原"不能全绿"的结论作废，见 `CHECK_ALARM_ROOT_CAUSE.md` |
| `run_id` 定义未动 | 把 `registry_version` 移出 run_id 哈希会改变所有历史身份，属另一个量级的改动，本轮不碰 |
| `adr` 仍 `wired` | 只是不进默认候选（库层 `include_adr=False`、CLI `--adr` 默认关、两脚本关）；研究入口保留 |
| 文档里的 `0.7.0` 引用 | `budget_params.py` 里多处 "registry 0.7.0" 是**契约世代名**（指那个形状的字段布局），不是动态版本，**不该改** |

## 6. 复现

```
# 三项行为
D:/autoeis_env/Scripts/python.exe cmd_compare.py <spectrum.csv> --budget quick --out /tmp/a   # 无 +adr
D:/autoeis_env/Scripts/python.exe cmd_compare.py <spectrum.csv> --budget quick --adr --out /tmp/b  # 有 +adr

# 版本与 digest
D:/autoeis_env/Scripts/python.exe -c "import circuit_registry as R; print(R.REGISTRY_VERSION, R.registry_digest())"
# 0.8.0 aeb6aca1c13dd2ff

# run_id 漂移证据
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-22/runid_generation_probe.py

# 快照核对（rc 0；3 项代际差异单列）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --check
# 冻结快照自检（10/10）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/test_stage0_freeze.py
```
