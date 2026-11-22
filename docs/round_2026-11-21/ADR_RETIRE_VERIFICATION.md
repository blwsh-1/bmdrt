# adr 退役相关说法核实（2026-11-22）

> 上一轮结尾提出三条落地动作，本轮逐项核实。结论：**三条里两条我说错了**，
> 另外查出一个更该处理的未登记漂移（§4）。

## 1. 核实结论总表

| # | 上一轮的说法 | 核实结果 | 依据 |
|---|---|---|---|
| ① | `build_candidates` 的 `include_adr` 默认 False，"其实已经是默认不进的" | **半对**：库层是 False，**CLI 层默认 True** | `circuit_runner.py:245` vs `cmd_compare.py:362-364` |
| ② | "改动在 `circuit_registry` 的 `deprecated` 标记" | **错**：没有这个字段；用 `maturity` 伪装会改 digest | `TopologySpec` 字段表 + 实测 |
| ③ | notes 里"nested 未收敛（R̂=22.07）而 adr 收敛" | **归因错**（旧配置），但 notes 可零代价修正 | 实测 digest 不变 |
| ④ | — | **新发现**：0.7.0 对应两个 digest，无人登记 | `map_stage0_records.py --check` |

## 2. ① `include_adr` 的实际默认值：两个口径相反

| 层 | 位置 | 默认值 |
|---|---|---|
| 库 | `circuit_runner.build_candidates(..., include_adr=False)` | **False**（不进） |
| CLI | `bmdrt compare` 的 `--no-adr`，`dest=include_adr, action=store_false, default=True` | **True**（进） |
| 脚本 | `batch_fit_21.py:130`、`std_recheck.py:140` 显式 `include_adr=True` | 进 |

所以"已经是默认不进"只在库层成立。`bmdrt compare` 手动跑一条谱时，adr 默认是参与的；
21 条那批也是脚本显式打开的。**要说"不进默认候选集"，得同时处理 CLI 与那两个脚本，
或者把话说清成"库层默认关、CLI 默认开"。**

## 3. ② `deprecated` 标记不存在

`TopologySpec` 的字段是：`id / name / skeleton / forward_ref / param_space / derived /
priors / init_profiles / slots / maturity / min_level / precheck_ref / consistency /
consistency_id / notes`。没有 `deprecated` / `retired` / `status`。

实测（本机 `registry_digest()`）：

| 改动 | digest | 是否变 |
|---|---|---|
| 现状 | `496946a5f5ea1b7d` | — |
| 只改 `notes` | `496946a5f5ea1b7d` | **不变** |
| 改 `maturity` → `frozen` | `3ddf0b23fe74ee18` | **变** |

`maturity` 在 digest payload 里，`notes` 不在。所以：

- 用 `maturity` 表达"退役"⇒ manifest 失效、239 条记录报"注册表被改动" ⇒ **不可用**
- **`notes` 是唯一零代价的记录渠道** ⇒ 退役理由只能写在 notes 里，不能写成字段

## 4. ③ notes 里的 `R̂=22.07`：归因错误，已订正

原文把 `AZ31-12Li_5` 的 `nested R̂=22.07` 与 `adr R̂≈1.002` 并列，读起来像
"nested 这个族采样不稳、adr 更稳"。事实上那个 22.07 出自 `include_L=False` 的旧配置
（`out/m4_holdout2.json` 14 条参数**全无 `L` 键**），接顶层 L 后同谱
`R̂=1.003 / div=0`。所以那是**缺 L 的表达力问题**，不是族的采样问题。

同源表述还在这些地方（**历史文档，本轮不改**）：

| 文件 | 行 | 表述 |
|---|---|---|
| `NESTED_TOPOLOGY_PROJECT.md` | 2448 / 2472 / 3625 / 3920 | "采样崩 R̂=22.07" |
| `round_2026-11-16/PROVISIONAL_ADMISSION.md` | 38 | M-c "`12Li_5` R̂=22.07" |
| `round_2026-11-16/TOPOLOGY_COVERAGE_AUDIT.md` | 68 | 归为 "P（主）+ D，**采样崩**…拓扑能表达，**是采样问题**" |

最后一条影响最大：它把"缺 L"归档成了"采样问题"。按项目纪律不改写历史文档，
但引用这三处时必须同时读本节。

**已改的只有 `circuit_registry.py` 的 adr `notes`**（原观测保留 + 追加订正 +
"作为候选已被支配" + "不进默认候选集"），digest 保持 `496946a5f5ea1b7d` 不变，
`test_entry_parity.py` **207/207**。

## 5. ④ 新发现：`0.7.0` 对应两个 digest，且无人登记

`docs/round_2026-11-13/map_stage0_records.py --check` 报两处不一致：

```
MANIFEST.json: 重算 fe08e70e / 冻结 37a43874 → 不一致
MANIFEST.registry_digest='b2675c8255bea716' ≠ 当前 '496946a5f5ea1b7d'
```

| | version | digest | 时间 / 来源 |
|---|---|---|---|
| MANIFEST 冻结 | `0.7.0` | `b2675c8255bea716` | 2026-11-14（commit `dc6d244`） |
| 当前注册表 | `0.7.0` | `496946a5f5ea1b7d` | 2026-11-20 起 |

差异来源：2026-11-20 接 Warburg/顶层 L 时给三族各加 `diffusion` 与
`series_inductor` 槽位（`git diff` 里 5 处 `SlotSpec(`），`slots` 进 digest。
**11-20 与 11-21 两轮的文档里没有任何一处提到 digest 变了。**

三点判断：

1. **不是数据被篡改**：`records.jsonl` 重算与冻结清单一致（`bf22f64b75ea8ea9`），
   239 条记录内容未变、等级未变。
2. **是版本号语义错误**：同一个 `0.7.0` 对应两套不同的注册表内容，
   "0.7.0 的注册表"不再唯一。`map_stage0_records.py` 的告警**机制正常工作了**，
   只是没人处理这个发现。
3. **不能靠重新冻结 MANIFEST 解决**：那等于把新注册表追认到 239 条历史记录上。

## 6. 待裁决（三项，都动生产行为）

| 项 | 动作 | 影响 |
|---|---|---|
| A | `REGISTRY_VERSION` 升 `0.8.0`（新增槽位属 schema 扩展） | digest 再变一次；但 0.7.0/`b2675c82` 与 0.8.0 的世代能区分开；MANIFEST 仍记 0.7.0 不重冻结 |
| B | `bmdrt compare --no-adr` 的默认从 `True` 改 `False` | 手动跑 compare 不再默认带 adr；既有产物不变 |
| C | `batch_fit_21.py` / `std_recheck.py` 的 `include_adr=True` | 只影响将来重跑；已跑的 21 条结果里 adr 从未当选 best，故不改数字 |

**本轮只做了一件不需要裁决的事**：订正 adr 的 notes（零 digest 代价）。
其余三项等确认。

## 7. 附：21 条里 adr 的真实消耗

| 项 | 值 |
|---|---|
| adr 候选被评估次数 | **21 次**（每谱 1 个） |
| rms 中位 | 22.23% |
| `R̂>1.05` 或 `div>0` | **10 / 21** |
| 当选 best | **0 次**（best 按 rms 定，不是 elpd） |

所以"报告里不再用它下结论"对 21 条的既有数字**零影响**——它本来就没进过结论；
但它确实消耗了 21 次采样，其中 10 次是无效后验。
