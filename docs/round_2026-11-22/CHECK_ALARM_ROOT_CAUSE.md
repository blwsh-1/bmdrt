# `--check` 永久报警：根因与处置（2026-11-22）

> 起因：`REGISTRY_0_8_0.md` §3 把"`--check` 从此报 3 处"写成**刻意的代价**，
> 并说"不要试图清零"。本轮查下来这个结论**站不住**：那不是代价，是判据失效。
> 本文给证据、给改法、给残留口子。

## 1. 结论

3 处报警**只有 1 个根因**：`--check` 第 3 层问的是"冻结快照的注册表指纹 == **当前代码**
的指纹"，而冻结快照是历史文物——它按构造永远追不上现役代码，所以只要注册表再动一次，
这层必然红。**永久性来自判据问法，不是来自谁没修完。**

## 2. 3 处只对应 1 个根因（实测）

把重算的 MANIFEST 抹掉 `registry_version`、`registry_digest`、
`files["records.jsonl"].sha256` 三项，再与冻结副本逐字段比 → **完全相同**。

即：

| 报出来的 | 实际身份 |
|---|---|
| `MANIFEST.json: 重算 11dcf721 ≠ 冻结 37a43874` | **派生项，零独立信息**（由下面两项决定） |
| `MANIFEST.registry_version='0.7.0' ≠ 当前 '0.8.0'` | 根因 |
| `MANIFEST.registry_digest='b2675c82…' ≠ 当前 'aeb6aca1…'` | 根因（同一个事实） |

`records.jsonl` 的字节差异早被 `crossgen` 收录（不计为问题），但它使
`files["records.jsonl"].sha256` 变化 —— 这就是 MANIFEST 那行的全部来源。

复现（`_strip_generation_manifest` 抹的就是这三项）：

```python
import json, sys
sys.path.insert(0, 'docs/round_2026-11-13')
import map_stage0_records as M, bm_select, circuit_registry as R
budgets = {k: {kk: vv for kk, vv in v.items() if kk in R.BUDGET_PARAM_KEYS}
           for k, v in bm_select.BUDGETS.items()}
M._HASHED_FILES.clear()
recs = M.build(budgets); cov = M.coverage(recs)
texts = {'records.jsonl': M._serialize(recs),
         'coverage.json': json.dumps(cov, ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
man = M._manifest(texts, cov, recs)
frozen = json.loads(open('evidence/stage0_records/MANIFEST.json', encoding='utf-8').read())
print(M._strip_generation_manifest(json.dumps(man))
      == M._strip_generation_manifest(json.dumps(frozen)))   # True
```

## 3. 根因链：`registry_version` 被放进了 `run_id` 的哈希

```
_check_software()   无论传入什么，一律盖成当前 REGISTRY_VERSION
                    ⇒ 实测：传 {'registry_version':'0.7.0'} 回吐 '0.8.0'
_run_id_from()      把整个 software 段（含 registry_version）哈希进 run_id
                    ⇒ 升版即 239/239 条 run_id 全变（D-4d61cf10ae → D-9751ecfac8）
```

`circuit_registry.py:214` 的注释写着"进 `run_id` 哈希的字段。**改这里必须改
`REGISTRY_VERSION`**"——方向反了。升版本号改的是**注册表内容**（加槽位），
不是**执行条件**，也不改 `run_id` 的字段集合。两件事被同一条注释绑在一起，
于是"补一个版本号"这个纯记账动作，变成了"重写 239 条执行身份"。

## 4. 真正的损失（比"红"重）

| # | 损失 | 状态 |
|---|---|---|
| L1 | 报警语义失效：永久红 rc=2，当不了门禁；真篡改只能靠"行数≠3"人眼分辨 | 已发生 |
| L2 | **冻结快照的 239 条 run_id 用当前代际复算不符**：把冻结记录的 `execution` 交给 `_check_execution()` 直接 `ValueError`（记录 `D-4d61cf10ae`，期望 `D-9751ecfac8`） | 已发生；**同日澄清为"写入约束"而非缺陷**，见 §4.1 |
| L3 | `test_stage0_freeze.py` 从 9/9 掉到 **5/9** | 已发生，但没人发现 |

L3 有两种病因，必须分开：

| 用例 | 病因 | 与升版的关系 |
|---|---|---|
| D0 基线通过 | 基线快照现在 rc=2 | 本轮升版导致 |
| D5a / D5b / D5e | 上一轮把 `verify_frozen` 改成返回 `(problems, crossgen)`，测试仍按单值用（`assert not probs` 拿到 `([], [])`，恒真→恒假） | **无关**，是改签名时漏改测试 |

也就是说：**守这道门的那把锁，在锁坏了之后一直没被跑过。**
（这也解释了为什么"警觉永久报警"没有触发任何修复：唯一的守卫已经是哑的。）

### 4.1 L2 的定性与处置：不改生产，改用只读审计

先证"能不能"：把 `_check_software()` 的覆盖行为临时绕开，按记录**自报**的
`registry_version` 复算，**239/239 全部一致**（`D-4d61cf10ae` 复算得出它自己）。
所以 L2 完全可修，代价是一个函数。

再判断"该不该"：三条事实说明**不值得动生产**。

1. **拟合链路不读 `run_id`。** 全仓 grep（`bm_*.py` / `nested_fit.py` /
   `cmd_compare.py` / `bmdrt.py`）只有 `circuit_runner.py` 的一处自检提到它，
   内容是"`run_id is None` 且状态 `unconfirmed`"——即**核对它没被伪造**，
   不是拿它做决策。`results.json` 不落 `run_id`，`bmdrt check` 18 项也不含它。
2. **拟合侧现在根本不发号。** `circuit_runner` 走 `execution=None` ⇒ 五项预算
   `unavailable` ⇒ `unconfirmed`、`run_id=None`。多模型多数据实验要真用上它，
   得先让 runner 传 data/seed/实际预算；那时是**新记录、新代际、自洽**，
   不需要"历史可复算"。
3. **改生产会放掉一条现有防护。** 现在"改 `registry_version` → run_id 复算不符
   → 当场抓住"。改成保留自报值后这条失效，得再引入一张版本白名单去补，
   于是 `circuit_registry` 和 `map_stage0_records` 各维护一份代际表 —— 新的漂移点，
   而且方向与"暂停身份扩展、回算法主线"的裁决相反。

所以分层这样切：**生产只管写**（新记录必须标当前代际，`_check_execution()` 维持原样），
**审计只管读**（`--recompute-runids` 显式声明"按记录自报代际重算"）。
证据一样拿到，生产语义不动，`_check_execution()` 维持原样（新记录必须标当前代际）。

```
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --recompute-runids
按记录自报代际复算 run_id：通过 239 / 不通过 0 / 未发号跳过 0 / 无代际标记无法复算 0
```

对应测试：**D6**（239/239 可复算）、**D6b**（改掉一个 `run_id` ⇒ 必须报不通过，
防"复算审计恒真"）。

## 5. 处置

### 已做：M1 + M2

| 项 | 改动 |
|---|---|
| M1 | `test_stage0_freeze.py` 三处 `probs, _ = verify_frozen(...)`；同时补 **D4c**（代际改成另一**已登记**代际 ⇒ 完整性通过、不阻断） |
| M2 | `verify_frozen` 第 3 层从"等于当前注册表"改成"**必须是已登记代际**"（`_KNOWN_GENERATIONS` 三行）；代际差归 `crossgen`，未知代际仍是 problem；**现役代际没登记也报 problem**（逼升版时补一行，而不是静默变绿） |
| M2 | 第 1 层 MANIFEST 字节差异按"抹三项后逐字段比"归入 `crossgen`；`records.jsonl` 逐行比对改严（按 `\n` 精确切行、不过滤空行；`read_bytes().decode()` 而非 `read_text()`，防通用换行吞掉 CRLF 篡改） |

结果：

```
registry: 现役 0.8.0 aeb6aca1c13dd2ff / 冻结 0.7.0 b2675c8255bea716（已登记代际）
代际差异 3 处（不计为问题，快照属旧代际）：
  ~ MANIFEST.json: …但除代际三项外逐字段相同
  ~ records.jsonl: …但内容逐行等价（差异仅 registry_version/run_id）
  ~ MANIFEST 代际 0.7.0/b2675c82… ≠ 现役 0.8.0/aeb6aca1…（已登记代际，属快照的历史归属）
完整性通过：239 条，239 独立（严格口径 237），237 个唯一 run_id；3 项代际差异已单列
rc=0
```

三条负对照（确认真篡改仍**只有**它红）：

| 篡改 | rc | 是否命中 |
|---|---|---|
| 往 `records.jsonl` 尾追加换行 | 2 | ✔ |
| 改一条记录的 `rms_rel` | 2 | ✔ |
| 把 `coverage.json` 转成 CRLF | 2 | ✔ |

测试：`test_stage0_freeze.py` **12/12**、`test_circuit_registry.py` **76/76**。

第一条负对照与第三条在改前是**漏的**：旧代码用 `splitlines()` + `if ln.strip()`
过滤空行，追加换行被读成"逐行等价"；旧代码用 `read_text()` 读冻结副本，
CRLF 被通用换行转成 LF。D5c 当时显示 `[ok]` 是**假通过**——它 `rc==2` 靠的是
另外 3 处永久报警，不是因为抓到了篡改。

### 挂账：M3 / M4

| 项 | 内容 | 现在怎么看 |
|---|---|---|
| **M3** | `evidence/stage0_records/GENERATION.json` 显式声明 `produced_under`，第 3 层改比"MANIFEST ↔ GENERATION 自洽" | 仍值得做：它关的是 §6 那个"换成另一已登记代际"的口子。但要动冻结目录，单独立项 |
| **M4** | `_run_id_from` 的 software 载荷只取 `{producer, code_version}`，另设 `IDENTITY_SCHEMA_VERSION`（照 `exec_identity.RUN_ID_CONFIG_KEYS_VERSION`） | **降级**：不再是止血项。原先说"只要 `registry_version` 还在哈希里，每次升版都要全量重算"——重算的是**重新生成的快照**，而冻结副本按自报代际 239/239 可复算（§4.1），所以"漂移"只发生在我们本来就要重新生成的那一份上，crossgen 已经把它标成代际差异。M4 现在只是"语义更干净"，无紧迫性 |

**不做：重冻结 MANIFEST。** 它能把红清零，但等于把 0.8.0 追认给 239 条 2026-11-14
的记录——`REGISTRY_0_8_0.md` 里"不重冻结"这条结论**仍然成立**。

## 6. 残留口子（必须写明）

| 口子 | 后果 | 关闭条件 |
|---|---|---|
| 把 `MANIFEST` 的代际改成**另一个已登记**代际（如 0.7.0/b2675c82 → 0.8.0/aeb6aca1） | 抓不到，且会把快照"洗成"现役代际 | M3（需要 MANIFEST 之外有锚点）。D4c 已把该口子钉进测试 |
| 改 `MANIFEST.registry_version`/`registry_digest` **同时**重算 `files[records].sha256` | 若目标是"抹掉代际痕迹"，`files` 会被一起改；但 `records.jsonl` 的内容比对是逐行抹平代际后比的，改不动内容 | 同上 |
| 冻结快照的 239 条 run_id | ~~只能逐字节相信~~ → 已澄清：按**自报代际** 239/239 可复算（§4.1）。只有"用当前代际去复核历史记录"才会不符，而那本来就是错误的操作 | 已关闭（`--recompute-runids` + D6/D6b） |

## 7. 复现

```
# 现状（rc 0）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --check
# 冻结快照自检（12/12）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/test_stage0_freeze.py
# 按记录自报代际复算 run_id（239/239，只读）
D:/autoeis_env/Scripts/python.exe docs/round_2026-11-13/map_stage0_records.py --recompute-runids
# 对照：用**当前**代际去复核历史记录 ⇒ 必然不符（这是预期，不是损坏；
# 正确的复核入口是上面的 --recompute-runids，它按记录自报代际算）
D:/autoeis_env/Scripts/python.exe -c "
import json, circuit_registry as R
rec=json.loads(open('evidence/stage0_records/records.jsonl',encoding='utf-8').readline())
o=rec['observation']
try: R._check_execution(o['execution'], candidate_id=o['candidate_id'], budget_params=o.get('budget_params'))
except ValueError as e: print(e)
"
```
