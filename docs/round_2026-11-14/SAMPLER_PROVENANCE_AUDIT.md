# ③ 采样参数来源审计（阶段 A：只读调用链）

> **本轮只做审计，不写 `sampler_provenance.py`，不改任何生产代码，不动 `exec_identity.py`。**
> 目标是回答一句话：**在采样调用点读到的五个值，是不是这一次拟合真实生效的值，以及
> 「读到了」能不能证明「它从哪来」。**

审计对象：`n_warmup` / `n_samples` / `num_chains` / `target_accept` / `max_tree_depth`。
边界：只覆盖**值的传递路径**，不涉及先验/似然/初始化策略，也不启动批量拟合。

---

## 1. 方法

| 手段 | 做了什么 | 证据强度 |
|---|---|---|
| 静态阅读 | 读完**全部** MCMC 构造点与**全部**到达它们的入口路径（含 `evidence/`），逐跳记行号 | 结论可复核，但属读码 |
| 机械核对 | 包一层 `MCMC.__init__` / `MCMC.run`（不做任何替换），用 `8/8`、`4/4` 的微预算跑 3 个工况，比对「声明值 → 构造实参 → 调用点属性」 | **实测**（3 工况 × 5 参数） |
| 未做 | 没有跑 `bm_real.fit_one` 全链路（它要跑 DRT/KK/诊断，成本与结论无关）；A 路径的 `kw` 构造与传递是**读码结论** | — |

`identity_contract_version` 仍 `1.4`，`exec_identity.py` 零改动，生产零改动。

---

## 2. 构造点：不是两处，是三处

| # | 位置 | 生产者 | 是否生产路径 |
|---|---|---|---|
| K1 | `bm_core.py:606` | `bm_core.fit()` | 是（voigt 族 / A 路径） |
| K2 | `bm_top.py:1001` | `bm_top.fit_top()` | 是（nested 族 / B 路径） |
| K3 | `evidence/nested_topology/m35_cond_n2.py:116` | 直接构造 | **否**（证据脚本，绕过两个 fit 函数） |

K3 的存在决定了一条结论：**任何「只在 `fit`/`fit_top` 里挂钩」的采集都会漏掉 K3**。
本轮不处理它（属 `evidence/`，不在 runner 范围），但也不能再说"只有两个构造点"。

---

## 3. 入口路径：三条生产路径，语义各不相同

| 路径 | 入口 | 到 K1/K2 的链 |
|---|---|---|
| **A** | `bm_real.run()`（`bm_real.py:2657`） | `kw=dict(KW)`(:2765) → `_fit_ks`(:2809) → `fit_one(..., kw=kw)` → `B.fit(**kw)`(:963) → **K1** |
| **B** | `nested_fit.run()`（`nested_fit.py:319`） | `budget_kw(budget)`(:107-112) → `T.fit_top(..., **kw)`(:339) → **K2**；`--with-voigt` 再走 `voigt_reference` → `C.fit(**kw)`(:216) → **K1** |
| **C** | `bm_select.select_across_families()`（`:1083`） | `fit_nested(kw=...)`(:388) → **K2**；`fit_voigt(kw=...)`(:345) → `R.fit_one` → **K1**；重跑走 `_refit(kw=rkw)`(:1084) → **K1/K2** |

另有**非生产**的两组调用（不进本轮判定，但要列出来避免误推广）：
`bm_experiments.py:282/454/487/555`、`bm_check.py:227`（自检），
以及 `evidence/` 下多处直接读 `S.BUDGETS`。

---

## 4. 五参数来源表

### 4.1 路径 A —— `bm_real.run()` / `bmdrt`

| 参数 | 用户入口 | 默认来源 | 中间转换 | 最终传给 |
|---|---|---|---|---|
| `n_warmup` | `--quick`（`:2771`） | `KW`（`:97`）=**3000** | `--quick` ⇒ 800；`--quick` 之外无路径 | K1 `:606` |
| `n_samples` | 同上 | `KW`=**1000** | `--quick` ⇒ 400 | K1 |
| `num_chains` | 同上 | `KW`=**3** | `--quick` ⇒ 2 | K1 |
| `target_accept` | `--ta`（`:2774`） | `KW`=**0.9** | `float(ta)`；`--quick` **不动它** | K1（经 `NUTS`） |
| `max_tree_depth` | **无入口** | `KW`=**13** | **无**（`--quick` 也不动） | K1（经 `NUTS`） |

A 路径有两个不对称：`max_tree_depth` 根本没有命令行入口；`--quick` 只改三个数，
`target_accept` 与 `max_tree_depth` 保持 `KW` 的值。

### 4.2 路径 B —— `nested_fit.run()`

| 参数 | 用户入口 | 默认来源 | 中间转换 | 最终传给 |
|---|---|---|---|---|
| `n_warmup` | `--budget` | `standard` ⇒ `VALIDATION_STANDARD`(:93)=**2000**；其它档 ⇒ `BUDGETS[name]`(:112) | `FIT_KEYS` 过滤 + `if k in b` | K2 |
| `n_samples` | 同上 | **1000**（standard）/ `BUDGETS` 档值 | 同上 | K2 |
| `num_chains` | 同上 | **2** | 同上 | K2 |
| `target_accept` | `--target-accept`(:336) | **0.98**（standard，`VALIDATION_STANDARD`）/ 档值 **0.9** | `float(...)`，非 `None` 才覆盖 | K2（经 `NUTS`） |
| `max_tree_depth` | 无入口 | **10** | — | K2 |

`budget_kw` 的 `if k in b` 是**静默降级通道**：档里缺某个键时不会报错，会落到
`fit_top` 的函数默认值（`num_chains` 的默认是 **4**）——见 §6 F-6。

### 4.3 路径 C —— `bm_select`

| 参数 | 用户入口 | 默认来源 | 中间转换 | 最终传给 |
|---|---|---|---|---|
| 五项全部 | `--budget` / `--retry-budget` / `kw=` | `QUICK`(:127)=**800/400/2/0.9/10**；重跑档 `BUDGETS["standard"]`=**2000/1000/2/0.9/10** | `fit_nested` 逐键 `int()`/`float()`(:390-393)；`fit_voigt` 整包 `**kw` | K2 / K1 |

C 路径与 B 路径对"缺键"的反应**相反**：`fit_nested` 用 `kw["n_warmup"]` 直接下标
⇒ `KeyError`；`budget_kw` 用 `if k in b` ⇒ 静默省略。同一个 `BUDGETS` 字典，两种语义。

---

## 5. 机械核对：值没被改写，但读的是**两个对象**

包一层构造与运行，用微预算跑（`8/8`、`4/4`），只观测不改值：

| 工况 | 声明值 | 构造实参（`MCMC.__init__`） | 调用点属性（`MCMC.run` 入口） | 五项相等 |
|---|---|---|---|---|
| `bm_core.fit` 单链 | 8/8/1/0.97/7 | `num_warmup=8, num_samples=8, num_chains=1, progress_bar=False` | 8/8/1, **sequential? 否 → `parallel`**, 0.97, 7 | **是** |
| `bm_top.fit_top` 单链 | 8/8/1/0.97/7 | 同上 | 同上 | **是** |
| `bm_top.fit_top` 三链 | 4/4/3/0.97/7 | `num_warmup=4, num_samples=4, num_chains=3` | 4/4/3, **`chain_method="sequential"`**, 0.97, 7 | **是** |

三个关键事实：

1. **值层面 `Resolved == Effective == Observed`**（15/15 全等）。没有任何一层对整数/浮点做
   改写，`numpyro 0.19.0` 的 `MCMC.run` 也不改 `num_warmup/num_samples/num_chains`
   （读过源码：只在 `_set_collection_params` 里读它们）。
2. **五个值分布在两个对象上**：`num_warmup/num_samples/num_chains` 在 `MCMC` 上，
   `target_accept/max_tree_depth` 在**内核**上（`mcmc.sampler._target_accept_prob`
   / `_max_tree_depth`）。"在采样入口读五个"实际是**跨对象读**。
3. **`chain_method` 被设备数静默解析**：`num_chains=3` + `local_device_count()=1`
   ⇒ `parallel` 被改成 `sequential`，并 `warnings.warn`。而 **A 路径的 `KW`
   是 3 条链，`bm_real.py:90` 有 `warnings.filterwarnings("ignore")`** ——
   这条警告在生产路径上被吞掉，且没有任何字段记录它。

`_max_tree_depth` 还有一个陷阱：`HMC.__init__` 里它被硬编码成 `10`，
只有 `NUTS.__init__` 会用实参覆盖。若将来换成非 NUTS 内核，"读 `sampler._max_tree_depth`"
会读到 10 而不是用户设的值——**属性路径是语义的一部分**，不能当通用读法。

---

## 6. 审计发现

每条都标了「今天会不会触发」。**不触发的也会写，因为它们是同族缺陷的入口**。

### F-1 五个值跨两个对象，读取位置不同构

- 证据：`mcmc.num_warmup/num_samples/num_chains` vs `mcmc.sampler._target_accept_prob/_max_tree_depth`。
- 后果：任何"读 `mcmc` 上的五个属性"的采集，会静默少两项（或读到内核的无关默认值）。
- 今天：`_grab` 已按两处分别读（`snapshot_probe.py:107/110`），`missing_attrs` 为空。
- 建议：`observed` 必须带**属性路径**，不能只带数值。

### F-2 值没有被任何一层改写

- 证据：§5 表，15/15 全等；`MCMC.run` 源码不写这三个属性。
- 后果：**这才是 ③ 的好消息**——"读到了 = 生效了"在**值**层面成立。
- 边界：只对"值"成立。执行方式（F-3）与诊断字段（F-8）不成立。

### F-3 `chain_method` / `progress_bar` 被静默解析，且不在五参数里

- 证据：`MCMC.__init__` 里 `if chain_method == "parallel" and local_device_count() < num_chains: chain_method = "sequential"`；`if "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ: self.progress_bar = False`。
- A 路径 3 条链、1 台设备 ⇒ **每次都是 sequential**，警告被 `bm_real.py:90` 的
  `filterwarnings("ignore")` 吞掉；记录里没有任何字段。
- 后果：同一个 `run_id` 可以对应两种执行方式。**没有测过两者是否逐位同数，此处不作断言**；
  但"没记录"是确定的。
- 今天：**会触发**（A 路径 `num_chains=3`）。不影响五参数的值，影响"执行条件是否等价"的可核验性。

### F-4 A 路径的 `sampling` 块自称"实际生效"，实为 `kw`

- 证据：`bm_real.py:1169` 注释写「实际生效的采样配置（不是命令行请求值）」，
  而 `:1172` 的取值是 `kw.get("n_warmup", 0)` / `float(kw.get("target_accept", nan))`。
  `kw` 是**解析后的声明值**，不是从 `mc` 读的；同函数作用域里 `mc` 就在手边。
- 后果：值与真值**今天恰好相同**（F-2），所以是"对的结论 + 错的证据链"。
  一旦中间层改写而 `kw` 不变（F-3 就是这么发生的），这个字段会说谎且不报错。
- 次要缺陷：`.get(k, 0)` / `float("nan")` 把"不知道"写成 `0`/`nan`。今天不会触发
  （所有调用方都给全 5 键），但它与 0.2.0 的 `hint or "uniform"` 是**同一类**：
  用哨兵值冒充观测值。

### F-5 同名档，不同数值：三个 `standard`、两个 `quick`

| 档名 | 路径 | 五元组 |
|---|---|---|
| `standard` | A（`KW` 默认） | 3000 / 1000 / **3** / 0.9 / **13** |
| `standard` | B（`VALIDATION_STANDARD`） | 2000 / 1000 / 2 / **0.98** / 10 |
| `standard` | C（`BUDGETS["standard"]`） | 2000 / 1000 / 2 / 0.9 / 10 |
| `quick` | A（`:2771`） | 800 / 400 / 2 / 0.9 / **13** |
| `quick` | B / C（`BUDGETS["quick"]`） | 800 / 400 / 2 / 0.9 / **10** |

`--quick` 只覆盖三个键，所以**A 的 quick 仍是 depth 13**。C 路径的注释
（`circuit_registry.py:1065`）只警告了 `target_accept` 0.98 vs 0.90 这一处；
`max_tree_depth` 13 vs 10 还没被写过。`bm_real.py:2771` 的 800/400/2 是**第二份字面量**，
与 `bm_select.QUICK` 不共享来源——改一处不会带动另一处。

### F-6 缺键时两条路径反应相反（静默注入默认值 vs 立刻报错）

- B：`nested_fit.budget_kw` 的 `{k: b[k] for k in FIT_KEYS if k in b}`（`:112`）
  ⇒ 档里缺 `num_chains` 时**静默省略**，下游 `fit_top` 用默认 **4**。
- C：`bm_select.fit_nested` 的 `kw["num_chains"]`（`:391`）⇒ `KeyError`。
- 今天：`BUDGETS` 三个档都齐，**不触发**；但这是"改一个常量就静默换档"的入口。

### F-7 同一个概念的默认值有三个：1 / 4 / 3

`bm_core.fit` 的 `num_chains=1`、`bm_top.fit_top` 的 `num_chains=4`、`bm_real.KW` 的 3。
省略 `num_chains` 的调用方会按"调了哪个函数"拿到不同链数。与 F-6 联动后，
"缺键"的后果是**4 条链**，而 A 路径的默认是 3、C 路径的档值是 2。
→ `origin=default` 必须能指到**哪个函数的默认**，否则 `defaulted` 只是一个好看的词。

### F-8 五参数之外还有三个被静默解析的执行条件

| 项 | 谁决定的 | 记录了吗 |
|---|---|---|
| `chain_method` | `local_device_count()`（`MCMC.__init__`） | ✗ |
| `progress_bar` | `os.environ` 的 `CI`/`PYTEST_XDIST_WORKER` | ✗ |
| `extra_fields` | **两族不同**：`fit_top` 硬编码 `("diverging",)`；voigt 走 `KW`/`BUDGETS` 的 `("num_steps",)`；B 路径 `budget_kw` 把它过滤掉了 ⇒ `voigt_reference` 里 K1 拿到 `()` | 部分（间接体现在「有没有树深统计」） |

第三条是已处理的不对称（`bm_select` 对 nested 的 `tree_cap` 记 `not_applicable`），
但它证明了一件事：**同一组对照里「档名相同」≠「执行条件相同」**。
所以只对五参数做 provenance，会给出一个**看起来完整**的 `complete`，
而这正是 §3.3 的 `unknown_keys` 机制抓不到的东西——它们不在任何接口签名里。

### F-9 `n_draw` 是从声明值推的，不是从采样器读的

`bm_select.py:1085` 的 `n_draw = int(kw["num_chains"]) * int(kw["n_samples"])`、
`bm_real.py:1143` 的 `n_draw=kw["num_chains"] * kw["n_samples"]`。
只有在 F-2 成立时它才等于真实抽样数。今天成立；但它把 F-6 那种"静默换档"
直接传播成**错误的样本量**，而这会影响 R̂/ESS 的解释。

---

## 7. 与契约 §3.3 的关系：③ 不是新造词，是给 F4 补证据

契约里**已经**写好了这件事的语义，审计要做的是验证它能不能落地：

- §3.3.1 表：`effective` = `{value, origin, effect}`，**在采样调用点捕获**，
  `origin ∈ {cli, config_file, default, tier, derived}`；
- §3.3.2 末段已经给了界线：**`origin=default` 的值是生效值，可以 `recorded`；
  区别在捕获位置**——在调用点把默认值物化成实参并记下来算"记录实际值"，
  事后从代码里读默认值算"猜"，不行；
- **F4** 就是"档名/默认值冒充实际值"，对策写着 `config.effective` 在采样调用点捕获。

所以本轮审计的净结论是：**F4 的对策在真实路径上可行**（F-2 证明值不被改写），
但**只做五个值还不够**（F-1/F-3/F-8）——"完整"的定义要包含读取位置与
被静默解析的执行条件，否则 `complete` 会变成一句空话。

用户提的四个概念与契约字段的映射（**不要新造第四套词**）：

| 用户的说法 | 本项目已有的字段 | 备注 |
|---|---|---|
| Declared | `budget_params` 的输入侧 / `origin ∈ {cli, config_file, tier}` | 档名只是 `tier` 标签 |
| Resolved | `budget_params`（数值）+ `origin` | 契约要求这是**权威**字段 |
| Effective | 同上 + `origin=default` 时表示"调用点物化的默认值" | 契约明确它是生效值 |
| Observed | **尚无字段** | ③ 要新增的唯一东西 |

---

## 8. 对纯函数层的接口建议（**待裁，本轮未实现**）

新模块 `sampler_provenance.py`，纯函数、不 import git / 不读环境 / 不碰磁盘；
`exec_identity.py` 一行不改，本模块只产出 `budget_params` 与一份核验报告。

**逐参数记录**（五项各一条，不合并）：

```
{ name, resolved: {value, origin}, observed: {value, attr_path},
  status, why }
```

- `attr_path` 必填，例如 `"mcmc.num_warmup"` / `"mcmc.sampler._target_accept_prob"`
  ——F-1 说明位置是语义的一部分；
- `resolved.value` 必须是**归一化后的数值**（不是 `"standard"`、不是 `None`）。

**状态判定（顺序即优先级）**：

| 序 | 条件 | 状态 |
|---|---|---|
| 1 | `observed` 不存在（属性路径失效 / 未采） | `incomplete`（**不许**用 `resolved` 反推） |
| 2 | `observed ≠ resolved` | `mismatch` |
| 3 | `resolved` 缺失（只知道档名） | `derived` |
| 4 | `resolved.origin == "default"` | `defaulted`（**不是** `derived`） |
| 5 | 其余（`cli`/`config_file`/`tier` 且一致） | `complete` |

**聚合**：一条采样事件的状态取**五项中最弱**（`incomplete > mismatch > derived >
defaulted > complete`），并**必须**同时保留五项明细——聚合值只用于筛选，不用于下结论。

**必须在同一份记录里单列的三个执行条件**（F-3/F-8，不参与五项状态，但缺了就降级）：

```
{ chain_method, local_device_count, progress_bar, extra_fields }
```

`chain_method != "parallel"` 且 `num_chains > 1` 时记 `sequentially_resolved`
（原因码，不改五项状态），因为**值相同、执行不同**。

**建议新增第六个状态 `unavailable`**（属性路径在**本 numpyro 版本**不存在）：
它与 `incomplete`（采集没跑到）原因不同，混在一起会让"库改名"看起来像"采集中断"。
是否接受，请你裁——我不擅自加。

**反例矩阵（建议 10 格，实现前先定）**：

| # | 场景 | 期望 |
|---|---|---|
| 1 | 五项全显式 | `complete` ×5 |
| 2 | 五项全走默认 | `defaulted` ×5（**且不等于 `derived`**） |
| 3 | 部分显式 | 逐项状态不同，聚合取最弱 |
| 4 | 中间入口改值（`--quick`） | 记**改后**值，`origin≠default` |
| 5 | 人为让传递路径改一个值 | `mismatch`（只有这一项） |
| 6 | `observed` 缺失 | `incomplete`，且**不出现** `resolved` 冒充 |
| 7 | 属性路径不存在（模拟改名） | `unavailable` / `incomplete`（按裁决） |
| 8 | 同工作流两次不同预算 | 两条独立记录，按 C-3 `event_id` 关联，互不覆盖 |
| 9 | `num_chains=3` + 1 设备 | 五项仍 `complete`，**但**附 `sequentially_resolved` |
| 10 | 档里缺键 ⇒ 静默用 `fit_top` 默认 | `defaulted` 且 `origin` 能指到**哪个函数**的默认（F-7） |

第 5 条是核心：**改了传递值必须被识别**，否则这套东西和 F-4 一样只是"看起来对"。

---

## 9. 明确不做

- 不改 `exec_identity.py`，不升 `identity_contract_version`；
- 不裁 `exec_identity.py` 的 `reference_only` 分类（仍挂账）；
- 不给补丁包加签名/真实性（完整性 ≠ 真实性的边界不动）；
- 不把补丁包扩到整棵 Git 工作树（仍只覆盖声明集）；
- 不写 runner，不接生产，不跑批量拟合；
- 不动 K3（`evidence/m35_cond_n2.py`）——列出来只是为了不自称"两处构造点"。

---

## 10. 复现

```bash
# 机械核对（微预算，不产出任何产物文件）
# 见本文件 §5 的 3 个工况；包 MCMC.__init__ / MCMC.run 只观测不改值，
# 谱用 archive/2EMC1-1_0005mA.csv（56 点）
```

读码入口（按此顺序可复核全文）：
`bm_real.py:97/2657/2765-2775/2809/963/1172` → `bm_core.py:560-606` →
`nested_fit.py:93/101-112/319/336/339` → `bm_top.py:913-1001` →
`bm_select.py:127/331-347/387-395/1083-1085/1131-1135` →
`circuit_registry.py:182-186`。
