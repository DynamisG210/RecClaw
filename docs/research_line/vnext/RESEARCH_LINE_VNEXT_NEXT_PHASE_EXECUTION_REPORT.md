# RecClaw Research Line vNext 下一阶段执行报告

- 状态：`EXECUTION_REFERENCE`
- 证据基线：`63b93013f04744d7a3b5b97fbd7c5cfba0633956`
- 适用范围：Research Line vNext 的远端集成、Q5 测量设计校正与 DEVELOPMENT_ONLY 前瞻 pilot
- 明确不包含：held-out、正式科学结论、发布、模型优越性声明、大规模 campaign

## 1. 执行摘要

Research Line vNext 已经达到“停止补架构”的工程节点。P0/R1/R2、Q0/Q0R2、F1、Q1、Q2、Q3、三轮 multi-round soak、Q4 和最终独立审计均已形成真实运行证据。当前缺口不再是模块缺失，而是比较设计仍然混合了实现随机性、IDEA 阶段同分选择、构造失败与机制不可识别。

下一阶段只做两条主线：

1. **远端集成**：保留完整证据链，将 vNext 通过无改写、无强推的显式 merge 纳入 `feat/research_line`。
2. **Q5 小规模前瞻验证**：先校正测量单位和消费者合同，再运行 Idea/Feasibility 与 Experiment/Utility 两段 pilot。

本报告的核心决策是：

- 不增加新的 Producer、Agent role、Meta head 或平行 schema。
- 保留 Q3 的 feasibility、mechanism-information、effect 三条 authority lane。
- 不用继续堆轮数弥补 Q4 的测量混杂。
- 在 Q5 主比较中，同一 OpenSpec 只实现一次；同一 package × seed 只产生一次 outcome。
- 只有 `NESTED_MECHANISM` 可进入机制归因；其他可执行候选仅进入 effect lane。
- Q5 完成前不进入 held-out 或正式大规模实验。

### 1.1 对 Pro 建议的吸收与修正

| 建议 | 决定 | 本报告中的落实 |
|---|---|---|
| 停止继续补架构 | 接受 | 后续只做集成、测量合同和有真实消费者的通用修复 |
| 同一 spec 使用共享实现 | 接受，并设为 Q5 主比较的基本实验单位 | §5.1、§6、§7 |
| 修复 Outcome-aware KeyError | 有条件接受 | 先区分候选缺陷与 shared consumer 缺陷；禁止预设根因或修旧候选 |
| 区分 nested mechanism 与 non-nested effect candidate | 接受 | §5.3 的 realization typing 与 off-equivalence gate |
| feasibility 分阶段建模 | 接受 | 在既有 feasibility authority lane 内做 stage-conditional projection，不增加平行 head |
| 稀疏 head 向 F1/static prior 收缩 | 接受 | §5.5；同时补充真实证据暴露出的 tie contract |
| 先 Q5-A，再 Q5-B，再决定正式实验 | 接受 | §6、§7、§8 |
| 使用独立 review 分支等待队友审查 | 不采用为阻塞门禁 | 以现有 independent final audit + merge 后 integration gate 替代 |
| 将 Q4 OA 失败解释为学习偏差 | 不接受 | 冻结记录显示 IDEA 四候选精确同分；当前更像信息不足和平分处理问题 |

## 2. 当前事实基线

### 2.1 已闭合的工程链

| 阶段 | 状态 | 当前能证明什么 | 不能证明什么 |
|---|---|---|---|
| P0 / R1 / R2 | Accepted | 冻结上下文、真实 Provider/Research contract 和早期运行链存在 | 后续策略有效性 |
| Q0 / Q0R2 | Accepted | 资源遥测、删失、非空准入和 deferred 语义闭合 | 候选效果 |
| F1 | Resource-compatible pass | 10 GB GPU 上的等价实现可真实完成；真实负向 Episode 被保存 | 原 sealed 字节候选可运行；科学优越 |
| Q1 | Pass | OpenSpec、Resolver、Implementer、Qualifier、resource admission 真实闭合 | Idea 质量优于 baseline |
| Q2 | Negative evidence accepted | 结构参与存在，负机制证据可进入后续学习 | 当前候选机制已识别 |
| Q3 | DEVELOPMENT_ONLY pass | 三条 authority lane、replay、activation 和真实 next-round consumer 闭合 | Outcome-aware 已经优于 F1/static |
| Multi-round soak | DEVELOPMENT_ONLY pass | 3 轮真实链、resume、failure isolation、跨轮 activation 闭合 | 多轮科学增益 |
| Q4 | DEVELOPMENT_ONLY pass | 同一 frozen spec pool 的三策略前瞻 pilot 真实执行 | 策略排名、机制结论、held-out 泛化 |
| Final audit | `PASS_WITH_FINDINGS` | lineage、hash、tests、missingness、0 held-out/retry 和负证据一致 | 正式科学结论 |

最终审计报告：[`Q4_FINAL_INDEPENDENT_AUDIT_REPORT.md`](Q4_FINAL_INDEPENDENT_AUDIT_REPORT.md)。

### 2.2 Q4 真实结果

| Policy | Episode | Parent-relative development effect | Mechanism state | 解释 |
|---|---:|---:|---|---|
| STATIC | 完整 | `-0.0064` | `NON_IDENTIFIABLE` | 单 seed development 负信号 |
| CURRENT_F1 | 完整 | `-0.0072` | `NON_IDENTIFIABLE` | 单 seed development 负信号 |
| OUTCOME_AWARE | Missing | 不可计算 | `NOT_ASSESSED` | Mechanical Qualifier `INTERFACE/KeyError`，上游失败后合法 deferred |

这些结果不支持 Outcome-aware 优越性。STATIC 与 CURRENT_F1 选择了同一个 spec，但各自独立实现，得到不同 source tree，因此 `-0.0064` 与 `-0.0072` 同时包含实现 realization variance，不能解释成政策因果差异。

### 2.3 Outcome-aware 选择的补充诊断

冻结文件：[`FROZEN_SELECTION_BEFORE_PHYSICAL.json`](../../../results/research_line/q4_prospective_policy_comparison_20260803_01/arms/03_outcome_aware/FROZEN_SELECTION_BEFORE_PHYSICAL.json)。

本轮 IDEA acquisition 不是随机探索：

- `exploration_probability = 0.15`
- `random_draw = 0.471136892567`
- `exploration_selected = false`

但四个候选的关键预测完全相同：

- feasibility posterior：`0.142857142857`
- mechanism-information posterior：`0.361111111111`
- selection score：`0.051587301587`

当前公式为：

```text
scientific_falsifiability
  × mechanism_information_probability
  × feasibility_probability
```

因此该次选择不是“模型明显偏好错误候选”，而是四个候选在 IDEA lane 没有区分度，确定性 exploitation tie-break 将 `0.8875` 的选择概率集中到一个平分候选。随后发生的 KeyError 不能反推为 outcome-aware 学习方向错误。

Q5 必须把这一现象显式化：

- 平分候选不得因输入顺序获得绝大部分 exploitation probability。
- 平分时应均匀分配 exploitation mass，或按冻结的 F1 prior 处理。
- `selected_by` 必须记录 `UNIFORM_TIE`、`F1_PRIOR_TIEBREAK`、`EXPLORATION` 或 `LEARNED_SCORE`。
- 如果 learned heads 缺乏支持，必须按样本支持度向冻结 prior 收缩。

## 3. 下一阶段的非目标

以下内容不是下一阶段任务：

- 增加 Agent role、Producer 或 Research lane。
- 新建与 OpenSpec、TypedResearchEpisode、Q3 authority 并行的 schema。
- 调整 candidate、seed、dataset、batch、provider model、endpoint 或 outcome 解释以追求 PASS。
- 用 retry、补样、筛成功或候选特定手修修复失败。
- 扩展到 50 轮 campaign 或直接运行 held-out。
- 把 Q4 的工程闭合写成 Outcome-aware 优越性。
- 为 receipt、hash 或形式完整性创建没有真实消费者的新层。

## 4. Git 交付与集成方案

### 4.1 当前分叉

截至本报告编写时：

- 远端主支 `origin/feat/research_line`：`6581a95a605deed34fc72ee4cfa3ecf33bf0b844`
- 最终审计提交：`63b93013f04744d7a3b5b97fbd7c5cfba0633956`
- merge base：`38995292ee472ae4d2b04fd4a5088b26199bd580`
- 主支独有 4 个提交，主要包含 F1 resource-compatible closure
- vNext 独有 7 个提交，包含 Q1、Q2、Q3、soak、Q4 和 final audit
- `git merge-tree --write-tree` 已成功生成无冲突合成树 `c491445692a3758b154daa02731763f6e77f8817`

结论：**结构上可合并，但不是 fast-forward，必须经过合并后验证。**

### 4.2 分支角色

- `feat/research_line_vnext`：vNext 交付线，保留 final audit 及本执行报告；不 rebase、不 squash、不改写历史。
- `feat/research_line`：正式工程主支，保留原有 F1 closure，并显式 no-ff merge vNext。

不另设远端 review 分支。队友无法 review 不构成阻塞；替代门禁是既有 independent final audit 加一次独立 integration gate。

### 4.3 合并顺序

1. 将当前 vNext 交付线推到远端 `feat/research_line_vnext`。
2. 从最新 `origin/feat/research_line` 创建本地临时 integration branch。
3. 使用 `--no-ff` 合并 vNext 精确提交，保留两条 first-parent 证据链。
4. 不修改任何 sealed Q1–Q4 evidence；冲突只允许在非 sealed 集成表面解决。
5. 完成下列 integration gate 后，才将 merge commit 推到 `feat/research_line`。

### 4.4 Integration gate

必须全部满足：

- `git diff --check` 通过。
- 合并结果工作树 clean。
- final audit focused suite：67 项通过。
- Q4 package `SHA256SUMS`：182/182 通过。
- F1 resource-compatible targeted/adjacent tests 通过。
- first-parent lineage 同时保留 F1 closure 和 Q1–Q4/audit chain。
- secret scan、held-out scan、retry ledger 和 sealed-path scan 无新增问题。
- clean runtime 完成 import、一个真实 Qualifier 和一个 bounded resource probe。
- 不重跑或覆盖 Q4 outcome；集成 probe 不进入 effect authority。

任何一项失败时不更新主支；保留 `feat/research_line_vnext` 作为可审计交付线。

## 5. Q5 前置校正

Q5 前置校正是测量合同修正，不是新架构开发。

### 5.1 Shared realization：消除实现随机性混杂

主比较采用以下实验单位：

1. 所有 policy 从同一个逐字节 frozen OpenSpec pool 选择。
2. 取各 policy selection 的并集。
3. 每个唯一 OpenSpec 只运行一次 Resolver、Implementer、materialize 和 Qualifier。
4. 每个唯一 spec 生成一个 frozen source tree、package、qualification receipt 和 resource receipt。
5. 所有 policy 在同一 executable package pool 上进行后续排序和归因。
6. 每个 `package × development seed` 只运行一次 matched outcome；若多个 policy 选择同一 package，共享该物理 outcome 和 cost attribution。

这套设计用于估计 selection policy 的价值。独立实现方差如需研究，应作为后续 secondary end-to-end experiment，而不是混入主 policy comparison。

### 5.2 Outcome-aware KeyError 根因处理

旧失败只用于诊断，不允许修补旧候选：

1. 从 sealed package、source tree、Qualifier input 和 traceback 重建失败路径。
2. 判断错误属于：
   - 候选 spec/实现本身违反通用合同；或
   - shared Implementer/Qualifier consumer 的通用缺陷。
3. 如果是候选缺陷：保留负证据，不修改候选，不重发 Provider。
4. 如果是通用 consumer 缺陷：只做一次最小通用修复，并建立去标识回归 fixture。
5. 修复必须在一个 fresh、非旧失败候选的 OpenSpec 上验证；旧候选仅允许 post-hoc diagnostic，不获得新 outcome authority。

若同一根因的通用修复失败两次，停止叠补丁并重新审视 consumer contract。

### 5.3 Realization typing：恢复机制可识别性

所有 executable realization 必须在 outcome 前分类：

#### `NESTED_MECHANISM`

必须具备冻结的 `mechanism_strength` 或等价 off switch，并证明 off 状态回到同一个 parent：

- loss 等价
- predict 等价
- full-sort predict 等价
- 所有相关梯度等价
- checkpoint/load 后等价
- evaluator、dataset、seed、optimizer 和 batch 不变

只有该类型可以进入 mechanism-information ablation 和 mechanism attribution。

#### `EFFECT_ONLY_NON_NESTED`

允许进入真实 matched development effect 比较，但：

- mechanism state 固定为 `NOT_ASSESSED`
- 不把无法构造 mechanism-off 解释为负机制证据
- 不计入 mechanism-identifiable Episode

### 5.4 Stage-conditional feasibility

保留现有 feasibility authority lane，但将预测和校准按真实阶段分解：

```text
P(full_episode)
  = P(materialize)
  × P(construct | materialized)
  × P(qualify | constructed)
  × P(resource_admitted | qualified)
  × P(full_episode | admitted)
```

每个阶段只消费在该阶段决策前可见的特征。Censored/deferred 只更新对应 feasibility stage，不进入 mechanism 或 effect。

### 5.5 Sparse-head shrinkage 与 tie contract

对样本稀疏的 task/direction 使用：

```text
learned_prediction
  = support_weight × outcome_head
  + (1 - support_weight) × frozen_F1_or_static_prior
```

`support_weight` 的定义、分母和阈值必须在 fresh pool 前冻结。平分容差也必须在 outcome 前冻结。

当候选在容差内同分时：

- exploitation mass 在并列候选间均匀分配，或使用冻结 prior；
- 不允许依赖 JSON、slot、candidate ID 或输入顺序破平分；
- 记录完整 tie set、分配概率和 `selected_by`；
- 15% exploration 独立于 tie handling。

### 5.6 可审计性与可移植性

在 Q5 首次物理执行前完成：

- 每次 GPU launch 记录 start/end timestamp、PID、GPU ID、hostname 和 exit class。
- artifact consumer 优先解析 package-relative path；历史 absolute path 仅作 provenance。
- clean environment 验证 install/import/hash、一个 Qualifier 和一个 bounded resource probe。
- 任何环境调用目录失败、依赖缺失或 pre-outcome invalid launch 单列保存，不冒充 candidate retry。

## 6. Q5-A：Idea / Feasibility pilot

### 6.1 固定规模

- 3 个 fresh、origin-blind pool。
- 每个 pool 8 个完整 OpenSpec，共 24 个 spec。
- STATIC、CURRENT_F1、OUTCOME_AWARE 每池各选择 2 个。
- 每池额外选择 1 个共享随机 exploration spec。
- 对选择并集中的每个唯一 spec 只实现一次。
- Provider/model/endpoint/temperature、单次 token ceiling、schema 和 consumer 与已接受 Q4 合同保持一致，除本报告列出的通用合同修正外不得改变。
- 每次 Provider/Implementer 物理调用最多一次；0 candidate retry、0 补样、0 成功筛选。

### 6.2 冻结顺序

1. 冻结 pool generation manifest、预算和 producer direction。
2. 生成完整 pool；任一 slot 不完整则该 pool 如实 failed，不补样。
3. 冻结三 policy 的 selection、probability、tie set、selected_by 和 exploration draw。
4. 形成 selection union。
5. 对 union 中的唯一 spec 做 shared realization。
6. 记录所有 stage denominator 和 missingness。
7. 只更新 feasibility 和 mechanism-information 的合法标签；Q5-A 不用 NDCG 作为主目标。

### 6.3 主指标

- `P(materialize)`、`P(construct)`、`P(qualify)`、`P(resource_admitted)`、`P(full_episode)` 的经验成功率。
- 每阶段 calibration；至少报告 Brier score 和分桶可靠性，不以小样本显著性包装结论。
- cost per executable package。
- cost per informative package。
- full Episode completion rate。
- exact/near tie 比例。
- prior shrinkage 使用比例。
- shared random exploration 与 exploitation 的阶段完成差异。

### 6.4 Q5-A Gate

只有全部满足才进入 Q5-B：

- 至少两个 policy 产生非空 executable package selection。
- shared realization 合同无跨 policy 字节漂移。
- 至少两个 `NESTED_MECHANISM` package 通过 off-equivalence gate。
- stage denominator、missingness 和 cost 完整。
- IDEA score 不是全池不可解释同分；如发生同分，tie contract 按冻结规则执行。
- 0 held-out、0 outcome leakage、0 candidate retry。

如果 3 个 pool 仍显示 IDEA learned score 无区分度，则 Q5-A 以负结果封存，不通过追加 pool 或 GPU outcome 掩盖。

## 7. Q5-B：Experiment / Utility pilot

### 7.1 实验单位

- 输入是 Q5-A 形成的同一个 frozen executable package pool。
- policy 只能读取 pre-outcome package/probe/resource features。
- 每个唯一 `package × seed` 只运行一次 candidate outcome。
- 每个 development seed/config 只运行一次共享 parent。
- 所有 policy 从同一 outcome ledger 计算被选 package 的结果。

### 7.2 固定运行协议

- 2 个 fresh paired development seeds；seed schedule 在任何 outcome 前冻结。
- gpu35 严格串行，同一时刻只允许一个训练 worker。
- 先完成 bounded resource admission，再进入 full matched execution。
- 单 package resource/implementation failure 不崩整个 policy 或 pool。
- `RESOURCE_DEFERRED` 保留未来资格，missing 不按 0 effect 处理。
- 0 retry、0 held-out、0 outcome reuse beyond the explicitly shared package × seed ledger。

### 7.3 主指标

继续使用 Q4 冻结的四项主指标：

1. best parent-relative development effect
2. mechanism-identifiable Episode count
3. cost per informative Episode
4. full Episode completion rate

补充报告 stage-conditional feasibility calibration，但不允许用补充指标改变四项主指标的结论。

### 7.4 允许的结论

Q5-B 最多支持：

- 某 policy 在该 DEVELOPMENT_ONLY pilot 中提高或未提高 executable/informative Episode 获取效率。
- 某 policy 在两个 development seeds 上显示一致、冲突或不确定的 parent-relative signal。
- 某 nested realization 的 mechanism-off ablation 为 supported、contradicted 或 non-identifiable。

Q5-B 仍不支持 held-out 泛化、正式科学优越或模型机制的最终结论。

## 8. Stop / Go 决策

### 8.1 进入正式实验协议冻结的最低条件

- Git 主支 integration gate 全部通过。
- Q5-A 和 Q5-B 均有 clean、sealed、可重算的 package。
- 至少两个 fresh pool 产生完整 Episode。
- 至少两个 nested mechanism candidate 具有合法 mechanism-off evidence。
- 同一 spec 的 implementation 和 outcome 不再跨 policy 漂移。
- feasibility calibration 至少能区分两个真实 stage-risk 层级。
- 负证据、missingness、resource censoring 和 cost 完整。

满足后只进入“正式协议冻结”，并不自动开放 held-out。

### 8.2 必须停止或回到第一原则的条件

- 同一通用修复假设失败两次。
- IDEA 在三个 fresh pool 仍完全同分且 prior/tie contract 无法产生可解释选择。
- shared realization 仍产生跨 policy 字节差异。
- nested mechanism-off 无法回到 declared parent。
- 为继续运行必须修改 candidate、seed、data、batch、model、endpoint、预算、denominator 或 outcome 解释。
- 发现 held-out read、outcome leakage、候选特定手修、补样或成功筛选。

这些情况必须封存精确负证据，不创建 retry/attempt family。

## 9. 预计顺序与时间预算

| 阶段 | 目标 | 预计耗时 | 主要退出条件 |
|---|---|---:|---|
| Integration | vNext 交付并无改写合入主支 | 2–4 小时 | merge tests/hash/smoke 全通过 |
| Q5 foundation | shared realization、KeyError 分类、realization typing、tie/stage contracts | 1–2 个工作日 | targeted/contract/fresh non-old-spec 验证通过 |
| Q5-A | 3 × 8 OpenSpec Idea/Feasibility pilot | 1–2 个工作日 | stage evidence 完整且满足 Q5-A Gate |
| Q5-B | 两 seed shared-package Experiment/Utility pilot | 2–4 个工作日 | 四主指标和 mechanism evidence 完整 |
| Formal freeze decision | 审计 Q5 并冻结或否决正式协议 | 0.5–1 个工作日 | 独立审计结论 |

在 Provider 与 gpu35 正常的情况下，达到“可以决定是否冻结正式实验协议”预计需要 4–7 个工作日。正式 campaign 的运行时间不包含在此估计内。

## 10. 交付物清单

### Integration

- `feat/research_line_vnext` 远端交付线
- `feat/research_line` no-ff merge commit
- integration verification receipt
- clean-environment smoke receipt

### Q5 foundation

- shared realization contract 与 targeted tests
- failure-classification report for Q4 Outcome-aware KeyError
- nested/effect-only realization schema extension和 equivalence tests
- stage-conditional feasibility authority matrix
- tie/prior shrinkage contract

### Q5-A

- 3 个 frozen pool manifests
- 完整 selection/tie/exploration ledgers
- shared implementation/package/qualification/resource receipts
- stage denominator、calibration 和 cost report

### Q5-B

- frozen package pool 与两 seed schedule
- shared parent/candidate outcome ledger
- TypedResearchEpisodes 与 mechanism-off ablations
- 四主指标 report
- formal-freeze input package

## 11. 最终报告口径

在 Q5 完成前，统一使用以下口径：

> Research Line vNext 的计划架构和真实多轮工程链已经 DEVELOPMENT_ONLY 闭合。Q4 未证明 Outcome-aware 优越性；它暴露了实现 realization variance、IDEA 同分、构造失败和机制不可识别四个测量问题。下一阶段不再扩架构，而是通过 shared realization、stage-conditional feasibility、nested mechanism gate 和 Q5 小规模前瞻 pilot，判断该系统是否具备冻结正式实验协议的条件。

## 12. Evidence anchors

| Evidence | Commit | 备注 |
|---|---|---|
| P0 | `ce9104ac67993ea5184ef4337403954ee50b543e` | pre-freeze research contract |
| R1 | `11dae330dbbbbf3a8108b19e7f9b9020326f0a20` | fresh R1 accepted |
| R2 | `3505885c69737064ba5bd59a5aa8c96e5309d15d` | fresh R2 accepted |
| Q0 | `ec8c419aa678bca1ab7468c1ae96b822256b53f0` | quality calibration |
| Q0R2 | `278391ab47c78508af211978045ad5573d6fc135` | resource admission pass |
| F1 | `1d3223104871bac6820ab1a550461bd142591a02` | resource-compatible realization pass |
| Q1 | `d10083908a64a0d05a0cbbb1a6eddc2055b2f710` | Idea/OpenSpec quality pass |
| Q2 | `49f9225517d76582500bcf80bf7fc6a5d112f137` | accepted negative mechanism evidence |
| Q3 | `fe1678d286fc710202265d145ed58940b1a373f7` | outcome-aware learning DEVELOPMENT_ONLY pass |
| Multi-round soak | `63e44341bd1da982f00a9db7de2be7e779c7a63d` | 3-round soak pass |
| Q4 | `3615a8c9b2cc6d473df6fb1348fd09c47a883426` | prospective policy comparison closure |
| Final audit | `63b93013f04744d7a3b5b97fbd7c5cfba0633956` | `PASS_WITH_FINDINGS` |

关键审计字节：

- Q4 canonical receipt：`304b0a0b1649983fefda9bbc3d50528eb75c40644c5e4d98d61661c851a30281`
- Q4 physical receipt：`5b0e6914d17960ffec025a65df6aaeb445f53ab7e3878616a60f0080891d6100`
- Q4 result package：`c7d246a88e3857fedbade67887fcdde431497c4dc960913fe2c686f240492f8b`
- Q4 fairness-v2 contract：`76e8ee755b457cc8b0d93af33e9b2d67fe2793100b22393378f665dd26dfce29`
