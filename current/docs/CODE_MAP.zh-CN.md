# 源码导读：从入口到反馈

以下阅读顺序对应真实调用关系，不需要先通读全部底层文件。

| 顺序 | 文件 / 符号 | 重点 |
|---|---|---|
| 1 | [`run_graph_ro_main.py`](../runtime/ro/run_graph_ro_main.py) 的 `main`、`execute` | 读绑定 → 生成空间 → adapter → compose campaign → `FrozenSearchPolicy` → run |
| 2 | [`graph_ro_generation.py`](../runtime/ro/graph_ro_generation.py) 的 `prepare`、`generate` | 一次 fresh Genesis/Critic、同成本账本、保留原请求，不静默重抽 |
| 3 | [`graph_autospace.py`](../runtime/ro/graph_autospace.py) | 任务合同、生成请求、装配和标准空间表示 |
| 4 | [`graph_assembler_engine.py`](../runtime/ro/graph_assembler_engine.py) | 源码锚点、schema、机制族与组合约束 |
| 5 | [`genesis_instructions.txt`](../runtime/ro/resources/genesis_instructions.txt)、[`critic_instructions.txt`](../runtime/ro/resources/critic_instructions.txt) | 实际提示词：效应路径、梯度、评分一致性、覆盖与组合 |
| 6 | [`graph_search_adapter.py`](../runtime/ro/graph_search_adapter.py) | 通用核心消费图任务、完整 root-relative recipe 和资格执行器 |
| 7 | [`standalone.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/standalone.py) 的 `compose_standalone_campaign` | 装配 provider、adapter、launch 和 campaign |
| 8 | [`producers.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/producers.py)、[`provider.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/provider.py) | 角色上下文、提案与实现请求；角色常量见 [`research_contracts.py`](../runtime/isolated_graph_core/src/recclaw_core/experiments/helix_abc_v1/research_contracts.py) |
| 9 | [`runtime.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/runtime.py) 的 `run_research_round` | 单轮提出、选择、实现、执行与解释 |
| 10 | [`graph_candidate_contract.py`](../runtime/graph_candidate_contract.py)、[`graph_qualification.py`](../runtime/graph_qualification.py) | 静态资格与真实资格入口；AST 检查不等于 OS 隔离 |
| 11 | [`graph_candidate_evaluator.py`](../runtime/graph_candidate_evaluator.py)、[`graph_worker.py`](../runtime/graph_worker.py) | 受保护训练/评价、最佳 checkpoint 与 reload |
| 12 | [`graph_ro_execution.py`](../runtime/graph_ro_execution.py) | evaluator → 核心结果，核对 NDCG@5，投影曲线与配置 |
| 13 | [`interpreter.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/interpreter.py)、[`campaign.py`](../runtime/isolated_graph_core/src/recclaw_core/research_line/campaign.py) | 解释、反馈投影与跨轮状态 |
| 14 | [`learning_components.py`](../runtime/ro/policy/learning_components.py) 的 `FrozenSearchPolicy` | 实际启用的策略冻结；文件中其他干预类不代表同时使用 |

## 历史名称如何理解

`research_line/` 是研究主链，仍导入 `experiments/helix_abc_v1/` 中的合同、实现、控制与证据模块；历史目录名不表示整块代码已退役。

`mechanism_space/`、`search_spaces/bl_icf_v1/` 包含通用接口和历史静态空间。部分资源在导入时加载，按依赖保留。**当前图任务使用 fresh AutoSpace，不是这个历史静态目录，也不是人工获胜算法清单。** 不要依据历史文件名推导当前实验方法。

`recclaw_evidence_guard/` 与实验合同处理来源身份和证据接入，可概括为“保证比较条件和反馈可追溯”；不必将所有 receipt/schema 当作研究创新。

## MoRec 版本

[`auto_space_v161.py`](../variants/morec/auto_space_v161.py) 是六轴 assembler；[`morec_axis_contract.py`](../variants/morec/morec_axis_contract.py) 限定各轴可改文件/区域；[`morec_ro_autospace.py`](../variants/morec/morec_ro_autospace.py) 从完整冻结的五文件 parent 生成空间；[`morec_ro_execution.py`](../variants/morec/morec_ro_execution.py) 是反馈接口。

五个适配模块远端实读 SHA256 均与 MoRec RO01 原始 `EXECUTION_BINDING` 对应项一致。共享核心 309 个可比文件中，308 个原始哈希相同；不同的 [`campaign.py`](../variants/morec/core/recclaw_core/research_line/campaign.py) 单独保存。这是版本阅读材料，不应整体加到图任务 PYTHONPATH，也不代表全部 MoRec 运行状态。
