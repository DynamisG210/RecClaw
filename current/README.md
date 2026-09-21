# 当前源码与架构材料

本目录用于理解当前 RecClaw，供后续整理与制作 PPT。主体来自已接纳、已部署的图任务 `formal_runtime_metric5_v1` 包，另保留正在使用的 MoRec AutoSpace 模块与一个不同的核心文件。来源与 SHA256 逐项列在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

| 目录 | 内容 |
|---|---|
| [`runtime/isolated_graph_core/src/`](runtime/isolated_graph_core/src/) | Research Line、机制接口、实现、反馈与证据基础设施 |
| [`runtime/ro/`](runtime/ro/) | 图 AutoSpace、任务适配器、研究入口及本次部署策略 |
| `runtime/graph_*.py` | 图任务资格、受保护训练、评价与结果回接 |
| [`variants/morec/`](variants/morec/) | 实际 MoRec 六轴 AutoSpace、反馈接口及核心差异 |
| [`docs/`](docs/) | 架构、源码导读、闭环示例、来源与边界 |
| [`examples/offline_walkthrough.py`](examples/offline_walkthrough.py) | 零 Provider、零训练的导入与合成反馈演示 |

这不是一键训练发行版：不包含数据、checkpoint、私有配置、凭据、实验结果库及完整外部执行环境。算法没有为展示重写。三个文件仅替换历史机器定位字符串，原始与发布哈希均保留。

## 最小检查

结构与哈希检查仅使用标准库，可在 Windows / Linux 上运行：

```bash
python -B current/tools/verify_snapshot.py
```

核心依赖 Linux 的 `fcntl`。在 Linux / WSL、Python 3.10+ 的独立环境中运行阅读式演示：

```bash
python3 -m venv .venv-course
.venv-course/bin/python -m pip install -r current/requirements-inspection.txt
.venv-course/bin/python -B current/examples/offline_walkthrough.py
```

示例检查真实源码契约、AutoSpace schema 和训练事实的反馈投影；数据是明确标记的合成样例，不能作为成绩。完整训练另需 PyTorch、RecBole、数据/划分、受保护执行环境、资源与成本模块和按任务绑定的 Provider 配置。参见 [依赖与归属](THIRD_PARTY.md) 和 [验证范围](docs/SNAPSHOT_SCOPE.md)。
