# DeepH-pack 训练参数文档

## 概览 (Overview)

训练过程的参数通过 TOML 格式文件进行配置。每一个键（key）都系统地控制着计算工作流的特定方面。

接下来，我们将详细介绍 TOML 文件中这些参数的含义。

TOML 配置文件包含四个核心部分：

- `system`：处理硬件和计算环境声明等。
- `data`：指定训练数据集的位置、特征和元数据等。
- `model`：定义基础架构组件和目标物理量（不限于哈密顿量——未来版本将逐步支持力场、原子间势、电荷密度、密度矩阵、GW 计算等），包括损失函数的选择。
- `process`：通过收敛标准、数据加载器配置、优化器、重启设置等控制训练/推理工作流。

首先，让我们来看几个关键参数的说明：

- `system.device`：设备配置遵循 `<type>*<num>:<id>` 的语法，其中 `<type>` 指定硬件类型（`cpu`、`gpu`、`tpu`、`rocm`、`dcu` 或 `cuda`），`<num>` 表示每个节点的设备总数（对于 GPU 等加速器）或 CPU 分区数（使用 `cpu` 时），`<id>` 定义目标设备索引（例如，`gpu*8:1-4,7` 表示从 8 设备节点中选择索引为 1,2,3,4,7 的 5 个 GPU，`gpu*3` 表示选择 3 设备节点的所有 GPU）。**注意**，对于 CPU 配置，`<id>` 会被忽略，而 `<num>` 控制线程分区。

- `model.net_type`：神经网络架构。（架构可选项见下文）

- `model.advanced.net_irreps`：神经网络特征的不可约表示，用于确保网络的等变性。需设置为 [`e3nn.Irreps`](https://e3nn-jax.readthedocs.io/en/latest/api/irreps.html) 的字符串形式，即*不可约表示*，它描述了输入特征的对称性。**注意**，对于哈密顿量预测任务，`Irreps` 中指定的最大 $l$ 必须至少是哈密顿量基组中存在的最高角动量量子数的两倍。这一要求是因为 Irreps 将哈密顿量的非耦合表示（直积基）转换为耦合表示（直和基）。例如，当包含 f 轨道（$l=3$）时，Irreps 必须支持 $l_{\text{max}} \geq 6$。

- `process.train.optimizer.init_learning_rate`：初始学习率。

- `process.train.scheduler.min_learning_rate_scale`：学习率的最小缩放因子。当学习率乘数达到此阈值时，训练将自动终止，此时有效学习率为 `scale` $\times$ `initial_learning_rate`。

## 配置选项详细说明

下面我们将分四个章节，分别介绍 `system`、`data`、`model` 和 `process` 的详细配置。

- `system` 部分用于配置计算资源、日志记录、随机数等。阅读 training_toml_system.md 或许全部配置说明信息。
- `data` 部分用于配置数据集、数据加载器等。阅读 training_toml_data.md 或许全部配置说明信息。
- `model` 部分用于配置神经网络架构、损失函数等。阅读 training_toml_model.md 或许全部配置说明信息。
- `process` 部分用于配置训练过程、优化器、学习率调度等。阅读 training_toml_process.md 或许全部配置说明信息。

## 配置文件结构

DeepX-Training 的配置文件采用 TOML 格式，其结构如下：

```toml
[system]
# system config

[data]
# data config

[model]
# model config

[process]
# process config
```