---
name: DeepH-pack 模型批量训练调参
description: "本技能详细阐述了如何使用 DeepH 进行模型训练并实施批量调参，以找到最优的超参组合。模型训练调参是 DeepH 机器学习任务中的关键环节，通过系统性调参，可显著优化模型性能并提高预测准确性。本指南涵盖了从环境配置、模型选择、参数设置到批量训练策略及结果评估的全流程。"
license: Proprietary. LICENSE.txt has complete terms
---

# DeepH-pack 模型批量训练调参指南

## 1. 背景介绍

DeepH-pack 是清华大学 DeepH 团队多年研究成果的集大成者。最新版本基于 JAX 和 Flax 框架重写，集成了先前所有方法，并提供了统一的调用接口。经过对神经网络模块的长期严格测试，DeepH-pack 在用户体验上已高度成熟。

得益于 JAX 的静态计算图和先进算法，当前版本在**运行时性能**、**计算精度**和**内存效率**方面均表现卓越。未来，DeepH-pack 将致力于无缝集成多物理量预测能力，旨在成为一个可扩展的高精度量子材料建模计算平台，确保哈密顿量构建达到量子化学级精度。

## 2. 环境准备与确认

在使用 DeepH-pack 开启训练任务前，请务必确认环境配置正确。

### 2.1 检查安装状态
可以通过检查 Python 环境中是否存在 `deepx` 包来验证安装。DeepH-pack 的安装包命名格式通常为 `deepx_pack-1.2.x-py3-none-any.whl`。

**版本检测代码：**
```python
import deepx
print(f"DeepH-pack version: {deepx.__version__}")
```

若未安装，请参考 [DeepH 安装指南](./deeph_install.md)。

### 2.2 安装辅助工具 Deepx-dock
`deepx-dock` 是 DeepH 训练流程中的重要辅助工具，提供了网络表示推荐（Analysis of Dataset Features）等必要功能。检查deepx-dock包是否安装, 如否，请执行以下命令进行安装。

**安装指令：**
```bash
# 请确保在 DeepH-pack 所在的 uv venv 环境中执行
uv pip install deepx-dock
```

安装完成后，命令行会新增 `dock` 指令，用于调用其内部定义的命令行工具。

## 3. 标准训练流程

### 3.1 配置训练参数 (TOML)
DeepH 使用 TOML 文件进行训练和推理配置。**默认训练配置模版可参考本 skill 文件夹内的CONFIG_TRAIN_DEFAULT.toml**。

**配置文件：**
*   用户配置文件TOML参考模版！：
    *   阅读 CONFIG_TRAIN_DEFAULT.toml 获取：包含默认训练配置模版。
    *   阅读 CONFIG_TRAIN_VALID.toml 获取：包含所有值合法类型。
*   用户可自定义配置文件名（如 `train_config.toml`）。
*   详细参数定义请参阅 [DeepH-pack 训练参数文档](./training_toml.md)。

### 3.2 数据集准备
训练数据需存放于 `train_config.toml` 中 `data.inputs_dir` 指定的目录。数据格式必须严格符合 [DeepH-pack 数据集格式文档](./dataset_format.md) 的要求。

特别注意!：如果在`train_config.toml`中使用相对路径，则该相对路径是相对于`train_config.toml`文件所在目录的路径。为避免谬误，大部分情况下推荐使用绝对路径。

重要提示：关于数据集划分，可以使用`train_config.toml`中的`train_size`,`validate_size`,`test_size` 三个参数进行划分，这个三个参数必须是整数，代表训练集、验证集、测试集中的样本数量。

重要提示：如果想要完全控制数据集划分样本严格一致，请设置`train_config.toml`中的`dataset_split_json`选项，该选项指向一个json文件，内部包含了训练集、验证集、测试集的样本id列表。dataset_split_json 更具体的细节可参考当前 `dock analyze dataset split` 命令说明。

### 3.3 任务提交方式

#### A. 裸机环境 (Bare Metal)
直接在终端运行指令即可，建议使用 `nohup ... &` 保持后台运行：
```bash
deeph-train train_config.toml
```

#### B. 超算环境 (HPC with Slurm/PBS)
在超算环境中，严禁在登录节点直接运行高负载任务。必须通过作业调度系统（如 Slurm, PBS）提交。

**注意事项：**
1.  **资源一致性**：申请的 GPU 卡数必须与 `train_config.toml` 中的 `system.device` 配置保持一致。
2.  **设备配置语法**：
    *   格式：`<type>*<num>:<id>`
    *   `<type>`: 硬件类型 (`cpu`, `gpu`, `rocm`, `dcu`, `cuda`)。
    *   `<num>`: 设备总数（GPU）或分区数量（CPU）。
    *   `<id>`: 目标设备索引。
    *   **示例**：`gpu*8:1-4,7` 表示从 8 卡节点中选择索引为 1, 2, 3, 4, 7 的 5 张 GPU。
    *   **默认值**：`"gpu*8:0"`。

## 4. 批量测试与调参策略

批量测试旨在通过自动化脚本一次性提交多组实验，以快速筛选最优超参。

### 4.1 自动化流程建议
1.  **资源评估**：根据服务器总资源确定并发作业数。单张 GPU 通常可承载 2-3 个小型任务（需关注显存占用）。
2.  **提交策略**：
    *   **裸机**：编写守护脚本（Daemon Script）监控 GPU 占用，有空闲资源时自动提交新任务。
    *   **超算**：利用作业队列系统一次性提交所有作业，但需注意不要超过用户的最大排队限制。
3.  **结果汇总**：必须编写一个汇总脚本（Summary Script），用于解析日志并生成报告。报告应包含当前进度、Loss 收敛情况及最优参数组合，以便随时查看。

### 4.2 最佳实践
*   **预测试 (Smoke Test)**：在大规模提交前，先跑通 1-2 个小任务进行 Debug，确保脚本逻辑无误。（对超大规模模型训练，例如单个epoch运行时间超过30分钟，则可跳过此步骤）。执行本测试任务时，需明确将执行超时阈值 (Timeout) 调整为 15 分钟，以确保任务顺利完成。
*   **环境整洁**：将中间临时文件统一存放或定期清理，避免目录混乱。
*   **进程安全**：批量脚本应具备精确的进程控制能力，**严禁误杀**非当前测试任务的 DeepH 进程。

### 4.3 文件夹结构
*   **数据文件**：`./inputs` (或其他名称，该文件夹由用户提供)
*   **预测试相关文件**：`./.test`
*   **批量测试配置文件**：`./configs`
    *   **基础模板配置**：`./configs/.base_config.toml`
    *   **第一个测试**：`./configs/config001.toml`
    *   **第二个测试**：`./configs/config002.toml`
    *   **第N个测试**：`./configs/config_N_.toml`
*   **功能性脚本**：`./script`
*   **批量测试输出日志**：`./logs`
*   **批量测试结果**：`./outputs`
    *   **第一个测试**：`./outputs/config001/`
    *   **第二个测试**：`./outputs/config002/`
    *   **第N个测试**：`./outputs/config_N_/`

## 5. 模型调参经验库

以下是 DeepH 团队总结的关键调参经验，分为网络结构、物理设置、优化策略三部分。

### 5.1 网络结构参数
*   **`net_irreps` (关键)**：定义神经网络中间层的 E3 变换表示。此参数直接决定模型的表示能力和参数量。
    *   **必须手动指定**，默认为空是非法的。
    *   **推荐做法**：使用 `dock analyze dataset features` 工具基于 `dft/` 原始数据分析推荐值。
*   **`num_blocks`**：决定网络深度。
    *   专用模型（单一材料）：3-5 层。
    *   通用模型（跨元素周期表）：5 层以上。
*   **`num_heads`**：注意力头数，通常设为 `2` 即可。
*   **`latent_irreps` / `latent_scalar_dim`**：在使用 `albatross` 等特定网络时非常关键，设置逻辑同 `net_irreps`。
*   **`enable_bs3b_layer` & `bs3b_orbital_types`**：**通用大模型必开**。这对提升泛化能力至关重要。

### 5.2 物理与数据设置
*   **`gaussian_basis_rmax`**：必须覆盖哈密顿量基组的半径范围，推荐值为 `7.5 - 10.0`。
*   **网络选择**：
    *   哈密顿量/密度矩阵任务：首选 `eagleplus` 或 `albatross`。
*   **Loss Function**：
    *   首选：`MAE` (Mean Absolute Error)。
    *   备选：`MSE` 偶尔也可尝试。
*   **Test Set**：**必须划分测试集**。没有测试集无法评估过拟合情况，模型结果将失去参考意义。

### 5.3 优化器与训练策略
*   **`batch_size`**：
    *   晶体材料：通常设为 `1`。
    *   分子体系：建议 `10` 以上。
    *   *注：需根据显存实际情况调整。*
*   **`dropout_rate`**：可尝试 `0.0` 或 `0.1`，观察对过拟合的影响。
*   **`optimizer`**：推荐 `adamw`。初始学习率范围 `1E-2` 至 `1E-4`。
*   **`scheduler` (学习率调度)**：
    *   大模型：推荐 `warmup_cosine_decay`，decay 步数建议 50 万步以上。
    *   专用模型：可尝试 `reduce_on_plateau`。
        *   分子（快速）：patience 设为 100-200。
        *   分子（慢速）：patience 设为 500-1000。
        *   晶体：patience 设为 50-200。
*   **`multi_way_jit_num`**：解决 JAX 静态图导致的编译慢问题。通过将边数分桶加速计算。
    *   通用模型：10-20 桶。
    *   专用模型：1 桶。
    *   *技巧：如果 JIT 时间过长，可开启 `ahead_of_time_compile` 进行并行预编译。*

## 6. 常见问题排查 (FAQ)

### Q1: 作业提交超过10分钟，为何一直卡在“模型建立”阶段？
**A:** 这是正常现象。DeepH 基于 JAX，首次运行时需要进行 **JIT (Just-In-Time)** 编译，该过程可能较为耗时（视模型复杂度而定，10分钟以上属正常范围）。请耐心等待。

### Q2: 遇到显存不足 (OOM) 怎么办？
**A:** 请按以下顺序排查优化：
1.  **检查模型规模**：`net_irreps` 或 `num_blocks` 是否过大？
2.  **降低 Batch Size**：这是最直接的手段。
3.  **多卡并行**：利用节点内多卡并行（DeepH 目前支持节点内多卡，不支持跨节点并行）。

### Q3: 浮点精度 (Float Type) 如何选择？
**A:** 强烈推荐 **FP32**。
*   FP64：速度过慢，通常不必要。
*   TF32/BF16：精度对于科学计算而言通常不足，可能导致训练发散或结果不准。

### Q4: 批量训练时 `data.outputs_dir` 有什么注意事项？
**A:** 虽然系统会自动按时间戳生成子目录，但**强烈建议**为每种测试配置手动指定唯一的 `outputs_dir`。
*   **原因**：防止批量提交速度过快导致时间戳重叠（撞车），造成不同实验输出到同一目录的严重 Bug，同时也便于后续的模型管理和对比。

### Q5: 建图模式 `disk` vs `memory` 如何选择？
**A:**
*   **Memory 模式**：将图数据全量加载到内存。**速度最快**，推荐内存充足时首选。
*   **Disk 模式**：数据存储在磁盘，随用随读。**节省内存**，但会显著增加 I/O 开销，降低训练速度。仅在内存不足时使用。

