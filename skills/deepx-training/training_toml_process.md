# DeepH-pack 训练参数文档: Process:Train 部分

---

**`[process.train.max_epoch]`**
*   **说明 (Description)**: 最大训练轮次 (Epoch)。当前的训练轮数 `epoch_number` 达到 `max_epoch` 时，训练将自动停止。
*   **默认值 (Default)**: `10000`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.multi_way_jit_num]`**
*   **说明 (Description)**: 多路 JIT 编译数量。当数据集中不同结构的边数差异巨大时，此参数有助于加速训练过程。推荐值为 10-20。
    *   **注意**: 这可能会导致训练的第一个 epoch 速度较慢，且如果设置过大可能引发显存溢出 (OOM) 错误。
*   **默认值 (Default)**: `1`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.ahead_of_time_compile]`**
*   **说明 (Description)**: 是否启用 [`ahead-of-time`](https://docs.jax.dev/en/latest/aot.html) (AOT，提前编译) 技术来加速 JIT (即时编译) 过程。在 AOT 方法的辅助下，多路 JIT 的编译耗时可从约 2 小时缩短至 10 分钟。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

### **Process: Train: Dataloader (数据加载器)**

---

**`[process.train.dataloader.batch_size]`**
*   **说明 (Description)**: 批大小 (Batch size)，即一个批次中包含的结构数量。推荐值：一般晶体材料设为1,分子材料设为20-200。
*   **默认值 (Default)**: `1`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.dataloader.train_size]`**
*   **说明 (Description)**: 训练集 (Training dataset) 中的结构数量。**是数量而不是比例！**
*   **默认值 (Default)**: `1`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.dataloader.validate_size]`**
*   **说明 (Description)**: 验证集 (Validation dataset) 中的结构数量。**是数量而不是比例！**
*   **默认值 (Default)**: `0`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.dataloader.test_size]`**
*   **说明 (Description)**: 测试集 (Test dataset) 中的结构数量。**是数量而不是比例！**
*   **默认值 (Default)**: `0`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.dataloader.dataset_split_json]`**
*   **说明 (Description)**: 自定义数据集划分规则的 JSON 文件路径。
    *   请将此参数设为空字符串 (`""`)，以使用`train_size`,`validate_size`,`test_size` 的数量配置随机划分数据集。
    *   请将此参数设为 JSON 文件路径，以使用 JSON 文件中定义的数据集划分规则，此时`train_size`,`validate_size`,`test_size`设置将失效。
    *   JSON 配置必须遵循以下格式规范（列表内为结构的索引）：
    ```json
    {"train": ["1", "3", "5"], "validate": ["2"], "test": ["4", "6"]}
    ```
*   **默认值 (Default)**: `""`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[process.train.dataloader.only_use_train_loss]`**
*   **说明 (Description)**: 是否仅基于训练损失 (Train loss) 来调整学习率，而不考虑验证损失。
    *   如果验证集为空（尽管这不常见），验证损失将始终是一个无意义的值，此时程序将强制设置该选项为 `true` 以忽略验证损失。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

### **Process: Train: Dropout (正则化丢弃)**

---

**`[process.train.drop.dropout_rate]`**
*   **说明 (Description)**: **标准 Dropout 丢弃率**。这是防止模型过拟合的主要手段，通过在训练过程中随机将部分神经元的输出置零来增强模型的泛化能力。一般可尝试设为 0.0 到 0.3 之间的数值。
*   **默认值 (Default)**: `0.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.drop.stochastic_depth]`**
*   **说明 (Description)**: **随机深度 (Stochastic Depth) 丢弃率**。这是一种针对网络深度进行正则化的进阶策略（即按层级 Layer-wise 随机丢弃整个残差块），通常作为标准 Dropout 的补充手段，使用频率相对较低。
*   **默认值 (Default)**: `0.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.drop.proj_rate]`**
*   **说明 (Description)**: **投影层 (Projection) 丢弃率**。专门应用于模型内部线性投影层（如注意力机制输出后的线性变换）的细粒度 Dropout 参数。作为更次级的正则化选项，通常仅在需要极强正则化约束时使用。
*   **默认值 (Default)**: `0.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

### **Process: Train: Optimizer (优化器)**

---

**`[process.train.optimizer.type]`**
*   **说明 (Description)**: 优化器核心类型，可选值包括：[`adamw`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [`adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), 或 [`sgd`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)。
*   **默认值 (Default)**: `"adamw"`
*   **类型 (Type)**: [`"sgd"`, `"adam"`, `"adamw"`]

---

**`[process.train.optimizer.init_learning_rate]`**
*   **说明 (Description)**: 初始学习率。
    *   具体最佳数值需要仔细测试。
    *   建议值：对于 `net_type`=`sparrow`，建议使用 `2E-3`；对于 `net_type`=`eagle` 或 `owl`，建议使用 `1E-3`。
    *   较大的学习率可以加速收敛，但可能面临训练不稳定的风险。
*   **默认值 (Default)**: `2E-3`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.optimizer.clip_norm_factor]`**
*   **说明 (Description)**: 启用优化器中的 [`clip`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip_by_global_norm) (梯度裁剪) 归一化算法，以防止大梯度导致神经元死亡。
    *   如果设置为负值，则禁用此功能。
*   **默认值 (Default)**: `-1.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.optimizer.momentum]`**
*   **说明 (Description)**: 仅适用于 `sgd` 优化器。控制先前梯度对当前梯度更新的影响（动量）。
*   **默认值 (Default)**: `0.8`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.optimizer.betas]`**
*   **说明 (Description)**: 适用于 `adam` 和 `adamw` 优化器。格式为 `betas = [beta1, beta2]`。
    *   `beta1` 控制一阶矩估计（即动量）的指数衰减率。
    *   `beta2` 控制二阶矩估计（即非中心化方差）的指数衰减率，通常用于计算梯度的平方移动平均值。
*   **默认值 (Default)**: `[0.9, 0.999]`
*   **类型 (Type)**: `<LIST-OF-FLOAT>` (浮点数列表)

---

**`[process.train.optimizer.weight]`**
*   **说明 (Description)**: 仅适用于 `adamw` 优化器。`adamw` 中的权重衰减率 (Weight decay)，用于防止过拟合。
*   **默认值 (Default)**: `0.001`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.optimizer.eps]`**
*   **说明 (Description)**: 适用于 `adam` 和 `adamw` 优化器。一个用于提高数值稳定性的小常数。在某些情况下，将其减小到 `1E-10` 或更低可能有助于训练。
*   **默认值 (Default)**: `1E-8`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

### **Process: Train: Scheduler (学习率调度器)**

在 [`optax`](https://github.com/google-deepmind/optax) 优化框架中，学习率衰减通过由缩放参数 (scale) 控制的解耦机制运行。神经网络更新的有效学习率由乘积决定：`learning_rate (lr)` = `init_lr` $\times$ `scale`。一个专用的调度器模块会在训练过程中系统地调节这个缩放因子，从而实现各种衰减策略（阶梯式、指数式或余弦衰减），同时保持初始值与衰减动态的架构隔离。

---

**`[process.train.scheduler.min_learning_rate_scale]`**
*   **说明 (Description)**: 最小学习率缩放因子。当学习率缩放因子 (scale) 降至 `min_learning_rate_scale` 以下时，训练将自动停止。
*   **默认值 (Default)**: `1E-4`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.scheduler.type]`**
*   **说明 (Description)**: 调度器类型，可选值：[`"ReduceOnPlateau"`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) (当指标停滞时衰减)，[`"WarmupCosineDecay"`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_cosine_decay_schedule) (预热余弦衰减)，和[`WarmupExponentialDecay`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_exponential_decay_schedule) (预热指数衰减)。
*   **默认值 (Default)**: `"reduce_on_plateau"`
*   **类型 (Type)**: [`"reduce_on_plateau"`, `"warmup_cosine_decay"`,`"warmup_exponential_decay"`]

---

#### Reduce LR On Plateau (停滞时衰减学习率)

---

**`[process.train.scheduler.factor]`**
*   **说明 (Description)**: 仅适用于 `"reduce_on_plateau"`。每次触发学习率调整时，学习率缩放因子将更新为 `factor` $\times$ `scale`。
*   **默认值 (Default)**: `0.5`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.scheduler.patience]`**
*   **说明 (Description)**: 仅适用于 `"reduce_on_plateau"`。耐心值，即在降低学习率之前等待的评估步数。这有助于确定模型性能是否确实不再提升。
    *   **重要提示**: 默认情况下，耐心步数是基于**训练步数 (Step/Batch)** 定义的，而**不是**基于轮次 (Epoch)。另外请着重参考参数`accum_size`的作用。
    *   建议根据您的数据集规模、计算资源和预期时间成本来定义此参数。
    *   通常，换算为 Epoch 后，`patience=120` 个 Epoch 能得到收敛良好的模型，`patience=60` 个 Epoch 或更少则能较快得到结果。
*   **默认值 (Default)**: `500`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.rtol]`**
*   **说明 (Description)**: 仅适用于 `"reduce_on_plateau"`。相对容差 (Relative tolerance)。如果损失相对于上一次最佳验证损失的改善幅度小于 `rtol`，则认为不再提升。
*   **默认值 (Default)**: `0.05`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.scheduler.cooldown]`**
*   **说明 (Description)**: 仅适用于 `"reduce_on_plateau"`。两次学习率调整之间的最小冷却评估周期。这可以防止学习率发生过于频繁的变动。
    *   **注意**: 默认情况下，冷却步数是基于**训练步数 (Step/Batch)** 定义的，而非 Epoch。
    *   大多数情况下，换算为 Epoch 后，`cooldown=20` $\sim$ `50` 个 Epoch 就足够了。
*   **默认值 (Default)**: `100`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.accum_size]`**
*   **说明 (Description)**: 仅适用于 `"reduce_on_plateau"`。用于调整 `patience` 和 `cooldown` 的计数基准。
    *   这两个参数本质上是基于梯度更新步数 (step) 运作的。本参数通过计算 *每 Epoch 的 Batch 数* $\times$ *目标 Epoch 数* 来设定基准值。
    *   **举例**: 假设每个 Epoch 有 100 个 Batch：
        *   若要监测验证损失 20 个 Epoch 且不使用累积 (`accum_size`=1)，需设置 `patience` = 100 $\times$ 20 = 2000。
        *   若设置 `accum_size`=100 (实际上创建了大小为 100 $\times$ `batch_size` 的宏批次)，此时每一个 "step" 等效于 1 个 Epoch，因此只需设置 `patience`=20。
    *   **自动配置**: 当设置为 `-1` 时，系统会自动将其配置为总训练 Batch 数，从而实现以完整的 Epoch 为周期进行监测。冷却步数 (Cooldown) 也遵循同样的计算逻辑。
*   **默认值 (Default)**: `-1`
*   **类型 (Type)**: `<INT>` (整数)

---

#### Warmup Cosine Decay (预热余弦衰减)

通过结合预热 (Warmup) 和余弦衰减，该调度器有助于深度学习模型更快收敛并提升性能。

---

**`[process.train.scheduler.init_scale]`**
*   **说明 (Description)**: 适用于 `"warmup_cosine_decay"` 和 `"warmup_exponential_decay"`。学习率缩放因子的初始值。
*   **默认值 (Default)**: `0.1`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[process.train.scheduler.warmup_steps]`**
*   **说明 (Description)**: 适用于 `"warmup_cosine_decay"`和 `"warmup_exponential_decay"`。预热步数，即学习率缩放因子从 `init_scale` 线性增加到 `1.0` 所需的步数。这有助于在训练初期稳定模型行为。
*   **默认值 (Default)**: `1000`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.decay_steps]`**
*   **说明 (Description)**: 仅适用于 `"warmup_cosine_decay"`。衰减步数，即学习率缩放因子从 `1.0` 衰减到 `end_scale` 所需的步数。
    *   注意：当 Epoch 数达到 `max_epoch` 时，训练会提前停止。
*   **默认值 (Default)**: `-1` (非法值)
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.end_scale]`**
*   **说明 (Description)**: 适用于 `"warmup_cosine_decay"`和 `"warmup_exponential_decay"`。学习率调度结束时的缩放因子。
    *   设置为 `-1.0` 表示让学习率衰减至 0。
    *   注意：当学习率缩放因子低于 `min_learning_rate_scale` 时，训练会提前停止。
*   **默认值 (Default)**: `-1.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---
#### Warmup Exponential Decay (预热指数衰减)

通过结合预热 (Warmup) 和指数衰减，该调度器有助于深度学习模型更快收敛并提升性能。

这组参数通常用于配置**指数衰减 (Exponential Decay)** 或**分段常数衰减**的学习率调度策略。以下是补全并润色后的描述：

---

**`[process.train.scheduler.transition_begin]`**
*   **说明 (Description)**: 仅适用于 `"warmup_exponential_decay"`。学习率衰减的**起始步数**。
    *   在当前的训练步数 (Global Step) 达到此值之前，学习率将保持初始值不变，不进行衰减。
    *   这允许模型在训练最开始阶段以固定的较高学习率进行学习，或者推迟衰减开始的时间。
*   **默认值 (Default)**: `0`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.transition_steps]`**
*   **说明 (Description)**: 仅适用于 `"warmup_exponential_decay"`。学习率衰减的**周期步数**（或衰减间隔）。
    *   该参数定义了衰减发生的频率。
    *   配合 `staircase` 参数使用：如果 `staircase=true`，则每过 `transition_steps` 步，学习率才会发生一次突变（乘以衰减系数）；如果 `staircase=false`，则以此为基准进行平滑衰减。
*   **默认值 (Default)**: `10000`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[process.train.scheduler.staircase]`**
*   **说明 (Description)**: 仅适用于 `"warmup_exponential_decay"`。是否启用**阶梯式 (Staircase) 衰减**模式。
    *   **`true`**: 启用离散衰减。学习率表现为台阶状下降，即每经过 `transition_steps` 步后，学习率才会突然降低一次。
    *   **`false`**: 启用连续衰减。学习率将在每一步训练中都进行微小的平滑指数更新。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

### **Process: Train: Continued (断点续训/微调)**

本节用于对现有模型进行微调 (Fine-tuning) 或继续训练 (Continued training)。

---

**`[process.train.continued.enable]`**
*   **说明 (Description)**: 设置为 `true` 以从现有模型继续训练或微调；设置为 `false` 则从头开始训练。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[process.train.continued.new_training_data]`**
*   **说明 (Description)**: 是否使用新的训练数据。
    *   对于类似微调 (fine-tuning) 的任务，设置为 `true`。
    *   如果在与上次相同的可以数据集上继续运行，则设置为 `false`。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[process.train.continued.new_optimizer]`**
*   **说明 (Description)**: 是否使用新的优化器。
    *   对于类似微调的任务，设置为 `true`。
    *   如果希望沿用上次的优化器状态和调度器继续运行，则设置为 `false`。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[process.train.continued.previous_output_dir]`**
*   **说明 (Description)**: 带有时间戳的前次输出目录路径，该目录应包含 `deepx.log` 文件和 `model` 文件夹。
*   **默认值 (Default)**: `<Invalid-Input>`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[process.train.continued.load_model_type]`**
*   **说明 (Description)**: 根据需要选择从 `best` (最佳) 或 `latest` (最新) 模型继续训练。
*   **默认值 (Default)**: `"latest"`
*   **类型 (Type)**: [`"best"`, `"latest"`]

---

**`[process.train.continued.load_model_epoch]`**
*   **说明 (Description)**: 仅当 `load_model_type` = `"latest"` 时有效。指定加载最新模型文件夹中保存的特定 Epoch 编号。使用 `-1` 表示加载最新的 Epoch。
*   **默认值 (Default)**: `"-1"`
*   **类型 (Type)**: `<INT>` (整数)


---

### **Process: Train: With Plugin (模组插件)**

本节用于配置训练过程中的外部插件，通常用于实现无监督学习（如自洽计算闭环）、在线数据生成或特定的物理量计算辅助功能。

---

**`[process.train.with_plugin.enable]`**
*   **说明 (Description)**: 是否启用插件模块。设置为 `true` 时，训练循环中将调用由 `script_path` 指定的外部脚本逻辑。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[process.train.with_plugin.plugin]`**
*   **说明 (Description)**: 指定要使用的插件名称或类型标识符。具体的插件逻辑需要在脚本中定义。插件需要事先通过python entries注册到系统中。
*   **默认值 (Default)**: `"null"`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[process.train.with_plugin.env_path]`**
*   **说明 (Description)**: 执行插件脚本所需的 Python 解释器路径或环境根目录路径。
    *   **注意**: 必须修改此默认值指向正确的环境路径，以确保插件能加载必要的依赖库。
*   **默认值 (Default)**: `"./need/user/set/this"` (需用户手动设置)
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[process.train.with_plugin.script_path]`**
*   **说明 (Description)**: 包含具体插件实现逻辑的 Python 脚本文件路径。
    *   **注意**: 必须修改此默认值指向实际存在的脚本文件。
*   **默认值 (Default)**: `"./need/user/set/this"` (需用户手动设置)
*   **类型 (Type)**: `<STRING>` (字符串)

---

#### Unsupervised Additional Options (无监督学习附加选项)

以下参数通常用于涉及物理计算后端（如 DFT 软件接口）的无监督训练场景。

---

**`[process.train.with_plugin.backend]`**
*   **说明 (Description)**: 指定插件在执行计算任务时使用的计算后端或接口类型（例如特定的 DFT 计算软件接口、线性代数库等）。
*   **默认值 (Default)**: `"null"`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[process.train.with_plugin.dump_intermediate]`**
*   **说明 (Description)**: 是否将插件运行过程中产生的中间数据（如中间时刻的密度矩阵、哈密顿量或临时张量）转储保存。
    *   该选项主要用于调试或在训练意外中断后恢复中间状态。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

