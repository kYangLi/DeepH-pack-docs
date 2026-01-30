# DeepH-pack 训练参数文档: Data 部分

---

**`[data.inputs_dir]`**
*   **说明 (Description)**: 指定包含结构化输入数据的根目录，该目录下必须包含必要的子文件夹 `dft/` 或 `graph/`（具体请参见 `sec::analysis_data_features` 节）。支持相对路径（基于toml配置文件所在文件夹位置解析）和绝对路径（系统原生格式）。
*   **默认值 (Default)**: `./inputs`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[data.outputs_dir]`**
*   **说明 (Description)**: 指定用于存储训练产物（如日志文件和序列化模型）的父目录。默认情况下，系统会在用户指定的路径下自动生成带有时间戳的子目录（采用 ISO 8601 扩展格式：`%Y-%m-%d_%H-%M-%S`），以隔离不同训练会话的输出内容。详细的目录结构规范请参阅 `sec::after_training` 节（训练后工作流）。
*   **默认值 (Default)**: `./outputs`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[data.dft.data_dir_depth]`**
*   **说明 (Description)**: 在组织 DFT 训练数据子目录时，如果结构配置数量过多（例如达到 100,000 个结构），在 `inputs_dir/dft/*` 下使用扁平目录结构将变得不切实际。此时，建议采用分层文件夹架构，例如 `dft/<t1>/<t2>/*`。这种配置建立的目录深度为 2（即设置 `data_dir_depth=2`），在保持系统访问有序性的同时，为海量结构数据集提供了可扩展的存储方案。
*   **默认值 (Default)**: `0`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.dft.validation_check]`**
*   **说明 (Description)**: 此标志用于启用或禁用 DFT 训练数据的验证检查。当设置为 `true` 时，系统将执行验证以确保训练数据格式正确且包含必要信息。虽然该检查对于保证训练数据的完整性和质量至关重要，但会显著降低图构建过程的速度（约10-100倍）。因此如果您对您数据的格式有信心，可以将其设置为 `false` 以加快建图速度。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[data.graph.dataset_name]`**
*   **说明 (Description)**: 数据集的名称，建议使用一个针对数据集特征的标志性名字。
*   **默认值 (Default)**: `"DATASET-DEMO"`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[data.graph.graph_type]`**
*   **说明 (Description)**: 指定用于训练的物理量类型，DeepH 将据此构建相应的图文件。可选值包括 `"H"`, `"HS"`, `"Rho"`, `"Sap"`, `"S"`, `"VrC"`, `"SapVc"`, `"FF"`, `"IEP"`。其中，`H` 代表哈密顿量，`Rho` 代表密度矩阵，`S` 代表交叠矩阵，`HS` 同时包含哈密顿量和重叠矩阵，`Sap` 代表仅考虑形状的重叠矩阵（不读取交叠矩阵的指，只从中读取哪些原子会建立连边）。此外，`"FF"` 代表力场图，`"IEP"` 代表机器学习原子间势能的图。图文件中顶点代表原子，若两个原子距离较近则存在连边。`"VrC"`, `"SapVc"` 适用于特殊任务，无需特别关注。
*   **默认值 (Default)**: `"H"`
*   **类型 (Type)**: [`"H"`, `"HS"`, `"Rho"`, `"Sap"`, `"S"`, `"VrC"`, `"SapVc"`, `"FF"`, `"IEP"`]

---

**`[data.graph.storage_type]`**
*   **说明 (Description)**: 指定图数据的存储位置。选择 `"memory"` 将数据存储在内存中；对于超大型数据集，请选择 `"disk"` 将数据存储在磁盘上。注意：数据存储在磁盘上，仅仅会缓解内存压力，不会解决显存不足问题。
*   **默认值 (Default)**: `"memory"`
*   **类型 (Type)**: [`"memory"`, `"disk"`]

---

**`[data.graph.disk_shards_num]`**
*   **说明 (Description)**: 仅当 `storage_type` 为 `"disk"` 时生效。在磁盘（`disk`）模式下构建图时，DeepH-pack 支持将数据集划分为多个分片（Shards/Sub-databases），以降低单个数据库文件的体积并提升管理效率。在训练阶段，系统会自动识别并将这些分片作为一个整体数据集进行调用。此外，该分片机制支持对不同分片任务进行并行处理（即并行建图）。本参数用于指定将数据集划分为分片的总数量。
*   **默认值 (Default)**: `1`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.graph.disk_shards_indices]`**
*   **说明 (Description)**: 仅当 `storage_type` 为 `"disk"` 时生效。该参数用于指定当前训练或建图任务所需处理（建立或调用）的具体数据库分片索引列表。索引从 1 开始计数。结合 `disk_shards_num` 使用，用户可以灵活地控制加载特定的数据子集，或在多节点环境下分配不同的分片构建任务以实现并行建图。留空(`[]`)则表示使用所有数据库分片。
*   **默认值 (Default)**: `[]`
*   **类型 (Type)**: `<LIST-OF-INT>` (整数列表)

---

**`[data.graph.disk_mem_buffer_size]`**
*   **说明 (Description)**: 仅当 `storage_type` 为 `"disk"` 时生效。磁盘建图采用"内存缓冲-批量写入"策略：即先在内存中构建一定数量的图数据，随后一次性写入磁盘数据库，循环往复。本参数定义了内存缓冲区的容量（即单次写入磁盘前的最大结构数量）。**注意**：更改此参数会改变数据写入的批次边界（Pagination），进而导致生成的数据库文件哈希值发生变化。
*   **默认值 (Default)**: `2048`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.graph.common_orbital_types]`**
*   **说明 (Description)**: 单个原子的轨道列表，按照 `l`（角量子数）的值排列，格式为`sNpNdNfN...`，例如 `s2p2d1`。默认值（`""`）为数据集中所有不同原子轨道类型的并集。例如，在 `OpenMX` 计算中，若基组为 `Mo-s3p2d1` 和 `S-s2p2d1`，则轨道类型的并集为 `[0, 0, 0, 1, 1, 2]`，对应于 `s3p2d1`。
*   **默认值 (Default)**: `""`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[data.graph.parallel_num]`**
*   **说明 (Description)**: 确定用于图构建的最大并发并行进程数。当配置为非正整数或超过可用计算资源的值（即超过主机的物理 CPU 核心数或加速器设备数量）时，系统将根据硬件可用性动态调整并行度。具体而言，将采用检测到的 CPU 核心数与加速器设备（GPU）数量两者中的较大值，以优化计算吞吐量。默认值`-1`表示自动确定并行数。
*   **默认值 (Default)**: `-1`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.graph.only_save_graph]`**
*   **说明 (Description)**: 如果设置为 `true`，程序将仅生成并将图文件保存到文件系统，随即退出。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[data.model_save.best]`**
*   **说明 (Description)**: 是否保存具有最低误差（loss）的模型。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[data.model_save.latest]`**
*   **说明 (Description)**: 是否保存最新训练轮次（epoch）的模型。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[data.model_save.latest_interval]`**
*   **说明 (Description)**: 仅当 `data.model_save.latest` 为 `true` 时有效。此参数控制训练期间模型状态保存（检查点）的频率。例如，设置 `latest_interval`=100，系统将在第 100、200、300 等轮次（epoch）生成训练状态存档，实现对完整模型状态（包括权重、优化器参数和元数据）的定期保存。
*   **默认值 (Default)**: `100`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.model_save.latest_num]`**
*   **说明 (Description)**: 仅当 `data.model_save.latest` 为 `true` 时有效。指定用户希望保留的最新检查点（checkpoint）的数量。
*   **默认值 (Default)**: `10`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[data.model_save.latest_cache_mid_step_interval]`**
*   **说明 (Description)**: 对于训练大型模型而言，该参数至关重要。由于大型模型的单个 epoch 训练时间可能极长（例如数周），仅按 epoch 保存存在较高风险。此参数设定了基于步数（step）的模型保存间隔。例如设置为 20000，则每训练 20000 步保存一次模型，从而防止训练中断导致模型丢失。若设置为 -1，则不保存epoch中间步骤模型。
*   **默认值 (Default)**: `-1`
*   **类型 (Type)**: `<INT>` (整数)