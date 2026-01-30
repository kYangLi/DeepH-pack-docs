# DeepH-pack 训练参数文档: System 部分

---

**`[system.note]`**
*   **说明 (Description)**: 本训练项目的名称。
*   **默认值 (Default)**: `"Enjoy DeepH-pack! ;-)"`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[system.device]`**
*   **说明 (Description)**: 设备配置遵循 `<type>*<num>:<id>` 语法，其中 `<type>` 指定硬件类型（`cpu`、`gpu`、`rocm`、`dcu` 或 `cuda`），`<num>` 表示每个节点的设备总数（对于 GPU 等加速器）或 CPU 分区数（使用 `cpu` 时），`<id>` 定义目标设备索引（例如，`gpu*8:1-4,7` 表示使用索引 1,2,3,4,7 从 8 设备节点中选择 5 个 GPU）。注意：对于 CPU 配置，`<id>` 被忽略，而 `<num>` 控制线程分区。
*   **默认值 (Default)**: `"gpu*8:0"`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[system.float_type]`**
*   **说明 (Description)**: 用于构建图和训练模型的浮点类型。对于大多数哈密顿量任务，`"fp32"` 足够精确，浮点误差约为 $0.001\sim 0.01$ meV。
*   **默认值 (Default)**: `"fp32"`
*   **类型 (Type)**: [`"bf16"`, `"tf32"`, `"fp32"`, `"fp64"`]

---

**`[system.random_seed]`**
*   **说明 (Description)**: 生成随机数的种子。
*   **默认值 (Default)**: `137`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[system.log_level]`**
*   **说明 (Description)**: `deepx.log` 的严重程度级别，按 debug、info、warning、error 和 critical 的顺序递增。
*   **默认值 (Default)**: `"info"`
*   **类型 (Type)**: [`"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"`]

---

**`[system.jax_memory_preallocate]`**
*   **说明 (Description)**: 是否在运行前预分配 $75\%$ 的剩余内存。正式训练时请使用 `true`；调试以观察 GPU 内存使用情况时请使用 `false`。在训练期间使用 `false` 可能会导致进程中途意外崩溃或减慢训练过程。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[system.show_train_process_bar]`**
*   **说明 (Description)**: 是否在命令行中显示图形化进度条。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---