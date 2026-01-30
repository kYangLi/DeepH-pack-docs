# DeepH-pack 训练参数文档: Model 部分

---

**`[model.net_type]`**
*   **说明 (Description)**: 神经网络架构。
    *   `sparrow` (也称为 `normal`)：一种轻量级架构（参数量通常 $<1$M），包含节点和边特征，适用于 DFT 哈密顿量学习的小型任务。
    *   `eagle`：一种高级架构（参数量 $\sim$5M），包含节点和边特征，适用于需要高精度的 DFT 哈密顿量学习任务。
    *   `eagleplus` (也称为 `accurate`)：一种高级架构（参数量 $\sim$10M），包含节点和边特征，适用于需要高精度的 DFT 哈密顿量学习任务。
    *   `albatross`：一种特级架构（参数量 $\sim$20M），包含节点和边特征，适用于需要超高精度超大数据规模的 DFT 哈密顿量学习任务。
    *   `owl`：一种高级架构（参数量 $\sim$5M），仅包含节点特征，适用于需要高精度的 DFT 力场学习任务。
    *   `petrel`：一种特级架构（参数量 $\sim$20M），仅包含节点特征，适用于需要超高精度超大数据规模的 DFT 力场学习任务。
    *   `swift`：一种自动编码器（auto-encoder）网络架构。
*   **默认值 (Default)**: `"normal"`
*   **类型 (Type)**: [`"sparrow"`, `"fast"`, `"eagle"`, `"accurate"`, `"eagleplus"`, `"albatross"`, `"owl"`, `"swift"`, `"petrel"`]

---

**`[model.target_type]`**
*   **说明 (Description)**: 待学习的物理量。`H` 代表哈密顿量，`Rho` 代表密度矩阵，`"FF"`代表力场，`"IEP"`代表原子间势能。
*   **默认值 (Default)**: `"H"`
*   **类型 (Type)**: [`"H"`, `"Rho"`, `"VrC"`, `"Vc"`, `"FF"`, `"IEP"`]

---

**`[model.loss_type]`**
*   **说明 (Description)**: 训练过程中的损失函数。损失类型不仅支持监督学习类（如 `mse`、`mae`、`"huber"`等），还支持即时（on-the-fly）非监督损失（如 `ai2dft`、`ai2dft_node`、`hopad` 和 `aims`）。
*   **默认值 (Default)**: `"mse"`
*   **类型 (Type)**: [`"mae"`, `"mse"`, `"wmae"`, `"huber"`, `"ai2dft"`, `"ai2dft_node"`, `"hopad"`, `"aims"`]

---

### **Model: Advanced (高级设置)**

---

**`[model.advanced.gaussian_basis_rmax]`**
*   **说明 (Description)**: 用于高斯基组采样的截断半径（单位：埃 Å）。建议将此截断值设置为您轨道基组最大截断半径的 2 倍。详情请参考 [`DeepH-E3`](https://www.nature.com/articles/s41467-023-38468-8) 论文。
*   **默认值 (Default)**: `7.5`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[model.advanced.net_irreps]`**
*   **说明 (Description)**: 用于定义神经网络特征的**不可约表示**（Irreducible Representations），以此保证网络的旋转等变性（Equivariance）。
    *   **配置格式**: 遵循 [`e3nn.Irreps`](https://e3nn-jax.readthedocs.io/en/latest/api/irreps.html) 语法标准，通常设置为形如 `"128x0e+64x1e+32x2e..."` 的字符串。
    *   **语法解析**: 例如 `"64x1e"` 表示该特征是一个 64 通道的张量，其中每个通道携带 $SO(3)$ 群的 $l=1$ 表示（即矢量），并具有偶（"e", even）宇称。`o`/`e` 分别代表奇（odd）/偶（even）宇称。
    *   **注意事项**: **所有通道通常必须设置为偶（even）宇称**。这描述了输入特征的对称性，其数学形式对应于：$$ [(mul\_l, (l, p\_val \cdot (p\_arg)^l)) \quad \text{for } l \in [0, \dots, l_{max}]] $$
*   **默认值 (Default)**: `<Invalid-Input>`
*   **类型 (Type)**: `<STRING>` (字符串)

---

**`[model.advanced.latent_irreps]`**
*   **说明 (Description)**: 用于定义**潜在空间**（Latent Space）特征的不可约表示，同样用于确保网络的等变性。
    *   **适用范围**: 专用于 `albatross` 和 `petrel` 等包含潜在层的高级网络架构。
    *   **配置说明**: 其格式语法与 `[model.advanced.net_irreps]` 完全一致，但具体数值（通道数、l阶数等）可以独立于 `net_irreps` 进行调整，以控制模型中间层的特征维度和表达能力。
*   **默认值 (Default)**: `<Invalid-Input>`
*   **类型 (Type)**: `<STRING>` (字符串)
---

**`[model.advanced.latent_edge_cutoff]`**
*   **说明 (Description)**: 隐空间神经网络的建图时连边的截断距离。默认为100.0埃, 也即不做任何额外截断。调整该参数可能对于某些任务有帮助。
*   **默认值 (Default)**: `100.0`
*   **类型 (Type)**: `<FLOAT>` (浮点数)

---

**`[model.advanced.latent_scalar_dim]`**
*   **说明 (Description)**: 中间层标量特征的维度。通过压缩或扩充中间层表示，可协调模型的效率、精度、表达能力之间的平衡。默认为-1,代表隐藏层不做压缩。
*   **默认值 (Default)**: `-1`
*   **类型 (Type)**: `<INT>` (字符串)

---

**`[model.advanced.num_blocks]`**
*   **说明 (Description)**: 模型中神经网络层（模块）的数量。对于常规任务，建议固体体系设置为 3，小分子体系设置为 4。
*   **默认值 (Default)**: `3`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[model.advanced.num_heads]`**
*   **说明 (Description)**: 多头注意力机制（Multi-head Attention）中的头数。
*   **默认值 (Default)**: `2`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[model.advanced.enable_bs3b_layer]`**
*   **说明 (Description)**: 是否启用 BS3B 层。启用后可使模型对化学元素的区分度更高。BS3B 层对通用模型训练至关重要。
*   **默认值 (Default)**: `2`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[model.advanced.bs3b_orbital_types]`**
*   **说明 (Description)**: BS3B 层的轨道类型设置。其格式与 `data.graph.common_orbital_types` 一致，单个原子的轨道列表，按照 `l`（角量子数）的值排列，格式为`sNpNdNfN...`，例如 `s2p2d1`。一般需要设置每个l上的channel数远大于 `common_orbital_types`。是十分关键的参数。
*   **默认值 (Default)**: `2`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[model.advanced.consider_parity]`**
*   **说明 (Description)**: 网络是否在宇称下保持等变。如果设置为 `false`，则 `net_irreps` 中不能出现奇宇称（例如 `2x3o`）表示。在当前版本的大部分网络中，`consider_parity` 必须设为 `false`。
*   **默认值 (Default)**: `true`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[model.advanced.standardize_gauge]`**
*   **说明 (Description)**: 训练模型时是否考虑规范（Gauge，此处指 DFT 中的零能量点）的任意性。如果设置为 `true`，`graph_type` 必须为 `HS`。建议对通用数据集（或当训练数据存在零能量点任意性问题时，例如包含不同厚度的二维平板的数据集）设置为 `true`，对特定用途数据集设置为 `false`。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[model.advanced.vr_focus_size]`**
*   **说明 (Description)**: 训练Vr、nr任务时的撒点数目，一般无需更改。
*   **默认值 (Default)**: `256`
*   **类型 (Type)**: `<INT>` (整数)

---

**`[model.advanced.enable_element_moe]`**
*   **说明 (Description)**: 启动元素混合多专家模型训练，构造MOE模型。
*   **默认值 (Default)**: `false`
*   **类型 (Type)**: `<BOOL>` (布尔值)

---

**`[model.advanced.moe_element_include]`**
*   **说明 (Description)**: 当前训练多专家模型时，需要包含的元素列表。例如 `["H", "C", "O"]`。默认为空，代表包含数据集中所有元素。
*   **默认值 (Default)**: `[]`
*   **类型 (Type)**: `<LIST-OF-STRING>` (字符串列表)
