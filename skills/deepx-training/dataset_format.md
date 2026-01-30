# DeepH 数据集格式

## 概览 (Overview)

要使用 DeepH-pack 训练模型，用户需要准备以下内容：

1. 一个配置文件，命名为 `<用户定义名称>.toml`，例如 `my_train.toml`；
2. 训练数据，可以是 [DeepH-pack 的统一 DFT 数据格式](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html)（请注意，文件夹名称 **必须** 为 `dft/`）：

```bash
inputs/
  |- dft/                # DFT 数据文件夹（可选，如果存在 graph 文件夹）
    |- <sid>               # 结构 ID (Structure ID)
       |- info.json        # 额外信息
       |- POSCAR           # 原子结构
       |- overlap.h5       # {R} 空间中的基组重叠矩阵
       |- (hamiltonian.h5) # {R} 空间中的哈密顿量条目
       |- (position.h5)
       |- (charge_density.h5)
       |- (density_matrix.h5)
       |- (force.h5)
    |- ...
```

或者使用 DeepH-pack 图文件格式（文件夹名称 **必须** 为 `graph/`）：

```bash
inputs/
  |- graph               # 图文件夹（可选）
    |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.npz
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.npz
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.npz

```

然后运行以下命令以开始训练。**如果用户从 DFT 数据格式开始，图文件将自动生成。**

```bash
deeph-train my_train.toml
```

## 原始训练数据与图文件

**开始 DeepH 训练的方式**：

- 要么准备并提供 DFT 原始训练数据目录 `dft/`，这允许在训练开始时自动构建图。更多详情，请参考我们的开源数据接口平台 [`DeepH-dock`](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html)。
- 要么提供预构建的图文件 `graph/`（例如，从外部来源传输到 GPU 集群）。

DeepH-pack 完全支持这两种方式。

```bash
inputs/
  |- dft/                # DFT 数据文件夹（可选，如果存在 graph 文件夹）
    |- <sid>               # 结构 ID
       |- info.json        # 额外信息
       |- POSCAR           # 原子结构
       |- overlap.h5       # {R} 空间中的基组重叠矩阵
       |- (hamiltonian.h5) # {R} 空间中的哈密顿量条目
       |- (position.h5)
       |- (charge_density.h5)
       |- (density_matrix.h5)
       |- (force.h5)
    |- ...
  |- graph               # 图文件夹（可选）
    |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.npz
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.npz
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.npz

```

作为一个基于 GNN 的框架，DeepH-pack 基于图文件运行。构建这些图文件是工作流中的必要步骤，既可以与训练程序一起执行，也可以作为单独的预处理任务执行。从技术上讲，图文件是直接从 DFT 数据转换而来的。与涉及分散的原始数据文件夹的传统存储方法相比，图文件系统提供了几个关键优势：

- **数值精度灵活性 (Numerical Precision Flexibility)：** DeepH-pack 支持 32 位和 64 位浮点精度，允许用户根据设备内存容量选择合适的设置。
- **统一的数据便携性 (Unified Data Portability)：** 图文件被打包为单个集成文件，比碎片化的原始数据文件夹更容易在服务器或集群之间传输。
- **通用兼容性 (Generalized Compatibility)：** 凭借其通用的数据结构，图文件格式不仅兼容 DeepH 框架，还具有扩展到其他神经网络架构的潜力。

在 DeepH-pack 中，图文件夹布局如下所示：

```bash
graph/
  |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.npz
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.npz
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.npz
```

存放原始 DFT 数据的根目录名称（此处指图文件目录）**必须为 `graph/`**，且所有图文件均需位于该目录下。

DeepH-pack 目前支持两种不同的图文件存储模式：

- `memory`（内存）模式。它在 DeepH 训练初始化期间将整个图文件预加载到节点内存中，优先考虑与可用内存资源兼容的数据集的运行效率。
- `disk`（磁盘）模式。它采用通过集成数据库-硬件存储解决方案的按需数据流传输，专为超出节点内存容量（例如 >10 TiB）的超大图文件设计。

这种双模式架构通过动态适应数据规模，确保了与内存无关的训练工作流。其中磁盘模式允许在计算期间进行实时访问，而无需占用全部内存，从而在不同的计算约束下保持系统的灵活性。

### 可选操作：单独构建图文件

启动标准 DeepH 训练会话时，框架会自动根据 `dft/` 目录中的 DFT 数据构建图文件并生成相应的图 `dataloader`。然而，鉴于图构建是 CPU 独占的性质以及图文件在数据便携性方面的固有优势，DeepH-pack 也支持将图生成与 GPU 加速的训练过程解耦。此外，如果图文件已存在，训练会话将跳过原始 DFT 数据，通过基于图的数据抽象简化训练工作流。

`build_graph.toml`:

```toml
# ----------------------------- SYSTEM -----------------------------
[system]
note = "Welcome to DeepH-pack!"
device = "cpu"
float_type = "fp32" # 或 `fp64`
random_seed = 137

# ----------------------------- DATA -------------------------------
[data]
inputs_dir = "."          # 包含 `dft` 和 `graph` 的输入路径
outputs_dir = "./logs"    # 日志路径

[data.graph]
dataset_name = "H2O_5K"
graph_type = "HS"         # 图将包含哈密顿量和重叠矩阵
storage_type = "memory"   # 或 `disk`
parallel_num = -1         # 构建图时的并行进程数
only_save_graph = true    # 仅生成并保存图的任务
```

然后，您可以使用以下命令在不开始训练过程的情况下构建数据图文件：

``` bash
deeph-train build_graph.toml
```

**注意：** 对于 `only_save_graph` 任务，不需要 GPU 版的 JAX 和 Flax的安装，只需要numpy等CPU端库即可。

```bash
uv pip install ./deepx-1.2.x-py3-none-any.whl[cpu]
```

### 可选（但强烈建议做的）操作：检查 DFT 数据集特征

完成数据准备后，您可以选择对数据集进行全面分析。深入了解数据集特征对于优化超参数配置和加速模型收敛至关重要。为此，[DeepH-dock](https://github.com/kYangLi/DeepH-dock) 提供了 `dock analyze dataset features` 命令用于数据集分析。详情请参阅 [文档](https://docs.deeph-pack.com/deeph-dock/en/latest/capabilities/analyze/dataset/demo.html#dft-data-features)。

```bash
# 如果DeepH-dock未安装，执行：
uv pip install deepx-dock # 在您的 uv 虚拟环境中安装
# 如果已安装，跳过上一步骤
cd ~/deeph-train/inputs # 您的 DeepH 训练任务根目录/inputs 文件夹
dock analyze dataset features . -p 8
```

执行后，您将收到类似于以下的输出：

```bash
📊 BASIC DATASET INFO (基本数据集信息)
-----------------------
  • Spinful:                False (有自旋: 否)
  • Parity consideration:   False (宇称考虑: 否)
  • Total data points:      4,999 (数据点总数)

🧪 ELEMENT & ORBITAL INFO (元素与轨道信息)
---------------------------
  • Elements included:      H, O (2 elements) (包含元素: H, O)
  • Orbital source:         auto_detected (轨道来源: 自动检测)
  • Common orbital types:   s3p2d1f1 (常见轨道类型)

🎯 IRREPS INFORMATION (不可约表示信息)
-----------------------
  Irreps Type          Irreps                                             Dimension
  .................... .................................................. ..........
  Common orbital       15x0e+24x1e+22x2e+18x3e+8x4e+3x5e+1x6e             441
  Suggested            16x0e+24x1e+24x2e+24x3e+8x4e+4x5e+2x6e             518
  Exp2                 32x0e+16x1e+8x2e+4x3e+2x4e+2x5e+2x6e               214
  Trivial              32x0e+32x1e+32x2e+32x3e+32x4e+32x5e+32x6e          1568

💡 RECOMMENDATIONS (建议)
--------------------
  1. Moderate dataset size - regular training recommended (数据集规模适中 - 建议常规训练)
  2. High-dimensional irreps - consider dimensionality reduction techniques (高维不可约表示 - 考虑降维技术)
```
