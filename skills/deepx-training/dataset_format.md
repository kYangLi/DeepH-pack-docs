# Model Training

## Overview

To train a model via DeepH-pack, the user needs to prepare:

1. a configuration file, named as `<user_defined_name>.toml`, such as `my_train.toml`;
2. the training data, either in [DeepH-pack's unified DFT data format](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html) (please note that the folder **must** have the name `dft/`):

```bash
inputs/
  |- dft/                # DFT data folder (optional, if graph folder exist)
    |- <sid>               # Structure ID
       |- info.json        # Additional information
       |- POSCAR           # Atomic structures
       |- overlap.h5       # Overlap matrices of basis in {R}
       |- (hamiltonian.h5) # Hamiltonian entries in {R}
       |- (position.h5)
       |- (charge_density.h5)
       |- (density_matrix.h5)
       |- (force.h5)
    |- ...
```

or in the DeepH-pack graph file format (the folder **must** have the name `graph/`):

```bash
inputs/
  |- graph               # Graph folder (optional)
    |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.pt
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.pt
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.pt

```

and run the command

```bash
deeph-train my_train.toml
```

to strat training. If the user starts from the unified DFT data format, the graph files will be generated automatically.

## The raw training data and Graph files

**To commence DeepH training**:

- Either prepare and provide the DFT raw training data directory `dft/`, which allows for automatic graph construction at the start of training. For further details, please refer to our open‑source data interface platform [`DeepH‑dock`](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html).
- Or supply pre‑built graph files `graph/` (e.g., transferred from external sources to the GPU cluster).

Both approaches are fully supported by DeepH-pack.

```bash
inputs/
  |- dft/                # DFT data folder (optional, if graph folder exist)
    |- <sid>               # Structure ID
       |- info.json        # Additional information
       |- POSCAR           # Atomic structures
       |- overlap.h5       # Overlap matrices of basis in {R}
       |- (hamiltonian.h5) # Hamiltonian entries in {R}
       |- (position.h5)
       |- (charge_density.h5)
       |- (density_matrix.h5)
       |- (force.h5)
    |- ...
  |- graph               # Graph folder (optional)
    |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.pt
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.pt
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
    |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.pt

```

As a GNN-based framework, DeepH-pack operates on graph files. Constructing these graph files is an essential step in the workflow, which can be performed either together with the training routine or as a separate pre-processing task. Technically, graph files are converted directly from DFT data. Compared to traditional storage methods involving scattered folders of raw data, the graph file system offers several key advantages:

- **Numerical Precision Flexibility:** DeepH-pack supports both 32-bit and 64-bit floating-point precision, enabling users to select the appropriate setting based on their device's memory capacity.
- **Unified Data Portability:** Packaged as single integrated files, graph files are significantly easier to transfer between servers or clusters than fragmented raw data folders.
- **Generalized Compatibility:** With its universal data structure, the graph file format is compatible not only with the DeepH framework but also potentially extensible to other neural network architectures.

In DeepH-pack, the graph folder layout looks like this:

```bash
graph/
  |- <GRAPH_NAME>.<GRAPH_TYPE>.memory.pt
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.pt
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.db/
  |- <GRAPH_NAME>.<GRAPH_TYPE>.disk.part1-of-1.info.pt
```

The root directory for the raw DFT data **must be named as `graph/`**, with all graph files residing within this directory.

DeepH-pack currently supports two distinct storage modes for graph files:

- The `memory` mode. It pre-loads the entire graph file into node memory during DeepH training initialization, prioritizing operational efficiency for datasets compatible with available memory resources.
- The `disk` mode. It employs on-demand data streaming through integrated database-hardware storage solutions, specifically designed for over-sized graph files exceeding node memory capacity (e.g., >10 TiB).

This dual-mode architecture ensures memory-agnostic training workflows by dynamically adapting to data scales, where disk mode enables real-time access during computation while bypassing full memory occupancy, thereby maintaining system flexibility across varying computational constraints.

### Building the Graph files separately (Optional)

Upon initiating a standard DeepH training session, the framework automatically constructs graph files from DFT data stored in the designated `dft/` directory and generates the corresponding graph `dataloader`. However, given the CPU-exclusive nature of graph construction and the inherent advantages of graph files in data portability, DeepH-pack also supports decoupled graph generation from the GPU-accelerated training process. And, if graph files already exist, the training sessions would skip raw DFT data, streamlining the training workflow through graph-based data abstraction.

`build_graph.toml`:

```toml
# ----------------------------- SYSTEM -----------------------------
[system]
note = "Welcome to DeepH-pack!"
device = "cpu"
float_type = "fp32" # or `fp64`
random_seed = 137

# ----------------------------- DATA -------------------------------
[data]
inputs_dir = "."          # Inputs path that contains `dft` and `graph`
outputs_dir = "./logs"    # Logging path

[data.graph]
dataset_name = "H2O_5K"
graph_type = "HS"         # Graph will include both Hamiltonian and Overlap matrices
storage_type = "memory"   # Or `disk`
parallel_num = -1         # Parallel processes during build graph
only_save_graph = true    # A task for generate and save graph only
```

You can then use the following command to build data graph file without starting the training process:

``` bash
deeph-train build_graph.toml
```

**NOTE:** For the `only_save_graph` task, GPUs are not required. You establish the DeepH-pack under CPU lib supporting as specified in [Installation & Setup](../installation_and_setup.md).

```bash
uv pip install ./deepx-1.0.6+light-py3-none-any.whl[cpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

### Inspect the DFT Dataset and Graph Set (Optional)

After completing data preparation, you may optionally perform a comprehensive analysis of your dataset. A thorough understanding of the dataset characteristics is crucial for optimizing hyperparameter configuration and accelerating model convergence. To facilitate this, the `deepx` package provides integrated utility tools. Once the package is installed, these analytical tools become readily accessible from your command-line interface.

[DeepH-dock](https://github.com/kYangLi/DeepH-dock) offers the `dock analyze dataset features` command for dataset analysis. [See the documentation](https://docs.deeph-pack.com/deeph-dock/en/latest/capabilities/analyze/dataset/demo.html#dft-data-features) for details.

```bash
# 如果DeepH-dock未安装，执行：
uv pip install deepx-dock # Install inside your uv venv
# 如果已安装，跳过上一步骤
cd ~/deeph-train/inputs # Your DeepH training task root/inputs folder
dock analyze dataset features . -p 8
```

After execution, you will receive output similar to the following:

```bash
📊 BASIC DATASET INFO
-----------------------
  • Spinful:                False
  • Parity consideration:   False
  • Total data points:      4,999

🧪 ELEMENT & ORBITAL INFO
---------------------------
  • Elements included:      H, O (2 elements)
  • Orbital source:         auto_detected
  • Common orbital types:   s3p2d1f1

🎯 IRREPS INFORMATION
-----------------------
  Irreps Type          Irreps                                             Dimension
  .................... .................................................. ..........
  Common orbital       15x0e+24x1e+22x2e+18x3e+8x4e+3x5e+1x6e             441
  Suggested            16x0e+24x1e+24x2e+24x3e+8x4e+4x5e+2x6e             518
  Exp2                 32x0e+16x1e+8x2e+4x3e+2x4e+2x5e+2x6e               214
  Trivial              32x0e+32x1e+32x2e+32x3e+32x4e+32x5e+32x6e          1568

💡 RECOMMENDATIONS
--------------------
  1. Moderate dataset size - regular training recommended
  2. High-dimensional irreps - consider dimensionality reduction techniques
```

**Note on Tool Location:** The graph analysis functionality is included in the `deeph-pack` (specifically within the `deepx` module) rather than in `dock` because graph processing is intrinsically closer to the core training workflow. Furthermore, analyzing graph data requires direct calls to PyTorch libraries, making it a more natural fit within the `deeph-pack` ecosystem.
