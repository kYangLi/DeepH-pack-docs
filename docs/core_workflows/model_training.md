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

- Either prepare and provide the DFT raw training data directory `dft\`, which allows for automatic graph construction at the start of training. For further details, please refer to our open‑source data interface platform [`DeepH‑dock`](https://docs.deeph-pack.com/deeph-dock/en/latest/key_concepts.html).
- Or supply pre‑built graph files `graph\` (e.g., transferred from external sources to the GPU cluster).

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

### Inspect the DFT dataset and Graph set (Optional)

Upon completion of data preparation, *an optional pre-computation step* remains: comprehensive dataset analysis. Thorough understanding of dataset characteristics enables optimized hyperparameter configuration and accelerated model convergence. To facilitate this, we provide integrated utility tools within the `deepx` package. These analytical tools become automatically accessible in your command-line interface after package installation.

The `deepx-1.0.6+light` package includes the `deeph-Tool-InspectDataset` command for analyzing raw DFT data features.

```bash
cd ~/deeph-train/inputs # Your deeph training task root/inputs folder
deeph-Tool-InspectDataset . --task dft -n 1
```

After which you will get the message like,

```bash
---------------------------------------------------------------
[info] Spinful:                False
[info] User needs parity:      False
[info] DFT data quantity:      4999
[info] Elements included:      ['H', 'O']
[info] Common orbital types:   s3p2d1f1
[info] Irreps common orbital:  15x0e+24x1e+22x2e+18x3e+8x4e+3x5e+1x6e (441, regrouped)
[info] Irreps in as suggested: 16x0e+24x1e+24x2e+24x3e+8x4e+4x5e+2x6e (518)
[info] Irreps in as exp2:      16x0e+8x1e+4x2e+2x3e+2x4e+2x5e+2x6e (140)
[info] Irreps in as trivial:   16x0e+16x1e+16x2e+16x3e+16x4e+16x5e+16x6e (784)
---------------------------------------------------------------
```

For a analyze the Graph data file, execute:

```bash
deeph-Tool-InspectDataset . --task graph -n 1
```

It will return,

```bash
---------------------------------------------------------------
[info] Dataset Name:              H2O_5K
[info] Graph Type:                train-HS
[info] Elements Included:         ['H', 'O']
[info] Spinful:                   False
[info] Common Orbital Types:      s3p2d1f1
[info] Common Fitting Types:      None
[info] Structure Quantity:        4999
[info] All Entries:               19841031
[info] Shape Masked in All:       27.4%
[info] Real Masked in Shape:      100.0%
[info] Real Entries:              5443911
---------------------------------------------------------------
```

## Example: DeepH model training on FHI-aims processed H2O-5k molecular dataset

With prepared training data in the `inputs/` directory:

- Create a minimal TOML configuration file `train.toml`
- For local GPU workstations, execute the following commands:

```bash
cd ~/deeph-train/H2O_5K_FHI-aims
deeph-train train.toml
```

- For HPC cluster environments (current configuration), prepare a job submission script,

```bash
#!/bin/bash
#
#SBATCH --job-name=H2O-5k-FHI-aims
#SBATCH --gpus=1
#SBATCH -p gpu

module load cuda/12.9               # Load CUDA
source ~/.uvenv/deeph/bin/activate  # Load DeepH venv

deeph-train train.toml
```

And submit the job,

```bash
sbatch submit.sh
```

---

The TOML configuration file comprises four core sections:

- `system`: handles hardware and computational environment declarations, etc.
- `data`: specifies training dataset locations, features, and metadata, etc.
- `model`: defines base architecture components and target physical quantities (not limited to the Hamiltonian – future releases will progressively support force fields, interatomic potentials, charge density, density matrices, GW calculations, etc.), including loss function selection.
- `process`: controls training/inference workflows through convergence criteria, data loader configurations, optimizers, restart settings, etc.

Due to space and time constraints, this section cannot exhaustively cover all TOML configuration details. Only the most critical parameters for the current model are presented below. You can refer the *User Guide Book* for more details.

- `system.device`: The device configuration follows the syntax `<type>*<num>:<id>`, where `<type>` specifies hardware type (`cpu`, `gpu`, `tpu`, `rocm`, `dcu`, or `cuda`), `<num>` denotes either the total devices per node (for accelerators like GPU) or the number of CPU partitions (when using `cpu`), and `<id>` defines target device indices (e.g., `gpu*8:1-4,7` selects 5 GPUs from an 8-device node using indices 1,2,3,4,7, `gpu*3` all GPUs for an 3-device node). **Note that**, for CPU configurations, `<id>` is ignored while `<num>` controls thread partitioning.

- `model.net_type`: The neural network architecture. In DeepH-pack Light version, two architectures are available:
  - `sparrow` is a light-weighted architecture (typically <1M parameters) with both node and edge features, which is suitable for small tasks of DFT Hamiltonian learning. (DeepH-E3 like networks)
  - `eagle` is an advanced architecture (typically $\sim$ 5M parameters) with both node and edge features, which is suitable for tasks of DFT Hamiltonian learning that requires high accuracy.

- `model.advanced.net_irreps`: Irreducible representations of the neural network features, which ensure the equivariance of the network. Set in the string form of [`e3nn.Irreps`](https://e3nn-jax.readthedocs.io/en/latest/api/irreps.html), namely *irreducible representations*, which describes the symmetry of input features. **Note that**, For Hamiltonian prediction tasks, the maximum $l$ specified in the `Irreps` must be at least twice the highest angular momentum quantum number present in the Hamiltonian's basis set. This requirement arises because Irreps transform the uncoupled representation (direct product basis) of the Hamiltonian into a coupled representation (direct sum basis). For example, when f-orbitals ($l=3$) are included, the Irreps must support $l_{\text{max}} \geq 6$.

- `process.train.optimizer.init_learning_rate`: Starting learning rate.

- `process.train.scheduler.min_learning_rate_scale`: The minimum scaling factor for the learning rate. Training automatically terminates when the learning rate multiplier reaches this threshold, at which point the effective learning rate becomes `scale` $\times$ `initial_learning_rate`.

The complete configuration file for GPU training is shown as follows:

```toml
# ---------------------------------- SYSTEM ----------------------------------
[system]
note = "DeepH-JAX"
device = "gpu*1"
float_type = "fp32"
random_seed = 137
log_level = "info"
jax_memory_preallocate = true
show_train_process_bar = false

# ----------------------------------- DATA ------------------------------------
[data]
inputs_dir = "./inputs"
outputs_dir = "./outputs"

[data.dft]
data_dir_depth = 0
validation_check = false

[data.graph]
dataset_name = "H2O_5K"
graph_type = "HS"
storage_type = "memory"
common_orbital_types = ""
parallel_num = -1
only_save_graph = false

[data.model_save]
best = true
latest = true
latest_interval = 1
latest_num = 10

# ----------------------------- MODEL -----------------------------------------
[model]
net_type = "eagle"
target_type = "H"
loss_type = "mae"

[model.advanced]
gaussian_basis_rmax = 10.0
net_irreps = "64x0e+48x1e+32x2e+16x3e+8x4e+8x5e+4x6e"
num_blocks = 3
num_heads = 2
enable_bs3b_layer = false
bs3b_orbital_types = ""
consider_parity = false
standardize_gauge = false

# ------------------------------ PROCESS --------------------------------------
[process.train]
max_epoch = 10000

multi_way_jit_num = 1
ahead_of_time_compile = true
do_remat = false

[process.train.dataloader]
batch_size = 100

train_size = 3000
validate_size = 1000
test_size = 999
dateset_split_json = ""
only_use_train_loss = false

[process.train.drop]
dropout_rate = 0.0
stochastic_depth = 0.0
proj_rate = 0.0

[process.train.optimizer]
type = "adamw"
init_learning_rate = 1E-3
clip_norm_factor = -1.0
betas = [0.9, 0.999]
weight = 0.001
eps = 1E-8

[process.train.scheduler]
min_learning_rate_scale = 1E-5
type = "reduce_on_plateau"
factor = 0.5
patience = 500
rtol = 0.05
cooldown = 120
accum_size = -1

[process.train.continued]
enable = false
new_training_data = false
new_optimizer = false
previous_output_dir = ""
load_model_type = "latest"
load_model_epoch = -1
```

**NOTE:**

- In the `process.train.dataloader` configuration, the sum of `train_size`, `validation_size`, and `test_size` must not exceed the total number of snapshots in the dataset. Violating this constraint will trigger a `ValueError()` exception.
- Currently, DeepH *does not* support cross-node CPU inference. This functionality will be implemented in future releases.

## Monitoring the training process

After training commences, DeepH automatically constructs a structured output directory with the following base hierarchy:

```bash
outputs/<timestamp>
  |- dataset_split.json
  |- deepx.log          # <-- DeepH logging file
  |- model/
      |- train.toml     # <-- Copy of training settings
      |- variables.json # <-- Model.__init__(variables)
      |- params/        # <-- Model parameters store here
        |- best.pytree
            |- epoch_124/
        |- latest.pytree
            |- epoch_120/
            |- epoch_110/
            |- epoch_100/
            |- ...
      |- states/        # <-- Optimizer states store here
        |- best.pytree
        |- latest.pytree
```

The `deepx.log` file enables real-time monitoring of training progress throughout the execution.

``` bash
tail -f outputs/2025-11-20_11-54-34/deepx.log
```

``` text
[ 11.20-11:54:34 ]
[ 11.20-11:54:34 ]            Welcome to DeepH-pack (deepx)!
[ 11.20-11:54:34 ]                  Version 1.0.6
[ 11.20-11:54:34 ]
[ 11.20-11:54:34 ] ...................................................
[ 11.20-11:54:34 ] ........_____....................._...._.[PACK]....
[ 11.20-11:54:34 ] .......|  __ \...................| |..| |..........
[ 11.20-11:54:34 ] .......| |  | | ___  ___ ._ _ ...| |..| |..........
[ 11.20-11:54:34 ] .......| |  | |/ _ \/ _ \| '_ \ .|X'><'X|..........
[ 11.20-11:54:34 ] .......| |__| |. __/. __/| |_) |.| |..| |..........
[ 11.20-11:54:34 ] .......|_____/ \___|\___|| .__/ .|_|..|_|..........
[ 11.20-11:54:34 ] .........................| |.......................
[ 11.20-11:54:34 ] .........................|_|.......................
[ 11.20-11:54:34 ] ...................................................
[ 11.20-11:54:34 ]
[ 11.20-11:54:34 ]             Copyright CMT@Phys.Tsinghua
[ 11.20-11:54:34 ]                  Powered by JAX
[ 11.20-11:54:34 ]
[ 11.20-11:54:34 ]
[ 11.20-11:54:34 ] [system] Under the machine `sppz4531@gn32317`, with `x86_64 (64 cores)` CPU, and `1007GB` RAM.
[ 11.20-11:54:41 ] [system] Use the GPU device(s) `[0]` of totally `1` device(s). Succeeded test on the head device `cuda:0`!
[ 11.20-11:54:41 ] [system] Totally use `64` CPU cores.
[ 11.20-11:54:42 ] [system] The calculation will be sharding across `Mesh('data': 1, axis_types=(Auto,))`.
[ 11.20-11:54:42 ] [system] Set random stream with seed `137`, type `key<fry>`.
[ 11.20-11:54:42 ] [system] Using the float type `fp32`. The testing results on JAX and PyTorch are `jnp.float32` and `torch.float32`
[ 11.20-11:55:14 ] [graph] Building the graph with type: `train-HS`.
[ 11.20-11:55:14 ] [rawdata] Read in features from json: `/home/sppz4531/workshop/4.Training/H2O_5K_FHI-aims/inputs/dft/features.json`.
[ 11.20-11:55:14 ] [graph] Establishing the graph from the DFT raw data: `/home/sppz4531/workshop/4.Training/H2O_5K_FHI-aims/inputs/dft`.
[ 11.20-11:55:14 ] [graph] Start processing `4999` structures with `32` processes.
[ 11.20-11:55:38 ] [graph] Finish processing `4999` structures.
[ 11.20-11:55:39 ] [graph] Saved the graph set in memory to `/home/sppz4531/workshop/4.Training/H2O_5K_FHI-aims/inputs/graph/H2O_5K.train-HS.memory.pt`.
[ 11.20-11:55:39 ] [graph] Using the common orbital types: `[0, 0, 0, 1, 1, 2, 3]`
[ 11.20-11:55:39 ] [graph] Split the graph set with batch size: `100`
[ 11.20-11:55:40 ] [dataloader] Train size: `3000`. Val size: `1000`. Test size: `999`.
[ 11.20-11:55:40 ] [dataloader] Data sharding way: `1`. Batch size: `100`. Number of nodes each batch: `[302]`, `[302]`, `[302]`. Number of edges each batch: `[1202]`, `[1202]`, `[1202]`.
[ 11.20-11:55:40 ] [dataloader] The training dataset encompasses `27000` edges, aggregating a total of `3267000` data entries.
[ 11.20-11:55:40 ] [model] Building the model `eagle-H` with loss `mae`.
[ 11.20-11:55:42 ] [model] Initializing the net parameters with dummy data...
[ 11.20-11:56:36 ] [model] The parameters size is `7684960`.
[ 11.20-11:56:39 ] [optimizer] Using the optimizer `AdamW` with: betas `[0.9, 0.999]`, eps `1e-08`, weight decay strength `0.001`, and initial learning rate `0.001`.
[ 11.20-11:56:39 ] [optimizer] The global CLIP norm algo factor is NOT USED.
[ 11.20-11:56:39 ] [optimizer] Using the scheduler `ReduceOnPlateau` with: factor `0.5`, patience `500`, rtol `0.05`, cooldown `120`, and accumulation size `30`.
[ 11.20-11:56:43 ] [model] We will save the model into `/home/sppz4531/workshop/4.Training/H2O_5K_FHI-aims/outputs/2025-11-20_11-54-09/model`. The best model will be saved. The latest model (keep `10` each `1` epoch) will be saved.
[ 11.20-11:56:43 ] [train] JAX networks: Parallel threads AOT-compiling `1` frameworks for training and `1` for validation.
[ 11.20-11:57:57 ] [train] Compile networks done!
[ 11.20-11:57:57 ] [train] Starting the training process ...
[ 11.20-11:58:29 ] [train] Epoch 1 | Time 26.27 s | Train-Loss 2.617852e+00 | Val-Loss 1.980336e+00 | Scale 1.0
[ 11.20-11:58:38 ] [train] Epoch 2 | Time 3.20 s | Train-Loss 1.600670e+00 | Val-Loss 1.274947e+00 | Scale 1.0
[ 11.20-11:58:46 ] [train] Epoch 3 | Time 3.17 s | Train-Loss 1.050974e+00 | Val-Loss 8.489931e-01 | Scale 1.0
[ 11.20-11:58:54 ] [train] Epoch 4 | Time 3.17 s | Train-Loss 7.525906e-01 | Val-Loss 6.280026e-01 | Scale 1.0
[ 11.20-11:59:02 ] [train] Epoch 5 | Time 3.16 s | Train-Loss 4.963389e-01 | Val-Loss 4.143508e-01 | Scale 1.0
[ 11.20-11:59:10 ] [train] Epoch 6 | Time 3.17 s | Train-Loss 3.706640e-01 | Val-Loss 3.402409e-01 | Scale 1.0
[ 11.20-11:59:19 ] [train] Epoch 7 | Time 3.17 s | Train-Loss 3.099693e-01 | Val-Loss 2.830099e-01 | Scale 1.0
[ 11.20-11:59:27 ] [train] Epoch 8 | Time 3.15 s | Train-Loss 2.531632e-01 | Val-Loss 2.190408e-01 | Scale 1.0
[ 11.20-11:59:35 ] [train] Epoch 9 | Time 3.70 s | Train-Loss 1.893313e-01 | Val-Loss 1.500694e-01 | Scale 1.0
[ 11.20-11:59:43 ] [train] Epoch 10 | Time 3.17 s | Train-Loss 1.100662e-01 | Val-Loss 6.166288e-02 | Scale 1.0
[ 11.20-11:59:52 ] [train] Epoch 11 | Time 3.22 s | Train-Loss 5.152119e-02 | Val-Loss 4.970534e-02 | Scale 1.0
[ 11.20-12:00:00 ] [train] Epoch 12 | Time 3.19 s | Train-Loss 4.407880e-02 | Val-Loss 4.198043e-02 | Scale 1.0
[ 11.20-12:00:09 ] [train] Epoch 13 | Time 3.16 s | Train-Loss 4.095507e-02 | Val-Loss 4.135586e-02 | Scale 1.0
[ 11.20-12:00:17 ] [train] Epoch 14 | Time 3.20 s | Train-Loss 4.356359e-02 | Val-Loss 4.571351e-02 | Scale 1.0
[ 11.20-12:00:26 ] [train] Epoch 15 | Time 3.19 s | Train-Loss 3.874674e-02 | Val-Loss 3.535024e-02 | Scale 1.0
[ 11.20-12:00:35 ] [train] Epoch 16 | Time 3.20 s | Train-Loss 3.541285e-02 | Val-Loss 3.941718e-02 | Scale 1.0
[ 11.20-12:00:43 ] [train] Epoch 17 | Time 3.19 s | Train-Loss 3.940658e-02 | Val-Loss 3.261374e-02 | Scale 1.0
[ 11.20-12:00:52 ] [train] Epoch 18 | Time 3.21 s | Train-Loss 3.621660e-02 | Val-Loss 3.793434e-02 | Scale 1.0
[ 11.20-12:01:00 ] [train] Epoch 19 | Time 3.75 s | Train-Loss 3.538929e-02 | Val-Loss 3.107030e-02 | Scale 1.0
[ 11.20-12:01:09 ] [train] Epoch 20 | Time 3.20 s | Train-Loss 2.974467e-02 | Val-Loss 2.745984e-02 | Scale 1.0
```

Meanwhile, the `model` directory stores all critical data for restarting computations, fine-tuning, and model inference – including complete model parameters. We will provide detailed guidance on utilizing this data in the next section.
