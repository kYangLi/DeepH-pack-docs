<!-- markdownlint-disable MD004 MD007 MD038 MD030 MD031 MD032 -->
# Training TOML

Parameters of the training process are configured through a TOML-formatted file. Each key systematically governs specific aspects of the computational workflow. An example TOML is shown below:

**`train.toml`:**

```toml
# ---------------------------------- SYSTEM ----------------------------------
[system]
note = "Enjoy DeepH-pack! ;-)"
device = "gpu*8:0"
float_type = "fp32"
random_seed = 137
log_level = "info"
jax_memory_preallocate = true
show_train_process_bar = true

# ----------------------------------- DATA ------------------------------------
[data]
inputs_dir = "./user/should/set/this/inputs"
outputs_dir = "./user/should/set/this/outputs"

[data.dft]
data_dir_depth = 0

[data.graph]
dataset_name = "DATASET-DEMO"
graph_type = "H"
storage_type = "memory"
common_orbital_types = ""
parallel_num = -1
only_save_graph = false

[data.model_save]
best = true
latest = true
latest_interval = 100
latest_num = 10

# ----------------------------- MODEL -----------------------------------------
[model]
net_type = "normal"
target_type = "H"
loss_type = "mse"

[model.advanced]
gaussian_basis_rmax = 7.5
net_irreps = ""
num_blocks = 3
consider_parity = true
standardize_gauge = false

# ------------------------------ PROCESS --------------------------------------
[process.train]
max_epoch = 10000

multi_way_jit_num = 1
ahead_of_time_compile = false

[process.train.dataloader]
batch_size = 1

train_size = 1
validate_size = 0
test_size = 0
dataset_split_json = ""
only_use_train_loss = false

[process.train.optimizer]
type = "adamw"
init_learning_rate = 2E-3
clip_norm_factor = -1.0
# sgd
momentum = 0.8
# adam(w)
betas = [0.9, 0.999]
weight = 0.001
eps = 1E-8

[process.train.scheduler]
min_learning_rate_scale = 1E-4
type = "reduce_on_plateau"
# Reduce on plateau
factor = 0.5
patience = 500
rtol = 0.05
cooldown = 100
accum_size = -1
# Warmup cosine decay
init_scale = 0.1
warmup_steps = 10
decay_steps = -1
end_scale = -1.0

[process.train.continued]
enable = false
new_training_data = false
new_optimizer = false
previous_output_dir = ""
load_model_type = "latest"
load_model_epoch = -1
```

---

Next, we will go through the semantics of these parameters in the TOML file in detail.

- [Training TOML](#training-toml)
  - [**System**](#system)
  - [**Data**](#data)
  - [**Model**](#model)
    - [**Model: Advanced**](#model-advanced)
  - [**Process: Train**](#process-train)
    - [**Process: Train: Dataloader**](#process-train-dataloader)
    - [**Process: Train: Optimizer**](#process-train-optimizer)
    - [**Process: Train: Scheduler**](#process-train-scheduler)
      - [Reduce LR On Plateau](#reduce-lr-on-plateau)
      - [Warmup Cosine Decay](#warmup-cosine-decay)
    - [**Process: Train: Continued**](#process-train-continued)

---

## **System**

---

**`[system.note]`**
*   **Description**: Name of this training project.
*   **Default**: `"Enjoy DeepH-pack! ;-)"`
*   **Type**: `<STRING>`

---

**`[system.device]`**
*   **Description**: The device configuration follows the syntax `<type>*<num>:<id>`, where `<type>` specifies hardware type (`cpu`, `gpu`, `rocm`, `dcu`, or `cuda`), `<num>` denotes either the total devices per node (for accelerators like GPU) or the number of CPU partitions (when using `cpu`), and `<id>` defines target device indices (e.g., `gpu*8:1-4,7` selects 5 GPUs from an 8-device node using indices 1,2,3,4,7). Note: For CPU configurations, `<id>` is ignored while `<num>` controls thread partitioning.
*   **Default**: `"gpu*8:0"`
*   **Type**: `<STRING>`

---

**`[system.float_type]`**
*   **Description**: The float type for building graph and training model. For most of the Hamiltonian tasks, `"fp32"` is accurate enough with float point error around $0.001\sim 0.01$ meV.
*   **Default**: `"fp32"`
*   **Type**: [`"bf16"`, `"tf32"`, `"fp32"`, `"fp64"`]

---

**`[system.random_seed]`**
*   **Description**: The seed for generating random numbers.
*   **Default**: `137`
*   **Type**: `<INT>`

---

**`[system.log_level]`**
*   **Description**: The degree of severity for `deepx.log`, increasing in the order of debug, info, warning, error, and critical.
*   **Default**: `"info"`
*   **Type**: [`"debug"`, `"info"`, `"warning"`, `"critical"`]

---

**`[system.jax_memory_preallocate]`**
*   **Description**: Whether to pre-allocate $75\%$ of the remaining memory before running. Use `true` for formal training; use `false` for debugging to observe memory usage on the GPU. Using `false` during training may cause the process to unexpectedly crash mid-way or slow down the training process.
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

**`[system.show_train_process_bar]`**
*   **Description**: Whether to display a graphical progress bar in the command line.
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

## **Data**

---

**`[data.inputs_dir]`**
*   **Description**: Specify the root directory containing structured input data with required sub-folders `dft/` or `graph/` (see Section `sec::analysis_data_features`). Accepts both relative paths (resolved from execution context) and absolute paths (system-native format).
*   **Default**: `<Invalid-Input>`
*   **Type**: `<STRING>`

---

**`[data.outputs_dir]`**
*   **Description**: The output directory designates the parent location for storing training artifacts (log files and serialized models). By default, the system automatically generates timestamped subdirectories (ISO 8601 extended format: `%Y-%m-%d_%H-%M-%S`) within user-specified paths, isolating each training session's outputs. Full structural specifications detailed in Section `sec::after_training` (Post-Training Workflows).
*   **Default**: `<Invalid-Input>`
*   **Type**: `<STRING>`

---

**`[data.dft.data_dir_depth]`**
*   **Description**: When organizing DFT training data subdirectories, if the number of structural configurations becomes excessive (e.g., reaching 100,000 structures), a flat directory structure under `inputs_dir/dft/*` becomes impractical. In such cases, a hierarchical folder architecture is recommended, such as `dft/<t1>/<t2>/*`. This configuration establishes a directory depth of 2,  `data_dir_depth`=2, providing a scalable storage solution for massive structural datasets while maintaining systematic accessibility.
*   **Default**: `0`
*   **Type**: `<INT>`

---

**`[data.graph.dataset_name]`**
*   **Description**: Name of your dataset.
*   **Default**: `"DATASET-DEMO"`
*   **Type**: `<STRING>`

---

**`[data.graph.graph_type]`**
*   **Description**: The physical quantity used for training. DeepH will build a corresponding graph file. One can choose from `H`, `S`, `Rho`, `HS`, and `Sap`. The `H` for Hamiltonian, `Rho` for density matrix, `S` for overlap, `HS` for both Hamiltonian and overlap, and `Sap` for shape-only overlap. The vertices of the graph are atoms, and there is an edge between two atoms if they are close.
*   **Default**: `"H"`
*   **Type**: [`"H"`, `"HS"`, `"Rho"`, `"Sap"`, `"S"`]

---

**`[data.graph.storage_type]`**
*   **Description**: Where to store the graph data. Choose `"memory"` to store in memory. For very large datasets, choose `"disk"` to store on disk.
*   **Default**: `"memory"`
*   **Type**: [`"memory"`, `"disk"`]

---

**`[data.graph.common_orbital_types]`**
*   **Description**: The list of one atom’s orbitals arranged according to the value of `l`(angular quantum number). e.g., `s2p2d1`. Default (set to `""`) is the union of orbital types of all different atoms in the dataset. For example, for basis sets `Mo-s3p2d1` and `S-s2p2d1` in the `OpenMX` calculation，the union of orbital types is `[0, 0, 0, 1, 1, 2]`, which corresponds to `s3p2d1`.
*   **Default**: `""`
*   **Type**: `<STRING>`

---

**`[data.graph.parallel_num]`**
*   **Description**: Determines the maximum concurrent parallel processes allocated for graph construction. When configured with non-positive integers or values exceeding the available compute resources (i.e., surpassing either the host's physical CPU core count or accelerator device quantity), the system will dynamically scale the parallelism based on hardware availability - specifically adopting the greater value between detected CPU cores and accelerator devices (GPUs) to optimize computational throughput.
*   **Default**: `-1`
*   **Type**: `<INT>`

---

**`[data.graph.only_save_graph]`**
*   **Description**: If set to `true`, the program will only generate and save the graph file to file-system and quit.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

**`[data.model_save.best]`**
*   **Description**: Whether to save the model with the lowest loss.
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

**`[data.model_save.latest]`**
*   **Description**: Whether to save the model in the latest training epoch.
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

**`[data.model_save.latest_interval]`**
*   **Description**: Only functional when `data.model_save.latest` is `true`. This parameter governs the checkpointing frequency for model state preservation during training. When configured with non-positive values or exceeding the maximum epoch count, the system enforces periodic snapshots at epoch multiples of the specified integer - e.g., setting `latest_interval`=10 systematically generates training state archives at epochs 10, 20, 30, etc., implementing epoch-aligned preservation of complete model states (weights, optimizer parameters, and metadata).
*   **Default**: `100`
*   **Type**: `<INT>`

---

**`[data.model_save.latest_num]`**
*   **Description**: Only functional when `data.model_save.latest` is `true`. The number of latest checkpoints that user wants to keep.
*   **Default**: `10`
*   **Type**: `<INT>`

---

## **Model**

---

**`[model.net_type]`**
*   **Description**: The neural network architecture.
    *   `sparrow` (also named `normal`) is a light-weighted architecture (typically $<$1M parameters) with both node and edge features, which is suitable for small tasks of DFT Hamiltonian learning.
    *   `eagle` (also named `accurate`) is an advanced architecture (typically $\sim$5M parameters) with both node and edge features, which is suitable for tasks of DFT Hamiltonian learning that requires high accuracy.
*   **Default**: `"normal"`
*   **Type**: [`"sparrow"`, `"normal"`, `"eagle"`, `"accurate"`]

---

**`[model.target_type]`**
*   **Description**: The physical quantity to learn. `H` for Hamiltonian, and `Rho` for density matrix.
*   **Default**: `"H"`
*   **Type**: [`"H"`, `"Rho"`]

---

**`[model.loss_type]`**
*   **Description**: Loss function during training. The loss type not only support the supervised ones (`mse`, `mae`, etc.) but also support on-the-fly unsupervised loss (like `ai2dft`, `ai2dft_node`, `hopad`, and `aims`).
*   **Default**: `"mse"`
*   **Type**: [`"mae"`, `"mse"`, `"wmae"`, `"huber"`, `"ai2dft"`, `"ai2dft_node"`, `"hopad"`, `"aims"`]

---

### **Model: Advanced**

---

**`[model.advanced.gaussian_basis_rmax]`**
*   **Description**: The cutoff radius used for Gaussian basis sampling, in angstrom. We suggest set this cutoff to 2$\times$the maximum cutoff radius of your orbital basis. Refer the paper of [`DeepH-E3`](https://www.nature.com/articles/s41467-023-38468-8) for details.
*   **Default**: `7.5`
*   **Type**: `<FLOAT>`

---

**`[model.advanced.net_irreps]`**
*   **Description**: Irreducible representations of the neural network features, which ensure the equivariance of the network. \\
    For `sparrow`, the `Irreps` can be set to ``$\cdots$\verb|x0e+|$\cdots$\verb|x1o+|$\cdots$\verb|x2e+|$\cdots$\verb|x3o+|$\cdots$\verb|x4e+|$\cdots$''. The channel $l$ has parity $(-1)^{l}$. \\
    For `eagle` and `owl`, set to ``$\cdots$\verb|x0e+|$\cdots$\verb|x1e+|$\cdots$\verb|x2e+|$\cdots$\verb|x3e+|$\cdots$\verb|x4e+|$\cdots$''. **All the channels must have even parity**. \\
    Note: ``64x1o'' means that the feature is a 64-channel tensor, where each channel has odd (``o'') parity and carries the $l=1$ representation of the $SO(3)$ group.\\
    `o`/`e` refers to odd/even parity.\\
    Set in the form of [`e3nn.Irreps`](https://e3nn-jax.readthedocs.io/en/latest/api/irreps.html), namely ``irreducible representations'', which describes the symmetry of input features. e.g. in the form of \([(mul\_l, (l, p\_val \cdot (p\_arg)^l)) \\ \text{for } l \in [0, \dots, l_{max}]]\).
*   **Default**: `<Invalid-Input>`
*   **Type**: `<STRING>`

---

**`[model.advanced.num_blocks]`**
*   **Description**: The number of neural network layers in the model. For usual task, we recommend set 3 for solids, 4 for small molecules.
*   **Default**: `3`
*   **Type**: `<INT>`

---

**`[model.advanced.consider_parity]`**
*   **Description**: Whether the network is equivariant under parity. If set to `false`, the `net_irreps` cannot appear odd (e.g., `2x3o`) representations. In `eagle` and `owl` net, the `consider_parity` must be `false`.
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

**`[model.advanced.standardize_gauge]`**
*   **Description**:  Whether to consider arbitrariness of gauge (zero-energy point in DFT) when training the model. If set to `true`, `graph_type` must be `HS`. We suggest set `true` for a general-purpose dataset (or whenever the training data has a zero-energy arbitrariness problem, e.g., a dataset with 2D slabs of different thicknesses), `false` for a special-purpose dataset.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

## **Process: Train**

---

**`[process.train.max_epoch]`**
*   **Description**: The maximum number of epochs. Training will automatically stop when `epoch_number` reaches `max_epoch`.
*   **Default**: `10000`
*   **Type**: `<INT>`

---

**`[process.train.multi_way_jit_num]`**
*   **Description**: This helps accelerate the training process when the number of edges between different structures in the dataset varies greatly. Recommended value is 10-20. This may cause the first epoch of training to be slow and may cause out-of-memory error.
*   **Default**: `1`
*   **Type**: `<INT>`

---

**`[process.train.ahead_of_time_compile]`**
*   **Description**: Whether to use [`ahead-of-time`](https://docs.jax.dev/en/latest/aot.html) (AOT) compilation to accelerate the JIT (or more precisely, compile) process. With the AOT method help, the multi-way JIT can speed up from 2 hours to 10 minutes.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

### **Process: Train: Dataloader**

---

**`[process.train.dataloader.batch_size]`**
*   **Description**: Batch size, number of structures in a batch.
*   **Default**: `1`
*   **Type**: `<INT>`

---

**`[process.train.dataloader.train_size]`**
*   **Description**: Number of structures in the training dataset.
*   **Default**: `1`
*   **Type**: `<INT>`

---

**`[process.train.dataloader.validate_size]`**
*   **Description**: Number of structures in the validation dataset.
*   **Default**: `0`
*   **Type**: `<INT>`

---

**`[process.train.dataloader.test_size]`**
*   **Description**: Number of structures in the test dataset.
*   **Default**: `0`
*   **Type**: `<INT>`

---

**`[process.train.dataloader.dataset_split_json]`**
*   **Description**: A JSON file path to execute dataset partitioning with customized rules. To employ the framework's default splitting mechanism, set this parameter to an empty string (`""`). The JSON configuration must adhere to the following schema specification:
    ```json
    {"train": ["1", "3", "5"], "validate": ["2"], "test":["4","6"]}
    ```
*   **Default**: `""`
*   **Type**: `<STRING>`

---

**`[process.train.dataloader.only_use_train_loss]`**
*   **Description**: Whether to adjust the learning rate based only on the train loss rather than considering both train and validation loss. If the validation set is empty (although this is not common), the validation loss is always a non-sense value, so it should be set to consider only train loss.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

### **Process: Train: Optimizer**

---

**`[process.train.optimizer.type]`**
*   **Description**: The optimizer core type, can choose from: [`adamw`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html), [`adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), or [`sgd`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
*   **Default**: `"adamw"`
*   **Type**: [`"sgd"`, `"adam"`, `"adamw"`]

---

**`[process.train.optimizer.init_learning_rate]`**
*   **Description**: Initial learning rate. We suggest use `2E-3` for `net_type`=`sparrow`, `1E-3` for `net_type`=`eagle` or `owl`. Larger learning rate can speed up convergence while may be confronted with instabilities.
*   **Default**: `2E-3`
*   **Type**: `<FLOAT>`

---

**`[process.train.optimizer.clip_norm_factor]`**
*   **Description**: Enable the [`clip`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip_by_global_norm) normalization algorithm in the optimizer to prevent large gradients that can cause neuron death. This feature will be disabled, if it was set with negative value.
*   **Default**: `-1.0`
*   **Type**: `<FLOAT>`

---

**`[process.train.optimizer.momentum]`**
*   **Description**: For `sgd` optimizer only. Controls the influence of previous gradients on the current gradient update.
*   **Default**: `0.8`
*   **Type**: `<FLOAT>`

---

**`[process.train.optimizer.betas]`**
*   **Description**: For `adam` and `adamw` optimizers. `betas = [beta1, beta2]`, where `beta1` controls the exponential decay rate of first moment estimate (i.e., momentum), and `beta2` controls the exponential decay rate of second moment estimate (i.e., the uncentered variance) usually used for calculating the moving average of squared gradients.
*   **Default**: `[0.9, 0.999]`
*   **Type**: `<LIST-OF-FLOAT>`

---

**`[process.train.optimizer.eps]`**
*   **Description**: For `adam` and `adamw` optimizers. A small constant to improve numerical stability. In some cases it is useful to decrease it to `1E-10` or lower.
*   **Default**: `1E-8`
*   **Type**: `<FLOAT>`

---

**`[process.train.optimizer.weight]`**
*   **Description**: For `adamw` optimizer only. The learning weight decay rate in `adamw`, to avoid overfitting.
*   **Default**: `0.001`
*   **Type**: `<FLOAT>`

---

### **Process: Train: Scheduler**

Within the [`optax`](https://github.com/google-deepmind/optax) optimization framework, learning rate decay operates through a decoupled control mechanism governed by the scale parameter. The effective learning rate for neural network updates is determined by the product: `learning_rate (lr)` = `init_lr` $\times$ `scale`. A dedicated scheduler module systematically modulates this scaling factor throughout training, enabling implementation of various decay strategies (step-wise, exponential, or cosine decay) while maintaining architectural isolation between initialization values and decay dynamics.

---

**`[process.train.scheduler.min_learning_rate_scale]`**
*   **Description**: The minimum learning rate scale. Training will automatically stop when the learning rate scale reaches `min_learning_rate_scale`.
*   **Default**: `1E-4`
*   **Type**: `<FLOAT>`

---

**`[process.train.scheduler.type]`**
*   **Description**: One can choose from: [`"ReduceOnPlateau"`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html), and [`"WarmupCosineDecay"`](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html#optax.schedules.warmup_cosine_decay_schedule).
*   **Default**: `"reduce_on_plateau"`
*   **Type**: [`"reduce_on_plateau"`, `"warmup_cosine_decay"`]

---

#### Reduce LR On Plateau

---

**`[process.train.scheduler.factor]`**
*   **Description**: For `"reduce_on_plateau"` only. Every time the learning rate adjustment is triggered, learning rate scale will be updated to `factor`$\times$`scale`.
*   **Default**: `0.5`
*   **Type**: `<FLOAT>`

---

**`[process.train.scheduler.patience]`**
*   **Description**: For `"reduce_on_plateau"` only. The number of evaluation steps to wait before reducing the learning rate. This helps determine when there is no further improvement in model performance. Be careful that by default, the number of patience step is defined based on the train steps (each batch), rather than epoch (each train-set). In most cases, we suggest to define this patience according to your own dataset scale, computing resources, and expected time cost. Usually `patience=120` epochs gives a well-converged model, and `patience=60` epochs or less gives a relatively quick result.
*   **Default**: `500`
*   **Type**: `<INT>`

---

**`[process.train.scheduler.rtol]`**
*   **Description**: For `"reduce_on_plateau"` only. Relative tolerance. A loss is considered no longer improving if its relative improvement over the previous best validation loss is less than `rtol`.
*   **Default**: `0.05`
*   **Type**: `<FLOAT>`

---

**`[process.train.scheduler.cooldown]`**
*   **Description**: For `"reduce_on_plateau"` only. The minimum number of evaluation cycles between two learning rate adjustments. This prevents frequent changes to the learning rate. Be careful that by default, the number of cooldown step is defined based on the train steps (each batch), rather than epoch (each train-set). In most cases, `cooldown=20$\sim$50` epochs is enough.
*   **Default**: `100`
*   **Type**: `<INT>`

---

**`[process.train.scheduler.accum_size]`**
*   **Description**: For `"reduce_on_plateau"` only. The `patience` and `cooldown` parameters operate on gradient update steps rather than epochs, with their baseline values calculated as *training batches per epoch* $\times$ *target epochs*. For instance, given 100 batches in each epoch:
    *   To monitor validation loss for 20 epochs without gradient accumulation (`accum_size`=1), set patience=100$\times$20=2000.
    *   With `accum_size`=100 (effectively creating macro-batches of 100$\times$`batch_size`), each "step" becomes equivalent to 1 epoch, thus `patience`=20 suffices.
    *   When patience=-1, the system auto-configures it as total training batches to implement full-epoch monitoring cycles.
    Cooldown steps follow equivalent computational logic based on gradient update granularity.
*   **Default**: `-1`
*   **Type**: `<INT>`

---

#### Warmup Cosine Decay

By combining warmup and cosine decay, the scheduler helps deep learning models converge faster and improves their performance.

---

**`[process.train.scheduler.init_scale]`**
*   **Description**: For `"warmup_cosine_decay"` only. The initial scaling factor for learning rate scale.
*   **Default**: `0.1`
*   **Type**: `<FLOAT>`

---

**`[process.train.scheduler.warmup_steps]`**
*   **Description**: For `"warmup_cosine_decay"` only. The number of steps for the learning rate scale to linearly increase from `init_scale` to `1.0`, just like "warmup" a model. This helps stabilize the model's behavior during the early stages of training.
*   **Default**: `1000`
*   **Type**: `<INT>`

---

**`[process.train.scheduler.decay_steps]`**
*   **Description**: For `"warmup_cosine_decay"` only. The number of steps for the learning rate scale to decay from `1.0` to `end_scale`. Note that training will stop early when the number of epochs reaches the `max_epoch`.
*   **Default**: `2E5`
*   **Type**: `<INT>`

---

**`[process.train.scheduler.end_scale]`**
*   **Description**: For `"warmup_cosine_decay"` only. The scaling factor at the end of the learning rate scheduling. Set to -1.0 for the learning rate to decay to 0. Note that training will stop early when the learning rate scale is lower than the `min_learning_rate_scale`.
*   **Default**: `-1.0`
*   **Type**: `<FLOAT>`

---

### **Process: Train: Continued**

This section is for fine-tuning or continuing training on an existing model.

---

**`[process.train.continued.enable]`**
*   **Description**: Set to `true` for continued training or fine-tuning from an existing model, or `false` for starting from scratch.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

**`[process.train.continued.new_training_data]`**
*   **Description**: Whether to use new training data. Set to `true` for fine-tuning like task, and `false` when running on the same dataset as the previous one.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

**`[process.train.continued.new_optimizer]`**
*   **Description**: Whether to use a new optimizer. Set to `true` for fine-tuning like task, and `false` when running on the same optimizer and scheduler as the previous one.
*   **Default**: `false`
*   **Type**: `<BOOL>`

---

**`[process.train.continued.previous_output_dir]`**
*   **Description**: Previous output directory with time stamp, which contains the `deepx.log` file and the `model` folder.
*   **Default**: `<Invalid-Input>`
*   **Type**: `<STRING>`

---

**`[process.train.continued.load_model_type]`**
*   **Description**: Continue training from the `best` or `latest` model as needed.
*   **Default**: `"latest"`
*   **Type**: [`"best"`, `"latest"`]

---

**`[process.train.continued.load_model_epoch]`**
*   **Description**: For `load_model_type` = `"latest"` only. Specify a number for particular epoch exist saved in latest model folder. Use `-1` for the most latest epoch.
*   **Default**: `"-1"`
*   **Type**: `<INT>`
