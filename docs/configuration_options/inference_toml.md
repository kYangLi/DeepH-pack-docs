<!-- markdownlint-disable MD004 MD007 MD030 MD032 -->
# Inference TOML

Inference process configurations are defined through a TOML-formatted file, where each key systematically governs specific aspects of the computational workflow.

**`infer.toml`:**

```toml
# ----------------------------- SYSTEM -----------------------------
[system]
note = "Enjoy DeepH-pack! ;-)"
device = "gpu*8:0"
float_type = "fp32"
random_seed = 137
log_level = "info"
jax_memory_preallocate = true


# ------------------------------ DATA -------------------------------
[data]
inputs_dir = "./user/should/set/this/inputs"
outputs_dir = "./user/should/set/this/outputs"

[data.dft]
data_dir_depth = 0

[data.graph]
dataset_name = "INFER-DEMO"
graph_type = "S"
storage_type = "memory"
parallel_num = -1
only_save_graph = false


# ----------------------------- MODEL -------------------------------
[model]
model_dir = "./user/should/set/this"
load_model_type = "best"
load_model_epoch = -1


# ---------------------------- PROCESS ------------------------------
[process.infer]
output_type = "h5"
output_into = "to_output"
target_symmetrize = true
multi_way_jit_num = 1

[process.infer.dataloader]
batch_size = 1
```

---

Next, we will go through the semantics of these parameters in the TOML file in detail.

- [Inference TOML](#inference-toml)
  - [**System**](#system)
  - [**Data**](#data)
  - [**Model**](#model)
  - [**Process: Infer**](#process-infer)
    - [**Process: Infer: Dataloader**](#process-infer-dataloader)

---

## **System**

---

**`[system.note]`** \
Description and behavior are the same as in the training configuration.

---

**`[system.device]`** \
Description and behavior are the same as in the training configuration.

---

**`[system.float_type]`** \
Description and behavior are the same as in the training configuration.

---

**`[system.random_seed]`** \
Description and behavior are the same as in the training configuration.

---

**`[system.log_level]`** \
Description and behavior are the same as in the training configuration.

---

**`[system.jax_memory_preallocate]`** \
Description and behavior are the same as in the training configuration.

---

## **Data**

---

**`[data.inputs_dir]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.outputs_dir]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.dft.data_dir_depth]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.graph.dataset_name]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.graph.graph_type]`**
*   **Description**: The physical quantities needed for inference. DeepH will build a corresponding graph file. One can choose from `Sap` and `S`. The `S` for overlap, `Sap` for overlap but do not calculate mask using overlap values.
*   **Default**: `"S"`
*   **Type**: [`"Sap"`, `"S"`]

---

**`[data.graph.storage_type]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.graph.parallel_num]`** \
Description and behavior are the same as in the training configuration.

---

**`[data.graph.only_save_graph]`** \
Description and behavior are the same as in the training configuration.

---

## **Model**

---

**`[model.model_dir]`**
*   **Description**: The directory storing the trained model, usually with the format of `<time_stamp>/model`.
*   **Default**: `<Invalid-Input>`
*   **Type**: `<STRING>`

---

**`[model.load_model_type]`**
*   **Description**: Infer with `best` or `latest` trained model.
*   **Default**: `"best"`
*   **Type**: [`"best"`, `"latest"`]

---

**`[model.load_model_epoch]`**
*   **Description**: For `load_model_type` = `"latest"` only. Specify a number for a particular epoch saved in the latest model folder. Use `-1` for the most recent epoch.
*   **Default**: `-1`
*   **Type**: `<INT>`

---

## **Process: Infer**

---

**`[process.infer.output_type]`**
*   **Description**: The output file format.
*   **Default**: `"h5"`
*   **Type**: [`"h5"`, `"petsc"`]

---

**`[process.infer.output_into]`**
*   **Description**: Location for storing the predicted data. One can choose from a new folder under the output path (`<time_stamp>/dft`) or the original data folder (`<inputs>/dft`). The output Hamiltonians are named as `hamiltonian_pred.h5`.
*   **Default**: `"to_output"`
*   **Type**: [`"to_output"`, `"to_input"`]

---

**`[process.infer.target_symmetrize]`**
*   **Description**: Whether to symmetrize the predicted target (e.g., to hermitianize the Hamiltonian).
*   **Default**: `true`
*   **Type**: `<BOOL>`

---

**`[process.infer.multi_way_jit_num]`** \
Description and behavior are the same as the training parameter `process.train.multi_way_jit_num`.

---

### **Process: Infer: Dataloader**

---

**`[process.infer.dataloader.batch_size]`**
*   **Description**: Batch size for inference. Can be significantly larger than the training batch size.
*   **Default**: `1`
*   **Type**: `<INT>`
