# DeepH 安装指南

## 1. 环境要求

### 1.1 操作系统与硬件支持
*   **操作系统**：DeepH 仅支持 **Linux** 系统，推荐使用 **Ubuntu 20.04** 及以上版本。
*   **硬件加速**：推荐使用 GPU (NVIDIA)、NPU 或 TPU 等加速硬件以获得最佳性能。
    *   *注：若无加速硬件，DeepH 亦支持仅使用 CPU 进行训练，但速度会显著受限。*

### 1.2 驱动与库依赖 (以 NVIDIA GPU 为例)
确保系统已正确安装 NVIDIA 驱动、CUDA Toolkit 和 cuDNN 库。根据运行环境的不同，配置方式如下：

*   **个人/裸机环境**：
    需自行安装与 GPU 型号匹配的 NVIDIA 驱动及 CUDA/cuDNN。
    > **特别提示**：针对 **RTX 5090** 等新一代显卡，强烈建议使用 **CUDA 12.9**。已知 CUDA 13.0 版本存在兼容性 Bug，会导致训练失败。

*   **超算/集群环境 (Slurm/PBS)**：
    若当前环境包含 Slurm 等调度器，您极大概率处于集群环境中。通常无需手动安装驱动，只需加载相应的环境变量模块即可。
    ```bash
    # 示例：加载 CUDA 12.9 模块
    module load cuda/12.9
    ```

### 1.3 Python 环境管理 (强烈推荐 uv)
DeepH 官方强烈建议使用 **uv 包管理器** 进行 Python 环境管理，以确保依赖解析的稳定性和速度。

**第一步：检查或安装 uv**
如果系统中尚未安装 uv，请在确保网络连接的情况下执行以下命令安装：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
更多安装方式请参考 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

**第二步：创建并激活虚拟环境**
安装完成后，建议在用户家目录下创建名为 `deeph` 的虚拟环境：
```bash
# 创建虚拟环境
uv venv ~/.uvenv/deeph

# 激活虚拟环境
source ~/.uvenv/deeph/bin/activate
```

## 2. 安装步骤

由于 DeepH-pack (DeepX) 核心库目前为闭源状态，无法通过公共 PyPI 源直接安装，需手动下载安装包。

### 2.1 获取安装包
请前往 DeepH-pack [官方网站](https://ticket.deeph-pack.com/) 申请并下载最新的加密 whl 安装包（例如：`deepx_pack-1.2.x-py3-none-any.whl`）。

### 2.2 执行安装
将下载的 whl 文件放置在服务器上的任意目录，并获取其**绝对路径**。根据您的硬件环境，选择相应的安装选项（Extras）：

**通用安装命令格式：**
```bash
uv pip install /绝对/路径/到/deepx_pack.whl[硬件选项]
```

**常用硬件选项说明：**
*   **NVIDIA GPU (推荐)**：使用 `[cuda12]` (适用于 CUDA 12.x)。
    ```bash
    # 示例：安装支持 CUDA 12 的版本
    uv pip install /path/to/deepx_pack.whl[cuda12]
    ```
*   **其他选项**：
    *   `[cuda13]`：适用于 CUDA 13.x 环境（*注意前文提到的兼容性警告*）。
    *   `[cpu]`：无 GPU 环境。
    *   `[tpu]`：Google TPU 环境。

## 3. 安装验证

安装完成后，请执行以下命令验证 DeepH 是否安装成功：

```bash
uv pip show deepx-pack
```

*   **成功**：终端将输出 DeepH-pack 的版本号、安装路径及依赖信息。
*   **失败**：若提示包未找到或报错，请检查 whl 路径是否正确及 Python 环境是否激活。

