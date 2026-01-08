# Installation & Setup

## Install UV

To begin, configure your environment with uv, a fast and versatile Python package manager written in Rust. Please follow the installation instructions on the [official uv website](https://docs.astral.sh/uv/#installation).

On Linux or macOS, you can install `uv` with a single command (requires an internet connection):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is highly recommended that configuring high-performance mirrors based on your IP location. For example, for users in China, you cloud using the mirror provided by [TUNA](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/).

```bash
# Add the following lines into ~/.config/uv/uv.toml
[[index]]
url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/"
default = true
```

## Create `deeph` python virtual environment

Create `python 3.13` environments with `uv`:

``` bash
mkdir ~/.uvenv
cd ~/.uvenv
uv venv deeph --python=3.13 # Create `deeph` venv in current dir
```

Then, the uv virtual environment can be activate with command,

```bash
source ~/.uvenv/deeph/bin/activate
```

Conveniently, all files installed into the `deeph` venv will be located in the `~/.uvenv/deeph` directory.

## Install DeepH-pack

Ensure you've activated the uv environment as described in the previous section, and that you're currently in the `deeph` environment you created.

``` bash
uv pip install ./deepx-1.0.6+light-py3-none-any.whl[gpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

This will install all the dependencies, including GPU version of jax, flax, etc. For CPU-only platforms, replace `[gpu]` with `[cpu]`.

**NOTE:**

- `./deepx-1.0.6+light-py3-none-any.whl` is the Python wheel file available for download from the [official DeepH-pack website](https://ticket.deeph-pack.com/?language=en).

- The `[gpu]` extra dependency tag indicates the GPU-accelerated version of the package, which is **strongly recommended** for optimal performance. If your system only supports CPU computation, replace `[gpu]` with `[cpu]`.

- The `--extra-index-url` flag is used to specify an additional package index (in this case, PyTorch's official repository) for resolving certain dependencies. The url ended with `"cpu"` because pytorch is ony used on the CPU side.

- Please do not confuse the [`DeepH-pack`](https://github.com/kYangLi/DeepH-pack-docs) in this tutorial with the other open-sourced DeepH packages. All the packages provided with this tutorial are named after `deepx`. It is short for `deeph-jax` and prevents user confusion between this *brand-new* JAX-based public release version and previously developed experimental programs (such as [`DeepH`](https://github.com/mzjb/DeepH-pack), [`DeepH-E3`](https://github.com/Xiaoxun-Gong/DeepH-E3)), whose packages are named `deeph`, upon future releases.
