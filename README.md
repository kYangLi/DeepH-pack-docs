<h1><p align="center">
  <img src="./docs/_image/logo-large.svg" alt="DeepH-pack Logo" width="500">
</p></h1>

<div align="center">

### *A General-purpose Neural Network Package for Deep-learning Electronic Structure Calculations*

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/DeepH-pack-docs.svg)](https://github.com/kYangLi/DeepH-pack-docs/issues)

*Drive Accuracy and Efficiency with Intelligence.*
</div>

- [Core Features](#core-features)
- [Quick Start](#quick-start)
  - [Get the Software](#get-software)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Citation](#citation)
- [Publications | DeepH Team](#publications--deeph-team)

## Core Features

The [modernized DeepH-pack](https://ticket.deeph-pack.com) is built upon the [solid foundation of its predecessor](https://github.com/mzjb/DeepH-pack) and has been re-engineered with [JAX](https://github.com/jax-ml/jax) and [FLAX](https://github.com/google/flax) to unlock new levels of efficiency and flexibility.

## Quick Start

### Get the Software

Please visit the [DeepH-pack official website](https://ticket.deeph-pack.com/) to apply for and obtain the software.

### Installation

First, ensure that [uv](https://docs.astral.sh/uv/) — a fast and versatile Python package manager — is properly installed and configured. Once set up, you can install DeepH-pack using the following command:

```bash
uv pip install ./deepx-1.0.6+light-py3-none-any.whl[gpu] --extra-index-url https://download.pytorch.org/whl/cpu
```

**Parameter explanation:**

- `./deepx-1.0.6+light-py3-none-any.whl` is the Python wheel file available for download from the official [DeepH-pack website](https://ticket.deeph-pack.com/).

- The `[gpu]` extra dependency tag indicates the GPU-accelerated version of the package, which is **strongly recommended** for optimal performance. If your system only supports CPU computation, replace `[gpu]` with `[cpu]`.

- The `--extra-index-url` flag is used to specify an additional package index (in this case, PyTorch's official repository) for resolving certain dependencies.

### Basic Usage

***Further online documentation will be available soon!***

## Citation

*Any and all use of this software, in whole or in part, should clearly acknowledge and link to this repository.*

If you use `DeepH-pack` in your work, please cite the following publications.

- **The original framework paper introduced the foundational methodology.**

    [He Li, Zun Wang, Nianlong Zou, *et al*. Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation. Nat. Comput. Sci. 2, 367 (2022)](https://doi.org/10.1038/s43588-022-00265-6)

- **Complete package featuring the latest implementation, methodology, and workflow.**

    [Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

```bibtex
@article{li2022deep,
    title={Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation},
    author={Li, He and Wang, Zun and Zou, Nianlong and Ye, Meng and Xu, Runzhang and Gong, Xiaoxun and Duan, Wenhui and Xu, Yong},
    journal={Nat. Comput. Sci.},
    volume={2},
    number={6},
    pages={367},
    year={2022},
    publisher={Nature Publishing Group US New York}
}

@article{li2026deeph,
    title={DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations},
    author={Li, Yang and Wang, Yanzhen and Zhao, Boheng and Gong, Xiaoxun and Wang, Yuxiang and Tang, Zechen and Wang, Zixu and Yuan, Zilong and Li, Jialin and Sun, Minghui and Chen, Zezhou and Tao, Honggeng and Wu, Baochun and Yu, Yuhang and Li, He and da Jornada, Felipe H. and Duan, Wenhui and Xu, Yong },
    journal={arXiv preprint arXiv:2601.02938},
    year={2026}
}
```

## Publications | DeepH Team

For a comprehensive overview of publications and research employing DeepH methods, please see the relevant section below. We also warmly welcome citations to our foundational papers if your work utilizes the DeepH framework or any of its modules (e.g., [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3), [HPRO](https://github.com/Xiaoxun-Gong/HPRO)).

1. **Latest Software Implementation**

    - [Yang Li, Yanzhen Wang, Boheng Zhao, *et al*. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026)](https://arxiv.org/abs/2601.02938)

2. **Architecture advancements**

    - **DeepH**: Original framework [Nat. Comput. Sci. 2, 367 (2022)](https://doi.org/10.1038/s43588-022-00265-6)
    - **DeepH-E3**: Integrating equivariant neural network [Nat. Commun. 14, 2848 (2023)](https://doi.org/10.1038/s41467-023-38468-8)
    - **DeepH-2**: Incorporating eSCN tensor product [arXiv:2401.17015 (2024)](https://arxiv.org/abs/2401.17015)
    - **DeepH-Zero**: Leveraging physics-informed unsupervised learning [Phys. Rev. Lett. 133, 076401 (2024)](https://doi.org/10.1103/PhysRevLett.133.076401)

3. **Improved compatibility with first-principles codes**

    - **HPRO**: Compatibility with plane-wave DFT [Nat. Comput. Sci. 4, 752 (2024)](https://doi.org/10.1038/s43588-024-00701-9)
    - **DeepH-hybrid**: Extension to hybrid DFT [Nat. Commun. 15, 8815 (2024)](https://doi.org/10.1038/s41467-024-53028-4)

4. **Exploration of application scenarios**

    - **xDeepH**: Dealing with magnetism with extended DeepH [Nat. Comput. Sci. 3, 321 (2023)](https://doi.org/10.1038/s43588-023-00424-3)
    - **DeepH-DFPT**: Investigating density functional perturbation theory [Phys. Rev. Lett. 132, 096401 (2024)](https://doi.org/10.1103/PhysRevLett.132.096401)
    - **DeepH-UMM**: Developing the universal model for electronic structures [Sci. Bull. 69, 2514 (2024)](https://doi.org/10.1016/j.scib.2024.06.011)

5. **Review of Recent Advancement**

    - From DeepH and ML-QMC to fast, accurate materials computation [Nat. Comput. Sci. 5, 1133 (2025)](https://doi.org/10.1038/s43588-025-00932-4)

---

*DeepH-pack is a general-purpose neural network package designed for deep-learning electronic structure calculations, empowering computational materials science with accelerated speed and enhanced efficiency through intelligent algorithms.*
