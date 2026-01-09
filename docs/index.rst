DeepH-pack
==========

.. div:: sd-text-left sd-font-italic

    *A general-purpose neural network package for deep-learning electronic structure calculations*

----

`DeepH-pack <https://arxiv.org/abs/2601.02938>`_ represents the culmination of **multi-generational research efforts** from the DeepH team at Tsinghua University. This latest iteration of DeepH unites all preceding methodologies into a cohesive, JAX-rewritten package, achieving comprehensive maturity through rigorous long-term testing across all neural modules.

At its core, `DeepH-pack` features a **JAX-based implementation with static computational graphs and advanced algorithms**, delivering exceptional performance in runtime, precision, and memory efficiency. Looking forward, the development roadmap envisions seamless integration of multi-framework backends, evolving into an extensible computational ecosystem for quantum materials modeling while preserving signature accuracy in Hamiltonian construction.

The platform is dedicated to constructing an expanded and more comprehensive toolkit for materials computation and predictive modeling, warmly welcoming community feedback. We have open-sourced the core data interfaces and standardization modules from DeepH calculations, establishing the `DeepH-dock <https://github.com/kYangLi/DeepH-dock>`_ project. This initiative delivers seamless interoperability with mainstream DFT software packages, while the codebase integrates community-contributed enhancements at the architecture level.

Features
^^^^^^^^

.. grid::

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Performance-Optimized
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Leverages JAX static computational graphs and innovative neural algorithms to achieve exceptional runtime efficiency, precision, and memory utilization for large-scale quantum materials simulations.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Mature & Reliable
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Built upon years of cumulative research, with rigorous testing across all neural modules ensuring comprehensive maturity and production-ready stability for scientific applications.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Extensible Ecosystem
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                Designed as a foundation for quantum materials modeling, with a roadmap for multi-framework backend integration and cross-platform compatibility while maintaining signature Hamiltonian accuracy.

    .. grid-item::
        :columns: 12 12 12 6

        .. card:: Unified Workflow
            :class-card: sd-border-0
            :shadow: none
            :class-title: sd-fs-5

            .. div:: sd-font-normal

                DeepH integrates diverse ab initio materials computation software, eliminating technical barriers to establish standardized workflows that accelerate convergence in computational materials science.


Installation
^^^^^^^^^^^^

Before installing `DeepH-pack`, ensure that `uv <https://docs.astral.sh/uv/>`_ — a fast and versatile Python package manager — is properly installed and configured, and that your `Python 3.13` environment is set up. If you plan to run DeepH in a GPU-accelerated environment, you must also pre-install `CUDA 12.8` or `12.9`.

.. code-block:: bash

    uv pip install ./deepx-1.0.6+light-py3-none-any.whl[gpu] --extra-index-url https://download.pytorch.org/whl/cpu

For step-by-step detailed procedures, please refer to the `Installation & Setup <./installation_and_setup.html>`_.


Basic usage
^^^^^^^^^^^

For command-line usage:

.. code-block:: bash

    deeph-train train.toml
    deeph-infer infer.toml

For comprehensive information beyond basic usage, refer to the following key sections of the documentation:

*   `Core Workflows <./core_workflows/index.html>`_: Details the essential computational steps of DeepH.
*   `Configuration Options <./configuration_options/index.html>`_: Explains all available parameters in the user input (TOML) files.
*   `Universal Material Model <./universal_material_model.html>`_: Describes the usage of generalized pre-trained models.
*   `Examples <examples/index.html>`_: Contains various practical training and inference examples.

Citation
^^^^^^^^

If you use ``DeepH-pack`` in your work, please cite the following publications.

- **The original framework paper introduced the foundational methodology.**

  `He Li, Zun Wang, Nianlong Zou, et al. Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation. Nat. Comput. Sci. 2, 367 (2022) <https://doi.org/10.1038/s43588-022-00265-6>`_

- **Complete package featuring the latest implementation, methodology, and workflow.**

  `Yang Li, Yanzhen Wang, Boheng Zhao, et al. DeepH-pack: A general-purpose neural network package for deep-learning electronic structure calculations. arXiv:2601.02938 (2026) <https://arxiv.org/abs/2601.02938>`_

.. code-block:: bibtex

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

----

.. toctree::
    :hidden:
    :maxdepth: 1
    
    theoretical_backgrounds
    getting_the_code
    installation_and_setup
    core_workflows/index
    universal_material_model
    configuration_options/index
    examples/index
    citation_and_community
    frequently_asked_questions
