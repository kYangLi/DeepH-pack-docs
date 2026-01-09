# Frequently Asked Questions (FAQ)

## General Questions

### What is the difference between this software and the DeepH-pack already published on GitHub?

[This software](https://github.com/kYangLi/DeepH-pack-docs) is a [JAX-based](https://github.com/jax-ml/jax) refactored version of the DeepH code, designed to provide users with a smoother and more streamlined experience. The DeepH team has spent over a year refactoring all modules and code of DeepH, integrating all previous achievements into a single software package. The new architecture aims to enhance computational efficiency, improve code maintainability, and offer a more user-friendly interface. At the same time, we extend our sincere gratitude to all developers of historical versions and the initiators of the DeepH project.

### Why is this package called `deepx`? It's a bit confusing to me

The name `deepx` was chosen for two primary reasons:

- The previously released DeepH-pack based on Pytroch has already occupied the `deeph` package name on Python package indices. To prevent users from accidentally installing the wrong package, we abbreviated `deeph-jax` to `deepx`.

- The name also hints at the expanded scope of the current DeepH-pack. While the original DeepH focused primarily on Hamiltonian prediction, the new architecture supports a broader range of electronic structure calculations and is designed to be extensible for future functionalities beyond DFT Hamiltonian prediction.

Regardless, our package is still named *DeepH-pack*.

### When will features like DFPT mentioned in the paper be released?

These are all on the DeepH team's timeline. The DeepH team is also collaborating with top technical teams such as [FHI-aims](https://www.fhi-aims.org/), [ABACUS](https://abacus.ustc.edu.cn/main.htm), and [EPW](https://epw-code.org/) to develop corresponding functionalities. Please stay tuned for updates.

## Installation & Distribution

### Can I download the new version's whl from GitHub currently?

No, not currently. The [open-source deeph-pack-docs project](https://github.com/kYangLi/DeepH-pack-docs) serves only as the public documentation for DeepH-pack. The only way to obtain the whl file is to apply through [our official website](https://ticket.deeph-pack.com/?language=en).

### How long does the software application process take?

Typically 1-2 business days.

### I see that my application has been approved and an email was sent, but I haven't received anything in my inbox

Please first double-check that the email address you provided is correct and ensure the email hasn't been filtered into your spam/junk folder. If both are confirmed to be correct, the email was likely intercepted by your institution/school's network center, as it contains download links. We recommend contacting your IT or network administration for assistance. If you have any further questions, feel free to reach out to us at `deeph-pack@outlook.com`.

### What CUDA and Python versions does this version support?

For `deepx-1.0.6+light`, Python `3.13`, `CUDA 12.8`, or `12.9`.

### When installing on CentOS 7, I encountered installation failures. What is the reason?

The reason is that the last version of glibc supported by CentOS 7 is too old (version 2.17). The currently released DeepH `deepx-1.0.6+light` version requires JAX minimum version `0.6.2` and PyTorch minimum version `2.7.0+cpu`. Both indeed cannot be installed on your platform, primarily due to the low glibc version, which leads to difficulties in installing the accompanying CUDA libraries and dependencies. The glibc `2.17` was released in 2012 and is the last version supported by CentOS 7, which has now completely ceased maintenance. Many of the latest stable versions of computational libraries and tools on the market claim to no longer support this version to ensure security and shed historical baggage. Regarding this issue, there are roughly four solutions:

1. **Delegate an engineer to gradually resolve dependencies.** This is cumbersome and has a high probability of failure.
2. **Use virtual containers like Docker or LXC to simulate Ubuntu, Rocky Linux, etc.** This may incur some performance loss but is compatible with your current production environment.
3. **Reinstall the operating system.** This will break your current production environment but will provide longer and more stable support for future software.
4. **Consider using the CPU-only version** if your computational needs allow, as it may have fewer dependencies.

### Does the new DeepH-pack also have requirements for underlying libraries? I always get errors when installing on `manylinux_2_17_x86_64`

Yes, the reason and solutions are the same as the previous question.

## Functionality & Usage

### I cannot find the data preprocessing conversion interface in the new version of DeepH. Where is it?

This functionality has been integrated into the open-source project [**DeepH-dock**](https://github.com/kYangLi/DeepH-dock), and we welcome contributions from the community to its development.

### Does the current version support spin-orbit coupling (SOC) calculations?

Yes, it does. After using DeepH-dock to [convert the raw DFT SOC data](https://deeph-dock.readthedocs.io/en/latest/capabilities/index.html) in [the DeepH format](https://deeph-dock.readthedocs.io/en/latest/key_concepts.html#spin-polarized-systems), DeepH-pack will automatically recognize and train on it.
