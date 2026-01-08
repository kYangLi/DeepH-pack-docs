# Data Preparation

In this part, we showcase examples of dataset generation with DFT interface.

## Interfacing to DeepH with DeepH-dock

[DeepH-dock](https://github.com/kYangLi/DeepH-dock) provides various interfaces to prepare the datasets required by DeepH. Current supported packages include OpenMX, SIESTA, ABACUS, FHI-aims and HONPAS. The DeepH team sincerely acknowledges the support from the development teams of these DFT codes.

### Overview of the DeepH-formatted files

For I/O efficiency considerations, the new version of DeepH has optimized the storage architecture of datasets with [a new format](https://deeph-dock.readthedocs.io/en/latest/key_concepts.html).

For format conversion from legacy to current DeepH versions, please refer to [DeepH-dock documentation](https://deeph-dock.readthedocs.io/en/latest/capabilities/convert/deeph/demo.html#upgrade).

The folder tree structure of DeepH-formatted datasets looks like this:

```bash
dft/ # DeepH ready data
  |- label-0/
     |- POSCAR
     |- info.json
     |- overlap.h5
     |- hamiltonian.h5
  |- label-1/
  |- ...
```

where `overlap.h5`, `hamiltonian.h5` are the corresponding matrices under localized atomic orbital (AO) basis, the `POSCAR` file contains structural information, while the `info.json` file stores critical metadata for the current structure. The `dft/` root name is mandatory for compatibility with the modern DeepH toolchain, whereas the sample subfolders can follow project-specific naming conventions (numeric indices, chemical prototypes, etc.). Conversion between legacy and modern layouts is discussed in section 3.2.

### Interface with OpenMX

For detailed instructions, see [DeepH-pack: Convert OpenMX](https://deeph-dock.readthedocs.io/en/latest/capabilities/convert/openmx/demo.html).

OpenMX (Open source package for Material eXplorer) is an open source DFT calculation package for nano-scale material simulations based on norm-conserving pseudopotentials and pseudo-atomic orbital basis.

To obtain the raw data required by DeepH-pack (Hamiltonian, overlap matrix, etc) from OpenMX calculations, the option

``` bash
HS.fileout ON
```

should be added to the OpenMX input file `*.in`.

After the calculation is done, the final structure and physical properties are stored in the `*.scfout` file.

The folder tree of the OpenMX calculations should be strutured like:

```bash
openmx_calculations/ # OpenMX raw data
  |- label-0/
  |- label-1/
  |- ...
```

Then the raw data can be converted to the DeepH-pack format by:

```bash
dock convert openmx to-deeph ./openmx_calculations ./dft -p 2
```

After the conversion, the folder tree structure should look like this:

```bash
openmx_calculations/ # OpenMX raw data
  |- label-0/
  |- label-1/
  |- ...
dft/ # converted data
  |- label-0/
     |- POSCAR
     |- info.json
     |- overlap.h5
     |- hamiltonian.h5
  |- label-1/
  |- ...
```

### Interface with SIESTA

For detailed instructions, see [DeepH-pack: Convert SIESTA](https://deeph-dock.readthedocs.io/en/latest/capabilities/convert/siesta/demo.html).

SIESTA is both a method and its computer program implementation to perform efficient electronic structure calculations with strictly-localized atomic orbital basis.

To obtain the raw data required by DeepH-pack from SIESTA calculations, the option

``` bash
SaveHS .true.
```

should be added to the siesta input file `*.fdf`.

After the calculation is done, the final structure is stored in the `*.XV` file and physical properties are stored in the `*.HSX`, `*.DM`, etc.

The folder tree of the SIESTA calculations should be strutured like:

```bash
siesta_calculations/ # SIESTA raw data
  |- label-0/
  |- label-1/
  |- ...
```

Then the raw data can be converted to the DeepH-pack format by:

```bash
dock convert siesta to-deeph ./siesta_calculations ./dft -p 2
```

After the conversion, the folder tree structure should look like this:

```bash
siesta_calculations/ # SIESTA raw data
  |- label-0/
  |- label-1/
  |- ...
dft/ # converted data
  |- label-0/
     |- POSCAR
     |- info.json
     |- overlap.h5
     |- hamiltonian.h5
  |- label-1/
  |- ...
```

Note that SIESTA allows different settings for the atoms of the same element (e.g., different basis sets and pseudopotentials for bulk and surface silicon atoms), but DeepH-pack can not distinguish these differences and treats them as the same species. To avoid this problem, do not use more than one settings for one element in the ChemicalSpeciesLabel block (e.g., do not include both "1 14 Si1" and "2 14 Si2").

### Interface with ABACUS

For detailed instructions, see [DeepH-pack: Convert ABACUS](https://deeph-dock.readthedocs.io/en/latest/capabilities/convert/abacus/demo.html).

ABACUS (Atomic-orbital Based Ab-initio Computation at USTC) is an open-source software package designed for large-scale electronic structure simulations from first principles. ABACUS supports both plane-wave and atomic orbital basis sets, while only the interface with atomic orbital basis mode is implemented in DeepH-dock.

To obtain the raw data required by DeepH-pack from ABACUS calculations, the options

```bash
basis_type lcao
out_mat_hs2 1
```

should be added in the ABACUS input file `INPUT`.

After the calculation is done, the final structure is stored in the `running_*.log` file and physical properties are stored in the `data-*.csr` files, all of which are dumped under `OUT.*` directory.

The folder tree of the ABACUS calculations should be strutured like:

```bash
abacus_calculations/ # ABACUS raw data
  |- label-0/
  |- label-1/
  |- ...
```

Then the raw data can be converted to the DeepH-pack format by:

```bash
dock convert abacus to-deeph ./abacus_calculations ./dft -p 2
```

After the conversion, the folder tree structure looks like this:

```bash
abacus_calculations/ # ABACUS raw data
  |- label-0/
  |- label-1/
  |- ...
dft/ # converted data
  |- label-0/
     |- POSCAR
     |- info.json
     |- overlap.h5
     |- hamiltonian.h5
  |- label-1/
  |- ...
```

Note that ABACUS also allows different settings for the atoms of the same element like SIESTA. To avoid this problem, the species labels in the ATOMIC_SPECIES block should be the standard element symbols instead of user-defined symbols (e.g., only Si is allowed for silicon atoms and "Si1" is not allowed).

The current interface is developed for ABACUS version 3.10 LTS (long time support version). The output filenames of other versions is different from the LTS version.

## Interfacing to DeepH with legacy interfaces

It is worth noting that, the current release of refactored DeepH-pack is accompanied with a more compact data format (referred to as the `DeepH` format), compared with the old format (referred to as the `DeepH-legacy` format). While the two formats contain same information, a data conversion interface is available in `deeph-dock` for converting `DeepH-legacy` to `DeepH`. The conversion is meaningful, regarding that:

1. existing DeepH users possess substantial legacy datasets
2. several interfaces currently output data in legacy format

### Data conversion with DeepH-dock

The data conversion is quite simple with DeepH-dock installed, with just one command

```bash
dock convert deeph upgrade ./collect_preprocessed ./dft -p 2
```

The converted data will be stored in `dft/`

The new format data can also be converted from the updated format to the legacy format for interaction with legacy format-based tools by

```bash
dock convert deeph downgrade ./dft ./collect_preprocessed -p 2
```

For detailed instructions, see [DeepH-pack: Convert DeepH](https://deeph-dock.readthedocs.io/en/latest/capabilities/convert/deeph/demo.html).

## Guidance and notes for dataset generation

### What to include in a dataset

It should be noted that, the generation of dataset for DeepH training is largely a open question and strongly depend on what you indend to research. Below we provide guidance for several simple scenarios for reference:

1. Dataset generation of **perturbed datasets**. Training DeepH on perturbed dataset would be useful in applications such as inspecting [electron-phonon coupling](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.132.096401). You can generate datasets either by random perturbations, or from ab inito molecular dynamics. DeepH has been trained on both setups in our example studies.
2. Dataset generation for **Moiré-twisted materials**. DeepH has capability to learn from Hamiltonians of small-sized structures and generalized to large-sized ones, as exemplified from non-twisted small-sized bilayers to large-scale Moiré-twisted ones. Due to the twisting, different local stacking (e.g. AA, AB and BA) will presence in the Moiré-twisted structure. The training dataset could therefore be generated by including strucutures with **random interlayer shift**, plus random atomic perturbations, in the training set.
3. Dataset generation for **more complicated structures**. It's meaningful to explore the DeepH's performance on existing databases based on database structures for formulating [**universal DeepH models**](https://www.sciencedirect.com/science/article/pii/S2095927324004079). Also inspecting the electronic structures of **defects/interfaces/alloys** with DeepH could be quite meaningful. For generating structures, we recommend using specialized structure-generation packages that can be seamlessly integrated with the DeepH@FHI-aims interface.

### Important tips regarding datasets

Several rule-of-thumbs guidance may be helpful regarding dataset generation:

1. **Dataset size**: DeepH typically requires 50 to 500 structures for perturbed structures or Moiré-twisted materials, depending on the complexity of the material and the perturbation magnitude you inposed. We still recommend users check data sufficience in their specific use.
2. **Hand-waving accuracy estimate**: A rough order of magnitude of DeepH models' being "accurate" is achieving <1 meV in terms of Hamiltonians' mean absolute error (MAE). Yet on most datasets you're expected to find DeepH far more accurate than that.
3. **Double-check data reliability**: In dataset generation please remember avoid including absurd structures such as structures with too-close-by atoms, or without SCF convergence. Even including one such structure could be disastrous to DeepH training.
4. **Ensure data quality**: It is recommended that you use well-converged DFT parameters for dataset generation, including real-space grid cutoff, k-mesh sampling, SCF convergence criteria, etc. With poor-converged parameters, the dataset will look like comprising random ``noise", which will pose challenge to DeepH training
5. **Avoid using too large basis sets**: Since the predicted Hamiltonian is under non-orthogonal basis, if the basis set is large, then the post-processing of the Hamiltonian is likely ill-conditioned, leading to large numerical error due to the near-linear-dependence of the basis sets.
6. **Avoid using bases with high angular momentum**: It will be time-consuming for DeepH to work with Hamiltonian matrix elements corresponding to high-angular-momentum basis sets, and we therefore recommend users use basis sets up to *f*-orbitals in practical use. Also if you include basis set up to $l_{\text{max}}$, the irreducible representations of DeepH should be at least up to $2l_{\text{max}}$, as will be described in the next section.
