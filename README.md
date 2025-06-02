# fermi

The **F**itn**E**ss, The **R**elatedness and The other **M**etr**I**cs

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)

---

`fermi` is a modular Python framework for analyzing the main Economic Complexity metrics and features.
It provides tools to explore the hidden structure of economies through:

- 📊 **Matrix preprocessing**: raw cleaning, sparse conversion, Comparative advantage RCA/ICA, transformation and thresholding.
- 🧠 **Fitness & complexity**: compute Fitness, Complexity ECI, PCI and other metrics via multiple methods.
- 🌐 **Relatedness metrics**: product space, taxonomy, assist matrix.
- 📈 **Prediction models**: GDP forecasting, density models, XGBoost.
- ✅ **Validation metrics**: AUC, confusion matrix, prediction@k.

---

## 📦 Getting Started
### Requirements
> ⚠️ Requires Python ≥ 3.0
To correnctly install and use the package, you need to have
```bash
numpy ≥ 1.24
pandas ≥ 1.5
scikit-learn ≥ 1.2
scipy ≥ 1.9
matplotlib ≥ 3.5
seaborn
bokeh ≥ 2.4
tqdm
networkx ≥ 2.6
bicm ≥ 3.3.1
```
///### Install the module 
To install prerequisites in a virtual environment you can use the pip installation.
/// This is commented
///```bash
///pip install fermi
///```

### Clone the Repository (if necessary) 
If you wan to create a virtual environment to avoid conflict with the installed version of the requirements, you can follow the code:
```bash
python -m venv fermi-env
source fermi-env/bin/activate
```
To install the `fermi` modules cloning the repository from GitHub, you can follow the code:
```bash
git clone https://github.com/EFC-data/fermi.git
cd fermi
pip install -e .
pip install -r requirements.txt
```

### Using colab
To use fermi on colab cloning the repository, use the following code:
```bash
!git clone https://github.com/EFC-data/fermi.git
%cd fermi
!pip install -r requirements.txt
```

---

## 🚀 Basic functionalities
### Fitness and Complexity module
The main module to generate an Economic Complexity object and initialize it (with a biadjacency matrix):

    import fermi
    myefc = fermi.efc()
    myefc.load(my_biadjacency_matrix, *possible kwargs*)

To compute the Revealed Comparative Advantage (Balassa index) and binarize its value

    myefc.compute_rca().binarize()

To compute the Fitness and the Complexity (using the original [Tacchella2012] algorithm)

    fitness, complexity = myefc.get_fitness_complexity()

To compute the diversification and the ubiquity

    div, ubi = myefc.get_diversification_ubiquity()

To compute the ECI index (using the eigenvalue method)

    eci, pci = myefc.get_eci_pci()

### Relatedness module
The module to generate cooccurrences and similar relatedness measures is

    myproj = fermi.RelatednessMetrics()
    myproj.load(my_biadjacency_matrix, *possible kwargs*)

The cooccurrence can be evaluated using

    relatedness = myproj.get_projection(projection_method="cooccurrence")
    validated_relatedness, validated_values = myproj.get_bicm_projection(projection_method="cooccurrence", validation_method="fdr")

See a more detailed description in the API in the documentation.

---

## 🌐 How to cite
If you use the `fermi` modules, please cite its location on Github
[https://github.com/EFC-data/fermi](https://github.com/EFC-data/fermi)


### References

[Tacchella2012] [A. Tacchella, M. Cristelli, G. Caldarelli, A. Gabrielli, L. Pietronero , *A New Metrics for Countries' Fitness and Products' Complexity*, SciRep vol. **2**, 723 (2012)](https://doi.org/10.1038/srep00723)

[Zaccaria2014] [Zaccaria A, Cristelli M, Tacchella A, Pietronero L, *How the Taxonomy of Products Drives the Economic Development of Countries*, PLoS ONE, (2014), 9(12): e113770](https://doi.org/10.1371/journal.pone.0113770)

[Tacchella2018] [Tacchella A., Mazzilli D., Pietronero L. *A dynamical systems approach to gross domestic product forecasting*. Nature Phys 14, 861–865 (2018)](https://doi.org/10.1038/s41567-018-0204-y)

[Pugliese2019] [Pugliese E., Cimini G., Patelli A. et al. *Unfolding the innovation system for the development of countries: coevolution of Science, Technology and Production*. Sci Rep vol. **9**, 16440 (2019)](https://doi.org/10.1038/s41598-019-52767-5)

[Mazzilli2024] [D Mazzilli, M S Mariani, F Morone and A Patelli, *Equivalence between the Fitness-Complexity and the Sinkhorn-Knopp algorithms*, J. Phys. Complex. 5 015010 (2024)](https://doi.org/10.1088/2632-072X/ad2697)

## Credits

__Authors__:
[CREF Team](www.cref.it)

- [Aurelio Patelli]()

- [Riccardo Piombo]()

- [Matteo Straccamore]()

- [Filippo Santoro]()

- [Valeria Secchini]()

- [Lorenzo Buffa]()

- [Daniele Cirulli]()

### Acknowledgements
We gratefully acknowledge the invaluable contributions, support, and foundational code provided by Andrea Tacchella, Emanuele Pugliese, Dario Mazzilli, and Andrea Zaccaria.