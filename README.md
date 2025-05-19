# fermi

The **F**itn**E**ss, The **R**elatedness and The other **M**etr**I**cs

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square)

---

FERMI is a modular Python framework for analyzing economic complexity using matrix-based techniques.
It provides tools to explore the hidden structure of economies through:

- 🧹 **Matrix preprocessing**: raw cleaning, validation, sparse conversion
- 📊 **Comparative advantage**: RCA/ICA transformation and thresholding
- 🧠 **Fitness & complexity**: compute Fitness, Complexity ECI, PCI and other metrics via multiple methods
- 🌐 **Relatedness metrics**: product space, taxonomy, assist matrix
- 📈 **Prediction models**: GDP forecasting, density models, XGBoost
- ✅ **Validation metrics**: AUC, confusion matrix, prediction@k

---

## 📦 Installation
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
To install prerequisites in a virtual environment you can use the pip installation:
```bash
python -m venv fermi-env
source fermi-env/bin/activate
git clone https://github.com/EFC-data/fermi.git
cd fermi
pip install -r requirements.txt
```
To use fermi on colab write instead:
```bash
!git clone https://github.com/EFC-data/fermi.git
%cd fermi
!pip install -r requirements.txt
```
