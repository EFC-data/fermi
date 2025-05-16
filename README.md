# FERMI

**FitnEss Relatedness and other MetrIcs**  

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](#)
[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)

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

## 🛠️ Installation

```bash
git clone https://github.com/EFC-data/fermi.git
cd fermi
pip install -e .
