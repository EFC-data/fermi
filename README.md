# fermi

**The FitnEss, The Relatedness and The other MetrIcs**  

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

### 🔄 From GitHub (latest version)

> ⚠️ Requires Python ≥ 3.0

```bash
pip install git+https://github.com/EFC-data/fermi.git
```
This will install fermi along with its core dependencies:

numpy ≥ 2.0 (⚠ may cause incompatibilities — see below)
pandas
scikit-learn ≥ 1.4.2
scipy
matplotlib
seaborn
bokeh
tqdm


⚠ Compatibility Note
FERMI currently requires NumPy ≥ 2.0, which may conflict with some libraries (e.g. tensorflow, numba, jax, etc.).
To avoid issues, consider installing FERMI in a virtual environment:

```bash
python -m venv fermi-env
source fermi-env/bin/activate
pip install git+https://github.com/EFC-data/fermi.git
```

🧪 Development
To install FERMI in editable mode with dev dependencies:

```bash
git clone https://github.com/EFC-data/fermi.git
cd fermi

# core install
pip install -e .

# dev tools (testing, linting, etc.)
pip install -r requirements-dev.txt
```
Then run the test suite:
```bash
pytest
```


