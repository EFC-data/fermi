# Fermi

**F**itn**E**ss, **R**elatedness, and other **M**etr**I**cs for economic-complexity analysis.

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Fermi is a Python toolkit for bipartite economic-complexity data. It provides:

- sparse matrix loading and transformations;
- comparative advantage through null-model-based ICA or RCA;
- Fitness--Complexity, ECI/PCI, diversification, ubiquity, density, and NODF;
- co-occurrence, proximity, taxonomy, and assist projections;
- Monte Carlo projection validation with WBNM;
- network and trajectory-based prediction;
- classification and ranking validation metrics.

The PyPI distribution is named `fermi-cref`; the Python package is imported as
`fermi`.

## Installation

```bash
python -m pip install fermi-cref
```

For editable development checkouts of both repositories:

```bash
git clone https://github.com/lbuffa/wbnm.git
git clone https://github.com/EFC-data/fermi.git
python -m pip install -e ./wbnm -e ./fermi
```

Fermi requires Python 3.10 or newer and `wbnm>=0.1.0`.

## Quickstart

```python
import pandas as pd
from fermi import MatrixProcessorCA, RelatednessMetrics, efc

exports = pd.DataFrame(
    [
        [8.0, 2.0, 0.0, 1.0],
        [1.0, 7.0, 3.0, 0.0],
        [0.0, 2.0, 6.0, 5.0],
        [3.0, 0.0, 1.0, 7.0],
    ],
    index=["A", "B", "C", "D"],
    columns=["p1", "p2", "p3", "p4"],
)

# Standard workflow: ICA with BiWCM and binary specialization matrix
binary = (
    MatrixProcessorCA()
    .load(exports)
    .compute_ica(model="biwcm")
    .binarize(threshold=1.0)
    .get_matrix()
)

# Alternative comparative-advantage workflow: RCA
binary_rca = (
    MatrixProcessorCA()
    .load(exports)
    .compute_rca()
    .binarize(threshold=1.0)
    .get_matrix()
)

# Fitness, Complexity, ECI, and PCI
economy = efc(binary)
fitness, complexity = economy.get_fitness_complexity(aspandas=True)
eci, pci = economy.get_eci_pci(aspandas=True)

# Product proximity network
relatedness = RelatednessMetrics(binary)
product_space = relatedness.get_projection(
    rows=False,
    projection_method="proximity",
)
```

## Null models

Fermi 0.2 uses WBNM for all bipartite null models:

| Model | Data | Constraints reproduced in expectation |
| --- | --- | --- |
| `BiCM` | Binary | Row and column degrees |
| `BiWCM` | Weighted | Row and column strengths |
| `BiECM` | Weighted | Degrees and strengths |
| `BiPECM` | Weighted | Strengths and total edge count |
| `BiCReMA` | Weighted | Degree stage and conditional strengths |

The models represent different null hypotheses. In particular, `BiCM`
binarizes weighted input; it is not a weighted substitute for `BiWCM`.

The standard ICA model is BiWCM:

```python
ica = MatrixProcessorCA().load(exports).compute_ica().get_matrix()
```

Select another model explicitly:

```python
ica = (
    MatrixProcessorCA()
    .load(exports)
    .compute_ica(
        model="biecm",
        solve_kwargs={"tol": 1e-8, "max_iter": 10_000},
    )
    .get_matrix()
)
```

Validate a projection with the binary configuration model:

```python
links, values = relatedness.get_bicm_projection(
    rows=False,
    projection_method="cooccurrence",
    validation_method="fdr",
    num_iterations=10_000,
    seed=42,
)
```

Or select a generic WBNM model:

```python
links, values = relatedness.get_null_model_projection(
    null_model="biecm",
    rows=False,
    projection_method="cooccurrence",
    validation_method="fdr",
    num_iterations=10_000,
    seed=42,
)
```

`get_bicm_projection()` remains backward compatible but now uses
`wbnm.BiCM`; Fermi no longer requires the separate `bicm` package.

## Documentation

The complete documentation is under [`docs/source`](docs/source/index.rst):

- [installation and upgrades](docs/source/installation.rst);
- [end-to-end quickstart](docs/source/quickstart.rst);
- [data and preprocessing](docs/source/data_and_preprocessing.rst);
- [null-model selection](docs/source/null_models.rst);
- [economic-complexity metrics](docs/source/economic_complexity.rst);
- [relatedness and projections](docs/source/relatedness.rst);
- [prediction](docs/source/prediction.rst);
- [validation metrics](docs/source/validation.rst);
- [migration to Fermi 0.2](docs/source/migration.rst);
- [API reference](docs/source/api.rst).

Build the HTML documentation locally:

```bash
python -m pip install -r requirements-dev.txt
python -m sphinx -W --keep-going -b html docs/source docs/build/html
```

## Tests

```bash
python -m pytest
```

## Citation

If you use Fermi, cite the repository and the scientific sources associated
with the methods used in your analysis. The documentation contains the full
[reference list](docs/source/references.rst).

## License

Fermi is distributed under the [MIT License](LICENSE).
