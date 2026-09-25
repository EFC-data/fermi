import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fermi import MatrixProcessorCA, RelatednessMetrics
from fermi.null_models import resolve_null_model


MATRIX = csr_matrix(
    [
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
BINARY_MATRIX = (MATRIX > 0).astype(float)


@pytest.mark.parametrize(
    ("name", "class_name"),
    [
        ("bicm", "BiCM"),
        ("BiWCM", "BiWCM"),
        ("bi_ecm", "BiECM"),
        ("bi-pecm", "BiPECM"),
        ("bicrema", "BiCReMA"),
    ],
)
def test_resolve_all_wbnm_models(name, class_name):
    assert resolve_null_model(name).__name__ == class_name


def test_compute_ica_uses_wbnm_and_restores_empty_nodes():
    processor = MatrixProcessorCA().load(MATRIX)

    result = processor.compute_ica(
        model="biwcm",
        solve_kwargs={"method": "coordinate", "max_iter": 1000},
    ).get_matrix()

    assert result.shape == MATRIX.shape
    assert result.getrow(3).nnz == 0
    assert np.isfinite(result.data).all()
    assert processor.null_model_.__class__.__name__ == "BiWCM"


def test_generic_projection_and_bicm_wrapper_restore_state():
    metrics = RelatednessMetrics(BINARY_MATRIX)
    before = metrics.get_matrix().copy()

    generic, generic_values = metrics.get_null_model_projection(
        projection_method="cooccurrence",
        validation_method="direct",
        null_model="bicm",
        num_iterations=3,
        seed=10,
        solve_kwargs={"method": "coordinate", "max_iter": 1000},
    )
    wrapped, wrapped_values = metrics.get_bicm_projection(
        projection_method="cooccurrence",
        validation_method="direct",
        num_iterations=3,
        seed=10,
        solve_kwargs={"method": "coordinate", "max_iter": 1000},
    )

    assert generic.shape == (MATRIX.shape[0], MATRIX.shape[0])
    np.testing.assert_array_equal(generic, wrapped)
    np.testing.assert_allclose(generic_values, wrapped_values)
    np.testing.assert_array_equal(metrics.get_matrix().toarray(), before.toarray())
