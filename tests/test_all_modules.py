import numpy as np
import warnings
from scipy.sparse import csr_matrix
from fermi import (
    MatrixProcessorCA,
    efc,
    RelatednessMetrics,
    ECPredictor,
    ValidationMetrics,
)

def test_matrix_processor_init():
    processor = MatrixProcessorCA()
    assert processor is not None

def test_matrix_processor_load():
    dummy_matrix = csr_matrix([[1, 0], [0, 1]])
    processor = MatrixProcessorCA().load(dummy_matrix)
    assert processor.get_matrix().shape == (2, 2)


def test_compute_rca_handles_empty_rows_and_columns_without_warning():
    matrix = csr_matrix(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = MatrixProcessorCA().load(matrix).compute_rca().get_matrix(dense=True)

    assert np.isfinite(result).all()
    assert np.count_nonzero(result[2, :]) == 0
    assert np.count_nonzero(result[:, 2]) == 0

def test_fitness_complexity_engine_init():
    dummy_matrix = csr_matrix([[1, 0], [1, 1]])
    engine = efc(dummy_matrix)
    assert engine is not None

def test_relatedness_metrics_init():
    dummy_matrix = csr_matrix([[0, 1], [1, 0]])
    metrics = RelatednessMetrics(dummy_matrix)
    assert metrics is not None

def test_prediction_module_init():
    dummy_matrix = csr_matrix([[0.1, 0.5], [0.2, 0.7]])
    module = ECPredictor(dummy_matrix)
    assert module is not None

def test_validation_metrics_init():
    M = np.array([[0, 1], [1, 0]])
    P = np.array([[0.2, 0.8], [0.9, 0.1]])
    metrics = ValidationMetrics(M, P)
    assert metrics is not None
