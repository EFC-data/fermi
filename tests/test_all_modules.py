import numpy as np
from scipy.sparse import csr_matrix
from fermi import (
    RawMatrixProcessor,
    ComparativeAdvantage,
    efc,
    RelatednessMetrics,
    ECPredictor,
    ValidationMetrics,
)

def test_raw_matrix_processor_init():
    processor = RawMatrixProcessor()
    assert processor is not None

def test_comparative_advantage_processor_init():
    dummy_matrix = csr_matrix([[1, 0], [0, 1]])
    processor = ComparativeAdvantage(dummy_matrix)
    assert processor is not None

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

