"""Helpers for fitting the null models exposed by :mod:`wbnm`."""

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import scipy.sparse as sp

from wbnm import BiCM, BiCReMA, BiECM, BiPECM, BiWCM
from wbnm.models.base import BipartiteModel


NullModelType = Type[BipartiteModel]
NullModelSpec = Union[str, NullModelType]

NULL_MODELS: Dict[str, NullModelType] = {
    "bicm": BiCM,
    "biwcm": BiWCM,
    "biecm": BiECM,
    "bipecm": BiPECM,
    "bicrema": BiCReMA,
}


def resolve_null_model(model: NullModelSpec) -> NullModelType:
    """Return the WBNM model class identified by ``model``."""
    if isinstance(model, str):
        normalized = model.lower().replace("-", "").replace("_", "")
        try:
            return NULL_MODELS[normalized]
        except KeyError as exc:
            available = ", ".join(cls.__name__ for cls in NULL_MODELS.values())
            raise ValueError(
                f"Unsupported null model {model!r}. Choose from: {available}."
            ) from exc

    if isinstance(model, type) and issubclass(model, BipartiteModel):
        return model

    raise TypeError("model must be a WBNM model name or BipartiteModel subclass")


def fit_null_model(
    matrix: Union[np.ndarray, sp.spmatrix],
    model: NullModelSpec,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[BipartiteModel], np.ndarray, np.ndarray]:
    """Fit a WBNM model after removing empty rows and columns.

    The masks returned alongside the fitted model allow callers to restore the
    original matrix shape.  An all-zero matrix returns ``None`` as its model.
    """
    sparse_matrix = sp.csr_matrix(matrix)
    sparse_matrix.eliminate_zeros()
    row_mask = np.asarray(sparse_matrix.getnnz(axis=1) > 0).ravel()
    col_mask = np.asarray(sparse_matrix.getnnz(axis=0) > 0).ravel()

    if not row_mask.any() or not col_mask.any():
        return None, row_mask, col_mask

    compact = sparse_matrix[row_mask][:, col_mask].toarray()
    model_class = resolve_null_model(model)
    fitted_model = model_class(compact, device=device)
    fitted_model.solve(**(solve_kwargs or {}))
    return fitted_model, row_mask, col_mask


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert a CPU or accelerator tensor (or array-like) to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def restore_matrix(
    compact: np.ndarray,
    shape: Tuple[int, int],
    row_mask: np.ndarray,
    col_mask: np.ndarray,
) -> np.ndarray:
    """Restore a compact matrix to its shape before empty-node removal."""
    restored = np.zeros(shape, dtype=np.asarray(compact).dtype)
    if row_mask.any() and col_mask.any():
        restored[np.ix_(row_mask, col_mask)] = compact
    return restored


def sample_null_model(
    fitted_model: Optional[BipartiteModel],
    shape: Tuple[int, int],
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw one sample and restore any empty rows and columns."""
    if fitted_model is None:
        return np.zeros(shape, dtype=float)
    compact = tensor_to_numpy(fitted_model.sample(n_samples=1, seed=seed)[0])
    return restore_matrix(compact, shape, row_mask, col_mask)
