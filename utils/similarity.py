"""A module for similarity methods.

This module includes 2 similarity methods, `Cosine Similarity` (CS) and 
`Pearson Correlation Coefficient` (PCC).
"""

from functools import wraps
from typing import Callable

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr

__all__ = [
    "cs",
    "pcc",
]


def _voxel_based_similarity(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x: np.ndarray
        y: np.ndarray
        x, y, *others = args
        assert x.shape == y.shape, (
            "The two arrays involved in the similarity comparison need "
            f"to have the same shape, but got {x.shape} and {y.shape}."
        )
        x, y = x.flatten(), y.flatten()
        xy = np.stack([x, y], axis=0)
        xy = xy[:, ~np.isnan(xy).any(axis=0)]
        x, y = xy
        return func(x, y, *others, **kwargs)

    return wrapper


@_voxel_based_similarity
def cs(x: np.ndarray, y: np.ndarray) -> float:
    return 1 - cosine_distance(x, y)


@_voxel_based_similarity
def pcc(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(x, y)[0]
