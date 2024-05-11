"""A module for similarity methods.

This module includes serval similarity methods, which can be used to
get the similarity between two brain activity patterns. These methods
are voxel-based, meaning that they compare the similarity between two
brain activity patterns by initially flattening them into 1D arrays.
"""

from functools import wraps
from typing import Callable

import numpy as np
import scipy.spatial.distance as distance
import scipy.stats as stats

__all__ = [
    "chebyshev",
    "cityblock",
    "cosine",
    "euclidean",
    "minkowski",
    "minkowski_5",
    "minkowski_10",
    "minkowski_50",
    "pearson",
    "spearman",
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
def chebyshev(x: np.ndarray, y: np.ndarray) -> float:
    return distance.chebyshev(x, y)


@_voxel_based_similarity
def cityblock(x: np.ndarray, y: np.ndarray) -> float:
    return distance.cityblock(x, y)


@_voxel_based_similarity
def cosine(x: np.ndarray, y: np.ndarray) -> float:
    return distance.cosine(x, y)


@_voxel_based_similarity
def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return distance.euclidean(x, y)


@_voxel_based_similarity
def minkowski(x: np.ndarray, y: np.ndarray, p: float) -> float:
    return distance.minkowski(x, y, p)


def minkowski_5(x: np.ndarray, y: np.ndarray) -> float:
    return minkowski(x=x, y=y, p=5)


def minkowski_10(x: np.ndarray, y: np.ndarray) -> float:
    return minkowski(x=x, y=y, p=10)


def minkowski_50(x: np.ndarray, y: np.ndarray) -> float:
    return minkowski(x=x, y=y, p=50)


@_voxel_based_similarity
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return stats.pearsonr(x, y)[0]


@_voxel_based_similarity
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return stats.spearmanr(x, y)
