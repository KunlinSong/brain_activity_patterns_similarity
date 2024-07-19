"""The module for randomly shuffling images in the dataset."""

from typing import Literal

import numpy as np
import pandas as pd

from utils.utils import get_process_method

from .config import (
    FIELDS_CONFIG,
    LABELS_CONFIG,
    ROIImgFieldsConfig,
    WholeBrainImgFieldsConfig,
)
from .utils import get_fields_config

__all__ = [
    "shuffle_all_imgs",
]


def _shuffle_img(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    non_nan_indices = np.nonzero(~np.isnan(img))
    shuffled_indices = np.stack(non_nan_indices, axis=0)
    rng.shuffle(shuffled_indices, axis=-1)
    shuffled_indices = tuple(shuffled_indices)
    img[non_nan_indices] = img[shuffled_indices]
    return img


def _shuffle_row_img(
    row: pd.Series,
    rng: np.random.Generator,
    fields_config: WholeBrainImgFieldsConfig | ROIImgFieldsConfig,
    process_method: str | None = None,
) -> pd.Series:
    row = row.copy()
    process_method = get_process_method(process_method=process_method)
    imgs: pd.Series = row[fields_config.images].copy()
    img: np.ndarray = imgs[process_method]
    shuffled_img = _shuffle_img(img, rng)
    imgs[process_method] = shuffled_img
    row[fields_config.images] = imgs
    row[fields_config.data_type] = LABELS_CONFIG.data_type.shuffled
    return row


def shuffle_all_imgs(
    img_df: pd.DataFrame,
    process_method: str | None = None,
    random_seed: int = 42,
    img_type: Literal["whole brain", "ROI"] = "ROI",
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    shuffled_df = img_df.copy()
    fields_config = get_fields_config(img_type)
    for idx, row in img_df.iterrows():
        shuffled_row = _shuffle_row_img(
            row=row,
            rng=rng,
            fields_config=fields_config,
            process_method=process_method,
        )
        shuffled_df.loc[idx, :] = shuffled_row
    return shuffled_df
