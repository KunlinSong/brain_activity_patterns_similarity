from typing import Callable, Literal

import pandas as pd

from .config import FIELDS_CONFIG, FORMATS_CONFIG, LABELS_CONFIG, ROIS_CONFIG
from .img_loader import get_img
from .similarity_dataset import compute_similarity_dataset
from .utils import get_fields_config

__all__ = [
    "concat_dfs",
    "process_original_imgs",
    "add_similarity",
]


def concat_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs, ignore_index=True).copy()


def process_original_imgs(
    img_df: pd.DataFrame,
    process_method: str,
    process_func: Callable,
    img_type: Literal["whole brain", "ROI"] = "ROI",
) -> pd.DataFrame:
    fields_config = get_fields_config(img_type)
    img_df = img_df.copy()
    for idx in img_df.index:
        original_img = get_img(
            idx=idx,
            img_df=img_df,
            process_method=process_method,
            fields_config=fields_config,
        )
        process_img = process_func(original_img)
        imgs: pd.Series = img_df.at[idx, fields_config.images].copy()
        imgs[process_method] = process_img
        img_df.at[idx, fields_config.images] = imgs
    return img_df


def add_similarity(
    img_df: pd.DataFrame,
    process_method: str | None,
    similarity_method: str,
    similarity_func: Callable,
    img_type: Literal["whole brain", "ROI"] = "ROI",
) -> pd.DataFrame:
    fields_config = get_fields_config(img_type)
    img_df = img_df.copy()
    for idx in img_df.index:
        similarity_dataset: pd.DataFrame | None = img_df.at[
            idx, fields_config.similarity
        ]
        similarity_dataset_add = compute_similarity_dataset(
            idx=idx,
            img_df=img_df,
            process_method=process_method,
            similarity_method=similarity_method,
            similarity_func=similarity_func,
            img_type=img_type,
        )
        similarity_dataset = concat_dfs(
            [similarity_dataset, similarity_dataset_add]
        )
        img_df.at[idx, fields_config.similarity] = similarity_dataset
    return img_df


# TODO
def assign_roi_pattern_labels(
    img_df: pd.DataFrame,
    n_patterns: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    img_df = img_df.copy()
