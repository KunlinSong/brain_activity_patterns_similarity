from functools import partial
from typing import Callable

import pandas as pd

from .config import get_basic_config
from .similarity_dataset import get_similarity_df

__all__ = [
    "concat_img_dfs",
]


def concat_img_dfs(img_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    img_df = pd.concat(img_dfs, ignore_index=True).copy()
    img_df = img_df.sort_values(
        by=[
            _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE,
            _BASIC_CONFIG.DATASET_FEATURES.STIMULATION,
            _BASIC_CONFIG.DATASET_FEATURES.SUBJECT,
        ],
        ignore_index=True,
    )
    return img_df


def process_original(
    img_df: pd.DataFrame, process_name: str, process_func: Callable
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    img_df = img_df.copy()
    img_df[process_name] = img_df[_BASIC_CONFIG.NAMES.PROCESS.ORIGINAL].apply(
        lambda original: process_func(original)
    )
    return img_df


def add_similarity(
    img_df: pd.DataFrame,
    process_name: str | None,
    similarity_name: str,
    similarity_func: Callable,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    img_df = img_df.copy()
    if _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY not in img_df.columns:
        img_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY] = None
    for img_idx, img_row in img_df.iterrows():
        similarity_df = get_similarity_df(
            img_row=img_row,
            img_df=img_df,
            process_name=process_name,
            similarity_name=similarity_name,
            similarity_func=similarity_func,
        )
        similarity_df = pd.concat(
            [
                img_row[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY],
                similarity_df,
            ],
            ignore_index=True,
        )
        img_df.at[img_idx, _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY] = (
            similarity_df
        )
    return img_df
