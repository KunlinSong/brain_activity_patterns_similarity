from typing import Callable

import pandas as pd

from .config import get_categories_config, get_features_config
from .similarity_dataset import get_similarity_df

__all__ = [
    "concat_img_dfs",
    "process_original",
    "add_similarity",
]


def concat_img_dfs(img_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    img_df = pd.concat(img_dfs, ignore_index=True).copy()
    img_df = img_df.sort_values(
        by=[
            _FEATURES_CONFIG.DATA_TYPE,
            _FEATURES_CONFIG.STIMULATION,
            _FEATURES_CONFIG.SUBJECT,
        ],
        ignore_index=True,
    )
    return img_df


def process_original(
    img_df: pd.DataFrame, process_name: str, process_func: Callable
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    img_df = img_df.copy()
    for img_idx, img_row in img_df.iterrows():
        imgs = img_row[_FEATURES_CONFIG.IMAGES].copy()
        original_img = imgs[_CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL]
        process_img = process_func(original_img)
        imgs[process_name] = process_img
        img_df.at[img_idx, _FEATURES_CONFIG.IMAGES] = imgs
    return img_df


def add_similarity(
    img_df: pd.DataFrame,
    process_name: str | None,
    similarity_name: str,
    similarity_func: Callable,
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    img_df = img_df.copy()
    if _FEATURES_CONFIG.SIMILARITY not in img_df.columns:
        img_df[_FEATURES_CONFIG.SIMILARITY] = None
    for img_idx, img_row in img_df.iterrows():
        similarity_df: pd.DataFrame | None = img_row[
            _FEATURES_CONFIG.SIMILARITY
        ]
        similarity_df_add = get_similarity_df(
            img_row=img_row,
            img_df=img_df,
            process_name=process_name,
            similarity_name=similarity_name,
            similarity_func=similarity_func,
        )
        similarity_df = pd.concat(
            [
                similarity_df,
                similarity_df_add,
            ],
            ignore_index=True,
        )
        img_df.at[img_idx, _FEATURES_CONFIG.SIMILARITY] = similarity_df
    return img_df
