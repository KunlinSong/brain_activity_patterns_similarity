import numpy as np
import pandas as pd

from .config import (
    get_categories_config,
    get_features_config,
    get_formats_config,
)

__all__ = [
    "get_random_img_df",
]


def _randomize_img(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = img.copy()
    non_nan_indices = np.nonzero(~np.isnan(img))
    random_indices = np.stack(non_nan_indices, axis=0)
    rng.shuffle(random_indices, axis=-1)
    random_indices = tuple(random_indices)
    img[non_nan_indices] = img[random_indices]
    return img


def get_random_img_df(
    img_df: pd.DataFrame, random_seed: int = 42
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    _FORMATS_CONFIG = get_formats_config()
    dtype_stim_feat = _FORMATS_CONFIG.format_datatype_stimulation_feature(
        stimulation_feat=_FEATURES_CONFIG.STIMULATION,
        data_type_feat=_FEATURES_CONFIG.DATA_TYPE,
    )

    rng = np.random.default_rng(random_seed)
    random_img_df = img_df.copy()
    for img_idx, img_row in img_df.iterrows():
        random_img_df.at[img_idx, _FEATURES_CONFIG.DATA_TYPE] = (
            _CATEGORIES_CONFIG.DATA_TYPE.RANDOM
        )

        stimulation = img_row[_FEATURES_CONFIG.STIMULATION]
        dtype_stim_category = (
            _FORMATS_CONFIG.format_datatype_stimulation_category(
                stimulation=stimulation,
                data_type=_CATEGORIES_CONFIG.DATA_TYPE.RANDOM,
            )
        )

        random_img_df.at[img_idx, dtype_stim_feat] = dtype_stim_category
        imgs = img_row[_FEATURES_CONFIG.IMAGES].copy()
        original_img = imgs[_CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL]
        random_img = _randomize_img(original_img, rng)
        imgs[_CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL] = random_img
        random_img_df.at[img_idx, _FEATURES_CONFIG.IMAGES] = imgs

    return random_img_df
