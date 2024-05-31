import numpy as np
import pandas as pd

from .config import get_basic_config

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
    _BASIC_CONFIG = get_basic_config()
    rng = np.random.default_rng(random_seed)
    data_type_feat = _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE
    stimulation_feat = _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
    dtype_stim_feat = eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_FEAT)
    data_type = _BASIC_CONFIG.NAMES.DATA_TYPE.RANDOM
    random_img_df = img_df.copy()
    random_img_df[_BASIC_CONFIG.NAMES.PROCESS.ORIGINAL] = random_img_df[
        _BASIC_CONFIG.NAMES.PROCESS.ORIGINAL
    ].apply(lambda img: _randomize_img(img, rng))
    random_img_df[data_type_feat] = _BASIC_CONFIG.NAMES.DATA_TYPE.RANDOM
    for img_idx, img_row in random_img_df.iterrows():
        stimulation = img_row[stimulation_feat]
        random_img_df.at[img_idx, dtype_stim_feat] = eval(
            _BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_NAME
        )
    return random_img_df
