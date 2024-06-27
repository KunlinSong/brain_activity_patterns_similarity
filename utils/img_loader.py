"""Module for loading images as `pandas.DataFrame` objects.

We use pandas to build a DataFrame object that contains the image data.
"""

import re
from itertools import product
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

from .config import (
    get_categories_config,
    get_features_config,
    get_formats_config,
    get_rois_config,
)

__all__ = ["load_brain_images", "load_roi_images"]


def _get_filename(stimulation: str, subject: str) -> str:
    match stimulation:
        case "auditory stimulation":
            n = re.findall(r"subject_(\d+)", subject)[0]
            return f"Words_{int(n)}.nii"
        case "visual stimulation":
            return "con_0006.img"
        case _:
            raise ValueError(f"Unknown stimulation: {stimulation}")


def _load_brain_img(
    dirname: Path, stimulation: str, subject: str
) -> pd.Series:
    _FEATURES_CONFIG = get_features_config()
    _FORMATS_CONFIG = get_formats_config()
    _CATEGORIES_CONFIG = get_categories_config()
    filename = _get_filename(stimulation=stimulation, subject=subject)
    img_path = dirname.joinpath(filename)
    if not img_path.exists():
        print(f"Warning: file not found. Skipping ({img_path}).")
        raise FileNotFoundError
    img = np.array(nibabel.load(img_path).get_fdata())
    dtype_stim_feat = _FORMATS_CONFIG.format_datatype_stimulation_feature(
        stimulation_feat=_FEATURES_CONFIG.STIMULATION,
        data_type_feat=_FEATURES_CONFIG.DATA_TYPE,
    )
    dtype_stim_category = _FORMATS_CONFIG.format_datatype_stimulation_category(
        stimulation=stimulation,
        data_type=_CATEGORIES_CONFIG.DATA_TYPE.REAL,
    )
    img_info_dict = {
        _FEATURES_CONFIG.DATA_TYPE: _CATEGORIES_CONFIG.DATA_TYPE.REAL,
        _FEATURES_CONFIG.STIMULATION: stimulation,
        dtype_stim_feat: dtype_stim_category,
        _FEATURES_CONFIG.SUBJECT: subject,
        _FEATURES_CONFIG.IMAGES: pd.Series(
            {_CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL: img}
        ),
    }
    return pd.Series(img_info_dict)


def load_brain_images(
    dirname: str,
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    img_info_lst = []
    dirname: Path = Path(dirname)
    for stimulation_dir in dirname.iterdir():
        if not stimulation_dir.is_dir():
            continue
        stimulation = stimulation_dir.name
        for subject_dir in stimulation_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name

            try:
                img_info_series = _load_brain_img(
                    dirname=subject_dir,
                    stimulation=stimulation,
                    subject=subject,
                )
            except FileNotFoundError:
                continue
            img_info_lst.append(img_info_series)

    brain_img_df = pd.DataFrame(img_info_lst)
    brain_img_df = brain_img_df.sort_values(
        by=[
            _FEATURES_CONFIG.STIMULATION,
            _FEATURES_CONFIG.SUBJECT,
        ]
    )
    brain_img_df = brain_img_df.reset_index(drop=True)
    return brain_img_df


def load_roi_images(dirname: str) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    rois_config = get_rois_config()
    brain_imgs_df = load_brain_images(dirname=dirname)
    roi_img_series_lst = []
    for (_, brain_img_info), region in product(
        brain_imgs_df.iterrows(), rois_config.regions
    ):
        original_brain_img = brain_img_info[_FEATURES_CONFIG.IMAGES][
            _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
        ]
        roi_img_series = rois_config.get_original_roi_img_series(
            original_brain_img=original_brain_img,
            region=region,
            stimulation=brain_img_info[_FEATURES_CONFIG.STIMULATION],
            data_type=brain_img_info[_FEATURES_CONFIG.DATA_TYPE],
        )
        roi_img_series[_FEATURES_CONFIG.SUBJECT] = brain_img_info[
            _FEATURES_CONFIG.SUBJECT
        ]
        roi_img_series_lst.append(roi_img_series)
    roi_img_df = pd.DataFrame(roi_img_series_lst)
    roi_img_df = roi_img_df.sort_values(
        by=[
            _FEATURES_CONFIG.REGION,
            _FEATURES_CONFIG.STIMULATION,
            _FEATURES_CONFIG.SUBJECT,
        ],
        ignore_index=True,
    )
    return roi_img_df
