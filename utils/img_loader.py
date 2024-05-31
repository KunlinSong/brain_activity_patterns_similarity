"""Module for loading images as `pandas.DataFrame` objects.

We use pandas to build a DataFrame object that contains the image data.
"""

import re
from collections import defaultdict
from itertools import product
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

from .config import get_basic_config, get_roi_config_df

__all__ = [
    "load_brain_images",
    "load_roi_images",
]


def _get_filename(stimulation: str, subject: str) -> str:
    match stimulation:
        case "auditory stimulation":
            n = re.findall(r"subject_(\d+)", subject)[0]
            return f"Words_{int(n)}.nii"
        case "visual stimulation":
            return "con_0006.img"
        case _:
            raise ValueError(f"Unknown stimulation: {stimulation}")


def load_brain_images(
    dirname: str,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    brain_images = defaultdict(list)
    dirname: Path = Path(dirname)
    data_type_feat = _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE
    stimulation_feat = _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
    dtype_stim_feat = eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_FEAT)
    data_type = _BASIC_CONFIG.NAMES.DATA_TYPE.REAL
    for stimulation_dir in dirname.iterdir():
        if not stimulation_dir.is_dir():
            continue
        stimulation = stimulation_dir.name
        for subject_dir in stimulation_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            filename = _get_filename(
                stimulation=stimulation_dir.name, subject=subject_dir.name
            )
            img_path = subject_dir.joinpath(filename)
            if not img_path.exists():
                print(f"Warning: file not found. Skipping ({img_path}).")
                continue
            img = np.array(nibabel.load(img_path).get_fdata())

            brain_images[data_type_feat].append(data_type)
            brain_images[stimulation_feat].append(stimulation)
            brain_images[dtype_stim_feat].append(
                eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_NAME)
            )
            brain_images[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT].append(
                subject_dir.name
            )
            brain_images[_BASIC_CONFIG.NAMES.PROCESS.ORIGINAL].append(img)

    brain_img_df = pd.DataFrame(brain_images)
    brain_img_df = brain_img_df.sort_values(
        by=[
            _BASIC_CONFIG.DATASET_FEATURES.STIMULATION,
            _BASIC_CONFIG.DATASET_FEATURES.SUBJECT,
        ]
    )
    brain_img_df = brain_img_df.reset_index(drop=True)
    return brain_img_df


def load_roi_images(dirname: str) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    roi_config_df = get_roi_config_df()
    brain_img_df = load_brain_images(dirname=dirname)
    data_type_feat = _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE
    stimulation_feat = _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
    dtype_stim_feat = eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_FEAT)
    roi_img_dict = defaultdict(list)
    for (_, brain_img_info), (_, roi_info) in product(
        brain_img_df.iterrows(), roi_config_df.iterrows()
    ):
        brain_img = brain_img_info[_BASIC_CONFIG.NAMES.PROCESS.ORIGINAL]
        roi_img = brain_img[
            roi_info[
                _BASIC_CONFIG.NAMES.ROI_CONFIG.X,
                _BASIC_CONFIG.NAMES.ROI_CONFIG.Y,
                _BASIC_CONFIG.NAMES.ROI_CONFIG.Z,
            ]
        ]
        for feat in [
            _BASIC_CONFIG.DATASET_FEATURES.REGION,
            _BASIC_CONFIG.DATASET_FEATURES.STRUCTURE,
            _BASIC_CONFIG.DATASET_FEATURES.HEMISPHERE,
        ]:
            roi_img_dict[feat].append(roi_info[feat])
        for feat in [
            data_type_feat,
            stimulation_feat,
            dtype_stim_feat,
            _BASIC_CONFIG.DATASET_FEATURES.SUBJECT,
        ]:
            roi_img_dict[feat].append(brain_img_info[feat])

        structure = roi_info[_BASIC_CONFIG.DATASET_FEATURES.STRUCTURE]
        stimulation = brain_img_info[
            _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
        ]
        roi_img_dict[_BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC].append(
            stimulation
            == _BASIC_CONFIG.REGION_SPECIFIC_STIMULATIONS[structure]
        )

        roi_img_dict[_BASIC_CONFIG.NAMES.PROCESS.ORIGINAL].append(roi_img)
    roi_img_df = pd.DataFrame(roi_img_dict)
    roi_img_df = roi_img_df.sort_values(
        by=[
            _BASIC_CONFIG.DATASET_FEATURES.REGION,
            _BASIC_CONFIG.DATASET_FEATURES.STIMULATION,
            _BASIC_CONFIG.DATASET_FEATURES.SUBJECT,
        ],
        ignore_index=True,
    )
    return roi_img_df
