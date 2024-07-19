"""the module for loading images as `pandas.DataFrame` objects.

We use pandas to build a DataFrame object that contains the image data.
"""

import re
from functools import partial
from itertools import product
from pathlib import Path
from typing import Literal

import nibabel
import numpy as np
import pandas as pd

from .config import (
    FIELDS_CONFIG,
    FORMATS_CONFIG,
    LABELS_CONFIG,
    ROIS_CONFIG,
    ROIImgFieldsConfig,
    WholeBrainImgFieldsConfig,
)
from .const import DATA_ROOT

__all__ = ["load_whole_brain_images", "load_roi_images"]


def _get_filename(
    stimulation: Literal["auditory stimulation", "visual stimulation"],
    subject: str,
) -> str:
    match stimulation:
        case "auditory stimulation":
            n = re.findall(r"subject_(\d+)", subject)[0]
            return f"Words_{int(n)}.nii"
        case "visual stimulation":
            return "con_0006.img"
        case _:
            raise ValueError(f"Unknown stimulation: {stimulation}")


def _load_brain_img(stimulation: str, subject: str) -> pd.Series:
    filename = _get_filename(stimulation=stimulation, subject=subject)
    img_path = DATA_ROOT.joinpath(stimulation, subject, filename)
    if not img_path.exists():
        print(f"Warning: file not found. Skipping ({img_path}).")
        raise FileNotFoundError
    img = np.array(nibabel.load(img_path).get_fdata())
    img_info = {
        FIELDS_CONFIG.whole_brain_img.data_type: LABELS_CONFIG.data_type.real,
        FIELDS_CONFIG.whole_brain_img.stimulation: stimulation,
        FIELDS_CONFIG.whole_brain_img.subject: subject,
        FIELDS_CONFIG.whole_brain_img.images: pd.Series(
            {
                LABELS_CONFIG.process_method.original: img,
            }
        ),
        FIELDS_CONFIG.roi_img.as_standard_pattern: None,
        FIELDS_CONFIG.roi_img.similarity: None,
    }
    return pd.Series(img_info)


def load_whole_brain_images() -> pd.DataFrame:
    img_infos = []
    for stimulation_dirname in DATA_ROOT.iterdir():
        if not stimulation_dirname.is_dir():
            continue
        stimulation = stimulation_dirname.name
        for subject_dirname in stimulation_dirname.iterdir():
            if not subject_dirname.is_dir():
                continue
            subject = subject_dirname.name

            try:
                img_infos.append(
                    _load_brain_img(
                        stimulation=stimulation,
                        subject=subject,
                    )
                )
            except FileNotFoundError as err:
                print(err)
                continue

    whole_brain_img_df = pd.DataFrame(img_infos)
    whole_brain_img_df = whole_brain_img_df.sort_values(
        by=[
            FIELDS_CONFIG.whole_brain_img.stimulation,
            FIELDS_CONFIG.whole_brain_img.subject,
        ],
        ignore_index=True,
        axis=0,
    )
    return whole_brain_img_df


def load_roi_images() -> pd.DataFrame:
    roi_img_infos = []
    whole_brain_images_df = load_whole_brain_images()
    structures = ["STG", "FFA"]
    hemispheres = ["L", "R"]
    for (_, whole_brain_img_info), structure, hemisphere in product(
        whole_brain_images_df.iterrows(), structures, hemispheres
    ):
        data_type = whole_brain_img_info[
            FIELDS_CONFIG.whole_brain_img.data_type
        ]
        stimulation = whole_brain_img_info[
            FIELDS_CONFIG.whole_brain_img.stimulation
        ]
        subject = whole_brain_img_info[FIELDS_CONFIG.whole_brain_img.subject]
        whole_brain_images: pd.Series = whole_brain_img_info[
            FIELDS_CONFIG.whole_brain_img.images
        ]
        region = FORMATS_CONFIG.format_region_category(
            structure=structure, hemisphere=hemisphere
        )
        if stimulation == ROIS_CONFIG.get_roi_specific_stimulation(
            structure=structure
        ):
            specificity_type = LABELS_CONFIG.specificity_type.specific
        else:
            specificity_type = LABELS_CONFIG.specificity_type.non_specific
        get_roi_axis_slice = partial(
            ROIS_CONFIG.get_roi_axis_slice,
            structure=structure,
            hemisphere=hemisphere,
        )
        roi_img_info = pd.Series(
            {
                FIELDS_CONFIG.roi_img.data_type: data_type,
                FIELDS_CONFIG.roi_img.stimulation: stimulation,
                FIELDS_CONFIG.roi_img.subject: subject,
                FIELDS_CONFIG.roi_img.images: pd.Series(
                    {
                        process_method: img[
                            get_roi_axis_slice(
                                axis="x", len_axis=img.shape[0]
                            ),
                            get_roi_axis_slice(
                                axis="y", len_axis=img.shape[1]
                            ),
                            get_roi_axis_slice(
                                axis="z", len_axis=img.shape[2]
                            ),
                        ]
                        for process_method, img in whole_brain_images.items()
                    }
                ),
                FIELDS_CONFIG.roi_img.region: region,
                FIELDS_CONFIG.roi_img.structure: structure,
                FIELDS_CONFIG.roi_img.hemisphere: hemisphere,
                FIELDS_CONFIG.roi_img.specificity_type: specificity_type,
                FIELDS_CONFIG.roi_img.as_standard_pattern: None,
                FIELDS_CONFIG.roi_img.similarity: None,
            }
        )
        roi_img_infos.append(roi_img_info)
    roi_imgs_df = pd.DataFrame(roi_img_infos)
    roi_imgs_df = roi_imgs_df.sort_values(
        by=[
            FIELDS_CONFIG.roi_img.region,
            FIELDS_CONFIG.roi_img.stimulation,
            FIELDS_CONFIG.roi_img.subject,
        ],
        ignore_index=True,
    )
    return roi_imgs_df


def get_img(
    idx: int,
    img_df: pd.DataFrame,
    process_method: str,
    fields_config: WholeBrainImgFieldsConfig | ROIImgFieldsConfig,
) -> np.ndarray:
    row = img_df.loc[idx, :]
    imgs: pd.Series = row[fields_config.images]
    return imgs[process_method].copy()
