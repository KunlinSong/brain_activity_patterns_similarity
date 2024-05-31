"""Module for loading data.

We use pandas to build a database to store the brain activity patterns 
in each ROI and the corresponding labels. By using the functions in 
`utils/similarity.py`, the database can be easily used to compute the
similarity between two brain activity patterns and store it as a new 
column in the database.
"""

import re
from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import nibabel
import numpy as np
import pandas as pd

from .config import _get_axis_max_feat, _get_axis_min_feat, get_basic_config

_CONFIG = get_basic_config()


def get_filename(stimulation: str, subject: str) -> str:
    match stimulation:
        case "auditory stimulation":
            n = re.findall(r"subject_(\d+)", subject)[0]
            return f"Words_{int(n)}.nii"
        case "visual stimulation":
            return "con_0006.img"
        case _:
            raise ValueError(f"Unknown stimulation: {stimulation}")


def load_brain_images(
    dirname: str, get_filename_func: Callable | None = None
) -> pd.DataFrame:
    brain_images = defaultdict(list)
    dirname: Path = Path(dirname)
    for stimulation in dirname.iterdir():
        if not stimulation.is_dir():
            continue
        for subject in stimulation.iterdir():
            if not subject.is_dir():
                continue

            if get_filename_func is None:
                get_filename_func = get_filename

            filename = get_filename_func(
                stimulation=stimulation.name, subject=subject.name
            )
            path = subject.joinpath(filename)
            if not path.exists():
                print(f"Warning: file not found. Skipping ({path}).")
                continue
            img = np.array(nibabel.load(path).get_fdata())

            brain_images[_CONFIG.DATASET_FEATURES.DATA_TYPE].append(
                _CONFIG.NAMES.DATA_TYPE.REAL
            )
            brain_images[_CONFIG.DATASET_FEATURES.STIMULATION].append(
                stimulation.name
            )
            brain_images[_CONFIG.DATASET_FEATURES.SUBJECT].append(subject.name)
            brain_images[_CONFIG.NAMES.PROCESS.ORIGINAL].append(img)

    brain_img_df = pd.DataFrame(brain_images)
    brain_img_df = brain_img_df.sort_values(
        by=[
            _CONFIG.DATASET_FEATURES.STIMULATION,
            _CONFIG.DATASET_FEATURES.SUBJECT,
        ]
    )
    brain_img_df = brain_img_df.reset_index(drop=True)
    return brain_img_df


def get_specific_stimulation(structure: str) -> str:
    return _CONFIG.REGION_SPECIFIC_STIMULATIONS[structure]


def is_specific(stimulation: str, structure: str) -> int:
    return int(get_specific_stimulation(structure=structure) == stimulation)


def get_roi_img_df(
    brain_img_df: pd.DataFrame, rois_df: pd.DataFrame
) -> pd.DataFrame:
    roi_dataset = defaultdict(list)
    for _, roi in rois_df.iterrows():
        x_min, x_max = (
            roi[_get_axis_min_feat(_CONFIG.NAMES.ROI_CONFIG.X)],
            roi[_get_axis_max_feat(_CONFIG.NAMES.ROI_CONFIG.X)],
        )
        y_min, y_max = (
            roi[_get_axis_min_feat(_CONFIG.NAMES.ROI_CONFIG.Y)],
            roi[_get_axis_max_feat(_CONFIG.NAMES.ROI_CONFIG.Y)],
        )
        z_min, z_max = (
            roi[_get_axis_min_feat(_CONFIG.NAMES.ROI_CONFIG.Z)],
            roi[_get_axis_max_feat(_CONFIG.NAMES.ROI_CONFIG.Z)],
        )
        for _, brain_img in brain_img_df.iterrows():
            img = brain_img[_CONFIG.NAMES.PROCESS.ORIGINAL]
            roi_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
            roi_dataset[_CONFIG.DATASET_FEATURES.REGION].append(
                roi[_CONFIG.DATASET_FEATURES.REGION]
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.STRUCTURE].append(
                roi[_CONFIG.DATASET_FEATURES.STRUCTURE]
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.HEMISPHERE].append(
                roi[_CONFIG.DATASET_FEATURES.HEMISPHERE]
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.DATA_TYPE].append(
                _CONFIG.NAMES.DATA_TYPE.REAL
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.STIMULATION].append(
                brain_img[_CONFIG.DATASET_FEATURES.STIMULATION]
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.IS_SPECIFIC].append(
                is_specific(
                    stimulation=brain_img[
                        _CONFIG.DATASET_FEATURES.STIMULATION
                    ],
                    structure=roi[_CONFIG.DATASET_FEATURES.STRUCTURE],
                )
            )
            roi_dataset[_CONFIG.DATASET_FEATURES.SUBJECT].append(
                brain_img[_CONFIG.DATASET_FEATURES.SUBJECT]
            )
            roi_dataset[_CONFIG.NAMES.PROCESS.ORIGINAL].append(roi_img)
    roi_dataset_df = pd.DataFrame(roi_dataset)
    roi_dataset_df[_CONFIG.DATASET_FEATURES.SIMILARITY] = None
    roi_dataset_df = roi_dataset_df.sort_values(
        by=[
            _CONFIG.DATASET_FEATURES.REGION,
            _CONFIG.DATASET_FEATURES.STIMULATION,
            _CONFIG.DATASET_FEATURES.SUBJECT,
        ]
    )
    roi_dataset_df = roi_dataset_df.reset_index(drop=True)
    return roi_dataset_df


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
    rng = np.random.default_rng(random_seed)
    random_img_df = img_df.copy()
    random_img_df[_CONFIG.NAMES.PROCESS.ORIGINAL] = random_img_df[
        _CONFIG.NAMES.PROCESS.ORIGINAL
    ].apply(lambda img: _randomize_img(img, rng))
    random_img_df[_CONFIG.DATASET_FEATURES.DATA_TYPE] = (
        _CONFIG.NAMES.DATA_TYPE.RANDOM
    )
    return random_img_df


def concat_img_dfs(img_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    img_df = pd.concat(img_dfs, ignore_index=True)
    img_df = img_df.sort_values(
        by=[
            _CONFIG.DATASET_FEATURES.REGION,
            _CONFIG.DATASET_FEATURES.DATA_TYPE,
            _CONFIG.DATASET_FEATURES.STIMULATION,
            _CONFIG.DATASET_FEATURES.SUBJECT,
        ]
    )
    img_df = img_df.reset_index(drop=True)
    return img_df


def process_original(
    img_df: pd.DataFrame, process_name: str, process_func: Callable
) -> pd.DataFrame:
    img_df = img_df.copy()
    img_df[process_name] = img_df[_CONFIG.NAMES.PROCESS.ORIGINAL].apply(
        lambda img: process_func(img)
    )
    return img_df


def _get_process_name(process_name: str | None = None) -> str:
    return (
        _CONFIG.NAMES.PROCESS.ORIGINAL
        if process_name is None
        else process_name
    )


def _get_similarity_sub_df(
    img_row: pd.Series,
    img_df: pd.DataFrame,
    similarity_func: Callable,
    similarity_name: str,
    process_name: str | None = None,
):
    process_name = _get_process_name(process_name)
    img_df = img_df.copy()
    img_df = img_df[
        img_df[_CONFIG.DATASET_FEATURES.REGION]
        == img_row[_CONFIG.DATASET_FEATURES.REGION]
    ]

    similarity_df = img_df.copy()
    similarity_df = similarity_df[
        [
            _CONFIG.DATASET_FEATURES.DATA_TYPE,
            _CONFIG.DATASET_FEATURES.STIMULATION,
            _CONFIG.DATASET_FEATURES.SUBJECT,
            _CONFIG.DATASET_FEATURES.IS_SPECIFIC,
            _CONFIG.DATASET_FEATURES.SIMILARITY,
        ]
    ].copy()
    similarity_df[_CONFIG.DATASET_FEATURES.SUBJECT_ID] = img_df.index.to_list()
    similarity_df[_CONFIG.DATASET_FEATURES.PROCESS_METHOD] = process_name
    similarity_df[_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD] = similarity_name
    similarity_df[_CONFIG.DATASET_FEATURES.SIMILARITY] = img_df[
        process_name
    ].apply(
        lambda process_img: similarity_func(img_row[process_name], process_img)
    )
    return similarity_df


def compute_similarity(
    img_df: pd.DataFrame,
    similarity_name: str,
    similarity_func: Callable,
    process_name: str | None = None,
) -> pd.DataFrame:
    get_similarity_sub_df = partial(
        _get_similarity_sub_df,
        img_df=img_df,
        similarity_func=similarity_func,
        similarity_name=similarity_name,
        process_name=process_name,
    )
    img_df = img_df.copy()

    for img_idx, img_row in img_df.iterrows():
        similarity_sub_df = get_similarity_sub_df(img_row=img_row)
        similarity_df: pd.DataFrame = pd.concat(
            [
                img_row[_CONFIG.DATASET_FEATURES.SIMILARITY],
                similarity_sub_df,
            ]
        )
        similarity_df = similarity_df.sort_values(
            by=[
                _CONFIG.DATASET_FEATURES.SIMILARITY_METHOD,
                _CONFIG.DATASET_FEATURES.PROCESS_METHOD,
                _CONFIG.DATASET_FEATURES.SUBJECT_ID,
            ],
            ignore_index=True,
        )
        img_df.at[img_idx, _CONFIG.DATASET_FEATURES.SIMILARITY] = similarity_df
    return img_df


def get_similarity_mat(
    img_df: pd.DataFrame,
    region: str,
    similarity_name: str,
    process_name: str | None = None,
) -> tuple[pd.DataFrame, list[list[int]]]:
    process_name = _get_process_name(process_name=process_name)
    img_df = img_df[img_df[_CONFIG.DATASET_FEATURES.REGION] == region].copy()
    img_df = img_df.sort_index()
    similarity_mat = np.empty((len(img_df), len(img_df)))
    for mat_idx, (_, img_row) in enumerate(img_df.iterrows()):
        similarity_df: pd.DataFrame = img_row[
            _CONFIG.DATASET_FEATURES.SIMILARITY
        ]
        similarity_df = similarity_df[
            (
                similarity_df[_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD]
                == similarity_name
            )
            & (
                similarity_df[_CONFIG.DATASET_FEATURES.PROCESS_METHOD]
                == process_name
            )
        ]
        similarity_df = similarity_df.sort_values(
            by=_CONFIG.DATASET_FEATURES.SUBJECT_ID
        )
        similarity_mat[mat_idx] = similarity_df[
            _CONFIG.DATASET_FEATURES.SIMILARITY
        ].values
    return similarity_mat, [
        img_df.index.tolist(),
        similarity_df[_CONFIG.DATASET_FEATURES.SUBJECT_ID].tolist(),
    ]


def _get_same_type_similarity_sub_df(
    img_idx: int, img_row: pd.Series
) -> pd.DataFrame:
    similarity_df: pd.DataFrame = img_row[
        _CONFIG.DATASET_FEATURES.SIMILARITY
    ].copy()
    similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.SUBJECT_ID] != img_idx
    ]
    same_type_df = similarity_df[
        (
            similarity_df[_CONFIG.DATASET_FEATURES.DATA_TYPE]
            == img_row[_CONFIG.DATASET_FEATURES.DATA_TYPE]
        )
        & (
            similarity_df[_CONFIG.DATASET_FEATURES.STIMULATION]
            == img_row[_CONFIG.DATASET_FEATURES.STIMULATION]
        )
    ].copy()
    same_type_df[_CONFIG.DATASET_FEATURES.REGION] = img_row[
        _CONFIG.DATASET_FEATURES.REGION
    ]
    same_type_df[_CONFIG.DATASET_FEATURES.STRUCTURE] = img_row[
        _CONFIG.DATASET_FEATURES.STRUCTURE
    ]
    same_type_df[_CONFIG.DATASET_FEATURES.HEMISPHERE] = img_row[
        _CONFIG.DATASET_FEATURES.HEMISPHERE
    ]
    return same_type_df


def get_same_type_similarity_df(img_df: pd.DataFrame) -> pd.DataFrame:
    sub_df_lst = []
    for img_idx, img_row in img_df.iterrows():
        sub_df = _get_same_type_similarity_sub_df(
            img_idx=img_idx, img_row=img_row
        )
        sub_df_lst.append(sub_df)
    return pd.concat(sub_df_lst, ignore_index=True)


def _combine_subject_ids(subject_ids: list, n_subjects: int = 0) -> Iterable:
    if not subject_ids:
        raise ValueError("There are subject IDs in `subject_ids`.")
    if n_subjects <= -len(subject_ids):
        n_subjects = 1
    elif n_subjects <= 0:
        n_subjects += len(subject_ids)
    else:
        n_subjects = min(n_subjects, len(subject_ids))

    yield from combinations(subject_ids, n_subjects)


def _compute_similarity_avg(
    similarity_df: pd.DataFrame,
    similarity_name: str,
    process_name: str | None,
    avg_ids: list,
) -> float:
    process_name = _get_process_name(process_name)
    similarity_df = similarity_df[
        (
            similarity_df[_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD]
            == similarity_name
        )
        & (
            similarity_df[_CONFIG.DATASET_FEATURES.PROCESS_METHOD]
            == process_name
        )
    ].copy()
    return similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.SUBJECT_ID].isin(avg_ids)
    ][_CONFIG.DATASET_FEATURES.SIMILARITY].mean()


def get_average_similarity(
    img_idx: int,
    img_row: pd.Series,
    similarity_process_pairs: list[str, str | None],
    n_subjects: int = 0,
) -> pd.DataFrame:
    avg_similarity_df_dict = defaultdict(list)
    similarity_df: pd.DataFrame = img_row[
        _CONFIG.DATASET_FEATURES.SIMILARITY
    ].copy()
    similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.SUBJECT_ID] != img_idx
    ]
    similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.DATA_TYPE]
        == _CONFIG.NAMES.DATA_TYPE.REAL
    ]
    specific_ids = (
        similarity_df[
            similarity_df[_CONFIG.DATASET_FEATURES.IS_SPECIFIC] == 1
        ][_CONFIG.DATASET_FEATURES.SUBJECT_ID]
        .unique()
        .tolist()
    )
    non_specific_ids = (
        similarity_df[
            similarity_df[_CONFIG.DATASET_FEATURES.IS_SPECIFIC] == 0
        ][_CONFIG.DATASET_FEATURES.SUBJECT_ID]
        .unique()
        .tolist()
    )

    def _append_avg_similarity(subject_ids: list, similarity_type: str):
        nonlocal avg_similarity_df_dict
        for similarity_name, process_name in similarity_process_pairs:
            for avg_ids in _combine_subject_ids(
                subject_ids=subject_ids, n_subjects=n_subjects
            ):
                avg_similarity_df_dict[
                    eval(_CONFIG.FORMATS.PROCESS_SIMILARITY_FEAT)
                ].append(
                    _compute_similarity_avg(
                        similarity_df=similarity_df,
                        similarity_name=similarity_name,
                        process_name=process_name,
                        avg_ids=avg_ids,
                    )
                )

    _append_avg_similarity(
        subject_ids=specific_ids,
        similarity_type=_CONFIG.NAMES.SIMILARITY_TYPE.SPECIFIC,
    )
    _append_avg_similarity(
        subject_ids=non_specific_ids,
        similarity_type=_CONFIG.NAMES.SIMILARITY_TYPE.NON_SPECIFIC,
    )
    avg_similarity_df = pd.DataFrame(avg_similarity_df_dict)
    avg_similarity_df[_CONFIG.DATASET_FEATURES.IS_SPECIFIC] = img_row[
        _CONFIG.DATASET_FEATURES.IS_SPECIFIC
    ]
    return avg_similarity_df


def get_average_similarity_dataset(
    img_df: pd.DataFrame,
    similarity_and_process_pairs: list[tuple[str, str | None]],
    only_real: bool = True,
    n_subjects: int = 0,
) -> pd.DataFrame:
    if only_real:
        img_df = img_df[
            img_df[_CONFIG.DATASET_FEATURES.DATA_TYPE]
            == _CONFIG.NAMES.DATA_TYPE.REAL
        ]
    avg_similarity_df_lst = []
    for img_idx, img_row in img_df.iterrows():
        row_avg_similarity_df = get_average_similarity(
            img_idx=img_idx,
            img_row=img_row,
            similarity_process_pairs=similarity_and_process_pairs,
            n_subjects=n_subjects,
        )
        avg_similarity_df_lst.append(row_avg_similarity_df)
    avg_similarity_df = pd.concat(avg_similarity_df_lst, ignore_index=True)
    return avg_similarity_df
