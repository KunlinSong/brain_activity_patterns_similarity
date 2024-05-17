"""Module for loading data.

We use pandas to build a database to store the brain activity patterns 
in each ROI and the corresponding labels. By using the functions in 
`utils/similarity.py`, the database can be easily used to compute the
similarity between two brain activity patterns and store it as a new 
column in the database.
"""

import re
from collections import defaultdict, namedtuple
from functools import partial
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable

import nibabel
import numpy as np
import pandas as pd
import yaml

from .config import get_config

_CONFIG = get_config()


def _get_axis_min_feat(axis: Any) -> str:
    return f"{axis} min"


def _get_axis_max_feat(axis: Any) -> str:
    return f"{axis} max"


def get_rois_df(path: str) -> pd.DataFrame:
    rois_df_dict = defaultdict(list)
    with open(path, "r") as f:
        roi_config: dict = yaml.safe_load(f)
    len_side = roi_config[_CONFIG.NAMES.ROI_CONFIG.SIDE_LENGTH]
    side_mid_front = len_side // 2
    side_mid_back = len_side - side_mid_front
    get_min = lambda x: x - side_mid_front
    get_max = lambda x: x + side_mid_back
    coords: dict = roi_config[_CONFIG.NAMES.ROI_CONFIG.COORDINATES]
    for structure in coords.keys():
        structure_coords: dict = roi_config[
            _CONFIG.NAMES.ROI_CONFIG.COORDINATES
        ][structure]
        for hemisphere in [
            _CONFIG.NAMES.ROI_CONFIG.LEFT_HEMISPHERE,
            _CONFIG.NAMES.ROI_CONFIG.RIGHT_HEMISPHERE,
        ]:
            region_coord: dict = structure_coords[hemisphere]

            rois_df_dict[_CONFIG.DATASET_FEATURES.REGION].append(
                f"{structure} {hemisphere}"
            )
            rois_df_dict[_CONFIG.DATASET_FEATURES.STRUCTURE].append(structure)
            rois_df_dict[_CONFIG.DATASET_FEATURES.HEMISPHERE].append(
                hemisphere
            )
            for axis in [
                _CONFIG.NAMES.ROI_CONFIG.X,
                _CONFIG.NAMES.ROI_CONFIG.Y,
                _CONFIG.NAMES.ROI_CONFIG.Z,
            ]:
                coord_axis = region_coord[axis]
                rois_df_dict[axis].append(coord_axis)
                rois_df_dict[_get_axis_min_feat(axis)].append(
                    get_min(coord_axis)
                )
                rois_df_dict[_get_axis_max_feat(axis)].append(
                    get_max(coord_axis)
                )
    rois_df = pd.DataFrame(rois_df_dict)
    rois_df = rois_df.sort_values(by=[_CONFIG.DATASET_FEATURES.REGION])
    rois_df = rois_df.reset_index(drop=True)
    return rois_df


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
            path = str(subject.joinpath(filename))
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
        if isinstance(
            img_row[_CONFIG.DATASET_FEATURES.SIMILARITY], pd.DataFrame
        ):
            img_df.at[img_idx, _CONFIG.DATASET_FEATURES.SIMILARITY] = (
                similarity_sub_df
            )
        else:
            similarity_df: pd.DataFrame = pd.concat(
                img_row[_CONFIG.DATASET_FEATURES.SIMILARITY], similarity_sub_df
            )
            similarity_df = similarity_df.sort_values(
                by=[
                    _CONFIG.DATASET_FEATURES.SIMILARITY_METHOD,
                    _CONFIG.DATASET_FEATURES.PROCESS_METHOD,
                    _CONFIG.DATASET_FEATURES.SUBJECT_ID,
                ]
            )
    return img_df


def get_similarity_mat(
    img_df: pd.DataFrame,
    region: str,
    similarity_name: str,
    process_name: str | None = None,
) -> pd.DataFrame:
    process_name = _get_process_name(process_name=process_name)
    img_df = img_df[img_df[_CONFIG.DATASET_FEATURES.REGION] == region].copy()
    similarity_mat = np.empty((len(img_df), len(img_df)))
    for img_idx, img_row in img_df.iterrows():
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
        similarity_mat[img_idx] = similarity_df[
            _CONFIG.DATASET_FEATURES.SIMILARITY
        ].values
    return similarity_mat


SimilarityVals = namedtuple(
    "SimilarityVals", ["specific", "non_specific"], defaults=[[], []]
)


def _compute_similarity_avg(
    similarity_df: pd.DataFrame, n_subjects: int = 0
) -> list[float]:
    average_similarity = []
    if n_subjects <= -len(similarity_df):
        return average_similarity
    elif n_subjects <= 0:
        n_subjects += len(similarity_df)
    else:
        n_subjects = min(n_subjects, len(similarity_df))

    for idxs in combinations(similarity_df.index, n_subjects):
        avg = similarity_df.loc[
            list(idxs), _CONFIG.DATASET_FEATURES.SIMILARITY
        ].mean()
        average_similarity.append(avg)
    return average_similarity


def get_average_similarity(
    img_idx: int,
    img_row: pd.Series,
    similarity_name: str,
    process_name: str | None = None,
    n_subjects: int = 0,
) -> SimilarityVals:
    compute_avg = partial(_compute_similarity_avg, n_subjects=n_subjects)
    similarity_df: pd.DataFrame = img_row[
        _CONFIG.DATASET_FEATURES.SIMILARITY
    ].copy()
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
    similarity_df = similarity_df.drop(img_idx, axis=0)
    similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.DATA_TYPE]
        == _CONFIG.NAMES.DATA_TYPE.REAL
    ]
    specific_similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.IS_SPECIFIC] == 1
    ].copy()
    non_specific_similarity_df = similarity_df[
        similarity_df[_CONFIG.DATASET_FEATURES.IS_SPECIFIC] == 0
    ].copy()
    specific_avg_similarity = compute_avg(
        similarity_df=specific_similarity_df,
    )
    non_specific_avg_similarity = compute_avg(
        similarity_df=non_specific_similarity_df
    )
    return SimilarityVals(
        specific=specific_avg_similarity,
        non_specific=non_specific_avg_similarity,
    )


def get_average_similarity_dataset(
    img_df: pd.DataFrame,
    similarity_and_process_names: list[tuple[str, str | None]],
    n_subjects: int = 0,
) -> pd.DataFrame:
    avg_similarity_dataset = defaultdict(list)
    for (img_idx, img_row), (similarity_name, process_name) in product(
        img_df.iterrows(), similarity_and_process_names
    ):
        specific_vals, non_specific_vals = get_average_similarity(
            img_idx=img_idx,
            img_row=img_row,
            similarity_name=similarity_name,
            process_name=process_name,
            n_subjects=n_subjects,
        )
        for specific_v, non_specific_v in product(
            specific_vals, non_specific_vals
        ):
            avg_similarity_dataset[_CONFIG.DATASET_FEATURES.SUBJECT_ID].append(
                img_idx
            )
            for feat in [
                _CONFIG.DATASET_FEATURES.DATA_TYPE,
                _CONFIG.DATASET_FEATURES.STIMULATION,
                _CONFIG.DATASET_FEATURES.SUBJECT,
                _CONFIG.DATASET_FEATURES.IS_SPECIFIC,
                _CONFIG.DATASET_FEATURES.SIMILARITY,
            ]:
                avg_similarity_dataset[feat].append(img_row[feat])

            avg_similarity_dataset[_CONFIG.DATASET_FEATURES.PROCESS_METHOD] = (
                process_name
            )
            avg_similarity_dataset[
                _CONFIG.DATASET_FEATURES.SIMILARITY_METHOD
            ] = similarity_name
            avg_similarity_dataset[
                _CONFIG.DATASET_FEATURES.SPECIFIC_SIMILARITY
            ] = specific_v
            avg_similarity_dataset[
                _CONFIG.DATASET_FEATURES.NON_SPECIFIC_SIMILARITY
            ] = non_specific_v
    return pd.DataFrame(avg_similarity_dataset)
