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
from itertools import combinations
from pathlib import Path
from typing import Callable

import nibabel
import numpy as np
import pandas as pd
import yaml

_SIDE_LENGTH_CONF = "side_length"
_COORDINATES_CONF = "coordinates"
_X_CONF = "x"
_Y_CONF = "y"
_Z_CONF = "z"
_SPECIFIC_DICT = {
    "STG": "auditory stimulation",
    "FFA": "visual stimulation",
}

_AXIS_MIN_FEAT = lambda axis: f"{axis} min"
_AXIS_MAX_FEAT = lambda axis: f"{axis} max"

_REGION_FEAT = "region"
_STRUCTURE_FEAT = "structure"
_HEMISPHERE_FEAT = "hemisphere"
_DATA_TYPE_FEAT = "data type"
_STIMULATION_FEAT = "stimulation"
_SUBJECT_FEAT = "subject"
_ORIGINAL_IMG_FEAT = "original"
_SIMILARITY_FEAT = "similarity"

_DATA_TYPE_REAL = "real"
_DATA_TYPE_RANDOM = "random"


def load_rois_config(path: str) -> pd.DataFrame:
    rois_config = defaultdict(list)
    with open(path, "r") as f:
        config: dict = yaml.safe_load(f)
    len_side = config[_SIDE_LENGTH_CONF]
    side_mid_front = len_side // 2
    side_mid_back = len_side - side_mid_front
    get_min = lambda x: x - side_mid_front
    get_max = lambda x: x + side_mid_back
    coords: dict = config[_COORDINATES_CONF]
    for structure in coords.keys():
        structure_coords: dict = config[_COORDINATES_CONF][structure]
        for hemisphere in structure_coords.keys():
            region_coord: dict = structure_coords[hemisphere]

            rois_config[_REGION_FEAT].append(f"{structure} {hemisphere}")
            rois_config[_STRUCTURE_FEAT].append(structure)
            rois_config[_HEMISPHERE_FEAT].append(hemisphere)
            for axis in [_X_CONF, _Y_CONF, _Z_CONF]:
                coord_axis = region_coord[axis]
                rois_config[axis].append(coord_axis)
                rois_config[_AXIS_MIN_FEAT(axis)].append(get_min(coord_axis))
                rois_config[_AXIS_MAX_FEAT(axis)].append(get_max(coord_axis))
    rois_df = pd.DataFrame(rois_config)
    rois_df = rois_df.sort_values(by=[_REGION_FEAT])
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

            brain_images[_DATA_TYPE_FEAT].append(_DATA_TYPE_REAL)
            brain_images[_STIMULATION_FEAT].append(stimulation.name)
            brain_images[_SUBJECT_FEAT].append(subject.name)
            brain_images[_ORIGINAL_IMG_FEAT].append(img)

    brain_img_df = pd.DataFrame(brain_images)
    brain_img_df = brain_img_df.sort_values(
        by=[_STIMULATION_FEAT, _SUBJECT_FEAT]
    )
    brain_img_df = brain_img_df.reset_index(drop=True)
    return brain_img_df


def get_roi_img_df(
    brain_img_df: pd.DataFrame, rois_df: pd.DataFrame
) -> pd.DataFrame:
    roi_dataset = defaultdict(list)
    for _, roi in rois_df.iterrows():
        x_min, x_max = (
            roi[_AXIS_MIN_FEAT(_X_CONF)],
            roi[_AXIS_MAX_FEAT(_X_CONF)],
        )
        y_min, y_max = (
            roi[_AXIS_MIN_FEAT(_Y_CONF)],
            roi[_AXIS_MAX_FEAT(_Y_CONF)],
        )
        z_min, z_max = (
            roi[_AXIS_MIN_FEAT(_Z_CONF)],
            roi[_AXIS_MAX_FEAT(_Z_CONF)],
        )
        for _, brain_img in brain_img_df.iterrows():
            img = brain_img[_ORIGINAL_IMG_FEAT]
            roi_img = img[x_min:x_max, y_min:y_max, z_min:z_max]
            roi_dataset[_REGION_FEAT].append(roi[_REGION_FEAT])
            roi_dataset[_STRUCTURE_FEAT].append(roi[_STRUCTURE_FEAT])
            roi_dataset[_HEMISPHERE_FEAT].append(roi[_HEMISPHERE_FEAT])
            roi_dataset[_DATA_TYPE_FEAT].append(_DATA_TYPE_REAL)
            roi_dataset[_STIMULATION_FEAT].append(brain_img[_STIMULATION_FEAT])
            roi_dataset[_SUBJECT_FEAT].append(brain_img[_SUBJECT_FEAT])
            roi_dataset[_ORIGINAL_IMG_FEAT].append(roi_img)
    roi_dataset_df = pd.DataFrame(roi_dataset)
    roi_dataset_df = roi_dataset_df.sort_values(
        by=[_REGION_FEAT, _STIMULATION_FEAT, _SUBJECT_FEAT]
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
    random_img_df[_ORIGINAL_IMG_FEAT] = random_img_df[
        _ORIGINAL_IMG_FEAT
    ].apply(lambda img: _randomize_img(img, rng))
    random_img_df[_DATA_TYPE_FEAT] = _DATA_TYPE_RANDOM
    return random_img_df


def concat_img_dfs(img_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    img_df = pd.concat(img_dfs, ignore_index=True)
    img_df = img_df.sort_values(
        by=[_REGION_FEAT, _DATA_TYPE_FEAT, _STIMULATION_FEAT, _SUBJECT_FEAT]
    )
    img_df = img_df.reset_index(drop=True)
    return img_df


def process_original(
    img_df: pd.DataFrame, process_name: str, process_func: Callable
) -> pd.DataFrame:
    img_df = img_df.copy()
    img_df[process_name] = img_df[_ORIGINAL_IMG_FEAT].apply(
        lambda img: process_func(img)
    )
    return img_df


def _get_process_feat(process_name: str | None = None) -> str:
    return _ORIGINAL_IMG_FEAT if process_name is None else process_name


def _get_similarity_feat(
    similarity_name: str, process_name: str | None
) -> str:
    return f"{_get_process_feat(process_name)} {similarity_name}"


def _get_similarity_to(
    img_row: pd.Series,
    img_df: pd.DataFrame,
    similarity_func: Callable,
    process_name: str | None = None,
):
    process_feat = _get_process_feat(process_name)
    img_df = img_df.copy()
    img_df = img_df[img_df[_REGION_FEAT] == img_row[_REGION_FEAT]]

    img_df[_SIMILARITY_FEAT] = img_df[process_feat].apply(
        lambda process_img: similarity_func(img_row[process_feat], process_img)
    )
    similarity_df = img_df[
        [
            _DATA_TYPE_FEAT,
            _STIMULATION_FEAT,
            _SUBJECT_FEAT,
            _SIMILARITY_FEAT,
        ]
    ].copy()
    return similarity_df


def compute_similarity(
    img_df: pd.DataFrame,
    similarity_name: str,
    similarity_func: Callable,
    process_name: str | None = None,
) -> pd.DataFrame:
    get_similarity_value = partial(
        _get_similarity_to,
        img_df=img_df,
        similarity_func=similarity_func,
        process_name=process_name,
    )
    similarity_feat = _get_similarity_feat(
        similarity_name=similarity_name, process_name=process_name
    )
    img_df = img_df.copy()
    img_df[similarity_feat] = None

    for img_idx, img_row in img_df.iterrows():
        img_df.at[img_idx, similarity_feat] = get_similarity_value(
            img_row=img_row
        )
    return img_df


def get_similarity_mat(
    img_df: pd.DataFrame,
    region: str,
    similarity_name: str,
    process_name: str | None = None,
) -> pd.DataFrame:
    img_df = img_df[img_df[_REGION_FEAT] == region].copy()
    similarity_feat = _get_similarity_feat(
        similarity_name=similarity_name, process_name=process_name
    )
    similarity_mat = np.empty((len(img_df), len(img_df)))
    for img_idx, img_row in img_df.iterrows():
        similarity_df: pd.DataFrame = img_row[similarity_feat]
        similarity_mat[img_idx] = similarity_df[_SIMILARITY_FEAT].values
    return similarity_mat


def get_specific_stimulation(structure: str) -> str:
    return _SPECIFIC_DICT[structure]


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
        avg = similarity_df.loc[list(idxs), _SIMILARITY_FEAT].mean()
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
    similarity_feat = _get_similarity_feat(
        similarity_name=similarity_name,
        process_name=process_name,
    )
    specific_stimulation = get_specific_stimulation(
        structure=img_row[_STRUCTURE_FEAT]
    )
    similarity_df: pd.DataFrame = img_row[similarity_feat]
    similarity_df = similarity_df.drop(img_idx, axis=0)
    specific_similarity_df = similarity_df[
        similarity_df[_STIMULATION_FEAT] == specific_stimulation
    ].copy()
    non_specific_similarity_df = similarity_df[
        similarity_df[_STIMULATION_FEAT] != specific_stimulation
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
