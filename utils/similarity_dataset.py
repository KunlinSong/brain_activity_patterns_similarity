from collections import defaultdict
from itertools import combinations, product
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import (
    get_categories_config,
    get_features_config,
    get_formats_config,
)

__all__ = [
    "get_similarity_df",
    "get_similarity_mat",
    "get_same_stimulation_similarity_df",
    "get_avg_similarity_df",
]


def get_similarity_df(
    img_row: pd.Series,
    img_df: pd.DataFrame,
    process_name: str | None,
    similarity_name: str,
    similarity_func: Callable,
) -> pd.DataFrame:
    _FORMATS_CONFIG = get_formats_config()
    _CATEGORIES_CONFIG = get_categories_config()
    _FEATURES_CONFIG = get_features_config()
    if process_name is None:
        process_name = _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
    img_row = img_row.copy()
    img_df = img_df.copy()
    processed_img: np.ndarray = img_row[_FEATURES_CONFIG.IMAGES][
        process_name
    ].copy()
    img_df = img_df.drop(_FEATURES_CONFIG.SIMILARITY, axis=1)
    if _FEATURES_CONFIG.REGION in img_row.keys():
        img_df = img_df[
            img_df[_FEATURES_CONFIG.REGION] == img_row[_FEATURES_CONFIG.REGION]
        ].copy()
    similarity_df = img_df.copy()
    dtype_stim_feat = _FORMATS_CONFIG.format_datatype_stimulation_feature(
        stimulation_feat=_FEATURES_CONFIG.STIMULATION,
        data_type_feat=_FEATURES_CONFIG.DATA_TYPE,
    )
    similarity_df = similarity_df[
        [
            _FEATURES_CONFIG.DATA_TYPE,
            _FEATURES_CONFIG.STIMULATION,
            dtype_stim_feat,
            _FEATURES_CONFIG.SUBJECT,
            _FEATURES_CONFIG.IS_SPECIFIC,
        ]
    ].copy()
    similarity_df.loc[:, _FEATURES_CONFIG.PROCESS_METHOD] = process_name
    similarity_df.loc[:, _FEATURES_CONFIG.SIMILARITY_METHOD] = similarity_name
    for img_idx_to, img_row_to in img_df.iterrows():
        similarity_df.at[img_idx_to, _FEATURES_CONFIG.SUBJECT_ID] = img_idx_to
        processed_img_to = img_row_to[_FEATURES_CONFIG.IMAGES][process_name]
        similarity = similarity_func(processed_img, processed_img_to)
        similarity_df.at[img_idx_to, _FEATURES_CONFIG.SIMILARITY] = similarity
    return similarity_df


def get_similarity_mat(
    img_df: pd.DataFrame,
    region: str,
    process_name: str | None,
    similarity_name: str,
) -> pd.DataFrame:
    _FORMATS_CONFIG = get_formats_config()
    _CATEGORIES_CONFIG = get_categories_config()
    _FEATURES_CONFIG = get_features_config()
    img_df = img_df.copy()
    if process_name is None:
        process_name = _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
    if _FEATURES_CONFIG.REGION in img_df.columns:
        img_df = img_df[img_df[_FEATURES_CONFIG.REGION] == region].copy()
    similarity_mat = np.empty((len(img_df), (len(img_df))))
    label_lst = []
    for mat_idx_0, (_, img_row) in enumerate(img_df.iterrows()):
        row_similarity_df: pd.DataFrame = img_row[_FEATURES_CONFIG.SIMILARITY]
        row_similarity_df = row_similarity_df[
            (
                row_similarity_df[_FEATURES_CONFIG.SIMILARITY_METHOD]
                == similarity_name
            )
            & (
                row_similarity_df[_FEATURES_CONFIG.PROCESS_METHOD]
                == process_name
            )
        ].copy()
        for mat_idx_1, img_idx in enumerate(img_df.index):
            id_series = row_similarity_df[
                row_similarity_df[_FEATURES_CONFIG.SUBJECT_ID] == img_idx
            ].iloc[0]
            value = id_series[_FEATURES_CONFIG.SIMILARITY]
            similarity_mat[mat_idx_0, mat_idx_1] = value
        label_lst.append(
            _FORMATS_CONFIG.format_datatype_stimulation_subject_label(
                data_type=img_row[_FEATURES_CONFIG.DATA_TYPE],
                stimulation=img_row[_FEATURES_CONFIG.STIMULATION],
                subject=img_row[_FEATURES_CONFIG.SUBJECT],
            )
        )
    return pd.DataFrame(similarity_mat, columns=label_lst, index=label_lst)


def get_same_stimulation_similarity_df(
    img_df: pd.DataFrame, similarity_name: str, include_self: bool = False
) -> pd.DataFrame:
    img_df = img_df.copy()
    _FEATURES_CONFIG = get_features_config()
    _FORMATS_CONFIG = get_formats_config()
    same_stim_similarity_series_lst = []
    for img_idx, img_row in img_df.iterrows():
        data_type = img_row[_FEATURES_CONFIG.DATA_TYPE]
        stimulation = img_row[_FEATURES_CONFIG.STIMULATION]
        dtype_stim = img_row[_FORMATS_CONFIG.datatype_stimulation_feature]
        structure = img_row[_FEATURES_CONFIG.STRUCTURE]
        hemisphere = img_row[_FEATURES_CONFIG.HEMISPHERE]
        region = img_row[_FEATURES_CONFIG.REGION]
        similarity_df = img_row[_FEATURES_CONFIG.SIMILARITY]
        similarity_df: pd.DataFrame = similarity_df[
            (
                similarity_df[_FORMATS_CONFIG.datatype_stimulation_feature]
                == dtype_stim
            )
            & (
                similarity_df[_FEATURES_CONFIG.SIMILARITY_METHOD]
                == similarity_name
            )
        ].copy()
        if not include_self:
            similarity_df = similarity_df[
                similarity_df[_FEATURES_CONFIG.SUBJECT_ID] != img_idx
            ].copy()
        for _, similarity_row in similarity_df.iterrows():
            process_method = similarity_row[_FEATURES_CONFIG.PROCESS_METHOD]
            similarity_method = similarity_row[
                _FEATURES_CONFIG.SIMILARITY_METHOD
            ]
            similarity = similarity_row[_FEATURES_CONFIG.SIMILARITY]

            same_stim_similarity_dict = {
                _FEATURES_CONFIG.DATA_TYPE: data_type,
                _FEATURES_CONFIG.STIMULATION: stimulation,
                _FORMATS_CONFIG.datatype_stimulation_feature: dtype_stim,
                _FEATURES_CONFIG.PROCESS_METHOD: process_method,
                _FEATURES_CONFIG.SIMILARITY_METHOD: similarity_method,
                _FEATURES_CONFIG.STRUCTURE: structure,
                _FEATURES_CONFIG.HEMISPHERE: hemisphere,
                _FEATURES_CONFIG.REGION: region,
                _FEATURES_CONFIG.SIMILARITY: similarity,
            }
            same_stim_similarity_series = pd.Series(same_stim_similarity_dict)
            same_stim_similarity_series_lst.append(same_stim_similarity_series)
    return pd.DataFrame(same_stim_similarity_series_lst)


def _get_subject_ids_combinations(
    subject_ids: list, n_subjects: int = 0
) -> Iterable:
    if not subject_ids:
        raise ValueError("No subject IDs in `subject_ids`.")
    if n_subjects <= -len(subject_ids):
        n_subjects = 1
    elif n_subjects <= 0:
        n_subjects += len(subject_ids)
    else:
        n_subjects = min(n_subjects, len(subject_ids))

    yield from combinations(subject_ids, n_subjects)


def _compute_subject_ids_avg_similarity(
    similarity_df: pd.DataFrame,
    similarity_name: str,
    process_name: str | None,
    avg_ids: list,
) -> float:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    if process_name is None:
        process_name = _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
    similarity_df = similarity_df[
        (similarity_df[_FEATURES_CONFIG.SIMILARITY_METHOD] == similarity_name)
        & (similarity_df[_FEATURES_CONFIG.PROCESS_METHOD] == process_name)
    ].copy()
    return similarity_df[
        similarity_df[_FEATURES_CONFIG.SUBJECT_ID].isin(avg_ids)
    ][_FEATURES_CONFIG.SIMILARITY].mean()


def _get_process_similarity_avg_series(
    similarity_df: pd.DataFrame,
    process_names: list[str | None],
    similarity_names: list[str],
    similarity_type: str,
    avg_ids: list,
) -> pd.Series:
    _CATEGORIES_CONFIG = get_categories_config()
    _FORMATS_CONFIG = get_formats_config()
    avg_similarity_series_dict = {}
    for process_name, similarity_name in zip(process_names, similarity_names):
        if process_name is None:
            process_name = _CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL
        avg_similarity_series_dict[
            _FORMATS_CONFIG.format_process_similarity_feature(
                similarity_name=similarity_name,
                process_name=process_name,
                similarity_type=similarity_type,
            )
        ] = _compute_subject_ids_avg_similarity(
            similarity_df=similarity_df,
            similarity_name=similarity_name,
            process_name=process_name,
            avg_ids=avg_ids,
        )
    return pd.Series(avg_similarity_series_dict)


def _get_row_similarity_type_similarity_df(
    img_idx: int,
    img_row: pd.Series,
    similarity_names: list[str],
    process_names: list[str | None],
    similarity_type: str,
    n_subjects: int = 0,
    include_self: bool = False,
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    similarity_df: pd.DataFrame = img_row[_FEATURES_CONFIG.SIMILARITY].copy()
    similarity_df = similarity_df[
        similarity_df[_FEATURES_CONFIG.DATA_TYPE]
        == _CATEGORIES_CONFIG.DATA_TYPE.REAL
    ]
    if not include_self:
        similarity_df = similarity_df[
            similarity_df[_FEATURES_CONFIG.SUBJECT_ID] != img_idx
        ]
    avg_series_list = []
    if similarity_type == _CATEGORIES_CONFIG.SIMILARITY_TYPE.SPECIFIC:
        subject_ids = (
            similarity_df[
                similarity_df[_FEATURES_CONFIG.IS_SPECIFIC].astype(bool)
            ][_FEATURES_CONFIG.SUBJECT_ID]
            .unique()
            .tolist()
        )
    elif similarity_type == _CATEGORIES_CONFIG.SIMILARITY_TYPE.NON_SPECIFIC:
        subject_ids = (
            similarity_df[
                ~similarity_df[_FEATURES_CONFIG.IS_SPECIFIC].astype(bool)
            ][_FEATURES_CONFIG.SUBJECT_ID]
            .unique()
            .tolist()
        )
    else:
        raise ValueError(f"Invalid similarity type: {similarity_type}")
    for avg_ids in _get_subject_ids_combinations(
        subject_ids=subject_ids, n_subjects=n_subjects
    ):
        avg_series_list.append(
            _get_process_similarity_avg_series(
                similarity_df=similarity_df,
                process_names=process_names,
                similarity_names=similarity_names,
                similarity_type=similarity_type,
                avg_ids=avg_ids,
            )
        )
    avg_similarity_df = pd.DataFrame(avg_series_list)
    return avg_similarity_df


def _get_row_avg_similarity(
    img_idx: int,
    img_row: pd.Series,
    similarity_names: list[str],
    process_names: list[str | None],
    n_subjects: int = 0,
    include_self: bool = False,
) -> pd.DataFrame:
    _CATEGORIES_CONFIG = get_categories_config()
    _FEATURES_CONFIG = get_features_config()
    similarity_type_lst = [
        _CATEGORIES_CONFIG.SIMILARITY_TYPE.SPECIFIC,
        _CATEGORIES_CONFIG.SIMILARITY_TYPE.NON_SPECIFIC,
    ]
    type_similarity_df_dict = {}
    for similarity_type in similarity_type_lst:
        type_similarity_df_dict[similarity_type] = (
            _get_row_similarity_type_similarity_df(
                img_idx=img_idx,
                img_row=img_row,
                similarity_names=similarity_names,
                process_names=process_names,
                similarity_type=similarity_type,
                n_subjects=n_subjects,
                include_self=include_self,
            )
        )
    idx_combinations = list(
        zip(
            *product(
                *[
                    type_similarity_df_dict[similarity_type].index
                    for similarity_type in similarity_type_lst
                ]
            )
        )
    )
    concat_lst = [
        type_similarity_df_dict[similarity_type]
        .loc[idx_combinations[similarity_type_lst.index(similarity_type)], :]
        .reset_index(drop=True)
        for similarity_type in similarity_type_lst
    ]
    avg_similarity_df = pd.concat(concat_lst, axis=1)
    avg_similarity_df.loc[:, _FEATURES_CONFIG.IS_SPECIFIC] = img_row[
        _FEATURES_CONFIG.IS_SPECIFIC
    ]
    return avg_similarity_df


def get_avg_similarity_df(
    img_df: pd.DataFrame,
    similarity_process_pairs: list[tuple[str, str | None]],
    only_real: bool = True,
    n_subjects: int = 0,
    include_self: bool = False,
) -> pd.DataFrame:
    _FEATURES_CONFIG = get_features_config()
    _CATEGORIES_CONFIG = get_categories_config()
    if only_real:
        img_df = img_df[
            img_df[_FEATURES_CONFIG.DATA_TYPE]
            == _CATEGORIES_CONFIG.DATA_TYPE.REAL
        ]
    avg_similarity_df_lst = []
    similarity_names, process_names = zip(*similarity_process_pairs)
    for img_idx, img_row in img_df.iterrows():
        avg_similarity_df_lst.append(
            _get_row_avg_similarity(
                img_idx=img_idx,
                img_row=img_row,
                similarity_names=list(similarity_names),
                process_names=list(process_names),
                n_subjects=n_subjects,
                include_self=include_self,
            )
        )
    return pd.concat(avg_similarity_df_lst, ignore_index=True)
