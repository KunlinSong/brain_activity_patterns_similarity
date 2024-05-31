from collections import defaultdict
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import get_basic_config


def get_similarity_df(
    img_row: pd.Series,
    img_df: pd.DataFrame,
    process_name: str | None,
    similarity_name: str,
    similarity_func: Callable,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    if process_name is None:
        process_name = _BASIC_CONFIG.NAMES.PROCESS.ORIGINAL
    img_row = img_row.copy()
    processed_img = img_row[process_name]
    img_df = img_df.copy().drop(
        columns=[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY]
    )
    if _BASIC_CONFIG.DATASET_FEATURES.REGION in img_row.keys():
        img_df = img_df[
            img_df[_BASIC_CONFIG.DATASET_FEATURES.REGION]
            == img_row[_BASIC_CONFIG.DATASET_FEATURES.REGION]
        ]
    similarity_df = img_df.copy()
    data_type_feat = _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE
    stimulation_feat = _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
    dtype_stim_feat = eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_FEAT)
    similarity_df = similarity_df[
        [
            data_type_feat,
            stimulation_feat,
            dtype_stim_feat,
            _BASIC_CONFIG.DATASET_FEATURES.SUBJECT,
            _BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC,
        ]
    ]
    similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID] = img_df.index
    similarity_df[_BASIC_CONFIG.DATASET_FEATURES.PROCESS_METHOD] = process_name
    similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD] = (
        similarity_name
    )
    similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY] = img_df[
        process_name
    ].apply(lambda img: similarity_func(processed_img, img))
    return similarity_df


def get_similarity_mat(
    img_df: pd.DataFrame,
    region: str,
    process_name: str | None,
    similarity_name: str,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    if process_name is None:
        process_name = _BASIC_CONFIG.NAMES.PROCESS.ORIGINAL
    img_df = img_df[
        img_df[_BASIC_CONFIG.DATASET_FEATURES.REGION] == region
    ].copyJ()
    img_df = img_df.sort_index(ignore_index=True)
    similarity_mat = np.empty((len(img_df), (len(img_df))))
    for idx_0, (_, img_row) in enumerate(img_df.iterrows()):
        similarity_df: pd.DataFrame = img_row[
            _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY
        ]
        similarity_df = similarity_df[
            (
                similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD]
                == similarity_name
            )
            & (
                similarity_df[_BASIC_CONFIG.DATASET_FEATURES.PROCESS_METHOD]
                == process_name
            )
        ]
        for idx_1 in img_df.index:
            idx_1_series = similarity_df[
                similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID]
                == idx_1
            ].iloc[0]
            value = idx_1_series[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY]
            similarity_mat[idx_0, idx_1] = value
    return pd.DataFrame(
        similarity_mat, columns=img_df.index, index=img_df.index
    )


def get_same_stimulation_similairty_df(
    img_df: pd.DataFrame, similarity_name: str, include_self: bool = False
) -> pd.DataFrame:
    img_df = img_df.copy()
    _BASIC_CONFIG = get_basic_config()
    same_similarity_df_dict = defaultdict(list)
    data_type_feat = _BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE
    stimulation_feat = _BASIC_CONFIG.DATASET_FEATURES.STIMULATION
    dtype_stim_feat = eval(_BASIC_CONFIG.FORMATS.DATATYPE_STIMULATION_FEAT)
    for img_idx, img_row in img_df.iterrows():
        data_type = img_row[data_type_feat]
        stimulation = img_row[stimulation_feat]
        dtype_stim = img_row[dtype_stim_feat]
        similarity_df = img_row[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY]
        similarity_df = similarity_df[
            (similarity_df[dtype_stim_feat] == dtype_stim)
            & (
                similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD]
                == similarity_name
            )
        ]
        if not include_self:
            similarity_df = similarity_df[
                similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID]
                != img_idx
            ]
        for _, similarity_row in similarity_df:
            same_similarity_df_dict[data_type_feat].append(data_type)
            same_similarity_df_dict[stimulation_feat].append(stimulation)
            same_similarity_df_dict[dtype_stim_feat].append(dtype_stim)
            same_similarity_df_dict[
                _BASIC_CONFIG.DATASET_FEATURES.PROCESS_METHOD
            ].append(
                similarity_row[_BASIC_CONFIG.DATASET_FEATURES.PROCESS_METHOD]
            )
            same_similarity_df_dict[
                _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD
            ].append(
                similarity_row[
                    _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD
                ]
            )
            same_similarity_df_dict[
                _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY
            ].append(similarity_row[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY])
    return pd.DataFrame(same_similarity_df_dict)


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
    _BASIC_CONFIG = get_basic_config()
    if process_name is None:
        process_name = _BASIC_CONFIG.NAMES.PROCESS.ORIGINAL
    similarity_df = similarity_df[
        (
            similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY_METHOD]
            == similarity_name
        )
        & (
            similarity_df[_BASIC_CONFIG.DATASET_FEATURES.PROCESS_METHOD]
            == process_name
        )
    ].copy()
    return similarity_df[
        similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID].isin(avg_ids)
    ][_BASIC_CONFIG.DATASET_FEATURES.SIMILARITY].mean()


def _get_process_similarity_avg_series(
    similarity_df: pd.DataFrame,
    process_names: list[str | None],
    similarity_names: list[str],
    similarity_type: str,
    avg_ids: list,
) -> pd.Series:
    _BASIC_CONFIG = get_basic_config()
    avg_similarity_series_dict = defaultdict(float)
    for process_name, similarity_name in zip(process_names, similarity_names):
        avg_similarity_series_dict[
            eval(_BASIC_CONFIG.FORMATS.PROCESS_SIMILARITY_FEAT)
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
    _BASIC_CONFIG = get_basic_config()
    similarity_df: pd.DataFrame = img_row[
        _BASIC_CONFIG.DATASET_FEATURES.SIMILARITY
    ].copy()
    similarity_df = similarity_df[
        similarity_df[_BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE]
        == _BASIC_CONFIG.NAMES.DATA_TYPE.REAL
    ]
    if not include_self:
        similarity_df = similarity_df[
            similarity_df[_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID] != img_idx
        ]
    avg_series_list = []
    if similarity_type == _BASIC_CONFIG.NAMES.SIMILARITY_TYPE.SPECIFIC:
        subject_ids = (
            similarity_df[
                similarity_df[
                    _BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC
                ].astype(bool)
            ][_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID]
            .unique()
            .tolist()
        )
    elif similarity_type == _BASIC_CONFIG.NAMES.SIMILARITY_TYPE.NON_SPECIFIC:
        subject_ids = (
            similarity_df[
                ~similarity_df[
                    _BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC
                ].astype(bool)
            ][_BASIC_CONFIG.DATASET_FEATURES.SUBJECT_ID]
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
    avg_similarity_df[_BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC] = img_row[
        _BASIC_CONFIG.DATASET_FEATURES.IS_SPECIFIC
    ]
    return avg_similarity_df


def _get_row_avg_similarity(
    img_idx: int,
    img_row: pd.Series,
    similarity_names: list[str],
    process_names: list[str | None],
    n_subjects: int = 0,
    include_self: bool = False,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    type_similarity_df_lst = []
    for similarity_type in [
        _BASIC_CONFIG.NAMES.SIMILARITY_TYPE.SPECIFIC,
        _BASIC_CONFIG.NAMES.SIMILARITY_TYPE.NON_SPECIFIC,
    ]:
        type_similarity_df_lst.append(
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
    return pd.concat(type_similarity_df_lst, ignore_index=True)


def get_avg_similarity_df(
    img_df: pd.DataFrame,
    similarity_process_pairs: list[tuple[str, str | None]],
    only_real: bool = True,
    n_subjects: int = 0,
    include_self: bool = False,
) -> pd.DataFrame:
    _BASIC_CONFIG = get_basic_config()
    if only_real:
        img_df = img_df[
            img_df[_BASIC_CONFIG.DATASET_FEATURES.DATA_TYPE]
            == _BASIC_CONFIG.NAMES.DATA_TYPE.REAL
        ]
    avg_similarity_df_lst = []
    similarity_names, process_names = zip(*similarity_process_pairs)
    for img_idx, img_row in img_df.iterrows():
        avg_similarity_df_lst.append(
            _get_row_avg_similarity(
                img_idx=img_idx,
                img_row=img_row,
                similarity_names=list(similarity_names),
                process_names=list[process_names],
                n_subjects=n_subjects,
                include_self=include_self,
            )
        )
    return pd.concat(avg_similarity_df_lst, ignore_index=True)
