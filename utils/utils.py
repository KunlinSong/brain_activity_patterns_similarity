from typing import Literal

import pandas as pd

from utils.config import (
    FIELDS_CONFIG,
    LABELS_CONFIG,
    ROIImgFieldsConfig,
    WholeBrainImgFieldsConfig,
)


def get_fields_config(
    img_type: Literal["whole brain", "ROI"]
) -> WholeBrainImgFieldsConfig | ROIImgFieldsConfig:
    match img_type:
        case "whole brain":
            fields_config = FIELDS_CONFIG.whole_brain_img
        case "ROI":
            fields_config = FIELDS_CONFIG.roi_img
        case _:
            raise ValueError(f"Invalid img_type: {img_type}")
    return fields_config


def get_process_method(process_method: str | None) -> str:
    process_method = (
        LABELS_CONFIG.process_method.original
        if process_method is None
        else process_method
    )


def get_similarity_dataset(
    idx: int,
    img_df: pd.DataFrame,
    img_type: Literal["whole brain", "ROI"],
) -> pd.DataFrame | None:
    fields_config = get_fields_config(img_type)
    similarity_dataset: pd.DataFrame | None = img_df.at[
        idx, fields_config.similarity
    ]
    return (
        similarity_dataset.copy() if similarity_dataset is not None else None
    )


def filter_similarity_dataset(
    similarity_dataset: pd.DataFrame,
    process_method: str | None,
    similarity_method: str | None,
) -> pd.DataFrame:
    similarity_dataset = similarity_dataset.copy()
    if process_method is not None:
        similarity_dataset = similarity_dataset[
            similarity_dataset[FIELDS_CONFIG.similarity_dataset.process_method]
            == process_method
        ]
    if similarity_method is not None:
        similarity_dataset = similarity_dataset[
            similarity_dataset[
                FIELDS_CONFIG.similarity_dataset.similarity_method
            ]
            == similarity_method
        ]
    return similarity_dataset.copy()
