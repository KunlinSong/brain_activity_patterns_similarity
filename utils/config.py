from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

_root_dirname = Path(__file__).parents[1]
_config_dirname = _root_dirname.joinpath("config")
del _root_dirname

__all__ = [
    "get_categories_config",
    "get_features_config",
    "get_formats_config",
    "get_rois_config",
]


@dataclass
class _DataTypeCategories:
    RANDOM: str
    REAL: str


@dataclass
class _ProcessMethodCategories:
    ORIGINAL: str


@dataclass
class _SimilarityTypeCategories:
    SPECIFIC: str
    NON_SPECIFIC: str


@dataclass
class _SubsetTypeCategories:
    TRAIN: str
    VALIDATION: str
    TEST: str


@dataclass
class CategoriesConfig:
    DATA_TYPE: _DataTypeCategories
    PROCESS_METHOD: _ProcessMethodCategories
    SIMILARITY_TYPE: _SimilarityTypeCategories
    SUBSET_TYPE: _SubsetTypeCategories


@dataclass
class FeaturesConfig:
    DATA_TYPE: str
    HEMISPHERE: str
    IS_SPECIFIC: str
    PROCESS_METHOD: str
    REGION: str
    SIMILARITY: str
    SIMILARITY_METHOD: str
    STRUCTURE: str
    STIMULATION: str
    SUBJECT: str
    SUBJECT_ID: str
    SUBSET_TYPE: str
    IMAGES: str


@dataclass
class FormatsConfig:
    _DATATYPE_STIMULATION_FEATURE: str
    _DATATYPE_STIMULATION_CATEGORY: str
    _GROUP_BY_PROCESS_TITLE: str
    _GROUP_BY_REGION_TITLE: str
    _PROCESS_SIMILARITY_FEATURE: str
    _REGION_CATEGORY: str
    _DATATYPE_STIMULATION_SUBJECT_LABEL: str

    def format_datatype_stimulation_feature(
        self, stimulation_feat: str, data_type_feat: str
    ) -> str:
        return eval(self._DATATYPE_STIMULATION_FEATURE)

    @property
    def datatype_stimulation_feature(self) -> str:
        _FEATURES_CONFIG = get_features_config()
        return self.format_datatype_stimulation_feature(
            stimulation_feat=_FEATURES_CONFIG.STIMULATION,
            data_type_feat=_FEATURES_CONFIG.DATA_TYPE,
        )

    def format_datatype_stimulation_category(
        self, stimulation: str, data_type: str
    ) -> str:
        return eval(self._DATATYPE_STIMULATION_CATEGORY)

    def format_group_by_process_title(
        self, similarity_name: str, region: str
    ) -> str:
        return eval(self._GROUP_BY_PROCESS_TITLE)

    def format_group_by_region_title(
        self, similarity_name: str, process_name: str
    ) -> str:
        return eval(self._GROUP_BY_REGION_TITLE)

    def format_process_similarity_feature(
        self, similarity_name: str, process_name: str, similarity_type: str
    ) -> str:
        return eval(self._PROCESS_SIMILARITY_FEATURE)

    def format_region_category(self, structure: str, hemisphere: str) -> str:
        return eval(self._REGION_CATEGORY)

    def format_datatype_stimulation_subject_label(
        self, data_type: str, stimulation: str, subject: str
    ) -> str:
        return eval(self._DATATYPE_STIMULATION_SUBJECT_LABEL)


def _get_config_dict(filename: str) -> dict:
    path = _config_dirname.joinpath(filename)
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_categories_config() -> CategoriesConfig:
    config_dict = _get_config_dict("categories.yaml")
    data_type_categories = _DataTypeCategories(**config_dict["DATA_TYPE"])
    process_method_categories = _ProcessMethodCategories(
        **config_dict["PROCESS_METHOD"]
    )
    similarity_type_categories = _SimilarityTypeCategories(
        **config_dict["SIMILARITY_TYPE"]
    )
    subset_type_categories = _SubsetTypeCategories(
        **config_dict["SUBSET_TYPE"]
    )
    return CategoriesConfig(
        DATA_TYPE=data_type_categories,
        PROCESS_METHOD=process_method_categories,
        SIMILARITY_TYPE=similarity_type_categories,
        SUBSET_TYPE=subset_type_categories,
    )


def get_features_config() -> FeaturesConfig:
    config_dict = _get_config_dict(filename="features.yaml")
    return FeaturesConfig(
        DATA_TYPE=config_dict["DATA_TYPE"],
        HEMISPHERE=config_dict["HEMISPHERE"],
        IS_SPECIFIC=config_dict["IS_SPECIFIC"],
        PROCESS_METHOD=config_dict["PROCESS_METHOD"],
        REGION=config_dict["REGION"],
        SIMILARITY=config_dict["SIMILARITY"],
        SIMILARITY_METHOD=config_dict["SIMILARITY_METHOD"],
        STRUCTURE=config_dict["STRUCTURE"],
        STIMULATION=config_dict["STIMULATION"],
        SUBJECT=config_dict["SUBJECT"],
        SUBJECT_ID=config_dict["SUBJECT_ID"],
        SUBSET_TYPE=config_dict["SUBSET_TYPE"],
        IMAGES=config_dict["IMAGES"],
    )


def get_formats_config() -> FormatsConfig:
    config_dict = _get_config_dict(filename="formats.yaml")
    return FormatsConfig(
        _DATATYPE_STIMULATION_FEATURE=config_dict[
            "DATATYPE_STIMULATION_FEATURE"
        ],
        _DATATYPE_STIMULATION_CATEGORY=config_dict[
            "DATATYPE_STIMULATION_CATEGORY"
        ],
        _GROUP_BY_PROCESS_TITLE=config_dict["GROUP_BY_PROCESS_TITLE"],
        _GROUP_BY_REGION_TITLE=config_dict["GROUP_BY_REGION_TITLE"],
        _PROCESS_SIMILARITY_FEATURE=config_dict["PROCESS_SIMILARITY_FEATURE"],
        _REGION_CATEGORY=config_dict["REGION_CATEGORY"],
        _DATATYPE_STIMULATION_SUBJECT_LABEL=config_dict[
            "DATATYPE_STIMULATION_SUBJECT_LABEL"
        ],
    )


class ROIsConfig:
    STRUCTURES = ["STG", "FFA"]
    HEMISPHERES = ["L", "R"]
    AXIS = ["x", "y", "z"]
    SPECIFIC_STIM_FEAT = "specific_stimulation"

    def __init__(self) -> None:
        self._FEATURES_CONFIG = get_features_config()
        self._FORMATS_CONFIG = get_formats_config()
        self._CATEGORIES_CONFIG = get_categories_config()

        self.rois_df = self._get_rois_df()
        self.regions = self.rois_df[self._FEATURES_CONFIG.REGION].unique()

    def get_original_roi_img_series(
        self,
        original_brain_img: np.ndarray,
        region: str,
        stimulation: str,
        data_type: str,
    ) -> pd.Series:
        roi_series = self.rois_df.loc[region]
        dtype_stim_feat = (
            self._FORMATS_CONFIG.format_datatype_stimulation_feature(
                data_type_feat=self._FEATURES_CONFIG.DATA_TYPE,
                stimulation_feat=self._FEATURES_CONFIG.STIMULATION,
            )
        )
        dtype_stim_category = (
            self._FORMATS_CONFIG.format_datatype_stimulation_category(
                data_type=data_type,
                stimulation=stimulation,
            )
        )
        original_roi_img = original_brain_img[
            roi_series[self.AXIS[0]],
            roi_series[self.AXIS[1]],
            roi_series[self.AXIS[2]],
        ]
        structure = roi_series[self._FEATURES_CONFIG.STRUCTURE]
        hemisphere = roi_series[self._FEATURES_CONFIG.HEMISPHERE]
        is_specific = int(
            (roi_series[self.SPECIFIC_STIM_FEAT] == stimulation)
            & (data_type == self._CATEGORIES_CONFIG.DATA_TYPE.REAL)
        )
        imgs = pd.Series(
            {self._CATEGORIES_CONFIG.PROCESS_METHOD.ORIGINAL: original_roi_img}
        )
        roi_img_series_dict = {
            self._FEATURES_CONFIG.REGION: region,
            self._FEATURES_CONFIG.STRUCTURE: structure,
            self._FEATURES_CONFIG.HEMISPHERE: hemisphere,
            self._FEATURES_CONFIG.IS_SPECIFIC: is_specific,
            self._FEATURES_CONFIG.DATA_TYPE: data_type,
            self._FEATURES_CONFIG.STIMULATION: stimulation,
            dtype_stim_feat: dtype_stim_category,
            self._FEATURES_CONFIG.IMAGES: imgs,
        }
        return pd.Series(roi_img_series_dict)

    def _get_rois_df(self) -> pd.DataFrame:
        rois_config_dict = _get_config_dict("rois.yaml")
        len_side = rois_config_dict["side_length"]
        coordinates_config_dict: dict = rois_config_dict["coordinates"]
        roi_series_lst = []
        for structure in self.STRUCTURES:
            structure_config_dict: dict = coordinates_config_dict[structure]
            specific_stimulation = structure_config_dict[
                self.SPECIFIC_STIM_FEAT
            ]
            for hemisphere in self.HEMISPHERES:
                center_coords_dict: dict = structure_config_dict[hemisphere]
                region = self._FORMATS_CONFIG.format_region_category(
                    structure=structure,
                    hemisphere=hemisphere,
                )
                roi_dict = {
                    self._FEATURES_CONFIG.REGION: region,
                    self._FEATURES_CONFIG.STRUCTURE: structure,
                    self._FEATURES_CONFIG.HEMISPHERE: hemisphere,
                    self.SPECIFIC_STIM_FEAT: specific_stimulation,
                }

                for axis in self.AXIS:
                    axis_center: int = center_coords_dict[axis]
                    axis_start: int = self._get_end_pos(
                        center=axis_center, len_side=len_side, end_point="min"
                    )
                    axis_end: int = self._get_end_pos(
                        center=axis_center, len_side=len_side, end_point="max"
                    )
                    roi_dict[axis] = slice(axis_start, axis_end)
                roi_series_lst.append(pd.Series(roi_dict))
        rois_df = pd.DataFrame(roi_series_lst)
        rois_df = rois_df.sort_values(
            by=self._FEATURES_CONFIG.REGION, ignore_index=True
        )
        rois_df = rois_df.set_index(self._FEATURES_CONFIG.REGION, drop=False)
        return rois_df

    @staticmethod
    def _get_end_pos(
        center: int, len_side: int, end_point: Literal["min", "max"]
    ) -> int:
        match end_point:
            case "min":
                len_end = len_side // 2
                end_pos = center - len_end
                return end_pos if end_pos > 0 else 0
            case "max":
                len_end = len_side - len_side // 2
                end_pos = center + len_end
                return end_pos
            case _:
                raise ValueError(f"Invalid end_point: {end_point}.")


def get_rois_config() -> ROIsConfig:
    return ROIsConfig()
