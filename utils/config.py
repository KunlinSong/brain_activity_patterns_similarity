import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import product

import pandas as pd
import yaml

_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_ROOT_PATH, "config")


__all__ = [
    "get_basic_config",
    "get_roi_config_df",
]


def _setattr_or_default(obj: object, kwargs: dict, attr: str, cls) -> None:
    if attr in kwargs.keys():
        setattr(obj, attr, cls(**kwargs[attr]))
    else:
        setattr(obj, attr, cls())


@dataclass
class DatasetFeatures:
    DATA_TYPE: str = "data type"
    HEMISPHERE: str = "hemisphere"
    IS_SPECIFIC: str = "is specific"
    PROCESS_METHOD: str = "process method"
    REGION: str = "region"
    SIMILARITY: str = "similarity"
    SIMILARITY_METHOD: str = "similarity method"
    STRUCTURE: str = "structure"
    STIMULATION: str = "stimulation"
    SUBJECT: str = "subject"
    SUBJECT_ID: str = "subject ID"
    SUBSET_TYPE: str = "subset type"


@dataclass
class DataType:
    REAL: str = "real"
    RANDOM: str = "random"


@dataclass
class Process:
    ORIGINAL: str = "original"


@dataclass
class SubsetType:
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"


@dataclass
class ROIConfig:
    SIDE_LENGTH: str = "side_length"
    COORDINATES: str = "coordinates"
    LEFT_HEMISPHERE: str = "L"
    RIGHT_HEMISPHERE: str = "R"
    X: str = "x"
    Y: str = "y"
    Z: str = "z"


@dataclass
class SimilarityType:
    SPECIFIC: str = "specific similarity"
    NON_SPECIFIC: str = "non-specific similarity"


@dataclass
class Formats:
    DATATYPE_STIMULATION_FEAT: 'f"{stimulation_feat} ({data_type_feat})"'
    DATATYPE_STIMULATION_NAME: 'f"{stimulation} ({data_type})"'
    PROCESS_SIMILARITY_FEAT: (
        'f"{similarity_name} ({process_name}, {similarity_type})"'
    )
    GROUP_BY_PROCESS_TITLE: 'f"{similarity_name} ({region})"'
    GROUP_BY_REGION_TITLE: 'f"{similarity_name} ({process_name})"'
    REGION_NAME: 'f"{structure} {hemisphere}"'


@dataclass
class Names:
    DATA_TYPE: DataType
    PROCESS: Process
    SUBSET_TYPE: SubsetType
    ROI_CONFIG: ROIConfig
    SIMILARITY_TYPE: SimilarityType

    def __init__(self, **kwargs) -> None:
        setattr_from_config = partial(
            _setattr_or_default, obj=self, config=kwargs
        )
        setattr_from_config(attr="DATA_TYPE", cls=DataType)
        setattr_from_config(attr="PROCESS", cls=Process)
        setattr_from_config(attr="SUBSET_TYPE", cls=SubsetType)
        setattr_from_config(attr="ROI_CONFIG", cls=ROIConfig)
        setattr_from_config(attr="SIMILARITY_TYPE", cls=SimilarityType)


@dataclass
class BasicConfig:
    REGION_SPECIFIC_STIMULATIONS: dict[str, str]
    NAMES: Names
    DATASET_FEATURES: DatasetFeatures
    FORMATS: Formats

    def __init__(self, **kwargs) -> None:
        setattr_from_config = partial(
            _setattr_or_default, obj=self, config=kwargs
        )
        setattr_from_config(attr="NAMES", cls=Names)
        setattr_from_config(attr="DATASET_FEATURES", cls=DatasetFeatures)
        self.REGION_SPECIFIC_STIMULATIONS = kwargs[
            "REGION_SPECIFIC_STIMULATIONS"
        ]
        setattr_from_config(attr="FORMATS", cls=Formats)


def get_basic_config() -> BasicConfig:
    """Load `basic.yaml` config as `BasicConfig`."""
    path = os.path.join(
        _CONFIG_PATH,
        "basic.yaml",
    )
    with open(path) as f:
        config_dict = yaml.safe_load(f)
        config = BasicConfig(**config_dict)
    return config


def get_roi_config_df() -> pd.DataFrame:
    """Load `roi.yaml` config as `pd.DataFrame`.

    Returns:
        Following columns are included: `region`, `structure`, `hemisphere`,
          `x`, `y`, `z`.  Note that `x`, `y`, `z`, here do not refer to the
          coordinates, but the range of roi on the axis (in `slice` type).
    """
    _BASIC_CONFIG = get_basic_config()
    rois_df_dict = defaultdict(list)
    path = os.path.join(
        _CONFIG_PATH,
        "roi.yaml",
    )
    with open(path, "r") as f:
        roi_config: dict = yaml.safe_load(f)
    len_side = roi_config[_BASIC_CONFIG.NAMES.ROI_CONFIG.SIDE_LENGTH]
    side_mid_front = len_side // 2
    side_mid_back = len_side - side_mid_front
    get_min = lambda x: x - side_mid_front
    get_max = lambda x: x + side_mid_back
    coords: dict = roi_config[_BASIC_CONFIG.NAMES.ROI_CONFIG.COORDINATES]
    for structure, hemisphere in product(
        coords.keys(),
        [
            _BASIC_CONFIG.NAMES.ROI_CONFIG.LEFT_HEMISPHERE,
            _BASIC_CONFIG.NAMES.ROI_CONFIG.RIGHT_HEMISPHERE,
        ],
    ):
        region_coord: dict = coords[structure][hemisphere]
        region_name = eval(_BASIC_CONFIG.FORMATS.REGION_NAME)
        rois_df_dict[_BASIC_CONFIG.DATASET_FEATURES.REGION].append(region_name)
        rois_df_dict[_BASIC_CONFIG.DATASET_FEATURES.STRUCTURE].append(
            structure
        )
        rois_df_dict[_BASIC_CONFIG.DATASET_FEATURES.HEMISPHERE].append(
            hemisphere
        )
        for axis in [
            _BASIC_CONFIG.NAMES.ROI_CONFIG.X,
            _BASIC_CONFIG.NAMES.ROI_CONFIG.Y,
            _BASIC_CONFIG.NAMES.ROI_CONFIG.Z,
        ]:
            coord_axis = region_coord[axis]
            coord_min = get_min(coord_axis)
            coord_max = get_max(coord_axis)
            rois_df_dict[axis].append(slice(coord_min, coord_max))
    rois_df = pd.DataFrame(rois_df_dict)
    rois_df: pd.DataFrame = rois_df.sort_values(
        by=[_BASIC_CONFIG.DATASET_FEATURES.REGION], ignore_index=True
    )
    return rois_df
