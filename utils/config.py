# %%
import os
from dataclasses import dataclass, fields
from functools import partial

import yaml


def _setattr_from_config(obj: object, config: dict, attr: str, cls) -> None:
    if attr in config.keys():
        setattr(obj, attr, cls(**config[attr]))
    else:
        setattr(obj, attr, cls())


@dataclass
class DatasetFeatures:
    REGION: str = "region"
    STRUCTURE: str = "structure"
    HEMISPHERE: str = "hemisphere"
    DATA_TYPE: str = "data type"
    STIMULATION: str = "stimulation"
    IS_SPECIFIC: str = "is specific"
    SUBJECT: str = "subject"
    SIMILARITY: str = "similarity"
    PROCESS_METHOD: str = "process method"
    SIMILARITY_METHOD: str = "similarity method"
    SUBJECT_ID: str = "subject ID"
    SPECIFIC_SIMILARITY: str = "specific similarity"
    NON_SPECIFIC_SIMILARITY: str = "non specific similarity"
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
class Names:
    DATA_TYPE: DataType = DataType()
    PROCESS: Process = Process()
    SUBSET_TYPE: SubsetType = SubsetType()
    ROI_CONFIG: ROIConfig = ROIConfig()

    def __init__(self, **kwargs) -> None:
        if config_dict is not None:
            setattr_from_config = partial(
                _setattr_from_config, obj=self, config=config_dict
            )
            setattr_from_config(attr="DATA_TYPE", cls=DataType)
            setattr_from_config(attr="PROCESS", cls=Process)
            setattr_from_config(attr="SUBSET_TYPE", cls=SubsetType)
            setattr_from_config(attr="ROI_CONFIG", cls=ROIConfig)


@dataclass
class Config:
    REGION_SPECIFIC_STIMULATIONS: dict[str, str]
    NAMES: Names
    DATASET_FEATURES: DatasetFeatures

    def __init__(self, **kwargs) -> None:
        setattr_from_config = partial(
            _setattr_from_config, obj=self, config=kwargs
        )
        setattr_from_config(attr="NAMES", cls=Names)
        setattr_from_config(attr="DATASET_FEATURES", cls=DatasetFeatures)
        self.REGION_SPECIFIC_STIMULATIONS = config_dict[
            "REGION_SPECIFIC_STIMULATIONS"
        ]


def get_config() -> Config:
    path = os.path.join("..", "config", "basic.yaml")
    with open(path) as f:
        config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
    return config
