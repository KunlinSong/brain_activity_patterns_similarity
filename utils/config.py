"""The module for loading the config files."""

from pathlib import Path
from typing import Literal, Optional, Self

import yaml
from pydantic import BaseModel

from .const import CONFIG_ROOT

__all__ = [
    "FIELDS_CONFIG",
    "FORMATS_CONFIG",
    "LABELS_CONFIG",
    "ROIS_CONFIG",
]


class YAMLConfig(BaseModel):
    @classmethod
    def from_yaml_file(cls, path: Path) -> Self:
        """Load the YAML config file and return a YAMLConfig object.

        If the file does not exist, return an empty YAMLConfig object.

        Args:
            path: the path to the YAML config file.

        Returns:
            The YAMLConfig object.
        """
        if not path.exists():
            config_dict = {}
        else:
            with open(path) as f:
                config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def load(cls):
        raise NotImplementedError()


class DataTypeLabelsConfig(BaseModel):
    shuffled: str = "shuffled"
    real: str = "real"


class ProcessMethodLabelsConfig(BaseModel):
    original: str = "original"


class SubsetTypeLabelsConfig(BaseModel):
    train: str = "train"
    validation: str = "validation"
    test: str = "test"


class SpecificityTypeLabelsConfig(BaseModel):
    specific: str = "specific"
    non_specific: str = "non_specific"


class LabelsConfig(YAMLConfig):
    data_type: DataTypeLabelsConfig = DataTypeLabelsConfig()
    process_method: ProcessMethodLabelsConfig = ProcessMethodLabelsConfig()
    subset_type: SubsetTypeLabelsConfig = SubsetTypeLabelsConfig()
    specificity_type: SpecificityTypeLabelsConfig = (
        SpecificityTypeLabelsConfig()
    )

    @classmethod
    def load(cls) -> Self:
        return cls.from_yaml_file(CONFIG_ROOT / "labels.yaml")


class _ImgInfoFields(BaseModel):
    data_type: str = "data type"
    stimulation: str = "stimulation"
    subject: str = "subject"


class _SpecificityFields(BaseModel):
    specificity_type: str = "specificity type"


class _SimilarityFields(BaseModel):
    similarity: str = "similarity"


class WholeBrainImgFieldsConfig(_ImgInfoFields, BaseModel, _SimilarityFields):
    images: str = "images"
    as_standard_pattern: str = "as standard pattern"


class ROIImgFieldsConfig(
    WholeBrainImgFieldsConfig, _SpecificityFields, BaseModel
):
    hemisphere: str = "hemisphere"
    region: str = "region"
    structure: str = "structure"


class SimilarityDatasetFieldsConfig(_SimilarityFields, BaseModel):
    subject_id: str = "subject id"
    process_method: str = "process method"
    similarity_method: str = "similarity method"


class AvgSimilarityDatasetFieldsConfig(_SpecificityFields, BaseModel):
    subset_type: str = "subset type"


class FieldsConfig(YAMLConfig):
    whole_brain_img: WholeBrainImgFieldsConfig = WholeBrainImgFieldsConfig()
    roi_img: ROIImgFieldsConfig = ROIImgFieldsConfig()
    similarity_dataset: SimilarityDatasetFieldsConfig = (
        SimilarityDatasetFieldsConfig()
    )
    avg_similarity_dataset: AvgSimilarityDatasetFieldsConfig = (
        AvgSimilarityDatasetFieldsConfig()
    )

    @classmethod
    def load(cls) -> Self:
        return cls.from_yaml_file(CONFIG_ROOT / "fields.yaml")


class FormatsConfig(YAMLConfig):
    datatype_stimulation_label: str = 'f"{stimulation} ({data_type})"'
    process_similarity_field: str = 'f"{similarity_name} ({process_name})"'
    region_label: str = 'f"{structure} {hemisphere}"'

    @classmethod
    def load(cls) -> Self:
        return cls.from_yaml_file(CONFIG_ROOT / "formats.yaml")

    def format_datatype_stimulation_label(
        self, data_type: str, stimulation: str
    ) -> str:
        return eval(self.datatype_stimulation_label)

    def format_process_similarity_field(
        self, process_name: str, similarity_name: str
    ) -> str:
        return eval(self.process_similarity_field)

    def format_region_category(self, structure: str, hemisphere: str) -> str:
        return eval(self.region_label)


class RegionCoordinatesConfig(BaseModel):
    x: int
    y: int
    z: int


class StructureConfig(BaseModel):
    L: RegionCoordinatesConfig
    R: RegionCoordinatesConfig
    specific_stimulation: str


class ROIsCoordinatesConfig(BaseModel):
    STG: StructureConfig
    FFA: StructureConfig


class ROIsConfig(YAMLConfig):
    side_length: int
    coordinates: ROIsCoordinatesConfig

    @classmethod
    def load(cls) -> Self:
        return cls.from_yaml_file(CONFIG_ROOT / "ROIs.yaml")

    def get_roi_axis_slice(
        self,
        structure: Literal["STG", "FFA"],
        hemisphere: Literal["L", "R"],
        axis: Literal["x", "y", "z"],
        len_axis: Optional[int] = None,
    ) -> slice:
        structure_config: StructureConfig = getattr(
            self.coordinates, structure
        )
        region_coordinates: RegionCoordinatesConfig = getattr(
            structure_config, hemisphere
        )
        center_pos: int = getattr(region_coordinates, axis)
        start = center_pos - self.side_length // 2
        end = start + self.side_length
        start = max(start, 0)
        if len_axis is not None:
            end = min(end, len_axis)
        return slice(start, end)

    def get_roi_specific_stimulation(
        self, structure: Literal["STG", "FFA"]
    ) -> str:
        structure_config: StructureConfig = getattr(
            self.coordinates, structure
        )
        return structure_config.specific_stimulation


FIELDS_CONFIG = FieldsConfig.load()
FORMATS_CONFIG = FormatsConfig.load()
LABELS_CONFIG = LabelsConfig.load()
ROIS_CONFIG = ROIsConfig.load()
