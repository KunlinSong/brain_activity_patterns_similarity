import sys
from pathlib import Path

_ROOT_DIRNAME = Path(__file__).parents[1]
sys.path.insert(0, _ROOT_DIRNAME.as_posix())

from utils.config import (
    FIELDS_CONFIG,
    FORMATS_CONFIG,
    LABELS_CONFIG,
    ROIS_CONFIG,
)
