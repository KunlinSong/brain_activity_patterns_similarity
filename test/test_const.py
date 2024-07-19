import sys
from pathlib import Path

_ROOT_DIRNAME = Path(__file__).parents[1]
sys.path.insert(0, _ROOT_DIRNAME.as_posix())

from utils.const import CONFIG_ROOT, DATA_ROOT, PROJECT_ROOT

print("finish")
