import sys
from pathlib import Path

_ROOT_DIRNAME = Path(__file__).parents[1]
sys.path.insert(0, _ROOT_DIRNAME.as_posix())

from utils.img_loader import load_roi_images, load_whole_brain_images

whole_brain_imgs = load_whole_brain_images()
roi_images = load_roi_images()

print(whole_brain_imgs)
print(roi_images)
print("finish")
