import sys
from pathlib import Path

_ROOT_DIRNAME = Path(__file__).parents[1]
sys.path.insert(0, _ROOT_DIRNAME.as_posix())

from utils.img_loader import load_roi_images, load_whole_brain_images
from utils.random import shuffle_all_imgs

whole_brain_imgs = load_whole_brain_images()
shuffled_whole_brain_imgs = shuffle_all_imgs(whole_brain_imgs)

print(shuffled_whole_brain_imgs)


roi_imgs = load_roi_images()
shuffled_roi_imgs = shuffle_all_imgs(roi_imgs)
print(shuffled_roi_imgs)
