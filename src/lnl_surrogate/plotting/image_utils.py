import glob
import os
import re

from PIL import Image

from ..logger import logger


def __get_fnames(rgx: str):
    files = glob.glob(rgx)
    files = sorted(
        files,
        key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()),
    )
    return files


def make_gif(
    img_regex: str = None,
    savefn: str = "out.gif",
    image_paths=None,
):
    assert savefn.endswith(".gif"), f"savefn must end with .gif"
    fnames = []
    if img_regex is not None:
        fnames = __get_fnames(img_regex)
    if image_paths is not None:
        fnames = image_paths
    if len(fnames) < 2:
        logger.warning(
            f"{len(fnames)} images found for {img_regex} (no gif created)"
        )
        return
    logger.info(f"Images {img_regex} ({len(fnames)}) -> {savefn}")
    images = [Image.open(i) for i in fnames]
    images[0].save(
        savefn, save_all=True, append_images=images[1:], duration=1000, loop=0
    )
