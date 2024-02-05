from PIL import Image
import glob

import re
from ..logger import logger


def __get_fnames(rgx: str):
    files = glob.glob(rgx)
    files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()))
    return files


def make_gif(img_regex: str, savefn: str):
    assert savefn.endswith('.gif'), f"savefn must end with .gif"
    fnames = __get_fnames(img_regex)
    logger.info(f"Images {img_regex} ({len(fnames)}) -> {savefn}")
    images = [Image.open(i) for i in __get_fnames(img_regex)]
    images[0].save(savefn, save_all=True, append_images=images[1:], duration=1000, loop=0)
