import os
from skimage import io
import sys
import numpy as np
from PIL import Image


def overlay_main(input_dir, output_dir, bg, seg=True):
    bg = io.imread(bg)
    for f in os.listdir(input_dir):
        fg = io.imread(os.path.join(input_dir, f))
        if seg:
            out_map = np.where(fg > 1, fg, bg)
            out_img = Image.fromarray(out_map)
        else:  # inst
            out_map = np.where(fg != 0, fg, bg)
            out_img = Image.fromarray(out_map.astype('int32'), mode='I')
        out_img.save(os.path.join(output_dir, f))

