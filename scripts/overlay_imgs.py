from skimage import io
import sys
import numpy as np
from PIL import Image


def overlay_map(front, back):
    overlay = np.where(front > 1, front, back)
    return overlay


def overlay_main(input_dir, output_dir, bg, seg=True):
    bg = io.imread(bg)
    for f in os.listdir(input_dir):
        fg = io.imread(os.path.join(input_dir, f))
        out_map = overlay_map(fg, bg)
        if seg:
            out_img = Image.fromarray(out_map)
        else:  # inst
            out_img = Image.fromarray(out_map.astype('int32'), mode='I')
        out_img.save(os.path.join(output_dir, f))


if __name__ == '__main__':
    output = overlay_map(img1, img2)

    if sys.argv[3] == "inst":
        output = Image.fromarray(output.astype('int32'), mode='I')
        output.save("overlay.png")
    else:
        output = Image.fromarray(output)
        output.save("overlay.png")
