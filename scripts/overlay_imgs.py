from skimage import io
import sys
import numpy as np
from PIL import Image


def overlay_semantic_map(front, back):
    overlay = np.where(front != 0, front, back)
    return overlay

if __name__ == '__main__':
    img1 = io.imread(sys.argv[1])
    img2 = io.imread(sys.argv[2])
    output = overlay_semantic_map(img1, img2)

    output = Image.fromarray(output)
    output.save("overlay.png")