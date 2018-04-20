import os
import sys

from PIL import Image
import numpy as np


cityscapes_mapping = {
    (  0,  0,  0): 0,
    (  0,  0,  0): 1,
    (  0,  0,  0): 2,
    (  0,  0,  0): 3,
    (  0,  0,  0): 4,
    (111, 74,  0): 5,
    ( 81,  0, 81): 6,
    (128, 64,128): 7,
    (244, 35,232): 8,
    (250,170,160): 9,
    (230,150,140): 10,
    ( 70, 70, 70): 11,
    (102,102,156): 12,
    (190,153,153): 13,
    (180,165,180): 14,
    (150,100,100): 15,
    (150,120, 90): 16,
    (153,153,153): 17,
    (153,153,153): 18,
    (250,170, 30): 19,
    (220,220,  0): 20,
    (107,142, 35): 21,
    (152,251,152): 22,
    ( 70,130,180): 23,
    (220, 20, 60): 24,
    (255,  0,  0): 25,
    (  0,  0,142): 26,
    (  0,  0, 70): 27,
    (  0, 60,100): 28,
    (  0,  0, 90): 29,
    (  0,  0,110): 30,
    (  0, 80,100): 31,
    (  0,  0,230): 32,
    (119, 11, 32): 33,
    (  0,  0,142): 34,
}

cache = {}


def process_all(input_dir, output_dir):
    for f in os.listdir(input_dir):
        print('Processing {}'.format(f))

        img = Image.open(os.path.join(input_dir, f))
        new_img_data = np.zeros(img.size)
        w, h = img.size

        for i in range(w):
            for j in range(h):
                class_id = get_nearest_class(img.getpixel((i, j)))
                new_img_data[i][j] = class_id

        new_img = Image.fromarray(new_img_data.astype('uint8'), mode='L')
        new_img.save(os.path.join(output_dir, 'clustered_{}'.format(f)))
        new_img.show()


def get_nearest_class(color):
    if color in cache:
        return cache[color]

    nearest_dist = float('inf')
    nearest_class = 0

    for class_color, class_id in cityscapes_mapping.items():
        d = dist(color, class_color)
        if d < nearest_dist:
            nearest_dist = d
            nearest_class = class_id

    if color not in cache:
        cache[color] = nearest_class

    return nearest_class


def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2) + ((c1[1] - c2[1]) ** 2) + ((c1[2] - c2[2]) ** 2)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    process_all(input_dir, output_dir)
