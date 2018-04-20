import os
import sys

from PIL import Image
import numpy as np


# I inverted https://github.com/facebookresearch/Detectron/blob/master/lib/datasets/cityscapes/coco_to_cityscapes_id.py
def coco_to_cityscapes_with_rider(coco_id):
    lookup = {
        0: 0,  # ... background
        2: 1,  # bicycle
        3: 2,  # car
        1: 3,  # person
        7: 4,  # train
        8: 5,  # truck
        4: 6,  # motorcycle
        6: 7,  # bus
        1: 8,  # rider ("person", *rider has human right!*)
    }
    return lookup[cityscapes_id]


GAN_SEG_MAP_DIR = 'test_label'
GAN_INST_DIR = 'test_inst'


def process_all(input_dir, output_dir):
    for f in os.listdir(input_dir):
        print('Processing {}'.format(f))

        img = Image.open(os.path.join(input_dir, f))
        seg_data = np.zeros(img.size)
        inst_data = np.zeros(img.size)
        w, h = img.size

        for i in range(w):
            for j in range(h):
                # Do work here, write to the np arrays
                pass

        seg_img = Image.fromarray(seg_data.astype('uint8'), mode='L')
        seg_img.save(os.path.join(output_dir, GAN_SEG_MAP_DIR, '{}'.format(f)))
        # seg_img.show()

        inst_img = Image.fromarray(inst_data.astype('int32'), mode='I')
        inst_img.save(os.path.join(output_dir, GAN_INST_MAP_DIR, '{}'.format(f)))
        # inst_img.show()


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    process_all(input_dir, output_dir)
