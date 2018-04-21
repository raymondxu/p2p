import os
import sys

from PIL import Image
import numpy as np


GAN_SEG_MAP_DIR = 'test_label'
GAN_INST_DIR = 'test_inst'


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
    return lookup[coco_id]


def process_all(input_mask_dir, input_class_map_dir, output_dir):
    for mask, class_map in zip(os.listdir(input_mask_dir), os.listdir(input_class_map_dir)):
        print('Processing {} and {}'.format(mask, class_map))
        make_seg(mask, input_mask_dir, class_map, input_class_map_dir, output_dir)
        make_inst(mask, input_mask_dir, class_map, input_class_map_dir, output_dir)


def make_seg(mask_name, mask_dir, class_map_name, class_map_dir, output_dir):
    """
    Produces a cityscapes segmentation map from an object mask and a class map.

    Identified pixels will have values corresponding to their cityscapes class id.
    Unidentified pixels will have value 0.
    """
    mask = np.load(os.path.join(mask_dir, mask_name))
    class_map = np.load(os.path.join(class_map_dir, class_map_name))

    w, h, objs = mask.shape
    seg_data = np.zeros((w, h))

    for mask_layer, coco_class_id in enumerate(class_map):
        seg_data += mask[:,:,mask_layer] * (coco_to_cityscapes_with_rider(coco_class_id) * np.ones((w, h)))

    seg_img = Image.fromarray(seg_data.astype('uint8'), mode='L')
    seg_img.save(os.path.join(output_dir, GAN_SEG_MAP_DIR, '{}.png'.format(mask_name[:-4])))


def make_inst(mask_name, mask_dir, class_map_name, class_map_dir, output_dir):
    """
    Produces a cityscapes instance map from an object mask and a class map.

    Each object will be colored a distinct arbitrary color such that the GAN edge processor
    will be able to draw object boundaries. Non-objects will have value 0.
    """
    mask = np.load(os.path.join(mask_dir, mask_name))
    class_map = np.load(os.path.join(class_map_dir, class_map_name))
    
    w, h, objs = mask.shape
    inst_data = np.zeros((w, h))
    object_counter = 100

    for mask_layer in range(len(class_map)):
        inst_data += mask[:,:,mask_layer] * (object_counter * np.ones((w, h)))

    inst_img = Image.fromarray(inst_data.astype('int32'), mode='I')
    inst_img.save(os.path.join(output_dir, GAN_INST_DIR, '{}.png'.format(mask_name[:-4])))


if __name__ == '__main__':
    """
    Usage: python3 rcnn_to_gan <input_masks_dir> <input_class_map_dir> <output_dir>

    Masks are one-hot npy files of shape (w, h, n_classes).
    Class maps are npy files of shape (n_classes, 1) and maps each layer of the mask to a coco class.
    The created files will be written to `<output_dir>/test_label` and `<output_dir>/test_inst`.
    """
    input_mask_dir = sys.argv[1]
    input_class_map_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    seg_path = os.path.join(output_dir, GAN_SEG_MAP_DIR)
    inst_path = os.path.join(output_dir, GAN_INST_DIR)
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)
    if not os.path.exists(inst_path):
        os.mkdir(inst_path)

    process_all(input_mask_dir, input_class_map_dir, output_dir)
