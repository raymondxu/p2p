import os
import sys

from PIL import Image
import numpy as np

from collections import defaultdict


GAN_SEG_MAP_DIR = 'test_label'
GAN_INST_DIR = 'test_inst'


def coco_to_cityscapes_id(coco_id):
    lookup = {
        0: 0,  # ... background
        1: 24,  # person
        2: 33,  # bike
        3: 26,  # car
        4: 32,  # motorcycle
        5: 0,  # airplane
        6: 28,  # bus
        7: 31,  # train
        8: 27,  # truck
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
        temp = coco_to_cityscapes_id(coco_class_id)
        seg_data += mask[:,:,mask_layer] * coco_to_cityscapes_id(coco_class_id)

    seg_data = np.clip(seg_data, 1, 30) # temp workaround
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
    object_counter = 1

    object_dict = defaultdict(int)

    for mask_layer, coco_class_id in enumerate(class_map):
        if coco_class_id > 0:
            obj_id = coco_to_cityscapes_id(coco_class_id) * 1000 + object_dict[coco_class_id]
            object_dict[coco_class_id] += 1
            inst_data += mask[:,:,mask_layer] * obj_id

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
