import argparse
import os
import shutil
import subprocess

from PIL import Image
import numpy as np

from Mask_RCNN.rcnn_master import rcnn_main
from scripts.rcnn_to_gan import rcnn_to_gan_main
from scripts.overlay_imgs import overlay_main
from test import gan_main

import torch
import sys

TEMP_DIR = '/tmp'

RCNN_OUTPUT_MASK_DIR = os.path.join(TEMP_DIR, 'rcnn_mask')
RCNN_OUTPUT_CLASS_MAP_DIR = os.path.join(TEMP_DIR, 'rcnn_class_map')

OVERLAY_INPUT_DIR = os.path.join(TEMP_DIR, 'overlay_input')
OVERLAY_INPUT_SEG_DIR = os.path.join(OVERLAY_INPUT_DIR, 'test_label')
OVERLAY_INPUT_INST_DIR = os.path.join(OVERLAY_INPUT_DIR, 'test_inst')

GAN_INPUT_DIR = 'datasets/master'
OVERLAY_OUTPUT_SEG_DIR = os.path.join(GAN_INPUT_DIR, 'test_label')
OVERLAY_OUTPUT_INST_DIR = os.path.join(GAN_INPUT_DIR, 'test_inst')
GAN_OUTPUT_DIR = 'results_rcnn'

DIRS = [
    TEMP_DIR,
    RCNN_OUTPUT_MASK_DIR,
    RCNN_OUTPUT_CLASS_MAP_DIR,
    OVERLAY_INPUT_DIR,
    OVERLAY_INPUT_SEG_DIR,
    OVERLAY_INPUT_INST_DIR,
    GAN_INPUT_DIR,
    OVERLAY_OUTPUT_SEG_DIR,
    OVERLAY_OUTPUT_INST_DIR,
    GAN_OUTPUT_DIR,
]


def process_all(input_dir, output_dir, bg_seg, bg_inst, part):
    for f in os.listdir(input_dir):
        print('Processing {}'.format(f))
        if part == 1:
            run_rcnn(input_dir)
            run_rcnn_to_gan()
            run_overlay(bg_seg, bg_inst)
        else:
            run_gan(output_dir)


def run_rcnn(input_dir):
    print('Running RCNN...')
    rcnn_main(input_dir)
    torch.cuda.empty_cache()
    from keras import backend as be
    be.clear_session()
    import pdb
    pdb.set_trace()

def run_rcnn_to_gan():
    print('Running rcnn_to_gan...')
    rcnn_to_gan_main(RCNN_OUTPUT_MASK_DIR, RCNN_OUTPUT_CLASS_MAP_DIR, OVERLAY_INPUT_DIR)


def run_overlay(bg_seg, bg_inst):
    print('Running overlay...')
    # Overlay seg maps
    overlay_main(OVERLAY_INPUT_SEG_DIR,
                 OVERLAY_OUTPUT_SEG_DIR,
                 bg_seg,
                 seg=True)

    # Overlay inst maps
    overlay_main(OVERLAY_INPUT_INST_DIR,
                 OVERLAY_OUTPUT_INST_DIR,
                 bg_inst,
                 seg=False)


def run_gan(output_dir):
    print('Running GAN...')

    # Run GAN
    command = 'bash scripts/test_master.sh'
    #ret = subprocess.call(command.split())
    gan_main()

    print('Moving results to {}...'.format(output_dir))
    # Move results to output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    shutil.move(os.path.join(GAN_OUTPUT_DIR), output_dir)


if __name__ == '__main__':
    parse_master = argparse.ArgumentParser(description='Segmentation-Driven Image Variation Generation Pipeline')
    parse_master.add_argument('-input', default='master_input', help='the path to the input directory containing photo images')
    parse_master.add_argument('-output', default='master_output', help='the path to the output directory where the image variants will be placed')
    parse_master.add_argument('bg_seg', help='the path to the background semantic map to generate variants in the foreground of')
    parse_master.add_argument('bg_inst', help='the path to the background instance map to generate variants in the foreground of')
    parse_master.add_argument('part', type=int, help='which part of the pipeline to run (1 or 2)')

    args = parse_master.parse_args()
    input_dir = args.input
    output_dir = args.output
    bg_seg = args.bg_seg
    bg_inst = args.bg_inst
    part = args.part

    sys.argv = sys.argv[:1]

    for d in DIRS:
        if not os.path.exists(d):
            os.mkdir(d)

    process_all(input_dir, output_dir, bg_seg, bg_inst, part)
