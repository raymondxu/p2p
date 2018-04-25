import argparse
import os
import shutil
import subprocess

from PIL import Image
import numpy as np

from Mask_RCNN.rcnn_master import rcnn_main
from scripts.rcnn_to_gan import rcnn_to_gan_main
from scripts.overlay_imgs import overlay_main


ROOT_DIR = os.path.abspath('./')
TEMP_DIR = os.path.join(ROOT_DIR, 'master_temp')

RCNN_OUTPUT_MASK_DIR = os.path.join(TEMP_DIR, 'rcnn_mask')
RCNN_OUTPUT_CLASS_MAP_DIR = os.path.join(TEMP_DIR, 'rcnn_class_map')

OVERLAY_INPUT_DIR = os.path.join(TEMP_DIR, 'overlay_input')
OVERLAY_INPUT_SEG_DIR = os.path.join(OVERLAY_INPUT_DIR, 'test_label')
OVERLAY_INPUT_INST_DIR = os.path.join(OVERLAY_INPUT_DIR, 'test_inst')

GAN_INPUT_DIR = os.path.join(ROOT_DIR, 'datasets/master')
OVERLAY_OUTPUT_SEG_DIR = os.path.join(GAN_INPUT_DIR, 'test_label')
OVERLAY_OUTPUT_INST_DIR = os.path.join(GAN_INPUT_DIR, 'test_inst')
GAN_OUTPUT_DIR = os.path.join(ROOT_DIR, 'results_rcnn/test_1024p')


def process_all(input_dir, output_dir, bg_seg, bg_inst):
    for f in os.listdir(input_dir):
        print('Processing {}'.format(f))
        run_rcnn(input_dir)
        run_rcnn_to_gan()
        run_overlay(bg_seg, bg_inst)
        run_gan(output_dir)


def run_rcnn(input_dir):
    print('Running RCNN...')
    rcnn_main(input_dir)


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
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Move results to output_dir
    shutil.move(GAN_OUTPUT_DIR, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation-Driven Image Variation Generation Pipeline')
    parser.add_argument('-input', default='master_input', help='the path to the input directory containing photo images')
    parser.add_argument('-output', default='master_output', help='the path to the output directory where the image variants will be placed')
    parser.add_argument('bg_seg', help='the path to the background semantic map to generate variants in the foreground of')
    parser.add_argument('bg_inst', help='the path to the background instance map to generate variants in the foreground of')

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    bg_seg = args.bg_seg
    bg_inst = args.bg_inst

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    process_all(input_dir, output_dir, bg_seg, bg_inst)
