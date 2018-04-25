import argparse
import os
import shutil
import subprocess

from PIL import Image
import numpy as np

from scripts.rcnn_to_gan import process_main


TEMP_DIR = 'master_temp'
RCNN_OUTPUT_MASK_DIR = os.path.join(TEMP_DIR, 'rcnn_mask')
RCNN_OUTPUT_CLASS_MAP_DIR = os.path.join(TEMP_DIR, 'rcnn_class_map')
GAN_INPUT_DIR = 'datasets/cityscapes'
GAN_OUTPUT_DIR = 'results_rcnn/test_1024p'


parser = argparse.ArgumentParser(description='Segmentation-Driven Image Variation Generation Pipeline')
parser.add_argument('-input', default='master_input', help='the path to the input directory containing photo images')
parser.add_argument('-output', default='master_output', help='the path to the output directory where the image variants will be placed')


def process_all(input_dir, output_dir):
    for f in os.listdir(input_dir):
        print('Processing {}'.format(f))


def run_rcnn(input_dir, output_dir):
    print('Running RCNN...')
    pass


def run_rcnn_to_gan():
    print('Running rcnn_to_gan...')
    process_main(RCNN_OUTPUT_MASK_DIR, RCNN_OUTPUT_CLASS_MAP_DIR, GAN_INPUT_DIR)


def run_gan(output_dir):
    print('Running GAN...')

    # Run GAN
    command = 'bash scripts/test_master.sh'
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Move results to output_dir
    shutil.move(GAN_OUTPUT_DIR, output_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(TEMP_DIR):
        os.mkdir(TEMP_DIR)

    process_all(input_dir, output_dir)
