#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
import argparse
import numpy as np
from PIL import Image


def run(load_path, save_path):

    for img_path in os.listdir(load_path):
        img = Image.open(os.path.join(load_path, img_path))
        img = img.resize((256, 128), Image.NEAREST)
        img.save(os.path.join(save_path, img_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, help='paths to real and fake images')
    parser.add_argument('--save_path', type=str, help='paths to real and fake images')
    args = parser.parse_args()
    run(args.load_path, args.save_path)
