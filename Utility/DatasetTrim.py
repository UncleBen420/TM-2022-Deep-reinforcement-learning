import argparse
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt

import numpy as np

dataset_classes = ['small-vehicle', 'large-vehicle', 'roundabout', 'ship', 'swimming-pool', 'bridge',
                   'soccer-ball-field', 'harbor', 'tennis-court', 'baseball-diamond', 'basketball-court',
                   'storage-tank', 'plane', 'helicopter', 'ground-track-field', 'container-crane', 'helipad']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")

    if os.path.exists(stored_image_path):
        exit(-1)

    os.makedirs(stored_image_path)

    img_list = os.listdir(args.img_path)
    # get all the class possible
    classes = {}
    for filename in img_list:

        label = filename.split('_')[0]

        label = dataset_classes[int(label)]
        if label == "harbor":
            label = "ship"

        if label != "ship":
            label = "not-ship"

        if label not in classes.keys():
            classes[label] = 0

        classes[label] += 1

        new_filename = label + '_' + str(classes[label]) + ".png"
        if classes[label] < 1100:
            shutil.copyfile(os.path.join(args.img_path, filename),
                            os.path.join(stored_image_path, new_filename))
