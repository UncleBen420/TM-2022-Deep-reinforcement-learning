#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt

import numpy as np

RES_MODEL = 224


def generate(img, labels, out_dir_img, name):
    for label in labels:
        sub_img = img[label["y_min"]:label["y_max"], label["x_min"]:label["x_max"]][:]

        sub_img = cv2.resize(sub_img, (RES_MODEL, RES_MODEL))

        img_filename = str(label["label"]) + '_' + name + '_' + str(label["label_nb"]) + '.png'
        cv2.imwrite(os.path.join(out_dir_img, img_filename), sub_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-l', '--label_path', help='the path to the labels folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")

    if os.path.exists(stored_image_path):
        exit(-1)

    os.makedirs(stored_image_path)

    img_list = os.listdir(args.img_path)
    label_list = os.listdir(args.label_path)
    # get all the class possible
    classes = {}
    image_filename = ""
    for filename in label_list:

        image_filename = filename.split('.')[0] + '.png'

        # check if the image file corresponding to the label file exist
        if image_filename in img_list:
            with open(os.path.join(args.label_path, filename)) as file:
                contents = file.read()
            img = cv2.imread(os.path.join(args.img_path, image_filename))

            if img.shape[0] < 640 or img.shape[1] < 640:
                continue

            img_bb = []

            item_number = 0
            for line in contents.split('\n')[2:]:

                data = line.split(' ')
                if len(data) != 10:
                    continue

                label = data[-2]
                X = [float(data[0]), float(data[2]), float(data[4]), float(data[6])]
                Y = [float(data[1]), float(data[3]), float(data[5]), float(data[7])]

                if label not in classes.keys():
                    classes[label] = len(classes)

                img_bb.append({
                    "x_min": int(np.min(X)),
                    "y_min": int(np.min(Y)),
                    "x_max": int(np.max(X)),
                    "y_max": int(np.max(Y)),
                    "label": classes[label],
                    "label_nb": item_number
                })
                item_number += 1
            print("parsing file:", filename)
            generate(img, img_bb, stored_image_path, filename.split('.')[0])

    print('number of class: {0}'.format(len(classes)))
    print(classes.keys())
