#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt

import numpy as np

classes = {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-l', '--label_path', help='the path to the labels folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')

    args = parser.parse_args()
    stored_label_path = os.path.join(args.out_path, "labels")

    if os.path.exists(stored_label_path):
        exit(-1)

    os.makedirs(stored_label_path)

    label_list = os.listdir(args.label_path)
    # get all the class possible

    image_filename = ""
    for filename in label_list:
        # check if the image file corresponding to the label file exist

        with open(os.path.join(args.label_path, filename)) as file:
            contents = file.read()

        img_bb = []

        for line in contents.split('\n')[2:]:

            data = line.split(' ')
            if len(data) != 10:
                continue

            if data[-2] != 'ship' and data[-2] != 'harbor':
                data[-2] = 'not_ship'

            label = data[-2]
            X = [float(data[0]), float(data[2]), float(data[4]), float(data[6])]
            Y = [float(data[1]), float(data[3]), float(data[5]), float(data[7])]

            if label not in classes.keys():
                classes[label] = len(classes)

            img_bb.append({
                "x_min": np.min(X),
                "y_min": np.min(Y),
                "x_max": np.max(X),
                "y_max": np.max(Y),
                "label": classes[label]
            })
        print("parsing file:", filename)
        text_label_file = ""
        for l in img_bb:
            if l['label'] == classes['ship'] or ('harbor' in classes.keys() and l['label'] == classes['harbor']):
                text_label_file += (str(l['label']) + ' '
                                    + str(l['x_min']) + ' '
                                    + str(l['x_max']) + ' '
                                    + str(l['y_min']) + ' '
                                    + str(l['y_max']) + '\n')

        with open(os.path.join(stored_label_path, filename), "w") as file:
            file.write(text_label_file)


    print('number of class: {0}'.format(len(classes)))
    print(classes.keys())