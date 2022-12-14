#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt

import numpy as np


def generate_random_image(img):
    frcnn_res = 100
    w = img.shape[1]
    h = img.shape[0]
    x_pad = random.randint(0, w - frcnn_res)
    y_pad = random.randint(0, h - frcnn_res)

    new_img = img[y_pad:(y_pad + frcnn_res), x_pad:(x_pad + frcnn_res)][:]
    return new_img, x_pad, y_pad


def sub_image_label(labels, x_pad, y_pad):
    frcnn_res = 320
    # filter out all element that are not contained in the sub image
    element_in_sub_image = list(filter(lambda bb: (x_pad <= bb['x_min'] and bb['x_max'] < x_pad + frcnn_res and
                                                   y_pad <= bb['y_min'] and bb['y_max'] < y_pad + frcnn_res), labels))
    # map the new coordinate of the center of gravity
    return list(map(lambda bb: (bb['label'],
                                bb['x_min'] - x_pad,
                                bb['y_min'] - y_pad,
                                bb['x_max'] - x_pad,
                                bb['y_max'] - y_pad,), element_in_sub_image))


def generate_frcnn_X_Y(img, labels, nb_sub_img, out_dir_label, out_dir_img, name):
    text_label_file = ""
    for i in range(nb_sub_img):
        sub_img, x_pad, y_pad = generate_random_image(img)
        sub_labels = sub_image_label(labels, x_pad, y_pad)

        text_label_file = ""
        for label in sub_labels:
            text_label_file += (str(label[0]) + ' '
                                + str(label[1]) + ' '
                                + str(label[2]) + ' '
                                + str(label[3]) + ' '
                                + str(label[4]) + '\n')

        # if there is no object in image
        if text_label_file != "":
            label_filename = name + '_' + str(i) + '.txt'
            with open(os.path.join(out_dir_label, label_filename), "w") as myfile:
                myfile.write(text_label_file)
                img_filename = name + '_' + str(i) + '.png'
                cv2.imwrite(os.path.join(out_dir_img, img_filename), sub_img)


if __name__ == '__main__':

    if os.path.exists(stored_image_path) or os.path.exists(stored_label_path):
        exit(-1)

    os.makedirs(stored_image_path)
    os.makedirs(stored_label_path)

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

            if img.shape[0] < 320 or img.shape[1] < 320:
                continue

            img_bb = []

            for line in contents.split('\n')[2:]:

                data = line.split(' ')
                if len(data) != 10:
                    continue

                if data[-2] != 'ship':
                    continue

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
            generate_frcnn_X_Y(img, img_bb, int(args.nb_sub_image), stored_label_path, stored_image_path, filename.split('.')[0])

    print('number of class: {0}'.format(len(classes)))
    print(classes.keys())