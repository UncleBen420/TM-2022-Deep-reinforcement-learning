#!/usr/bin/env python3

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt

import numpy as np

classes = {}


class BoundingBox:
    def __init__(self, points):
        self.points = points

    def get_center(self):
        x = np.mean(self.points[:, 0])
        y = np.mean(self.points[:, 1])
        return int(x), int(y)

    def get_width_height(self):
        width = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        height = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        return int(width), int(height)


def generate_random_image(img):
    yolo_resolution = 640
    w = img.shape[1]
    h = img.shape[0]
    x_pad = random.randint(0, w - yolo_resolution)
    y_pad = random.randint(0, h - yolo_resolution)

    new_img = img[y_pad:(y_pad + yolo_resolution), x_pad:(x_pad + yolo_resolution)][:]
    return new_img, x_pad, y_pad


def sub_image_label(labels, x_pad, y_pad):
    yolo_res = 640
    # filter out all element that are not contained in the sub image
    element_in_sub_image = list(filter(lambda bb: (x_pad <= bb['x'] < x_pad + yolo_res and
                                                   y_pad <= bb['y'] < y_pad + yolo_res), labels))
    # map the new coordinate of the center of gravity
    return list(map(lambda bb: (bb['label'],
                                bb['x'] - x_pad,
                                bb['y'] - y_pad,
                                bb['w'],
                                bb['h']), element_in_sub_image))


def label_to_yolo_format(labels):
    yolo_res = 640
    return list(map(lambda bb: (bb[0],
                                bb[1] / yolo_res,
                                bb[2] / yolo_res,
                                bb[3] / yolo_res,
                                bb[4] / yolo_res), labels))


def generate_yolo_X_Y(img, labels, nb_sub_img, out_dir_label, out_dir_img, name):
    text_label_file = ""
    background_count = 0
    normal_count = 0
    patience = 0
    while True:
        sub_img, x_pad, y_pad = generate_random_image(img)
        sub_labels = sub_image_label(labels, x_pad, y_pad)
        sub_labels = label_to_yolo_format(sub_labels)

        text_label_file = ""
        nb_boat = 0
        for label in sub_labels:
            if label[0] == classes['ship']:
                nb_boat += 1
            text_label_file += (str(label[0]) + ' '
                                + str(label[1]) + ' '
                                + str(label[2]) + ' '
                                + str(label[3]) + ' '
                                + str(label[4]) + '\n')

        # if there is no object in image, it creates a background image
        if text_label_file == "" and background_count < 2:
            background_count += 1
            img_filename = 'b_' + name + '_' + str(background_count) + '.png'
            cv2.imwrite(os.path.join(out_dir_img, img_filename), sub_img)
        elif text_label_file == "" and background_count >= 2:
            continue

        if nb_boat > 0:
            print("there is a boat")
            label_filename = name + '_' + str(normal_count) + '.txt'
            with open(os.path.join(out_dir_label, label_filename), "w") as myfile:
                myfile.write(text_label_file)
                img_filename = name + '_' + str(normal_count) + '.png'
                cv2.imwrite(os.path.join(out_dir_img, img_filename), sub_img)
                normal_count += 1
                if normal_count == nb_sub_img:
                    break

        patience += 1

        if patience >= 1000:
            break



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-l', '--label_path', help='the path to the labels folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')
    parser.add_argument('-n', '--nb_sub_image', help='the number of sub images that will be created per images')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")
    stored_label_path = os.path.join(args.out_path, "labels")

    if os.path.exists(stored_image_path) or os.path.exists(stored_label_path):
        exit(-1)

    os.makedirs(stored_image_path)
    os.makedirs(stored_label_path)

    img_list = os.listdir(args.img_path)
    label_list = os.listdir(args.label_path)
    # get all the class possible

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

            for line in contents.split('\n')[2:]:

                data = line.split(' ')
                if len(data) != 10:
                    continue

                if data[-2] != 'ship' and data[-2] != 'harbor':
                    data[-2] = 'not_ship'

                label = data[-2]
                points = np.zeros((4, 2))
                points[0] = [float(data[0]), float(data[1])]
                points[1] = [float(data[2]), float(data[3])]
                points[2] = [float(data[4]), float(data[5])]
                points[3] = [float(data[6]), float(data[7])]
                bb = BoundingBox(points)

                x, y = bb.get_center()
                w, h = bb.get_width_height()
                if label not in classes.keys():
                    classes[label] = len(classes)

                img_bb.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "label": classes[label]
                })
            print("parsing file:", filename)
            generate_yolo_X_Y(img, img_bb, int(args.nb_sub_image), stored_label_path, stored_image_path, filename.split('.')[0])

    print('number of class: {0}'.format(len(classes)))
    print(classes.keys())