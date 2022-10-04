#!/usr/bin/env python3

import argparse
import os
import random

import cv2
from matplotlib import pyplot as plt

classes = {}

def generate_random_image(img):
    mobilenet_res = 224
    w = img.shape[1]
    h = img.shape[0]
    random_zoom = random.randint(30, min(w, h))
    x_pad = random.randint(0, w - mobilenet_res)
    y_pad = random.randint(0, h - mobilenet_res)

    new_img = img[y_pad:(y_pad + mobilenet_res), x_pad:(x_pad + mobilenet_res)][:]
    new_img = cv2.resize(new_img,(mobilenet_res, mobilenet_res))
    return new_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')
    parser.add_argument('-n', '--nb_sub_image', help='the number of sub images that will be created per images')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")
    if os.path.exists(stored_image_path):
        exit(-1)

    os.makedirs(stored_image_path)

    img_list = os.listdir(args.img_path)

    for filename in img_list:
        print("parsing file: {0}".format(filename))
        img = cv2.imread(os.path.join(args.img_path, filename))
        for i in range(int(args.nb_sub_image)):
            sub_img = generate_random_image(img)

            plt.imshow(sub_img)
            plt.show()
            label = input("Enter class: w (water) or g (ground): ")
            img_filename = label + str(i) + '_' + filename
            cv2.imwrite(os.path.join(stored_image_path, img_filename), sub_img)

g