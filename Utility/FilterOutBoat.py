#!/usr/bin/env python3


# This is a sample Python script.
import argparse
import os
import shutil

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to select images in Dota dataset with ship in it')
    parser.add_argument('-i', '--img_path', help='the path to the images folder')
    parser.add_argument('-l', '--label_path', help='the path to the labels folder')
    parser.add_argument('-o', '--out_path', help='the path where images will be stored')

    args = parser.parse_args()

    stored_image_path = os.path.join(args.out_path, "images")
    stored_label_path = os.path.join(args.out_path, "labels")

    if os.path.exists(stored_image_path) or os.path.exists(stored_label_path):
        print("already exist")
        exit(-1)

    os.makedirs(stored_image_path)
    os.makedirs(stored_label_path)

    img_list = os.listdir(args.img_path)
    label_list = os.listdir(args.label_path)

    for filename in label_list:

        image_filename = filename.split('.')[0] + '.png'

        # check if the image file corresponding to the label file exist
        if image_filename in img_list:
            with open(os.path.join(args.label_path, filename)) as file:
                contents = file.read()
                search_word = "ship"
                if search_word in contents:
                    shutil.copy(os.path.join(args.label_path, filename), os.path.join(stored_label_path, filename))
                    shutil.copyfile(os.path.join(args.img_path, image_filename),
                                    os.path.join(stored_image_path, image_filename))

                    print('ship label found')
                else:
                    print('ship label not found')
        else:
            print("image not found")

