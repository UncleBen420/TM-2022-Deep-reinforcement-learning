import os
import random

import cv2
import numpy as np

def generate_random_image(img, bb):

    frcnn_res = 200
    w = img.shape[1]
    h = img.shape[0]

    x, y = random.choice(bb)
    x = int(float(x))
    y = int(float(y))


    x_pad = random.randint(-64, 0)
    y_pad = random.randint(-64, 0)

    new_x = x + x_pad
    new_y = y + y_pad

    new_img = img[new_y:(new_y + frcnn_res), new_x:(new_x + frcnn_res)][:]

    new_bbs = []
    for b in bb:
        x_, y_ = b
        x_ = int(float(x_)) - new_x
        y_ = int(float(y_)) - new_y
        if 0 < x_ < 100 and 0 < y_ < 100:
            new_bbs.append((x_, y_))

    return new_img, new_bbs

def generate_random_image2(img, mask, bb):

    frcnn_res = 200
    w = img.shape[1]
    h = img.shape[0]

    x, y = random.choice(bb)
    x = int(float(x))
    y = int(float(y))


    x_pad = random.randint(-64, 0)
    y_pad = random.randint(-64, 0)

    new_x = x + x_pad
    new_y = y + y_pad

    new_img = img[new_y:(new_y + frcnn_res), new_x:(new_x + frcnn_res)]
    new_mask = mask[new_y:(new_y + frcnn_res), new_x:(new_x + frcnn_res)]

    return new_img, new_mask


if __name__ == '__main__':
    img_list = os.listdir("../../TrainSmall2/TrainDotted")
    for file in img_list:

        label = open(os.path.join("../../TrainSmall2/labels", file.split(".")[0] + ".txt"), 'r')
        Lines = label.readlines()
        bb = []
        for line in Lines:
            bb.append(line.split(" "))

        count = 0
        # Strips the newline character
        for line in Lines:
            count += 1

        print(file)
        img = cv2.imread(os.path.join("../../TrainSmall2/Train", file))
        mask = cv2.imread(os.path.join("../../TrainSmall2/TrainDotted/", file))

        for i in range(50):
            #img_new, bb_new, = generate_random_image(img, bb)
            img_new, mask_new = generate_random_image2(img, mask=mask, bb=bb)

            filename = os.path.join("../../Seal/Classification/Train/img", file.split('.')[0] + "_" + str(i) + ".jpg")
            cv2.imwrite(filename, img_new)

            new_file = os.path.join("../../Seal/Classification/Train/mask", file.split('.')[0] + "_" + str(i) + ".jpg")
            cv2.imwrite(new_file, mask_new)




