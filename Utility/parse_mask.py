import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

RED = [255, 0, 0]
BLUE = [0, 0, 255]
BROWN = [255, 255, 255]
GREEN = [0, 255, 0]

if __name__ == '__main__':

    img_list = os.listdir("../../Seal/Train/mask")

    for file in img_list:
        print(file)
        img = cv2.imread(os.path.join("../../Seal/Train/mask", file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img <= 5] = 0
        img[img >= 250] = 255
        gray_img = cv2.bitwise_or(img[:,:,0], img[:,:,1])
        gray_img = cv2.bitwise_or(gray_img, img[:, :, 2])

        bboxes = []

        threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)[1]

        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(threshold, 1, cv2.CV_32S)

        new_file = open("../../Seal/Train/bboxes/" + file.split(".")[0] + ".txt", "w")
        print(totalLabels)
        for i in range(1, totalLabels):
            # Area of the component
            x_clr = int(centroid[i][0])
            y_clr = int(centroid[i][1])
            color_class = img[y_clr][x_clr]

            if np.array_equal(color_class, BROWN):
                l = 0
            elif np.array_equal(color_class, RED):
                l = 1
            elif np.array_equal(color_class, GREEN):
                l = 2
            elif np.array_equal(color_class, BLUE):
                l = 3

            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]


            new_file.writelines('{0} {1} {2} {3} {4}\n'.format(str(l), str(x), str(y), str(w), str(h)))

        new_file.close()


