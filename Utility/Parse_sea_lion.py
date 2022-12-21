import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_label(img, mask, threshold=0.96):
    # Perform match operations.
    res = cv2.matchTemplate(img, mask, cv2.TM_CCORR_NORMED)
    w, h, _ = mask.shape
    # Specify a threshold
    # Store the coordinates of matched area in a numpy array
    loc = np.where(res >= threshold)

    # Draw a rectangle around the matched region.
    bb = []
    for pt in zip(*loc[::-1]):
        img = cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), -1)
        bb.append((pt[0] + w / 2, pt[1] + h / 2))

    return img, bb

if __name__ == '__main__':

    img_list = os.listdir("../../TrainSmall2/TrainDotted")
    dot_brown = cv2.imread(os.path.join("../../TrainSmall2/dot.jpg"))
    dot_brown = cv2.cvtColor(dot_brown, cv2.COLOR_BGR2RGB)

    dot_red = cv2.imread(os.path.join("../../TrainSmall2/red.jpg"))
    dot_red = cv2.cvtColor(dot_red, cv2.COLOR_BGR2RGB)

    dot_pink = cv2.imread(os.path.join("../../TrainSmall2/pink.jpg"))
    dot_pink = cv2.cvtColor(dot_pink, cv2.COLOR_BGR2RGB)

    dot_blue = cv2.imread(os.path.join("../../TrainSmall2/blue.jpg"))
    dot_blue = cv2.cvtColor(dot_blue, cv2.COLOR_BGR2RGB)

    seal = [dot_red, dot_pink, dot_brown]

    for file in img_list:
        print(file)
        img = cv2.imread(os.path.join("../../TrainSmall2/TrainDotted", file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = []
        # red point
        for s in seal:
            img, bb =  get_label(img, s)
            for b in bb:

                labels.append(b)

        new_file = open("../../TrainSmall2/labels/" + file.split(".")[0] + ".txt", "w")
        for value in labels:
            x, y = value
            new_file.writelines(str(x) + ' ' + str(y) + '\n')
        new_file.close()


