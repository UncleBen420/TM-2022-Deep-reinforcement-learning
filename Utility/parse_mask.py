import os

import cv2

if __name__ == '__main__':

    img_list = os.listdir("../../dataset_marker/mask")

    for file in img_list:
        print(file)
        img = cv2.imread(os.path.join("../../dataset_marker/mask", file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bboxes = []

        totalLabels, label_ids, values, centroid = cv2.connectedComponentsWithStats(gray_img,  4, cv2.CV_32S)
        new_file = open("../../dataset_marker/bboxes/" + file.split(".")[0] + ".txt", "w")
        for i in range(1, totalLabels):
            # Area of the component
            x = values[i, cv2.CC_STAT_LEFT]
            y = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]


            new_file.writelines('0 {0} {1} {2} {3}\n'.format(str(x), str(y), str(w), str(h)))

        new_file.close()


