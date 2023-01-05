"""
This file contain the implementation of the real environment.
"""
import math
import os
import random
import re
import cv2
import imageio
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

import gc
from matplotlib import pyplot as plt

TASK_MODEL_RES = 200


class Transform:

    def __init__(self, min_shift, max_shift):
        self.min_shift = min_shift
        self.max_shift = max_shift

    def shift_img(self, img, x_shift, y_shift):
        T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(img, T, (TASK_MODEL_RES, TASK_MODEL_RES))

    def shift_bbox(self, bboxes, x_shift, y_shift):
        new_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x += x_shift
            y += y_shift

            if x > TASK_MODEL_RES:
                continue
            if y > TASK_MODEL_RES:
                continue
            if x < 0:
                continue
            if y < 0:
                continue



            new_bboxes.append((x, y, w, h))
        return new_bboxes

    def __call__(self, img, bboxes):
        x_shift = random.randint(self.min_shift, self.max_shift)
        y_shift = random.randint(self.min_shift, self.max_shift)
        img = self.shift_img(img, x_shift, y_shift)
        bboxes = self.shift_bbox(bboxes, x_shift, y_shift)
        return img, bboxes


def check_cuda():
    """
    check if opencv can use cuda
    :return: return True if opencv can detect cuda. False otherwise.
    """
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self):
        self.history = None
        self.min_res = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.nb_max_conv_action = None
        self.full_img = None
        self.dim = None
        self.pq = None
        self.sub_images_queue = None
        self.current_node = None
        self.Queue = None
        self.conventional_policy_nb_step = None
        self.base_img = None
        self.nb_actions_taken = 0
        self.action_space = 10
        self.cv_cuda = check_cuda()
        self.transform = Transform(-50, 50)
        self.difficulty = 0.
        self.agent = None

    def reload_env(self, img, bb):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        self.objects_coordinates = []
        self.history = []
        self.prepare_img(img)
        self.prepare_coordinates(bb)
        #self.full_img, self.objects_coordinates = self.transform(self.full_img, self.objects_coordinates)
        self.heat_map = np.zeros((TASK_MODEL_RES, TASK_MODEL_RES))


        self.nb_actions_taken = 0
        return self.get_state()

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        H, W, channels = self.full_img.shape
        # check which dimention is the bigger
        max_ = np.max([W, H])
        # check that the image is divisble by 2
        if max_ % 2:
            max_ += 1

        self.full_img = cv2.copyMakeBorder(self.full_img, 0, max_ - H, 0,
                                           max_ - W, cv2.BORDER_CONSTANT, None, value=0)
        self.base_img = self.full_img.copy()

    def get_state2(self):
        return np.squeeze(cv2.resize(self.full_img, (100, 100))) / 255.
        #return np.squeeze(self.full_img) / 255.

    def get_state(self):
        if len(self.objects_coordinates) > 0:
            nb_masked = random.randint(0, len(self.objects_coordinates) - 1)
        else:
            nb_masked = 0

        masked_bb = random.choices(self.objects_coordinates, k=nb_masked)
        self.bboxes = list(set(self.objects_coordinates) - set(masked_bb))

        temp = cv2.resize(self.full_img, (100, 100))
        for bb in masked_bb:
            x, y, w, h, label = bb
            p1 = (int(x/ 2), int(y/ 2) )
            p2 = (int((x + w)/ 2), int((y + h)/ 2))
            temp = cv2.rectangle(temp, p1, p2, [0., 0., 0.], -1)


        return temp / 255
        #return np.squeeze(self.full_img) / 255.


    def prepare_coordinates(self, bb):
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        for line in lines:
            if line is not None:
                values = line.split()
                self.objects_coordinates.append(((float(values[1]), float(values[2]),
                                                 float(values[3]), float(values[4]), int(float(values[0])))))


    # https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    def intersection_over_union(self, boxA, boxB):

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
        yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def non_max_suppression(self, boxes, conf_threshold=0.1, iou_threshold=0.1):
        """
        The function performs nms on the list of boxes:
        boxes: [box1, box2, box3...]
        box1: [x1, y1, x2, y2, Class, Confidence]
        """
        bbox_list_thresholded = []  # List to contain the boxes after filtering by confidence
        bbox_list_new = []  # List to contain final boxes after nms
        # Stage 1: (Sort boxes, and filter out boxes with low confidence)
        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[5])  # Sort boxes according to confidence
        for box in boxes_sorted:
            if box[5] > conf_threshold:  # Check if the box has a confidence greater than the threshold
                bbox_list_thresholded.append(box)  # Append the box to the list of thresholded boxes
            else:
                break
        # Stage 2: (Loop over all boxes, and remove boxes with high IOU)
        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)  # Remove the box with highest confidence
            bbox_list_new.append(current_box)  # Append it to the list of final boxes
            for box in bbox_list_thresholded:
                if True:  # Check if both boxes belong to the same class
                    iou = self.intersection_over_union(current_box[:4], box[:4])  # Calculate the IOU of the two boxes
                    if iou > iou_threshold:  # Check if the iou is greater than the threshold defined
                        bbox_list_thresholded.remove(box)  # If there is significant overlap, then remove the box
        return bbox_list_new

    def action_to_box(self, actions):
        pad = TASK_MODEL_RES / self.action_space
        x = pad * actions[0]
        y = pad * actions[1]

        w = pad * (actions[2] + 1)
        w = TASK_MODEL_RES - x if x + w > TASK_MODEL_RES else w

        h = pad * (actions[3] + 1)
        h = TASK_MODEL_RES - y if y + h > TASK_MODEL_RES else h
        return x, y, w, h

    def difficulty_up(self):
        self.difficulty += 0.1

    def bb_to_x2(self, img, bboxes):
        x, y, w, h = bboxes
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cropped_img = img[y:y + h, x:x + w]
        H, W, C = cropped_img.shape

        max_ = max(W, H)

        if H + W == 0.:
            cropped_img = np.zeros((64, 64, 3))
        pad_h = int((max_ - H) / 2)
        pad_w = int((max_ - W) / 2)
        cropped_img = cv2.copyMakeBorder(cropped_img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, None, value=0)

        cropped_img = cv2.resize(cropped_img, (32, 32))

        return cropped_img / 255.

    def bb_to_x(self, img, bboxes):
        x, y, w, h = bboxes
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cropped_img = img[y:y + h, x:x + w]

        cropped_img = cv2.resize(cropped_img, (64, 64))

        return cropped_img / 255.

    def take_action(self, actions, conf):

        reward = 0.
        self.nb_actions_taken += 1
        is_terminal = False

        agent_bbox = self.action_to_box(actions)
        max_i = 0
        for i, bbox in enumerate(self.bboxes):
            iou = self.intersection_over_union(agent_bbox, bbox)

            if iou > reward:
                reward = iou
                max_i = i

        x, y, w, h = agent_bbox
        self.history.append((x, y, w, h, 1, reward))

        X = self.bb_to_x(self.full_img, agent_bbox)

        if reward >= 0.4:
            x, y, w, h, label = self.objects_coordinates.pop(max_i)

            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            self.full_img = cv2.rectangle(self.full_img, p1, p2, [0., 0., 0.], -1)
            state_change = True
            Y = 1.
            Y_classification = label
            self.agent.add_to_ds(X, Y, label)
        elif reward < 0.1:
            #reward = 0.
            Y = 0.
            self.agent.add_to_ds(X, Y, 0)

        iou, cat_pred = self.agent.pred_class(X)

        self.add_to_history(agent_bbox, reward, cat_pred)
        #reward = iou

        S_prime = self.get_state()
        state_change = True

        if self.nb_actions_taken >= 50:
            is_terminal = True

        if len(self.objects_coordinates) <= 0:
            is_terminal = True

        return S_prime, reward, is_terminal, state_change

    def add_to_history(self, bbox, iou, label):
        x, y, w, h = bbox
        self.history.append((x, y, w, h, label, iou))

    def get_heat_map(self):

        bboxes = self.non_max_suppression(self.history)
        history_img = self.base_img.copy()

        for bb in bboxes:
            x, y, w, h, label, conf = bb
            print(label)
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            history_img = cv2.rectangle(history_img, p1, p2, [255., 0., 0.], 2)
            history_img = cv2.putText(history_img, str(label) + ' ' + str(conf), (int(x + w / 2), int(y + 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                [255, 0, 0], 1, cv2.LINE_AA)

        return history_img

    if __name__ == '__main__':
        # Pointing out a wrong IoU implementation in https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        boxA = [0., 0., 10., 10.]
        boxB = [9., 9., 11., 11.]

        correct = intersection_over_union(boxA, boxB)
        print(correct)
