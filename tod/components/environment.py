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
from sklearn.cluster import MeanShift
from skimage.feature import hog
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

import gc
from matplotlib import pyplot as plt

TASK_MODEL_RES = 200
ANCHOR_AGENT_RES = 64


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

    def __init__(self, train_tod=False, record=False):
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
        self.difficulty = 0.
        self.train_tod = train_tod
        self.tod = None
        self.nb_classes = 4
        self.record = record
        self.steps_recorded = None

        self.colors = [[255, 0, 0],
                       [0, 0, 255],
                       [0, 255, 0],
                       [255, 255, 0],
                       [255, 0, 255]]

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
        self.steps_recorded = []
        # self.full_img, self.objects_coordinates = self.transform(self.full_img, self.objects_coordinates)
        self.heat_map = np.zeros((TASK_MODEL_RES, TASK_MODEL_RES))
        if self.train_tod:
            self.bboxes = []
            self.bboxes_y = []

        self.nb_actions_taken = 0
        return self.get_state()

    def reload_env_tod(self, index_bb):
        self.index_bb = index_bb
        self.nb_actions_taken_tod = 0
        self.steps_recorded = []
        return self.get_tod_state()

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        pad = int(ANCHOR_AGENT_RES / 2)
        self.full_img = cv2.copyMakeBorder(self.full_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, value=0)

        self.base_img = self.full_img.copy()

    def get_current_coord(self):
        x = int(self.nb_actions_taken % 20)
        y = int(self.nb_actions_taken / 20)
        pad = int((200) / 20)

        return x * pad, y * pad

    def get_state(self):

        x, y = self.get_current_coord()

        temp = self.full_img[y: y + ANCHOR_AGENT_RES,
               x: x + ANCHOR_AGENT_RES]

        return cv2.resize(temp, (ANCHOR_AGENT_RES, ANCHOR_AGENT_RES)) / 255.
        # return np.squeeze(self.full_img) / 255.

    def get_tod_state(self):
        bb_x, bb_y, bb_w, bb_h, _, _ = self.bboxes[self.index_bb]
        temp = self.base_img[bb_y: bb_h + bb_y, bb_x: bb_w + bb_x]
        return cv2.resize(temp, (ANCHOR_AGENT_RES, ANCHOR_AGENT_RES))

    def get_tod_visualisation(self):
        x, y, w, h, conf, label = self.bboxes[self.index_bb]
        return cv2.rectangle(self.base_img.copy(), (x, y), (x + w, y + h), self.colors[label], 2)

    def prepare_coordinates(self, bb):
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        pad = ANCHOR_AGENT_RES / 2
        for line in lines:
            if line is not None:
                values = line.split()

                x = int(float(values[1]) + float(values[3]) / 2 + pad)
                y = int(float(values[2]) + float(values[4]) / 2 + pad)

                self.objects_coordinates.append(((float(values[1]) + pad, float(values[2]) + pad,
                                                  float(values[3]), float(values[4]), int(float(values[0])), (x, y))))

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

    def non_max_suppression(self, boxes, conf_threshold=0.5, iou_threshold=0.1):

        bbox_list_thresholded = []  # List to contain the boxes after filtering by confidence
        bbox_list_new = []  # List to contain final boxes after nms

        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[4])  # Sort boxes according to confidence
        for box in boxes_sorted:
            if box[4] > conf_threshold * 100:  # Check if the box has a confidence greater than the threshold
                bbox_list_thresholded.append(box)  # Append the box to the list of thresholded boxes
            else:
                break
        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)
            bbox_list_new.append(current_box)
            for i, box in enumerate(bbox_list_thresholded):
                iou = self.intersection_over_union(current_box[:4], box[:4])  # Calculate the IOU of the two boxes
                if iou > iou_threshold:
                    bbox_list_thresholded.pop(i)
        return bbox_list_new

    def distance(self, centroid1, centroid2):
        dist = ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) // 2
        if dist == 0.:
            dist = 1
        dist = 1 / dist
        return dist

    def reduce(self):
        clustering = MeanShift(bandwidth=10).fit(self.bboxes)
        self.bboxes = []
        _, count = np.unique(clustering.labels_, return_counts=True)
        for i, bb in enumerate(clustering.cluster_centers_):
            if count[i] < 3:
                continue

            bb_x, bb_y = bb
            bb_x -= 16
            bb_y -= 16
            bb_w, bb_h = 32, 32
            self.bboxes.append((bb_x, bb_y, bb_w, bb_h, 0, 0))

        self.bboxes = np.array(self.bboxes, dtype=int)

    def take_action(self, action):

        reward = 0.
        self.nb_actions_taken += 1
        is_terminal = False

        x, y = self.get_current_coord()

        x += 32
        y += 32

        for i, bbox in enumerate(self.objects_coordinates):
            _, _, _, _, _, centroid = bbox

            dist = self.distance(centroid, (x, y))

            if dist > reward:
                reward = dist

        if action == 1 and reward >= 0.01:
            reward = 1.

        elif action == 0 and reward <= 0.01:
            reward = 0.1
        else:
            reward = 0.

        if action == 1:
            self.full_img = cv2.circle(self.full_img, (x, y), 5, [0., 0., 0.], -1)
            self.history.append((x, y, reward))
            if self.train_tod:
                self.bboxes.append((x, y))

        S_prime = self.get_state()

        if self.nb_actions_taken >= 400:
            is_terminal = True

        if self.record:
            self.steps_recorded.append(self.full_img.copy())

        return S_prime, reward, is_terminal

    def add_bbox_for_class(self):
        for i, bbox in enumerate(self.objects_coordinates):
            x, y, w, h, Y, _ = bbox

            x = int(x + random.randint(-5, 5))
            y = int(y + random.randint(-5, 5))
            w = int(w + random.randint(-5, 5))
            h = int(h + random.randint(-5, 5))

            temp = self.base_img[y: h + y, x: w + x]

            # temp = cv2.copyMakeBorder(temp, 0, pad_w, 0, pad_h, cv2.BORDER_CONSTANT, None, value=0)
            X = cv2.resize(temp, (ANCHOR_AGENT_RES, ANCHOR_AGENT_RES))

            self.tod.add_to_ds(X, Y)

    def take_action_tod(self, A, conf, label):
        is_terminal = False

        if self.record:
            self.steps_recorded.append(self.get_tod_visualisation())

        self.nb_actions_taken_tod += 1
        pad = 2

        agent_bbox = self.bboxes[self.index_bb]
        old_iou = 0.
        for bbox in self.objects_coordinates:
            x, y, w, h, _, _ = bbox

            iou = self.intersection_over_union((x, y, w, h), agent_bbox)
            if iou > old_iou:
                old_iou = iou

        if A == 0:
            self.bboxes[self.index_bb][2] += pad
        elif A == 1:
            self.bboxes[self.index_bb][3] += pad
        elif A == 2:
            if self.bboxes[self.index_bb][2] >= 30:
                self.bboxes[self.index_bb][2] -= pad
        elif A == 3:
            if self.bboxes[self.index_bb][3] >= 30:
                self.bboxes[self.index_bb][3] -= pad
        elif A == 4:
            if self.bboxes[self.index_bb][0] >= pad:
                self.bboxes[self.index_bb][0] -= pad
                self.bboxes[self.index_bb][2] += pad
        elif A == 5:
            if self.bboxes[self.index_bb][2] >= 30:
                self.bboxes[self.index_bb][0] += pad
                self.bboxes[self.index_bb][2] -= pad
        elif A == 6:
            if self.bboxes[self.index_bb][1] >= pad:
                self.bboxes[self.index_bb][1] -= pad
                self.bboxes[self.index_bb][3] += pad
        elif A == 7:
            if self.bboxes[self.index_bb][3] >= 30:
                self.bboxes[self.index_bb][1] += pad
                self.bboxes[self.index_bb][3] -= pad

        self.bboxes[self.index_bb][4] = int(conf * 100)
        self.bboxes[self.index_bb][5] = label

        if self.nb_actions_taken_tod >= 50:
            is_terminal = True

        agent_bbox = self.bboxes[self.index_bb]
        new_iou = 0.
        label = 0
        for bbox in self.objects_coordinates:
            x, y, w, h, current_label, _ = bbox

            iou = self.intersection_over_union((x, y, w, h), agent_bbox)
            if iou > new_iou:
                new_iou = iou
                label = current_label

        reward = (new_iou - old_iou) * 10.
        if reward < 0.:
            reward = 0.

        next_state = self.get_tod_state()

        if is_terminal:
            Y = None
            if new_iou > 0.7:
                Y = label
            if new_iou <= 0.:
                Y = self.nb_classes
            if Y is not None:
                self.tod.add_to_ds(next_state, Y)

        return next_state, reward, is_terminal

    def add_to_history(self, bbox, iou, label):
        x, y, w, h = bbox
        self.history.append((x, y, w, h, label, iou))

    def DOT_history(self):
        history_img = self.base_img.copy()
        for coord in self.history:
            x, y, dist = coord
            history_img = cv2.circle(history_img, (x, y), 5, [255., 0., 0.], 2)
        return history_img

    def TOD_history(self):

        bboxes = self.non_max_suppression(self.bboxes)
        history_img = self.base_img.copy()

        for bb in bboxes:
            x, y, w, h, conf, label = bb

            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            history_img = cv2.rectangle(history_img, p1, p2, self.colors[label], 2)
            history_img = cv2.putText(history_img, str(label) + " " + str(conf / 100.), (int(x + w / 2), int(y + 10)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.3,
                                      self.colors[label], 1, cv2.LINE_AA)

        return history_img

    if __name__ == '__main__':
        # Pointing out a wrong IoU implementation in https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        boxA = [0., 0., 10., 10.]
        boxB = [9., 9., 11., 11.]

        correct = intersection_over_union(boxA, boxB)
        print(correct)
