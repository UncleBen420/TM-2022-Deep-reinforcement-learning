"""
This file contain the implementation of the real environment.
"""

import re
import cv2
import numpy as np

TASK_MODEL_RES = 200
ANCHOR_AGENT_RES = 64

class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, train_tod=False, record=False):
        self.bboxes = None
        self.nb_actions_taken_tod = None
        self.current_bbox = None
        self.history = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.full_img = None
        self.base_img = None
        self.nb_actions_taken = 0
        self.difficulty = 0.
        self.train_tod = train_tod
        self.tod = None
        self.nb_classes = 4
        self.record = record
        self.steps_recorded = None
        self.iou_final = []
        self.iou_base = []
        self.eval_tod =False

        # TOD metric
        self.tod_rewards = []
        self.tod_class_loss = []
        self.tod_conf_loss = []
        self.tod_policy_loss = []

        self.colors = [[255, 0, 0],
                       [0, 0, 255],
                       [0, 255, 0],
                       [255, 255, 0],
                       [255, 0, 255]]

        self.truth_values = []
        self.predictions = []

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

        self.bboxes = []

        self.nb_actions_taken = 0
        return self.get_state()

    def reload_env_tod(self, bb):

        bb_x, bb_y = bb
        bb_x -= 48
        bb_x = bb_x if bb_x >= 0 else 0
        bb_y -= 48
        bb_y = bb_y if bb_y >= 0 else 0
        bb_w, bb_h = 32, 32

        if bb_x + bb_w >= 200:
            bb_x = 168

        if bb_y + bb_h >= 200:
            bb_y = 168

        self.current_bbox = {
            'x': bb_x,
            'y': bb_y,
            'w': bb_w,
            'h': bb_h,
            'conf': 0.,
            'label': 0.
        }

        self.nb_actions_taken_tod = 0

        return self.get_tod_state()

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        pad = int(ANCHOR_AGENT_RES / 2)
        self.base_img = self.full_img.copy()
        self.full_img = cv2.copyMakeBorder(self.full_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    def get_current_coord(self):
        x = int(self.nb_actions_taken % 10)
        y = int(self.nb_actions_taken / 10)
        pad = int((200) / 10)

        return x * pad, y * pad

    def get_state(self):

        x, y = self.get_current_coord()

        temp = self.full_img[y: y + ANCHOR_AGENT_RES,
               x: x + ANCHOR_AGENT_RES]

        temp = cv2.resize(temp, (ANCHOR_AGENT_RES, ANCHOR_AGENT_RES)) / 255.
        return temp

    def get_tod_state(self):
        temp = self.base_img[self.current_bbox['y']: self.current_bbox['y'] + self.current_bbox['h'],
                             self.current_bbox['x']: self.current_bbox['x'] + self.current_bbox['w']]
        temp = cv2.resize(temp, (ANCHOR_AGENT_RES, ANCHOR_AGENT_RES)) / 255.
        return temp

    def get_tod_visualisation(self):
        return cv2.rectangle(self.base_img.copy(), (self.current_bbox['x'], self.current_bbox['y']),
                             (self.current_bbox['x'] + self.current_bbox['w'],
                              self.current_bbox['y'] + self.current_bbox['h']),
                             self.colors[int(self.current_bbox['label'])], 2)

    def prepare_coordinates(self, bb):
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        pad = ANCHOR_AGENT_RES / 2
        for line in lines:
            if line is not None:
                values = line.split()

                x = int(float(values[1]) + float(values[3]) / 2 + pad)
                y = int(float(values[2]) + float(values[4]) / 2 + pad)

                self.objects_coordinates.append(((float(values[1]), float(values[2]),
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

    def non_max_suppression(self, boxes, conf_threshold=0., iou_threshold=0.1):

        bbox_list_thresholded = []  # List to contain the boxes after filtering by confidence
        bbox_list_new = []  # List to contain final boxes after nms

        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[4])  # Sort boxes according to confidence
        for box in boxes_sorted:
            if box[4] >= conf_threshold:  # Check if the box has a confidence greater than the threshold
                bbox_list_thresholded.append(box)  # Append the box to the list of thresholded boxes
            else:
                break
        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)
            bbox_list_new.append(current_box)
            for box in bbox_list_thresholded:
                iou = self.intersection_over_union(current_box[:4], box[:4])  # Calculate the IOU of the two boxes
                if iou > iou_threshold:
                    bbox_list_thresholded.remove(box)

        return bbox_list_new

    def distance_euclidian(self, centroid1, centroid2):
        return ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) // 2

    def take_action(self, action):

        x, y = self.get_current_coord()
        x += 32
        y += 32

        if action:
            self.history.append((x, y, 0., action))

        label_max = 0
        dist_min = 1000.
        max_bbox = None

        is_already_marked = False
        for agent_bb in self.bboxes:
            abb_x, abb_y, abb_w, abb_h, label, centroid = agent_bb
            is_already_marked += abb_x <= x - 32 <= abb_x + abb_w and abb_y <= y - 32 <= abb_y + abb_h

        if not is_already_marked:
            for i, bbox in enumerate(self.objects_coordinates):

                bb_x, bb_y, bb_w, bb_h, label, centroid = bbox
                in_bbox = bb_x <= x - 32 <= bb_x + bb_w and bb_y <= y - 32 <= bb_y + bb_h
                dist = self.distance_euclidian(centroid, (x, y))

                if in_bbox and dist < dist_min:
                    dist_min = dist
                    label_max = label
                    max_bbox = (bb_x, bb_y, bb_w, bb_h)

        if action == 1 and label_max == 1:
            reward = 1.
        elif action == 0 and label_max == 0:
            reward = 0.5
        else:
            reward = 0.

        # --------------------------------------------------------------------------------------------------------------
        # CALLING TOD IF TARGET IS DETECTED
        # --------------------------------------------------------------------------------------------------------------
        if action and (self.train_tod or self.eval_tod):
            first_state_tod = self.reload_env_tod((x, y))
            if self.train_tod:
                bonus_iou, reward_tod, loss_policy, loss_class, loss_conf = self.tod.fit_one_episode(first_state_tod)
                self.tod_rewards.append(reward_tod)
                self.tod_conf_loss.append(loss_conf)
                self.tod_class_loss.append(loss_class)
                self.tod_policy_loss.append(loss_policy)
            else:
                bonus_iou, _ = self.tod.exploit_one_episode(first_state_tod)

            cv2.rectangle(self.full_img, (self.current_bbox['x'] + 32, self.current_bbox['y'] + 32),
                          (self.current_bbox['x'] + 32 + self.current_bbox['w'],
                           self.current_bbox['y'] + 32 + self.current_bbox['h']), [0, 0, 0], -1)

            reward += bonus_iou

            self.bboxes.append((self.current_bbox['x'], self.current_bbox['y'],
                                self.current_bbox['w'], self.current_bbox['h'],
                                self.current_bbox['conf'], self.current_bbox['label']))

        self.nb_actions_taken += 1
        is_terminal = False

        x, y = self.get_current_coord()
        x += 32
        y += 32

        S_prime = self.get_state()

        if self.nb_actions_taken >= 100:
            is_terminal = True

        if self.record:
            self.steps_recorded.append(self.DOT_history())

        return S_prime, reward, is_terminal


    def get_iou_error(self):
        error = 0.

        for agent_bbox in self.bboxes:
            max_iou = 0.
            for bbox in self.objects_coordinates:
                x, y, w, h, _, _ = bbox
                iou = self.intersection_over_union((x, y, w, h), agent_bbox)
                if iou > max_iou:
                    max_iou = iou

            error += 1. - max_iou
        return error / len(self.bboxes)

    def take_action_tod(self, A, conf, label_pred):
        is_terminal = False

        self.nb_actions_taken_tod += 1
        pad = 3

        agent_bbox = (self.current_bbox['x'], self.current_bbox['y'],
                      self.current_bbox['w'], self.current_bbox['h'])
        old_iou = 0.
        for bbox in self.objects_coordinates:
            x, y, w, h, _, _ = bbox

            iou = self.intersection_over_union((x, y, w, h), agent_bbox)
            if iou > old_iou:
                old_iou = iou

        if self.record:
            if self.nb_actions_taken_tod == 1:
                self.iou_base.append(old_iou)

        if A == 0:
            if self.current_bbox['x'] + self.current_bbox['w'] < 200 - pad:
                self.current_bbox['w'] += pad
        elif A == 1:
            if self.current_bbox['y'] + self.current_bbox['h'] < 200 - pad:
                self.current_bbox['h'] += pad
        elif A == 2:
            if self.current_bbox['w'] >= 32:
                self.current_bbox['w'] -= pad
        elif A == 3:
            if self.current_bbox['h'] >= 32:
                self.current_bbox['h'] -= pad
        elif A == 4:
            if self.current_bbox['x'] >= pad:
                self.current_bbox['x'] -= pad
                self.current_bbox['w'] += pad
        elif A == 5:
            if self.current_bbox['x'] + self.current_bbox['w'] < 200 - pad:
                self.current_bbox['x'] += pad
                self.current_bbox['w'] -= pad
        elif A == 6:
            if self.current_bbox['y'] >= pad:
                self.current_bbox['y'] -= pad
                self.current_bbox['h'] += pad
        elif A == 7:
            if self.current_bbox['y'] + self.current_bbox['h'] < 200 - pad:
                self.current_bbox['y'] += pad
                self.current_bbox['h'] -= pad

        if self.nb_actions_taken_tod >= 25:
            is_terminal = True

        self.current_bbox['conf'] = conf
        self.current_bbox['label'] = label_pred

        agent_bbox = (self.current_bbox['x'], self.current_bbox['y'],
                      self.current_bbox['w'], self.current_bbox['h'])
        new_iou = 0.
        label = 0
        for bbox in self.objects_coordinates:
            x, y, w, h, current_label, _ = bbox

            iou = self.intersection_over_union((x, y, w, h), agent_bbox)
            if iou > new_iou:
                new_iou = iou
                label = current_label

        reward = (new_iou - old_iou)
        if reward < 0.:
            reward = 0.

        next_state = self.get_tod_state()

        if self.record:

            self.steps_recorded.append(self.get_tod_visualisation())

            if is_terminal:
                self.iou_final.append(new_iou)
                if new_iou <= 0.:
                    Y = self.nb_classes
                else:
                    Y = label

                self.truth_values.append(Y)
                self.predictions.append(label_pred)

        if old_iou <= 0.:
            label = self.nb_classes

        return next_state, reward, is_terminal, old_iou, label

    def DOT_history(self):
        history_img = self.base_img.copy()
        for coord in self.history:
            x, y, dist, label = coord
            history_img = cv2.circle(history_img, (x - 32, y - 32), 5, self.colors[label], 2)

        return history_img

    def TOD_history(self):

        bboxes = self.non_max_suppression(self.bboxes)
        history_img = self.base_img.copy()

        for bb in bboxes:
            x, y, w, h, conf, label = bb

            if label == self.nb_classes:
                continue

            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            history_img = cv2.rectangle(history_img, p1, p2, self.colors[label], 2)
            history_img = cv2.putText(history_img,
                                      str(label) + ' ' + str(round(conf, 2)),
                                      (int(x + 5), int(y + 10)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.3,
                                      self.colors[label], 1, cv2.LINE_AA)

        return history_img
