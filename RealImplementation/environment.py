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
import gc
from matplotlib import pyplot as plt

MODEL_RES = 64
HIST_RES = 255
ZOOM_DEPTH = 4


class PriorityQueue(object):
    """
    Implementation of a priority queue
    """
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        """
        :return: True if the queue is empty
        """
        return len(self.queue) == 0

    # for inserting an element in the queue
    def append(self, node, rank):
        """
        append an element to the queue in respect to the priority
        :param node: element to be added
        :param rank: priority of the element
        """
        self.queue.append([rank, node])

    # for popping an element based on Priority
    def pop(self):
        """
        delete the first element with respect of the priority and return it.
        :return: return the element with the most priority.
        """
        try:
            max_val_i = 0

            for i in range(len(self.queue)):
                if self.queue[i][0] >= self.queue[max_val_i][0]:
                    max_val_i = i

            return self.queue.pop(max_val_i)[1]
        except IndexError:
            print("error")
            exit()


class Node:
    """
    Node used in the environment.
    """
    def __init__(self, img, pos, parent, number):
        x, y, z = pos
        self.number = number
        self.childs = []
        self.parent = parent
        self.visited = False
        self.img = img
        self.resized_img = cv2.resize(img, (MODEL_RES, MODEL_RES)) / 255.
        self.x = x
        self.y = y
        self.z = z
        self.proba = None
        self.V = None
        self.nb_childs = 0

    def get_state(self):
        """
        return a state that the agent can use to predict an action.
        :return: a state in the format of an image
        """
        return np.array(self.resized_img.squeeze())

    def get_child(self, action, number):
        """
        in function of an action, return a child node.
        :param action: a number between 0 and 3
        :param number: the node number of the child
        :return: the new node
        """
        self.nb_childs += 1

        sub_z = self.z + 1

        h = int(self.img.shape[0] / 2)
        w = int(self.img.shape[1] / 2)

        if action == 0:
            i = 0
            j = 0
        elif action == 1:
            i = 1
            j = 0
        elif action == 2:
            i = 0
            j = 1
        else:
            i = 1
            j = 1

        x_ = self.x + (i * w)
        y_ = self.y + (j * h)

        return Node(self.img[h * j:h + h * j, w * i: w + w * i], (x_, y_, sub_z), self.number, number)


class Environment:
    """
    this class implement a probleme of "where is Waldo" the agent must find waldo in a minimal amount of steps.
    """

    def __init__(self, dataset_path, difficulty=0):
        self.current_node = None
        self.nb_good_choice = None
        self.nb_bad_choice = None
        self.Queue = None
        self.conventional_policy_nb_step = None
        self.full_img = None
        self.root = None
        self.sub_images_queue = None
        self.base_img = None
        self.gpu_full_img = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.history = None
        self.nb_actions_taken = 0
        self.zoom_padding = 2
        self.nb_action = 4
        self.difficulty = difficulty
        self.charlie = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, "waldo.png")), cv2.COLOR_BGR2RGB)
        self.image_list = sorted(os.listdir(os.path.join(dataset_path, "images")))
        self.mask_list = sorted(os.listdir(os.path.join(dataset_path, "masks")))
        self.dataset_path = dataset_path
        self.evaluation_mode = False
        self.action_space = np.arange(4)
        self.img_res = MODEL_RES

    def place_charlie(self):
        """
        this method place change the charlie's position on the map.
        """
        while True:
            x = random.randint(0, self.W - self.charlie.shape[1])
            y = random.randint(0, self.H - self.charlie.shape[0])
            if self.mask[y][x][0] == 0:
                self.charlie_x = x
                self.charlie_y = y
                self.full_img[self.charlie_y:self.charlie_y + self.charlie.shape[0],
                              self.charlie_x:self.charlie_x + self.charlie.shape[1]] = self.charlie
                break

        nb_line = self.W / self.min_res
        nb_col = int(x / self.min_res)
        last = int(y / self.min_res)
        self.conventional_policy_nb_step = nb_line * nb_col + last + 1

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history
        del self.sub_images_queue

        self.sub_images_queue = []
        self.history = []
        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.nb_max_zoom = 0
        self.nb_bad_choice = 0
        self.nb_good_choice = 0


        if self.difficulty == 1:
            del self.full_img
            self.full_img = self.base_img.copy()
            self.place_charlie()

        self.current_node = Node(self.full_img, (0, 0, 0), -1, self.nb_actions_taken)
        return self.current_node.get_state()

    def init_env(self):
        """
        This method is used to load the image representing the environment
        It place charlie on it has well
        """
        del self.full_img

        if self.evaluation_mode and self.difficulty > 2:
            img = os.path.join(self.dataset_path, "images", self.validation_image)
            mask = os.path.join(self.dataset_path, "masks", self.validation_mask)
        else:
            index = random.randint(0, len(self.image_list) - 1)
            img = os.path.join(self.dataset_path, "images", self.image_list[index])
            mask = os.path.join(self.dataset_path, "masks", self.mask_list[index])

        self.base_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.base_mask = cv2.imread(mask)

        self.H, self.W, self.channels = self.base_img.shape
        # check which dimention is the bigger
        max_ = np.max([self.W, self.H])
        # check that the image is divisble by 2
        if max_ % 2:
            max_ += 1

        self.base_img = cv2.copyMakeBorder(self.base_img, 0, max_ - self.H, 0,
                                           max_ - self.W, cv2.BORDER_CONSTANT, None, value=0)
        self.full_img = self.base_img.copy()

        self.base_mask = cv2.copyMakeBorder(self.base_mask, 0, max_ - self.H, 0, max_ - self.W, cv2.BORDER_CONSTANT,
                                            None, value=[255, 255, 255])
        self.mask = self.base_mask.copy()
        self.W = max_
        self.H = max_
        self.ratio = HIST_RES / self.H

        self.min_res = max_

        self.min_res = self.min_res >> (ZOOM_DEPTH + 1)

        self.place_charlie()
        self.compute_mask_map()
        self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES))
        self.heat_map = np.zeros((ZOOM_DEPTH + 1, HIST_RES, HIST_RES))
        self.V_map = np.full((ZOOM_DEPTH + 1, HIST_RES, HIST_RES), np.inf)

    def compute_mask_map(self):
        """
        Compute the coordinate at which Waldo can be placed.
        """
        pad = int(self.H / 100)
        h = int(self.H / pad)
        w = int(self.W / pad)

        mask_map = cv2.cvtColor(cv2.resize(self.mask, (h, w)), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), 'uint8')
        mask_map = cv2.erode(mask_map, kernel, iterations=2)

        _, mask_map = cv2.threshold(mask_map, 10, 255, cv2.THRESH_BINARY)
        self.ROI = np.array(np.where(mask_map == False))
        self.ROI[0, :] *= pad
        self.ROI[1, :] *= pad

    def record(self, x, y, z, window, V):
        """
        record a step in a cubic representation of the agent trajectory.
        :param x: x coordinate of the agent
        :param y: y coordinate of the agent
        :param z: z coordinate (level of zoom) of the agent
        :param window: size of the window.
        :param V: State value predicted by the agent.
        """
        ratio = HIST_RES / self.W
        x *= ratio
        y *= ratio
        window *= ratio
        x = int(x)
        y = int(y)
        window = int(window)
        self.heat_map[ZOOM_DEPTH + 1 - z][y:window + y, x: window + x] += 1
        self.V_map[ZOOM_DEPTH + 1 - z][y:window + y, x: window + x] = V

    def follow_policy(self, probs, epsilon=0.1):
        """
        Save the Q predicted by the agent in the node and return an action according the e-greedy function.
        :param epsilon: epsilon factor of the e-greedy function.
        :param probs: probability given by the agent.
        :return: an action.
        """
        p = random.random()
        if p < epsilon:
            idx = np.where(probs > -1000.)[0]
            A = random.choice(idx)
        else:
            A = np.argmax(probs)

        V = probs[A]
        probs[A] = -1000.
        self.current_node.proba = probs
        self.current_node.V = V
        return A

    def exploit(self, probs):
        """
        Save the Q predicted by the agent in the node and return the action with the most probability.
        :param probs: probability given by the agent.
        :return: an action.
        """
        A = np.argmax(probs)
        V = probs[A]
        probs[A] = -1000.
        self.current_node.proba = probs
        self.current_node.V = V
        return A

    def sub_img_contain_charlie(self, x, y, window):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """
        return ((x <= self.charlie_x < x + window) or (x <= self.charlie_x + self.charlie.shape[1] < x + window)) \
            and \
            ((y <= self.charlie_y < y + window) or (y <= self.charlie_y + self.charlie.shape[0] < y + window))

    def sub_image_contain_roi(self, x, y, window):
        """
        return if the agent position is on a position where Waldo can be.
        :param x: x coordinate of the agent.
        :param y: y coordinate of the agent.
        :param window: size of the agent window.
        :return: True if the agent is on a position where Waldo can be.
        """
        for i in range(self.ROI.shape[1]):
            if (x <= self.ROI[1][i] <= x * window or x <= self.ROI[1][i] <= x + window)\
                    and \
                    (y <= self.ROI[0][i] <= y + window or y <= self.ROI[0][i] <= y + window):
                return True
        return False

    def take_action(self, action):
        """
        allow the agent to take an action that influence the environment.
        :param action: action that the agent take
        :return: next state, reward, if the state is terminal, information on the given state and prior state prediction
                 if any.
        """
        reward = 0.
        self.nb_actions_taken += 1
        is_terminal = False

        parent_n = self.current_node.parent
        current_n = self.current_node.number

        child = self.current_node.get_child(action, self.nb_actions_taken)

        node_info = (parent_n, current_n, self.nb_actions_taken)

        if self.current_node.nb_childs < 4:
            self.pq.append(self.current_node, self.current_node.V)

        if self.current_node.z < ZOOM_DEPTH:
            self.pq.append(child, self.current_node.V)
        else:
            self.nb_max_zoom += 1

        if self.pq.isEmpty():
            is_terminal = True

        x = child.x
        y = child.y
        z = child.z

        if self.sub_image_contain_roi(x, y, child.img.shape[0]):
            self.nb_good_choice += 1
        else:
            self.nb_bad_choice += 1

        if z > ZOOM_DEPTH and self.sub_img_contain_charlie(x, y, child.img.shape[0]):
            reward = 100.
            is_terminal = True
        elif z > ZOOM_DEPTH:
            reward = -1.

        self.history.append((x, y, child.img.shape[0]))

        if self.evaluation_mode:
            self.record(x, y, z, child.img.shape[0], self.current_node.V)

        if not self.pq.isEmpty():
            self.current_node = self.pq.pop()

        S_prime = self.current_node.get_state()

        return S_prime, reward, is_terminal, node_info, self.current_node.proba

    def get_gif_trajectory(self, name):
        """
        This function allow the user to create a gif of all the moves the
        agent has made along the episodes
        :param name: the name of the gif file
        """
        frames = []
        ratio = HIST_RES / self.W
        hist_img = cv2.resize(self.full_img, (255, 255))
        for step in self.history:
            hist_frame = hist_img.copy()
            x, y, window = step
            x *= ratio
            y *= ratio
            window *= ratio
            x = int(x)
            y = int(y)
            window = int(window)

            hist_frame[y:window + y, x: window + x] = [255, 0, 0]
            frames.append(hist_frame)

        imageio.mimsave(name, frames, duration=0.2)
