"""
This file contain the implementation of the real environment.
"""
import math
import os
import random
import re
from enum import Enum
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

class Tree:
    def __init__(self, img, pos):
        x, y, z = pos
        self.number = -1
        self.childs = []
        self.parent = None
        self.visited = False
        self.img = img
        self.resized_img = cv2.resize(img, (MODEL_RES, MODEL_RES)) / 255.
        self.x = x
        self.y = y
        self.z = z
        self.proba = None
        
    def add_child(self, child):
        child.parent = self
        self.childs.append(child)

    def get_parent_number(self):
        return -1 if self.parent is None else self.parent.number

    def get_state(self):
        imgs = []
        for child in self.childs:
            imgs.append(child.resized_img)

        ab = np.concatenate((imgs[0], imgs[1]), axis=0)
        cd = np.concatenate((imgs[2], imgs[3]), axis=0)
        state = np.concatenate((ab, cd), axis=1)
        return np.array(state.squeeze())

    def visit(self, nb_action):
        if self.visited:
            return
        self.resized_img = np.zeros((MODEL_RES, MODEL_RES, 3))
        self.number = nb_action
        del(self.img)
        self.img = None

        self.visited = True

    def get_childs(self, min_zoom):

        if len(self.childs) != 0:
            return

        if self.z - 1 < min_zoom:
            return

        sub_z = self.z - 1
        sub_x = self.x << 1
        sub_y = self.y << 1

        h = int(self.img.shape[0] / 2)
        w = int(self.img.shape[1] / 2)

        for i in range(2):
            for j in range(2):
                x_ = sub_x + i
                y_ = sub_y + j
                child = Tree(self.img[h * j:h + h * j, w * i: w + w * i], (x_, y_, sub_z))
                self.add_child(child)

    def is_leaf(self):
        is_leaf = True
        for child in self.childs:
            is_leaf *= child.visited
        return is_leaf

def check_cuda():
    """
    check if opencv can use cuda
    :return: return True if opencv can detect cuda. False otherwise.
    """
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0

MODEL_RES = 20
HIST_RES = 100
ZOOM_DEPTH = 4


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, dataset_path, nb_max_actions=100, difficulty=0):
        self.sub_images_queue = None
        self.base_img = None
        self.gpu_full_img = None
        self.charlie_y = 0
        self.charlie_x = 0
        self.history = None
        self.heat_map = np.zeros((ZOOM_DEPTH + 1, HIST_RES, HIST_RES))
        self.nb_actions_taken = 0
        self.nb_max_actions = nb_max_actions
        self.zoom_padding = 2
        self.z = 1
        self.x = 0
        self.y = 0
        self.nb_action = 4
        self.cv_cuda = check_cuda()
        self.difficulty = difficulty
        self.charlie = cv2.cvtColor(cv2.imread(os.path.join(dataset_path, "waldo.png")), cv2.COLOR_BGR2RGB)
        self.image_list = sorted(os.listdir(os.path.join(dataset_path, "images")))
        self.mask_list = sorted(os.listdir(os.path.join(dataset_path, "masks")))
        self.dataset_path = dataset_path
        self.evaluation_mode = False
        self.action_space = np.arange(4)


    def place_charlie(self):
        """
        this method place change the charlie's position on the map.
        """     
        while True:
            x = random.randint(0, self.W - 1)
            y = random.randint(0, self.H - 1)
            if self.mask[y][x][0] == 0:
                self.charlie_x = x
                self.charlie_y = y
                self.full_img[self.charlie_y:self.charlie_y + self.charlie.shape[0],
                self.charlie_x:self.charlie_x + self.charlie.shape[1]] = self.charlie
                break

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history
        del self.sub_images_queue
        self.sub_images_queue = []
        self.reward_tree = []
        self.history = []
        self.nb_actions_taken = 0
        self.nb_choice = 0
        self.nb_bad_choice = 0
        self.nb_good_choice = 0
        self.z = self.max_zoom
        self.x = 0
        self.y = 0
        self.nb_mark = 0


        if self.difficulty > 0:
            self.full_img = self.base_img.copy()
       	    self.place_charlie()
        
        if self.difficulty > 1:
       	    self.init_env()

        self.root = Tree(self.full_img, (self.x, self.y, self.z))
        self.current_node = self.root
        self.current_node.get_childs(self.min_zoom)
        self.current_node.visit(-1)
        S = self.current_node.get_state()
        return S

    def load_env(self):
        """
        This method read a image file and the mask (where waldo can be placed).
        :param img: image representing the environment.
        :param mask: image corresponding to the mask where waldo can be placed.
        """
        index = random.randint(0, len(self.image_list) - 1)
        img = os.path.join(self.dataset_path, "images", self.image_list[index])
        mask = os.path.join(self.dataset_path, "masks", self.mask_list[index])

        self.base_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.H, self.W, self.channels = self.base_img.shape
        self.ratio = HIST_RES / self.H
        self.base_mask = cv2.imread(mask)
        self.max_distance = math.sqrt(self.W ** 2 + self.H ** 2)
        min_dim = np.min([self.W, self.H])
        self.max_zoom = int(math.log(min_dim, 2))
        self.min_zoom = self.max_zoom - ZOOM_DEPTH

    def init_env(self):
        """
        This method is used to load the image representing the environment to the gpu
        It place charlie on the image has well
        """
        self.load_env()
        if self.difficulty > 1 and not self.evaluation_mode:
            angle = random.randint(0, 2)
            cv_angle = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            if self.cv_cuda:
                self.full_img = cv2.cuda.rotate(self.base_img.copy(), cv_angle[angle])
                self.mask = cv2.cuda.rotate(self.base_mask, cv_angle[angle])
            else:
                self.full_img = cv2.rotate(self.base_img.copy(), cv_angle[angle])
                self.mask = cv2.rotate(self.base_mask, cv_angle[angle])

            self.compute_mask_map()
            self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES))
            
        else:
            self.mask = self.base_mask
            self.full_img = self.base_img.copy()

        self.place_charlie()
        self.compute_mask_map()
        self.hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES))

    def compute_mask_map(self):
        """
        Compute the sub grid at the agent position given the x, y and z axis.
        """
        h = int(self.H / 100)
        w = int(self.W / 100)

        mask_map = cv2.cvtColor(cv2.resize(self.mask, (h, w)), cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), 'uint8')
        mask_map = cv2.erode(mask_map, kernel, iterations=2)

        _, mask_map = cv2.threshold(mask_map, 10, 255, cv2.THRESH_BINARY)
        self.ROI = np.array(np.where(mask_map == False))
        self.ROI[0, :] *= 100
        self.ROI[1, :] *= 100

    def record(self, x, y, z):
        window = self.zoom_padding << (z - 1)
        self.heat_map[z - self.min_zoom + 1][int(window * y * self.ratio):
                      int((window + window * y) * self.ratio),
        int(window * x * self.ratio):
        int((window + window * x) * self.ratio)] += 1

    def get_current_state_deep(self):
        """
        give to the agent 2 images (the sub image and the hist image). they are squeeze into
        a single array.
        :return: the current state.
        """

        return np.array(self.sub_vision.squeeze() / 255)

    def follow_policy(self, probs):
        A = np.random.choice(self.action_space, p=probs)
        p = probs[A]
        probs[A] = 0.
        giveaway = p / (np.count_nonzero(probs) + 0.00000001)
        probs[probs != 0.] += giveaway
        self.current_node.proba = probs
        return A

    def exploit(self, probs):
        A = np.argmax(probs)
        p = probs[A]
        probs[A] = 0.
        giveaway = p / (np.count_nonzero(probs) + 0.00000001)  
        probs[probs != 0.] += giveaway
        self.current_node.proba = probs
        return A

    def sub_img_contain_charlie(self, x, y, z):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """
        window = self.zoom_padding << (z - 1)
        charlie_w = self.charlie.shape[1] / 2
        charlie_h = self.charlie.shape[0] / 2
        return ((x * window <= self.charlie_x < x * window + window - charlie_w or
                x * window <= self.charlie_x + charlie_w < x * window + window)
                and
                (y * window <= self.charlie_y < y * window + window - charlie_h or
                y * window <= self.charlie_y + charlie_h < y * window + window))

    def sub_image_contain_roi(self, x, y, z):
        window = self.zoom_padding << (z - 1)
        roi_pad = 100
        for i in range(self.ROI.shape[1]):

            if ((x * window <= self.ROI[1][i] < x * window + window or
                 x * window <= self.ROI[1][i] + roi_pad < x * window + window)
                and
                (y * window <= self.ROI[0][i] < y * window + window or
                 y * window <= self.ROI[0][i] + roi_pad < y * window + window)):
                return True

        return False

    def take_action(self, action):

        reward = -1
        is_terminal = False
        proba = None

        if not self.current_node.childs[action].visited:
            self.current_node = self.current_node.childs[action]

        self.current_node.get_childs(self.min_zoom)
        self.current_node.visit(self.nb_actions_taken)

        x = self.current_node.x
        y = self.current_node.y
        z = self.current_node.z

        # Different Checks
        if self.sub_image_contain_roi(x, y, z):
            self.nb_good_choice += 1
        else:
            self.nb_bad_choice += 1
   
        if z <= self.min_zoom and self.sub_img_contain_charlie(x, y, z):
            reward = 10
            is_terminal = True 

        self.history.append((x, y, z))

        if self.evaluation_mode:
            self.record(x, y, z)

        #if the current node is a leaf we need to go up the tree
        while self.current_node.is_leaf():
            if self.current_node.parent is not None:
                self.current_node = self.current_node.parent 
                proba = self.current_node.proba
            else:
                is_terminal = True
                break

        S_prime = self.current_node.get_state()

        self.nb_actions_taken += 1

        return S_prime, reward, is_terminal, self.current_node.get_parent_number(), self.current_node.number, proba


    def get_gif_trajectory(self, name):
        """
        This function allow the user to create a gif of all the moves the
        agent has made along the episodes
        :param name: the name of the gif file
        """
        frames = []
        for hist in self.history:
            x, y, z = hist
            mm = self.hist_img.copy()

            color = [0, 255, 0]

            window = (self.zoom_padding ** z)
            mm[int(window * y * self.ratio):
               int((window + window * y) * self.ratio),
            int(window * x * self.ratio):
            int((window + window * x) * self.ratio)] = color

            frames.append(mm)

        imageio.mimsave(name, frames, duration=0.5)
