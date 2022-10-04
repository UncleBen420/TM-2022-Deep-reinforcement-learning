"""
This file implement a dummy environment to train the agents on and compare them. The term "Soft" mean that the
states of the environment are not linked to it's size (contrary to a grid world for exemple).
"""

import math
import random
from enum import Enum

import cv2
import re
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

from MobileNetV3.model import build_model

class Event(Enum):
    """
    this enum class simplify the different state of the grid
    """
    UNKNOWN = 0
    VISITED = 1
    MARKED = 2
    BLOCKED = 3


class Action(Enum):
    """
    this enum class represent all the action that the agent is allowed to do.
    """
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    ZOOM = 4
    DEZOOM = 5
    MARK = 6




def check_cuda():
    """
    check if opencv can detect cuda
    :return: return True if opencv can detect cuda. False otherwise.
    """
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0


class NormalisationMobileNet:
    """
    Class use to transform the input image in a normalised one.
    """
    def __init__(self):
        self.transforms = A.Compose([
            # A.Resize(resize_to, resize_to),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']




class SoftEnv:
    """
    this class implement the grid world problem as a frozen lake problem.
    """
    def __init__(self, model_resolution, item_model, surface_model, max_nb_actions=1000):
        self.bb_map = None
        self.marked_image = None
        self.max_zoom = 0
        self.nb_actions_taken = None
        self.channels = 0
        self.W = 0
        self.H = 0
        self.marked_map = None
        self.marked = None
        self.history = None
        self.full_img = None
        self.sub_img = None
        self.bboxes = None
        self.x = 0
        self.y = 0
        self.z = 1
        self.transform = NormalisationMobileNet()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: Models are using {0}'.format(self.device))
        self.min_zoom = 6
        self.model_resolution = model_resolution
        self.cv = cv2.cuda if check_cuda() else cv2
        self.states = np.arange(2 * 2 * (4 ** 6) * 3).reshape((2, 2, 4, 4, 4, 4, 4, 4, 3))
        self.pred_boat = 0
        self.pred_surf = 0
        self.nb_action = 7
        self.vision = np.zeros(7, dtype=int)
        self.nb_max_actions = max_nb_actions
        self.pad = 2

        self.item_model = build_model(False).to(self.device)
        checkpoint = torch.load(item_model, map_location=torch.device(self.device))
        self.item_model.load_state_dict(checkpoint['model_state_dict'])
        self.item_model.eval()

        self.surface_model = build_model(False).to(self.device)
        checkpoint = torch.load(surface_model, map_location=torch.device(self.device))
        self.surface_model.load_state_dict(checkpoint['model_state_dict'])
        self.surface_model.eval()

    def reload_env(self):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        del self.history
        del self.marked
        del self.marked_map

        self.history = []
        self.marked = []
        self.marked_map = np.zeros((self.W, self.H), dtype=bool)
        self.nb_actions_taken = 0
        self.z = self.min_zoom
        self.x = random.randint(0, int(self.W / (self.pad ** self.z) - 1))
        self.y = random.randint(0, int(self.H / (self.pad ** self.z) - 1))
        self.compute_sub_img()
        img = self.transform(self.sub_img)
        self.pred_boat = self.get_boat_prediction(img)
        self.pred_surf = self.get_surface_prediction(img)
        self.get_vision()

        return self.get_current_state()

    def init_env(self, image, labels):
        """
        need to be called to initialize the environment over a given image
        :param image: the image in high resolution which will be analysed by the agent.
        :param labels: the bounding boxes in YOLO format
        """
        self.full_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        self.H, self.W, self.channels = self.full_img.shape
        self.max_zoom = int(math.log(np.min([self.W, self.H]),self.pad))

        self.bb_map = np.zeros((self.H, self.W))
        with open(labels, "r") as file:
            lines = file.read()
        for i, line in enumerate(lines.split('\n')):
            if len(line.split(' ')) > 1:
                _, x_min, x_max, y_min, y_max = line.split(' ')
                x_min = int(float(x_min))
                x_max = int(float(x_max))
                y_min = int(float(y_min))
                y_max = int(float(y_max))

                self.bb_map[y_min:y_max, x_min:x_max] = i + 1

        self.marked_image = np.zeros((self.H, self.W, self.channels), dtype=np.uint8)
        self.marked_image[self.bb_map > 0] = [255, 0, 0]
        self.marked_image = self.cv.addWeighted(self.marked_image, 0.3, self.full_img, 0.7, 0)

    def compute_sub_img(self):
        """
        Get the current sub images given by the x, y and z axis.
        """
        window = self.pad ** self.z
        self.sub_img = self.full_img[window * self.y:window + window * self.y, window * self.x:window + window * self.x]
        self.sub_img = self.cv.resize(self.sub_img, (self.model_resolution, self.model_resolution))

    def transform(self, img):
        """
        Normalize and turn the images into a tensor.
        :param img:
        :return:
        """
        img_pil = Image.fromarray(img)
        return self.transform(img_pil)

    def get_prediction(self, model, img):
        """
        Get the prediction for the given model of the given image
        :param model: the model which analyse the image
        :param img: the image that will be analyse
        :return: the prediction (0 or 1)
        """
        with torch.no_grad():
            img = img.unsqueeze(0).to(self.device)
            outputs = model(img)
            _, preds = torch.max(outputs.data, 1)
            pred = preds.cpu().detach().numpy()[0]
        return pred

    def get_boat_prediction(self, img):
        """
        return True if a boat is detected.
        :param img: the image that will be analysed
        """
        return self.get_prediction(self.item_model, img)

    def get_surface_prediction(self, img):
        """
        return True if the sub image is on the water.
        :param img: the image that will be analysed
        """
        return self.get_prediction(self.surface_model, img)

    def get_current_state(self):
        """
        Return a number representing the current state of the environment.
        :return: the current state
        """
        return \
            self.states[self.pred_boat][self.pred_surf][self.vision[0]][self.vision[1]][
                self.vision[2]][
                self.vision[3]][self.vision[4]][self.vision[5]][self.vision[6]]

    def is_already_marked(self, position):
        """
        if the current location is already marked
        :param position: the current x, y, z axis
        """
        x, y, z = position
        window = self.pad ** z
        sub_mark = self.marked_map[window * y:window + window * y, window * x:window + window * x]
        return True if np.count_nonzero(sub_mark) else False

    def mark(self):
        """
        this method allow the agent to mark a position
        """
        window = self.pad ** self.z
        self.marked.append((self.x, self.y, self.z))
        self.marked_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y] = True

    def get_marked_percent(self):
        """
        return the percent of unmarked pixel boat left
        :return: a percent indicating the percent of boat unmarked
        """
        total_piece = np.count_nonzero(self.bb_map)
        marked_piece = np.count_nonzero(self.bb_map[self.marked_map])
        return (total_piece - marked_piece) / total_piece * 100

    def get_vision(self):

        move_set = [(self.x - 1, self.y, self.z),
                    (self.x + 1, self.y, self.z),
                    (self.x, self.y - 1, self.z),
                    (self.x, self.y + 1, self.z),
                    (self.x, self.y, self.z - 1),
                    (self.x, self.y, self.z + 1),
                    (self.x, self.y, self.z)]
        # check if a place had already been visited or marked
        for i in range(7):
            if move_set[i] in self.history:
                self.vision[i] = Event.VISITED.value
            elif move_set[i] in self.marked:
                self.vision[i] = Event.MARKED.value
            else:
                self.vision[i] = Event.UNKNOWN.value

        self.vision[0] = Event.BLOCKED.value if self.x <= 0 else self.vision[0]
        self.vision[1] = Event.BLOCKED.value if (self.x + 1) >= self.W / (
                self.pad ** self.z) else self.vision[1]

        self.vision[2] = Event.BLOCKED.value if self.y <= 0 else self.vision[2]
        self.vision[3] = Event.BLOCKED.value if (self.y + 1) >= self.H / (
                self.pad ** self.z) else self.vision[3]

        self.vision[4] = Event.BLOCKED.value if self.z - 1 < self.min_zoom else self.vision[4]
        self.vision[5] = Event.BLOCKED.value if self.z + 1 >= self.max_zoom else self.vision[5]

    def get_nb_state(self):
        return self.states.size

    def get_reward(self, action):
        reward = -1


        if action == Action.MARK and not self.marked[-1] in self.marked[:-1]:
            window = self.pad ** self.z
            marked = self.bb_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y]
            nb_boat_marked = np.count_nonzero(marked > 0)
            reward += nb_boat_marked / marked.size * 100

        elif action == Action.MARK:
            reward -= 100

        elif action == Action.ZOOM:
            window = self.pad ** self.z
            marked = self.bb_map[window * self.x:window + window * self.x, window * self.y:window + window * self.y]
            reward += 5 if np.count_nonzero(marked > 0) else 0

        if self.history[-1] in self.history[:-1]:
            reward -= 10

        return reward

    def take_action(self, action):
        action = Action(action)

        self.history.append((self.x, self.y, self.z))

        if action == Action.LEFT:
            self.x -= 1 if self.vision[0] != Event.BLOCKED.value else 0
        elif action == Action.UP:
            self.y -= 1 if self.vision[2] != Event.BLOCKED.value else 0
        elif action == Action.RIGHT:
            self.x += 1 if self.vision[1] != Event.BLOCKED.value else 0
        elif action == Action.DOWN:
            self.y += 1 if self.vision[3] != Event.BLOCKED.value else 0
        elif action == Action.ZOOM:
            self.z -= 1 if self.vision[4] != Event.BLOCKED.value else 0
        elif action == Action.DEZOOM:
            if self.vision[5] != Event.BLOCKED.value:
                self.x = int(self.x / self.pad)
                self.y = int(self.y / self.pad)
                self.z += 1
        elif action == Action.MARK:
            self.mark()

        self.compute_sub_img()
        self.get_vision()
        img = self.transform(self.sub_img)
        self.pred_boat = self.get_boat_prediction(img)
        self.pred_surf = self.get_surface_prediction(img)
        self.nb_actions_taken += 1
        is_terminal = self.nb_max_actions <= self.nb_actions_taken or self.get_marked_percent() < 5.

        return self.get_current_state(), self.get_reward(action), is_terminal



    def compute_boat_prediction_map(self):
        max_pad_y = int(self.W / self.model_resolution)
        max_pad_x = int(self.H / self.model_resolution)
        prediction_map = np.zeros_like(self.full_img)
        for i in range(max_pad_x):
            for j in range(max_pad_y):
                window = self.model_resolution
                img = self.full_img[window * j:window + window * j, window * i:window + window * i]
                img = self.cv.resize(img, (self.model_resolution, self.model_resolution))
                img = self.transform(img)
                color = [0, 255, 0] if self.get_boat_prediction(img) else [255, 0, 0]
                prediction_map[window * j:window + window * j, window * i:window + window * i] = color
        return self.cv.addWeighted(prediction_map, 0.3, self.full_img, 0.7, 0)

    def compute_surface_prediction_map(self):
        max_pad_y = int(self.W / self.model_resolution)
        max_pad_x = int(self.H / self.model_resolution)
        prediction_map = np.zeros_like(self.full_img)
        for i in range(max_pad_x):
            for j in range(max_pad_y):
                window = self.model_resolution
                img = self.full_img[window * j:window + window * j, window * i:window + window * i]
                img = self.cv.resize(img, (self.model_resolution, self.model_resolution))
                img = self.transform(img)
                color = [255, 0, 0] if self.get_surface_prediction(img) else [0, 255, 0]
                prediction_map[window * j:window + window * j, window * i:window + window * i] = color
        return self.cv.addWeighted(prediction_map, 0.3, self.full_img, 0.7, 0)

    def render_marked_map(self):
        marked_image = np.zeros((self.H, self.W, self.channels), dtype=np.uint8)
        marked_image[self.bb_map > 0] = [0, 255, 0]
        marked_image[self.marked_map] = [255, 0, 0]
        return self.cv.addWeighted(marked_image, 0.4, self.full_img, 0.6, 0)


