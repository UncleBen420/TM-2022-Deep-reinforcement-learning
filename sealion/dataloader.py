import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.feature import hog


class TOD_DataLoader(torch.utils.data.Dataset):
    def __init__(self, root, train=True, batch_y_max_size=20):
        self.root = root
        self.images = list(sorted(os.listdir(os.path.join(root, 'img'))))
        self.train = train
        self.batch_y_max_size = batch_y_max_size

    def __getitem__(self, idx):
        image_left = os.path.join(self.root, 'img', self.images[idx])
        image_left = Image.open(image_left).convert("RGB")
        coord = os.path.join(self.root, 'bboxes', self.images[idx].split('.')[0] + ".txt")
        file1 = open(coord, 'r')

        img = cv2.imread(os.path.join(self.root, 'img', self.images[idx]))

        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(5, 5),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        plt.imshow(hog_image)
        plt.show()

        Lines = file1.readlines()
        # Strips the newline character
        counter = 0
        for i, line in enumerate(Lines):
            counter += 1

        resize = transforms.Resize(size=(100, 100))
        image_left = resize(image_left)

        if self.train:

            # Random crop
            #i, j, h, w = transforms.RandomCrop.get_params(image_left, output_size=(100, 100))
            #image_left = TF.crop(image_left, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image_left = TF.hflip(image_left)

            # Random vertical flipping
            if random.random() > 0.5:
                image_left = TF.vflip(image_left)

        image_left = TF.to_tensor(image_left)
        #counter = torch.FloatTensor(counter)
        return image_left, np.float32(counter)

    def __len__(self):
        return len(self.images)
