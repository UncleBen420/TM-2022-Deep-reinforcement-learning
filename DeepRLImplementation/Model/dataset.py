import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MobileRLNetDataset:
    """Tabular and Image dataset."""

    def __init__(self, size_max=1024, batch_size=16):
        self.image = []
        self.image_prime = []
        self.vision = []
        self.vision_prime = []
        self.rewards = []
        self.dones = []
        self.actions = []

        self.size_max = size_max
        self.batch_size = batch_size

    def __len__(self):
        return len(self.actions)

    def append(self, iter):
        S, A, R, S_prime, done = iter
        i1, v1 = S
        i2, v2 = S_prime
        self.image.append(i1)
        self.image_prime.append(i2)
        self.vision.append(v1)
        self.vision_prime.append(v2)
        self.rewards.append(R)
        self.dones.append(done)
        self.actions.append(A)

        if self.__len__():
            self.image.pop(0)
            self.image_prime.pop(0)
            self.vision.pop(0)
            self.vision_prime.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.actions.pop(0)

    def get_batch(self):
        idX = np.random.randint(2, self.__len__() - 1, size=self.batch_size)


        I1 = torch.cat(self.image[idX])
        V1 = torch.cat([v1 for (i1, v1, a, r, i2, v2, d) in batch])
        A = torch.Tensor([a for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)
        R = torch.Tensor([r for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)
        I2 = torch.cat([i2 for (i1, v1, a, r, i2, v2, d) in batch])
        V2 = torch.cat([v2 for (i1, v1, a, r, i2, v2, d) in batch])
        done = torch.Tensor([d for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)

        return (I1, V1), A, R, (I2, V2), done