from torch import nn
from torch.autograd import Variable

MODEL_RES = 100

import argparse
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from dataloader import TOD_DataLoader

class TinyObjectDetection(nn.Module):
    def __init__(self, img_res=100, nb_classes=10, learning_rate=0.001, batch_max=20):
        super(TinyObjectDetection, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.img_res = img_res
        self.batch_max=batch_max

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=9),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def model_summary(self):
        print("RUNNING ON {0}".format(self.device))
        print(self)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.parameters())))

    def forward(self, X):
        x = self.backbone(X)
        return x

    def predict(self, X):
        X = X.unsqueeze(0)
        with torch.no_grad():
            x = self.backbone(X)
        return x.item()

    def update(self, X, Y):
        self.optimizer.zero_grad()
        preds = self.forward(X)
        loss = torch.nn.functional.mse_loss(preds.squeeze(), Y)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()

        return loss.item()


def train(path_train, path_test, num_epochs):
    # use our dataset and defined transformations
    dataset = TOD_DataLoader(path_train)
    # dataset_test = TOD_DataLoader(path_test, train=False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    #data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and boat
    num_classes = 2

    # get the model using our helper function
    model = TinyObjectDetection()
    # move model to the right device
    model.to(device)

    model.model_summary()

    history = []
    for epoch in range(num_epochs):
        sum_loss = 0.
        for images, masks in data_loader:
            X = list(image.to(device) for image in images)
            Y = list(mask.to(device) for mask in masks)
            X = torch.stack(X, dim=0)
            Y = torch.stack(Y, dim=0)

            sum_loss += model.update(X, Y)
        history.append(sum_loss)
        print(sum_loss)

    plt.plot(history)
    plt.show()

    for images, masks in data_loader:
        X = images[0].to(device)
        pred = model.predict(X)
        print(pred)
        img = np.transpose(X.cpu().numpy(), (1, 2, 0))
        img = (img * 255).astype(dtype=np.uint8)
        plt.imshow(img)
        plt.show()
