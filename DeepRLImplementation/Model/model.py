import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torchvision.models as models

VISION_SIZE = 5
MOBILENET_RES = 224


class MobileRLNET(nn.Module):
    def __init__(self, n_hidden_nodes=32, learning_rate=0.01, device='cpu', fine_tune=True):
        super(MobileRLNET, self).__init__()

        self.device = device
        self.n_inputs = 1000
        self.n_outputs = 7
        self.n_hidden_nodes = n_hidden_nodes
        self.learning_rate = learning_rate
        self.action_space = np.arange(self.n_outputs)

        self.loss_fn = torch.nn.MSELoss()

        self.mb_net = models.mobilenet_v3_large()
        if fine_tune:
            print('[INFO]: Fine-tuning all layers...')
            for params in self.mb_net.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print('[INFO]: Freezing hidden layers...')
            for params in self.mb_net.parameters():
                params.requires_grad = False

        # Change the classification head.
        self.mb_net.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=n_hidden_nodes),
            nn.ReLU()
        )

        # Define policy head
        self.Q = nn.Sequential(
            nn.Linear(self.n_hidden_nodes + VISION_SIZE, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_outputs)
        )

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, vision, img):
        with torch.no_grad():
            mb_output = self.mb_net(img)
            x = torch.cat((mb_output, vision), dim=1)
            return self.Q(x)

    def update(self, batch):
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        iters = len(trainloader)
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            torch.cuda.empty_cache()
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
            # Backpropagation.
            loss.backward()
            # Update the weights.
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + i / iters)

        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = (train_running_correct / len(trainloader.dataset))
        return epoch_loss, epoch_acc
        pred = self.model(state1_batch)