import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.dataset import MobileRLNetDataset

VISION_SIZE = 7
MOBILENET_RES = 224
BATCH_SIZE = 16
NUM_WORKERS = 4


class MobileRLNET(nn.Module):
    def __init__(self, n_hidden_nodes=32, learning_rate=0.01, fine_tune=True):
        super(MobileRLNET, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def predict(self, img, vision):
        with torch.no_grad():
            mb_output = self.mb_net(img)
            x = torch.cat((mb_output, vision), dim=1)
            return self.Q(x)

    def prepare_batch(self, batch):
        dataset = MobileRLNetDataset(batch)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def update(self, batch):
        dataloader = self.prepare_batch()

        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        iters = len(dataloader)
        with tqdm(enumerate(dataloader), unit="batch", total=len(dataloader)) as batches:
            for i, data in batches:
                counter += 1
                image, vision, y = data
                image = image.to(self.device)
                vision = vision.to(self.device)
                y = y.to(self.device)

                # Reset the gradient
                self.optimizer.zero_grad()

                # Forward pass.
                outputs = self.predict(image, vision)

                # Calculate the loss.
                loss = self.loss_fn(outputs, y)
                train_running_loss += loss.item()

                # Backpropagation.
                loss.backward()

                # Update the weights.
                self.optimizer.step()
                batches.set_postfix(loss=train_running_loss)

        # Loss for the complete epoch.
        return train_running_loss / counter
