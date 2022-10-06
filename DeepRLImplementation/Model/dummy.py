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
MODEL_RES = 16
BATCH_SIZE = 16
NUM_WORKERS = 4


class DummyNET(nn.Module):
    def __init__(self, n_hidden_nodes=64, learning_rate=0.01):
        super(DummyNET, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_inputs = 1000
        self.n_outputs = 7
        self.n_hidden_nodes = n_hidden_nodes
        self.learning_rate = learning_rate
        self.action_space = np.arange(self.n_outputs)

        self.loss_fn = torch.nn.MSELoss()

        self.dummy_net = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(4, 8, 5),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Conv2d(8, 16, 7),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_hidden_nodes),
            nn.ReLU()
        )

        self.Q = nn.Sequential(
            nn.Linear(self.n_hidden_nodes + VISION_SIZE, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs)
        )

        self.V = nn.Sequential(
            nn.Linear(self.n_hidden_nodes + VISION_SIZE, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        if self.device == 'cuda':
            self.net.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, State):
        img, vision = State
        with torch.no_grad():
            mb_output = self.dummy_net(img.to(self.device))
            x = torch.cat((mb_output, vision.to(self.device)), dim=1)
            return self.Q(x), self.V(x)

    def predict_with_grad(self, State):
        img, vision = State
        mb_output = self.dummy_net(img.to(self.device))
        x = torch.cat((mb_output, vision.to(self.device)), dim=1)
        return self.Q(x), self.V(x)

    def prepare_batch(self, batch):
        dataset = MobileRLNetDataset(batch)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    def update(self, batch):
        dataloader = self.prepare_batch(batch)

        Q_running_loss, V_running_loss = 0., 0.

        counter = 0
        iters = len(dataloader)
        for i, data in enumerate(dataloader):
            counter += 1
            S, Qy, Vy, A = data
            Qy = Qy.to(self.device)
            Vy = Vy.to(self.device)

            # Reset the gradient
            self.optimizer.zero_grad()

            # Forward pass.
            Q, V = self.predict_with_grad(S)
            Qx = Q.gather(dim=1, index=A.long().unsqueeze(dim=1)).squeeze()
            Vx = V.squeeze()

            # Calculate the loss.
            loss_Q = self.loss_fn(Qx, Qy)
            loss_V = self.loss_fn(Vx, Vy)
            Q_running_loss += loss_Q.item()
            V_running_loss += loss_V.item()

            # Backpropagation.
            loss_V.backward(retain_graph=True)
            loss_Q.backward()


            # Update the weights.
            self.optimizer.step()

        # Loss for the complete epoch.
        return Q_running_loss / counter, V_running_loss / counter
