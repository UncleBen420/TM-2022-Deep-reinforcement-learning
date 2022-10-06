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
    def __init__(self, n_hidden_nodes=64, learning_rate=0.01, batch_size=32):
        super(DummyNET, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_inputs = 1000
        self.n_outputs = 7
        self.n_hidden_nodes = n_hidden_nodes
        self.learning_rate = learning_rate
        self.action_space = np.arange(self.n_outputs)
        self.batch_size = batch_size
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
        self.dummy_net.to(self.device)

        self.Q = nn.Sequential(
            nn.Linear(self.n_hidden_nodes + VISION_SIZE, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs)
        )
        self.Q.to(self.device)

        self.V = nn.Sequential(
            nn.Linear(self.n_hidden_nodes + VISION_SIZE, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.V.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_no_grad(self, State):
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

    def prepare_batch(self, dataset):
        #dataset = MobileRLNetDataset(batch)
        #return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        batch = random.sample(dataset, self.batch_size)

        I1 = torch.cat([i1 for (i1, v1, a, r, i2, v2, d) in batch])
        V1 = torch.cat([v1 for (i1, v1, a, r, i2, v2, d) in batch])
        A = torch.Tensor([a for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)
        R = torch.Tensor([r for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)
        I2 = torch.cat([i2 for (i1, v1, a, r, i2, v2, d) in batch])
        V2 = torch.cat([v2 for (i1, v1, a, r, i2, v2, d) in batch])
        done = torch.Tensor([d for (i1, v1, a, r, i2, v2, d) in batch]).to(self.device)

        return (I1, V1), A, R, (I2, V2), done


    def update(self, dataset, gamma):
        S, A, R, S_prime, done = self.prepare_batch(dataset)
        i, v = S

        Q, V = self.predict_no_grad(S)
        Q_prime, V_prime = self.predict_with_grad(S_prime)

        Qy = R + gamma * (1 - done) * torch.max(Q_prime, 1)[0]
        Vy = R + gamma * (1 - done) * V_prime.squeeze()
        Qx = Q.gather(dim=1, index=A.long().unsqueeze(dim=1)).squeeze()
        Vx = V.squeeze(1)

        #S, Qy, Vy, A = data
        #Qy = Qy.to(self.device)
        #Vy = Vy.to(self.device)

        # Reset the gradient
        #self.optimizer.zero_grad()

        # Forward pass.
        #Q, V = self.predict_with_grad(S)
        #Qx = Q.gather(dim=1, index=A.long().unsqueeze(dim=1)).squeeze()
        #Vx = V.squeeze()

        # Calculate the loss.
        loss_Q = self.loss_fn(Qx, Qy)
        loss_V = self.loss_fn(Vx, Vy)
        Q_running_loss = loss_Q.item()
        V_running_loss = loss_V.item()

        # Backpropagation.
        loss_V.backward(retain_graph=True)
        loss_Q.backward()


        # Update the weights.
        self.optimizer.step()

        # Loss for the complete epoch.
        return Q_running_loss, V_running_loss
