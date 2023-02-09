import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

VISION_SIZE = 7
MODEL_RES = 16
BATCH_SIZE = 16
NUM_WORKERS = 4


class DummyNET(nn.Module):
    def __init__(self, n_inputs, n_hidden_nodes=64, learning_rate=0.001, batch_size=64):
        super(DummyNET, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_inputs = n_inputs
        self.n_outputs = 4
        self.n_hidden_nodes = n_hidden_nodes
        self.learning_rate = learning_rate
        self.action_space = np.arange(self.n_outputs)
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()

        self.dummy_net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.dummy_net.to(self.device)

        self.Q = nn.Sequential(
            nn.Linear(32, self.n_outputs)

        )
        self.Q.to(self.device)
        self.Q.apply(self.init_weights)
        self.dummy_net.apply(self.init_weights)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def predict_no_grad(self, State):
        with torch.no_grad():
            x = self.dummy_net(State.to(self.device))
            return self.Q(x)

    def predict_with_grad(self, State):
        x = self.dummy_net(State.to(self.device))
        return self.Q(x)

    def split_dataset(self, dataset):
        return [dataset[x:x + self.batch_size] for x in range(0, len(dataset), self.batch_size)]

    def prepare_batch(self, batch):
        S1 = torch.stack([s1 for (s1, a, r, s2, d) in batch])
        A = torch.LongTensor([a for (s1, a, r, s2, d) in batch]).to(self.device)
        R = torch.FloatTensor([r for (s1, a, r, s2, d) in batch]).to(self.device)
        S2 = torch.stack([s2 for (s1, a, r, s2, d) in batch])
        done = torch.FloatTensor([d for (s1, a, r, s2, d) in batch]).to(self.device)
        return S1, A, R, S2, done

    def update(self, dataset, gamma):
        counter = 0
        for batch in self.split_dataset(dataset):

            if len(batch) < self.batch_size:
                continue

            S, A, R, S_prime, done = self.prepare_batch(batch)
            self.optimizer.zero_grad()

            Q_prime = self.predict_no_grad(S_prime)
            Q = self.predict_with_grad(S)

            Qy = R + gamma * (1 - done) * torch.max(Q_prime, 1)[0]
            Qx = Q.gather(1, A.unsqueeze(1)).squeeze()

            # Calculate the loss.
            loss_Q = self.loss_fn(Qx, Qy.detach())
            Q_running_loss = loss_Q.item()

            loss_Q.backward()

            # Update the weights.
            self.optimizer.step()
            counter += 1

        # Loss for the complete epoch.
        return Q_running_loss / counter
