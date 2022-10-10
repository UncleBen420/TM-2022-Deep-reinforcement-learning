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
    def __init__(self, n_inputs, n_hidden_nodes=256, learning_rate=0.01, batch_size=128):
        super(DummyNET, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_inputs = n_inputs
        self.n_outputs = 7
        self.n_hidden_nodes = n_hidden_nodes
        self.learning_rate = learning_rate
        self.action_space = np.arange(self.n_outputs)
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()

        self.dummy_net = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_hidden_nodes),
            nn.ReLU()
        )
        self.dummy_net.to(self.device)

        self.Q = nn.Sequential(
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs)
        )
        self.Q.to(self.device)

        self.V = nn.Sequential(
            nn.Linear(self.n_hidden_nodes, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.V.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_no_grad(self, State):
        with torch.no_grad():
            x = self.dummy_net(State)
            return self.Q(x), self.V(x)

    def predict_with_grad(self, State):
        x = self.dummy_net(State)
        return self.Q(x), self.V(x)

    def split_dataset(self, dataset):
        return [dataset[x:x + self.batch_size] for x in range(0, len(dataset), self.batch_size)]

    def prepare_batch(self, batch):
        S = torch.stack([s for (s, a, r, s_prime, d) in batch]).to(self.device)
        A = torch.LongTensor([a for (s, a, r, s_prime, d) in batch]).to(self.device)
        R = torch.FloatTensor([r for (s, a, r, s_prime, d) in batch]).to(self.device)
        S_prime = torch.stack([s_prime for (s, a, r, s_prime, d) in batch]).to(self.device)
        done = torch.FloatTensor([d for (s, a, r, s_prime, d) in batch]).to(self.device)

        return S, A, R, S_prime, done

    def split_random(self, dataset):
        return [random.sample(dataset, self.batch_size)]

    def update(self, batches, gamma):
        counter = 0
        for batch in batches:

            if len(batch) < self.batch_size:
                print("yo")
                continue

            S, A, R, S_prime, done = self.prepare_batch(batch)
            print(S)
            print(A)
            print(S_prime)
            Q, V = self.predict_with_grad(S)
            Q_prime, V_prime = self.predict_no_grad(S_prime)

            Qy = R + gamma * (1 - done) * torch.max(Q_prime, 1)[0]
            print(Qy)
            print(Q)
            print("yo")
            Vy = R + gamma * (1 - done) * V_prime.squeeze()
            Qx = Q.gather(1, A.unsqueeze(1)).squeeze()
            Vx = V.squeeze(1)

            # Calculate the loss.
            loss_Q = self.loss_fn(Qx, Qy.detach())
            loss_V = self.loss_fn(Vx, Vy.detach())
            Q_running_loss = loss_Q.item()
            V_running_loss = loss_V.item()

            # Backpropagation.
            self.optimizer.zero_grad()

            loss_Q.backward(retain_graph=True)
            loss_V.backward()

            # Update the weights.
            self.optimizer.step()

            counter += 1

        # Loss for the complete epoch.
        return Q_running_loss / counter, V_running_loss / counter