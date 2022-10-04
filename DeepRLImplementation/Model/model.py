import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torchvision.models as models


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
            nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.n_hidden_nodes, self.n_outputs)
        )

        if self.device == 'cuda':
            self.net.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, vision, img):
        mb_output = self.mb_net(img)
        #HAPPEND /!\
        return self.Q(mb_output)

    def predict_action(self):
        if np.random.binomial(1, self.e):
            return random.randrange(self.n_outputs)
        return np.argmax(self.agent.Q[state])

    def update(self):
        pred = self.model(state1_batch)