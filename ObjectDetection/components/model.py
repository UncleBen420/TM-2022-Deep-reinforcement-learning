import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


class PolicyNet(nn.Module):
    def __init__(self, img_res=64, n_hidden_nodes=1024, n_kernels=32):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.img_res = img_res
        self.batch = None

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=n_kernels >> 3, kernel_size=(3, 3)),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(in_channels=n_kernels >> 3, out_channels=n_kernels >> 2, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(in_channels=n_kernels >> 2, out_channels=n_kernels >> 1, kernel_size=(3, 3)),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(5184, n_hidden_nodes),
            torch.nn.ReLU(),

        )

        self.IoU_estimator = torch.nn.Sequential(
            torch.nn.Linear(4 n_hidden_nodes >> 3, 1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.IoU_estimator.to(self.device)

        self.middle.apply(self.init_weights)
        self.IoU_estimator.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, images):
        images = torch.float(images)
        return images.permute(0, 3, 1, 2)

    def forward(self, img):
        x = self.backbone(img)
        x = self.middle(x)
        return self.IoU_estimator(x)

    def train(self):


