import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MobileRLNetDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, idx):
        image, vision, Qy, Vy, A = self.trajectory[idx]
        return (image[0], vision[0]), Qy, Vy, A