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
        image, vision, g = self.trajectory[idx]
#        if torch.is_tensor(idx):
#            idx = idx.tolist()

        y = g

        image = transforms.functional.to_tensor(image)

        vision = torch.FloatTensor(vision)

        return image, vision, y