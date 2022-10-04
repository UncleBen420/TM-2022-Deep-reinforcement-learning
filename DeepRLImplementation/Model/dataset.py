import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, idx):
        image, vision = self.trajectory[idx]
#        if torch.is_tensor(idx):
#            idx = idx.tolist()

        tabular = self.tabular.iloc[idx, 0:]

        y = tabular["price"]

        image = Image.open(f"{self.image_dir}/{tabular['zpid']}.png")
        image = np.array(image)
        image = image[..., :3]

        image = transforms.functional.to_tensor(image)

        tabular = tabular[["latitude", "longitude", "beds", "baths", "area"]]
        tabular = tabular.tolist()
        tabular = torch.FloatTensor(tabular)

        return image, tabular, y