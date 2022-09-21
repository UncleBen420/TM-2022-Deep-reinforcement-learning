import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class DOTA(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root,'images'))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label_filename = self.imgs[idx].split('.')[0] + '.txt'
        label_path = os.path.join(self.root, 'labels', label_filename)

        lines = []
        with open(label_path) as file:
            lines = file.readlines()

        # get bounding box coordinates for each mask
        num_objs = len(lines)
        boxes = []
        for line in lines:
            data = np.array(line.split(' '), dtype=float).astype(int)
            boxes.append([data[1], data[2], data[3], data[4]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = []
        if num_objs != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes,
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd
                  }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)