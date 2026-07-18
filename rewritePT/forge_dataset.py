from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ForgeDataset(Dataset):
    def __init__(self, directory: str):
        self.pairs: list[tuple[int, str]] = []

        # saving label and images into pairs
        for image in Path(directory).glob("*.jpg"):
            label = int(image.stem.split("_")[0])
            img = str(image)

            self.pairs.append((label, img))

    # returning the number of pairs we have
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        image = self.pairs[item][1]
        _img = Image.open(image).convert('L')
        pixels = np.array(_img)

        # normalization
        normalized = pixels / 255
        tensor = torch.from_numpy(normalized).to(torch.float32)

        return tensor, self.pairs[item][0]
