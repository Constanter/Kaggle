from PIL import Image
from torch.utils import data
import numpy as np


class LargeScaleFishDataset(data.Dataset):
    def __init__(self, inputs: list, targets: list, transform=None):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.transform = transform

    def __len__(self, ):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        input_image = self.inputs[idx]
        target_image = self.targets[idx]

        image = np.array(Image.open(input_image).convert("RGB"))
        mask = np.array(Image.open(target_image).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask