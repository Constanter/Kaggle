import torch
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import os
from PIL import Image 
import matplotlib.pyplot as plt
from Variables import IMAGE_WIDTH, IMAGE_HEIGHT


class FishDataset(torch.utils.data.Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        super().__init__() 
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self, ):
        return len(self.labels)
        
    def __getitem__(self, index):
        input_image = self.images[index]
        label = self.labels[index]

        image = np.array(Image.open(input_image).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label    


class Data:
    def get_path(self, folder_path):
        """Function takes path folder,where place images,
        and return list of image paths"""
        path = Path(folder_path)
        path_images = list(path.glob('**/*.png'))
        images_paths = [str(path_image) for path_image in path_images if 'GT' not in str(path_image)]
        print(f'Number of images :{len(images_paths)}')
        labels = [os.path.split(os.path.split(name)[0])[1] for name in images_paths]
        print(f'Number of labels :{len(labels)}')
        classes = list(set(labels))
        labels_str_to_int = {label: i for i, label in enumerate(classes)}
        labels_int = [labels_str_to_int[str_label] for str_label in labels]
        return images_paths, labels_int, classes

    def get_transforms(self, ):
        train_transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
                ),
            ToTensorV2(),
            ],)

        val_transforms = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,),
            ToTensorV2(),
            ],)
        return train_transform, val_transforms

    def get_loaders(
        self,
        train_data,
        val_data,
        train_labels,
        val_labels,
        test_images,
        test_labels_int,
        batch_size,
        train_transform,
        test_transform,
        num_workers=4,
        pin_memory=True,
            ):
        train_ds = FishDataset(
            images=train_data,
            labels=train_labels,
            transform=train_transform,)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,)

        val_ds = FishDataset(
            images=val_data,
            labels=val_labels,
            transform=test_transform,)

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,)

        test_ds = FishDataset(
            images=test_images,
            labels=test_labels_int,
            transform=test_transform,)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,)

        return train_loader, val_loader, test_loader

    def show_batch(self, loader, batch_size, labels_dic):
        rows_number = 1
        cols_number = 2
        if batch_size > 2:
            cols_number = 4
            rows_number = batch_size//cols_number

        fig = plt.figure(figsize=(48, 30))
        images, labels = next(iter(loader))
        for i, data in enumerate(images, 1):
            ax = fig.add_subplot(rows_number, cols_number, i)
            plt.imshow(data.permute(1, 2, 0).numpy())
            ax.set_title(labels_dic[labels[i-1].item()])
        plt.show()
