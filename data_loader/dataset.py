from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import nrrd
from torchvision import transforms
from torchvision.transforms import Compose, Lambda
import sys
sys.path.append("..")
from utils.AugmentationClass import *
from utils.helper import *


class MRIDataset(Dataset):
    def __init__(self, root_dir, batch_dir, target_size=(128, 128), phase="train", transform=None):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.batch_dir = batch_dir
        self.phase = phase
        self.transform = transform

        # todo fix leakage
        csv = pd.read_csv(self.batch_dir)
        separation_index = 19
        if self.phase == "train":
            self.image_files = csv['Image'].tolist()[:separation_index]
            self.mask_files = csv['Mask'].tolist()[:separation_index]
        else:
            self.image_files = csv['Image'].tolist()[separation_index + 1:]
            self.mask_files = csv['Mask'].tolist()[separation_index + 1:]

        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        print(len(self.image_files))
        for index in range(len(self.image_files)):
            images, _ = nrrd.read(self.root_dir / self.image_files[index])
            masks, _ = nrrd.read(self.root_dir / self.mask_files[index])
            self.images.extend(images.astype(np.float32))
            self.masks.extend(masks.astype(np.float32))

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)


def main():
    root_dir = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData")
    batch_path = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData/batch.csv")

    transform_train_mask = Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Resize((128, 128)),
    ])
    transform_train_image = Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Resize((128, 128)),
    ])

    transform_train = JointTransformTrain(60, transform_image=transform_train_image,
                                          transform_mask=transform_train_mask)

    dataset = MRIDataset(phase='train', root_dir=root_dir, batch_dir=batch_path, transform=transform_train)
    print("dataset length: ", len(dataset))
    data_loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=1)
    for batch_idx, data in enumerate(data_loader):
        test_image, test_mask = data
        for num in range(test_image.shape[0]):
            plt.figure(figsize=(10, 5))
            image = test_image[0].squeeze()
            label = test_mask[0].squeeze()
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Image size{image.shape}')
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap='gray')
            plt.title(f'Mask size{label.shape}')
            plt.show()


if __name__ == "__main__":
    main()
