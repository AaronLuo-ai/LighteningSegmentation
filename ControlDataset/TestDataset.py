import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import nrrd
import pandas as pd
from torch.utils.data import DataLoader
import re
from glob import glob
from PIL import Image


total_rows = 28
split = (3 * total_rows) // 4  # This gives the split point
TestIndex = list(range(split + 1, total_rows))  # From split + 1 to total_rows - 1
TrainIndex = list(range(split))  # From 0 to split - 1

print("Test Index:", TestIndex)
print("Train Index:", TrainIndex)


class TrainDataset(Dataset):
    def __init__(self):
        self.root_dir = Path("C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\TestData")
        self.label_file = []
        self.image_file = []
        jpg_files = Path("C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\TestData").glob("*")
        print(jpg_files)  # Ensure that files are found
        print("TrainIndex elements type: ", type(TrainIndex[0]))
        print("TrainIndex[0]: ", TrainIndex[0])
        for path in jpg_files:
            # Get number
            # print(file_path)
            file_name = path.stem
            print("file_name type: ", type(file_name))
            print("file_name: ", file_name)
            last_part = file_name[-2:]
            print("last_part type: ", type(last_part))
            print("last_part: ", last_part)
            number = int(file_name[-2:])
            print("Number:", number)
            print("type of number:", type(number))
            print("file_path type:", type(file_name))
            print("file_name:", file_name)
            if number in TrainIndex:
                image_name = f"train-volume{number:02}.jpg"
                label_name = f"train-labels{number:02}.jpg"
                label_name = self.root_dir / "labels" / label_name
                image_name = self.root_dir / "images" / image_name
                img = Image.open(image_name)
                label = Image.open(label_name)
                img_array = np.array(img)
                label_array = np.array(label)
                print(f"Processing {file}, shape: {img_array.shape}")
                print(f"Processing {label}, shape: {label.shape}")
                self.label_file.append(label_array)
                self.image_file.append(img_array)

    def __getitem__(self, index):
        return image_file[index], label_file[index]

    def __len__(self):
        return len(self.image_file)

def test_dataset(batch_size=1):
    train_dataset = TrainDataset()
    print("len(data_loader): ", len(train_dataset))

    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, data in enumerate(data_loader):
        test_image, test_mask = data
        print(f"Batch {batch_idx + 1}")
        image = test_image[0, 0].cpu().numpy()  # Extract the first channel for visualization
        mask = test_mask[0, 0].cpu().numpy()  # Extract the first channel for visualization

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Image size{ image.shape}')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask size{ mask.shape}')

        plt.show()


def main():
    test_dataset(batch_size=1)


if __name__ == '__main__':
    main()
