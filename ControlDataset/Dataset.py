import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

# first load the images and labels according to the phase.
# create a list consisting pairs of image and labels ending in the same number.
# getitem can be implemented by returning the i-th element of the above list
# len() is the length of the array

class Dataset(Dataset):
    def __init__(self, phase, transform=None):
        self.phase = phase
        self.root_dir = Path("/Users/luozisheng/Documents/Zhu_lab/LighteningSegmentation/ControlData")
        if self.phase == 'train':
            self.label_path = self.root_dir / 'labels' / 'train_label'
            self.image_path = self.root_dir / 'images' / 'train_image'
        elif self.phase == 'test':
            self.label_path = self.root_dir / 'labels' / 'test_label'
            self.image_path = self.root_dir / 'images' / 'test_image'
        else:
            FileNotFoundError(f"Error: The specified path '{root_dir}' was not found.")
        print("self.root_dir", self.root_dir)
        print("self.label_path", self.label_path)
        print("self.image_path", self.image_path)
        self.array = []
        for jpg_path in self.image_path.glob('*.jpg'):
            print("jpg_path", jpg_path)
            image = Image.open(jpg_path)
            image_np = np.array(image)
            num = str(jpg_path.stem[-2:])
            print("num", num)
            tmp_path = self.label_path / f"train-labels{num}.jpg"
            label = Image.open(tmp_path)
            label_np = np.array(label)
            self.array.append((image_np, label_np))
        # self.label_array = []
        # for jpg_path in self.label_path.glob('*.jpg'):
        #     print("jpg_path", jpg_path)
        #     label = Image.open(jpg_path)
        #     label_np = np.array(label)
        #     self.label_array.append(label_np)

    def __getitem__(self, index):
        return self.array[index]
    def __len__(self):
        return len(self.array)

def test_dataset(batch_size=1):
    dataset = Dataset(phase='train')
    print("len(dataset): ", len(dataset))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, data in enumerate(data_loader):
        test_image, test_mask = data
        print(f"Batch {batch_idx + 1}")
        # image = test_image[0, 0].cpu().numpy()  # Extract the first channel for visualization
        # mask = test_mask[0, 0].cpu().numpy()  # Extract the first channel for visualization
        print(test_image.shape)
        print(test_mask.shape)
        print("type(test_image): ", type(test_image))
        print("type(test_mask): ", type(test_mask))
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(test_image.squeeze(), cmap='gray')
        plt.title(f'Image size{ test_image.shape}')

        plt.subplot(1, 2, 2)
        plt.imshow(test_mask.squeeze(), cmap='gray')
        plt.title(f'Mask size{ test_mask.shape}')

        plt.show()

def main():
    test_dataset(batch_size=1)


if __name__ == '__main__':
    main()