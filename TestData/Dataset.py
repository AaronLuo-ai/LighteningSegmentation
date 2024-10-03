from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from utils.helper import normalize
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, phase, root_dir, transform=None):
        self.phase = phase
        self.root_dir = root_dir
        if self.phase == 'train':
            self.label_path = self.root_dir / 'labels' / 'train_label'
            self.image_path = self.root_dir / 'images' / 'train_image'
        elif self.phase == 'test':
            self.label_path = self.root_dir / 'labels' / 'test_label'
            self.image_path = self.root_dir / 'images' / 'test_image'
        else:
            FileNotFoundError(f"Error: The specified path '{root_dir}' was not found.")

        self.label_array = []
        self.image_array = []
        for jpg_path in self.image_path.glob('*.jpg'):
            image = Image.open(jpg_path)
            label = Image.open(self.label_path / f"train-labels{str(jpg_path.stem[-2:])}.jpg")
            if transform is not None:
                image = transform(image)
                label = transform(label)
                self.image_array.append(image)
                self.label_array.append(label)

    def __getitem__(self, index):
        return np.expand_dims(self.image_array[index], axis=0), np.expand_dims(self.label_array[index], axis=0)

    def __len__(self):
        return len(self.label_array)

def test_dataset():
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.Lambda(lambda img: np.array(img)),
        transforms.Lambda(lambda img: normalize(img)),
    ])
    root_dir = Path("C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\ControlData")
    dataset = MRIDataset(phase = 'train', root_dir = root_dir, transform = transform)
    data_loader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=1)
    for batch_idx, data in enumerate(data_loader):
        test_image, test_mask = data

        for num in range(test_image.shape[0]):
            plt.figure(figsize=(10, 5))
            image = test_image[0].squeeze()
            label = test_mask[0].squeeze()
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title(f'Image size{ image.shape}')
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap='gray')
            plt.title(f'Mask size{ label.shape}')
            plt.show()

def main():
    test_dataset()


if __name__ == '__main__':
    main()