import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import nrrd
import pandas as pd
from torch.utils.data import DataLoader

batch_path = Path("/Users/luozisheng/Documents/Zhu‘s Lab/PyTorch_Lightening/batch.csv")
df = pd.read_csv(batch_path)
total_rows = len(df)
separation_index = (3 * total_rows) // 4
test_df = df.iloc[separation_index + 1:]

class TestDataset(Dataset):
    def __init__(self):
        self.root_dir = Path("/Users/luozisheng/Documents/Zhu‘s Lab/PyTorch_Lightening/first_20_3D_resampled")
        self.test_image_files = test_df['Image'].tolist()
        self.test_mask_files = test_df['Mask'].tolist()
        self.target_size = (512, 512)
        self.mask_list = []
        self.image_list = []
        self.seg_list = []
        num_files = len(self.test_image_files)
        for index in range(num_files):
            test_nrrd_array , _ = nrrd.read(self.root_dir /  Path(self.test_image_files[index]))
            print(self.root_dir /  Path(self.test_image_files[index]))
            mask_nrrd_array, _ = nrrd.read(self.root_dir /  Path(self.test_mask_files[index]))
            print(self.root_dir /  Path(self.test_mask_files[index]))
            image_padded = self.pad_image_top_left(test_nrrd_array, self.target_size)
            mask_padded = self.pad_image_top_left(mask_nrrd_array, self.target_size)

            for i in range(image_padded.shape[0]):
                image_slice = image_padded[i, :, :]
                mask_slice = mask_padded[i, :, :]
                print("image_slice.shape: ", image_slice.shape)
                print("mask_slice.shape: ", mask_slice.shape)
                seg_image = image_slice * mask_slice
                seg_image = (seg_image - seg_image.min()) / (seg_image.max() - seg_image.min())
                mask_slice = (mask_slice - mask_slice.min()) / (mask_slice.max() - mask_slice.min())
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                self.seg_list.append(seg_image)
                self.mask_list.append(mask_slice)
                self.image_list.append(image_slice)

    def __getitem__(self, index):
        # test_image_name = Path(self.test_image_files[index])
        # test_mask_name = Path(self.test_mask_files[index])
        # test_image_path = self.root_dir / test_image_name
        # test_mask_path = self.root_dir / test_mask_name
        # test_nrrd_array, _ = nrrd.read(test_image_path)  # The first element is the data (NumPy array)
        # mask_nrrd_array, _ = nrrd.read(test_mask_path)    # The first element is the data (NumPy array)
        # train_padded = self.pad_image_top_left(test_nrrd_array, self.target_size)
        # mask_padded = self.pad_image_top_left(mask_nrrd_array, self.target_size)

        # print("test_image_path: ", test_image_path)
        # print("test_mask_path: ", test_mask_path)
        # print("type of test_nrrd_array: ", type(test_nrrd_array))
        # print("type of mask_nrrd_array: ", type(mask_nrrd_array))

        image = self.image_list[index]
        mask = self.mask_list[index]
        seg = self.seg_list[index]
        # print("image.shape in dataloader: ", image.shape)
        # print("mask.shape in dataloader: ", mask.shape)
        # print("image.shape in dataloader: ", image.shape)
        # print("mask.shape in dataloader: ", mask.shape)
        return np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0)


    def __len__(self):
        return len(self.mask_list)

    def pad_image_top_left(self, img_array, target_size):
        """Pads a NumPy array to the target size with the original image aligned to the top-left."""
        # print(img_array.shape)
        _, h, w = img_array.shape  # Get the current height and width of the image
        target_h, target_w = target_size
        pad_h = target_h - h  # Padding for the bottom
        pad_w = target_w - w  # Padding for the right
        padding = ((0, 0), (0, pad_h), (0, pad_w))  # No padding for the depth/channel dimension
        padded_img = np.pad(img_array, padding, mode='constant', constant_values=0)
        return padded_img

def test_test_loader(batch_size=1, num_workers=0):
    train_loader = TestDataset()
    data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("len(data_loader): ", len(data_loader))
    for batch_idx, data in enumerate(data_loader):
        test_image, test_mask = data
        print(f"Batch {batch_idx + 1}")
        # print(f"Train Image Shape: {train_image.shape}")
        # print(f"Train Massk Shape: {train_mask.shape}")
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
    test_test_loader(batch_size=1)


if __name__ == '__main__':
    main()
