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
    def __init__(
        self, root_dir, batch_dir, target_size=(128, 128), phase="train", transform=None
    ):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.batch_dir = batch_dir
        self.phase = phase
        self.transform = transform

        # todo fix leakage
        csv = pd.read_csv(self.batch_dir)
        num_lines = len(csv)
        separation_index = int(0.75 * num_lines)
        if self.phase == "train":
            self.image_files = csv["Image"].tolist()[:separation_index]
            self.mask_files = csv["Mask"].tolist()[:separation_index]
        else:
            self.image_files = csv["Image"].tolist()[separation_index + 1 :]
            self.mask_files = csv["Mask"].tolist()[separation_index + 1 :]

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
        # print("shape of image inside dataloader: ", image.shape, "shape of mask inside dataloader: ", mask.shape)
        return image, mask

    def __len__(self):
        return len(self.images)


def main():
    root_dir = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData")
    batch_path = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData/batch.csv")
    root_dir = Path("C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple")
    batch_path = Path(
        "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv"
    )

    transform_train_mask = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_train_image = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_test_image = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_test_mask = Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize((128, 128)),
        ]
    )

    transform_train = JointTransformTrain(
        60, transform_image=transform_train_image, transform_mask=transform_train_mask
    )
    transform_test = JointTransformTest(transform_test_image, transform_test_mask)

    dataset = MRIDataset(
        phase="train", root_dir=root_dir, batch_dir=batch_path, transform=transform_test
    )
    # print("dataset length: ", len(dataset))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for batch_idx, (images, masks) in enumerate(
        data_loader
    ):  # Assuming the DataLoader returns (images, masks)
        print(f"Visualizing batch {batch_idx + 1}")

        # Loop through all images and masks in the batch
        for img_idx in range(images.size(0)):
            # Get the image and mask
            image = (
                images[img_idx].squeeze().permute(1, 2, 0).cpu().numpy()
            )  # Shape: [3, height, width]
            mask = (
                masks[img_idx].squeeze().permute(1, 2, 0).cpu().numpy()
            )  # Shape: [height, width] or [1, height, width]
            # Permute dimensions for the image (from [C, H, W] to [H, W, C])
            print(
                "After dataloader image shape and mask shape: ", image.shape, mask.shape
            )

            # Create a figure with two subplots for image and mask
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Display the image
            axes[0].imshow(image)
            axes[0].set_title(f"Image {img_idx + 1}")
            axes[0].axis("off")

            # Display the mask
            axes[1].imshow(mask)  # Assuming the mask is grayscale
            axes[1].set_title(f"Mask {img_idx + 1}")
            axes[1].axis("off")

            # Show the figure
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()


