# datasets
from data_loader import test_dataset, train_dataset
from Lightening_module import Segmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import albumentations as A
import random

import os
from tqdm import tqdm
from collections import OrderedDict

import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from tabulate import tabulate

def main():
    batch_size = 3

    TrainDataset = train_dataset.TrainDataset()
    TestDataset = test_dataset.TestDataset()

    train_dl = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(TestDataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Select GPU device for the training if available
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

    arch = 'unet'
    enc_name = 'efficientnet-b0'
    classes = 1

    model = smp.create_model(arch=arch,  # Use U-Net architecture
                             encoder_name="resnet34",  # Use ResNet34 as the encoder
                             encoder_weights="imagenet",  # Initialize with pre-trained ImageNet weights
                             in_channels=3,  # Input is an RGB image (3 channels)
                             classes=2).to("cuda")  # Model predicts 2 classes and runs on a GPU

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    criterion = criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    cbs = pl.callbacks.ModelCheckpoint(dirpath=f'./checkpoints_{arch}',
                                       filename=arch,
                                       verbose=True,
                                       monitor='valid_loss',
                                       mode='min')

    pl_model = Segmentation(model, optimizer, criterion)
    trainer = pl.Trainer(callbacks=cbs, accelerator='gpu', max_epochs=2, auto_lr_find=True)
    trainer.fit(pl_model, train_dl, test_dl)

if __name__ == '__main__':
    main()