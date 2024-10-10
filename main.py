import wandb
from Lightening_module import Segmentation
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from data_loader import dataset
from torchvision import transforms
from utils.helper import *
from torchmetrics.segmentation import GeneralizedDiceScore
import torch.nn as nn
import os
from utils import *
# from utils.helper import normalize
# from utils.AugmentationClass import JointTransformTrain, JointTransformTest
from torchvision.transforms import Compose, RandomRotation


# import pdb
# pdb.set_trace()

def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    # Local Computer Path
    root_dir = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData")
    batch_path = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData/batch.csv")
    # Lab Computer Path
    # root_dir = Path("C:\\Users\\aaron.l\\Documents\\data")
    # batch_path = Path("C:\\Users\\aaron.l\\Documents\\data\\batch.csv")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("CUDA_LAUNCH_BLOCKING =", os.environ.get('CUDA_LAUNCH_BLOCKING'))

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

    transform_test_image = Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Resize((128, 128)),
    ])
    transform_test_mask = Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])

    transform_train = JointTransformTrain(60, transform_image=transform_train_image,
                                          transform_mask=transform_train_mask)
    transform_test = JointTransformTest(transform_test_image, transform_test_mask)

    train_dataset = dataset.MRIDataset(phase='train', root_dir=root_dir, batch_dir=batch_path,
                                       transform=transform_train)
    test_dataset = dataset.MRIDataset(phase='test', root_dir=root_dir, batch_dir=batch_path, tranform=transform_test)

    batch_size = 2
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True)

    # Select GPU device for the training if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Current device:", device)
    else:
        device = torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03, weight_decay=1e-04)
    metrics = GeneralizedDiceScore(num_classes=2)
    loss_fn = nn.BCELoss()
    wandb_logger = WandbLogger(log_model=False, project="Tumor Segmentation")
    pl_model = Segmentation(model=model, optimizer=optimizer, loss_fn=loss_fn, metrics=metrics)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100, log_every_n_steps=100)

    trainer.fit(pl_model, train_dl, test_dl)
    wandb.finish()


if __name__ == '__main__':
    main()
