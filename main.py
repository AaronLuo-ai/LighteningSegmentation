import wandb
from Lightening_module import Segmentation
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from data_loader import dataset
from torchmetrics.segmentation import GeneralizedDiceScore
from utils import *
from datetime import datetime
import os

# from utils.helper import normalize
# from utils.AugmentationClass import JointTransformTrain, JointTransformTest
from torchvision.transforms import Compose
from pytorch_lightning.callbacks import EarlyStopping

# from nnunet.network_architecture.generic_UNetPlusPlus import Generic_UNetPlusPlus
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

from model_zoo.unet import UNet


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    # root_dir = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData")
    # batch_path = Path("/Users/luozisheng/Documents/Zhu_lab/MRIData/batch.csv")
    root_dir = Path("C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple")
    batch_path = Path(
        "C:\\Users\\aaron.l\\Documents\\nrrd_images_masks_simple\\batch.csv"
    )
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # print("CUDA_LAUNCH_BLOCKING =", os.environ.get('CUDA_LAUNCH_BLOCKING'))

    input_size = (512, 512)

    transform_train_mask = Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize(input_size),
        ]
    )

    transform_train_image = Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize(input_size),
        ]
    )

    transform_test_image = Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),
            transforms.Resize(input_size),
        ]
    )

    transform_test_mask = Compose(
        [
            transforms.ToTensor(),
            # transforms.Lambda(grayscale_to_rgb),
            transforms.Lambda(normalize),  # No normalize originally
            transforms.Resize(input_size),
        ]
    )

    transform_train = JointTransformTrain(
        60, transform_image=transform_train_image, transform_mask=transform_train_mask
    )
    transform_test = JointTransformTest(transform_test_image, transform_test_mask)

    train_dataset = dataset.MRIDataset(
        phase="train",
        root_dir=root_dir,
        batch_dir=batch_path,
        transform=transform_train,
    )
    test_dataset = dataset.MRIDataset(
        phase="test", root_dir=root_dir, batch_dir=batch_path, transform=transform_test
    )

    batch_size = 5
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Current device:", device)
    else:
        device = torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

    # model_weight_path =Path("C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\model_zoo\\brain_unet.pt")
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    # model.load_state_dict(torch.load(model_weight_path, weights_only=True))
    # model = Generic_UNetPlusPlus(input_channels=3, base_num_features=64, num_classes=1, num_pool=3)
    # model.load_state_dict(torch.load("C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\model_zoo\\unet.pt"))

    optimizer = torch.optim.SGD(
        model.parameters(), lr=3e-3, momentum=0.9, weight_decay=0.0001
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)

    metrics = GeneralizedDiceScore(num_classes=2)
    early_stopping = EarlyStopping(
        monitor="validation/dice_score", patience=40, mode="max"
    )
    loss_fn = smp.losses.DiceLoss(mode="binary")
    run_name = f"segmentation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_batch={batch_size}"
    wandb_logger = WandbLogger(
        log_model=False, project="rectal-mri-Segmentation", name=run_name
    )

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all", log_freq=300, log_graph=False)

    pl_model = Segmentation(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler=scheduler,
        lr=3e-3,  # Automatically logged
        optimizer_type="SGD",  # Automatically logged
        scheduler_type="CosineAnnealingLR",  # Automatically logged
        encoder_name="resnet34",  # Automatically logged
        batch_size=batch_size,  # Additional custom hyperparameter
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=100,
        callbacks=[early_stopping],
        min_epochs=30,
        num_sanity_val_steps=0,
    )
    trainer.fit(pl_model, train_dl, test_dl)

    save_path = "C:\\Users\\aaron.l\\Documents\\LighteningSegmentation\\saved_models"
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
    torch.save(model.state_dict(), os.path.join(save_path, "unet_trained.pth"))

    wandb_logger.experiment.unwatch(model)
    wandb.finish()


if __name__ == "__main__":
    main()
