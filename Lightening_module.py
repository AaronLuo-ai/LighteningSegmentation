import pytorch_lightning as pl
import torch
import wandb


class Segmentation(pl.LightningModule):
    def __init__(self, model, optimizer, loss_fn, metrics):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metrics
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage: str):
        images, masks = batch
        masks = torch.clamp(masks, 0, 1)
        outputs = self.model(images)
        outputs = torch.sigmoid(outputs)
        loss = self.loss_fn(outputs, masks)
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        outputs = outputs.clamp(0, 1)
        masks.clamp(0, 1)
        dice_score = self.metric(preds=outputs.long(), target=masks.long())    # Error
        # print(f"stage: {stage}, dice_score: {dice_score}")
        # print(f"stage {stage} loss: {loss.item()}")

        self.log(f"{stage}/dice_score", dice_score)
        self.log(f"{stage}/loss", loss.item())
        table = wandb.Table(columns=["Image ID", "Overlay Image", "Mask", "Predicted"])

        # print("images shape", images.shape)
        # print("images.shape[0]: ", images.shape[0])
        # print("masks shape", masks.shape)
        # print("output shape", outputs.shape)
        images_np = images.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        masks_np = masks.cpu().detach().numpy()
        class_labels = {
            0: "background",
            1: "tumor"
        }
        for index in range(images_np.shape[0]):
        # for img, mask, pred in zip(images_np, masks_np, outputs_np):
            overlay_img = wandb.Image(images_np[index].squeeze(), masks={
                "ground_truth": {
                    "mask_data": masks_np[index].squeeze(),
                    "class_labels": class_labels
                },
                "predictions": {
                    "mask_data": outputs_np[index].squeeze(),
                    "class_labels": class_labels
                }
            })
            table.add_data(index, overlay_img, wandb.Image(masks_np[index].squeeze()), wandb.Image(outputs_np[index].squeeze()))
        wandb.log({f"batch_idx {batch_idx} Table": table})

    def training_step(self, batch, batch_idx):
        print("stage training")
        self._common_step(batch, batch_idx, stage='training')

    def validation_step(self, batch, batch_idx):
        print("stage validation")
        self._common_step(batch, batch_idx, stage='validation')

    def configure_optimizers(self):
        return self.optimizer