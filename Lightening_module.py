from typing import Any
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
        self.training_step_inputs = []
        self.training_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.training_step_targets = []  # save targets in each batch to compute metric overall epoch
        self.val_step_inputs = []
        self.val_step_outputs = []  # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []  # save targets in each batch to compute metric overall epoch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = torch.clamp(masks, 0, 1)
        predictions = torch.sigmoid(self.model(images))
        loss = self.loss_fn(predictions, masks)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        predicted_masks = predictions.clone()
        predicted_masks[predicted_masks > 0.5] = 1
        predicted_masks[predicted_masks <= 0.5] = 0
        predicted_masks = predicted_masks.clamp(0, 1)
        masks.clamp(0, 1)
        self.training_step_outputs.extend(predicted_masks)
        self.training_step_targets.extend(masks)
        self.training_step_inputs.extend(images)
        return loss

    def on_train_epoch_end(self, *arg, **kwargs):
        predicted_masks = torch.stack(self.training_step_outputs[-5:])
        masks = torch.stack(self.training_step_targets[-5:])
        images = torch.stack(self.training_step_inputs[-5:])
        dice_score = self.metric(preds=predicted_masks.long(), target=masks.long())
        self.log("training dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        images_np = images.cpu().detach().numpy()
        predicted_masks_np = predicted_masks.cpu().detach().numpy()
        masks_np = masks.cpu().detach().numpy()
        table = wandb.Table(columns=["Image ID", "Training Overlay Image", "Training Mask", "Predicted"])
        class_labels = {0: "background", 1: "tumor"}
        for index in range(images_np.shape[0]):
            overlay_img = wandb.Image(images_np[index].squeeze(), masks={
                "ground_truth": {
                    "mask_data": masks_np[index].squeeze(),
                    "class_labels": class_labels
                },
                "predictions": {
                    "mask_data": predicted_masks_np[index].squeeze(),
                    "class_labels": class_labels
                }
            })
            table.add_data(index, overlay_img, wandb.Image(masks_np[index].squeeze()),
                           wandb.Image(predicted_masks_np[index].squeeze()))
        wandb.log({f"training_epoch_{self.current_epoch}": table})
        self.training_step_outputs.clear()
        self.training_step_targets.clear()
        self.training_step_inputs.clear()

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = torch.clamp(masks, 0, 1)
        predictions = torch.sigmoid(self.model(images))
        loss = self.loss_fn(predictions, masks)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        predicted_masks = predictions.clone()
        predicted_masks[predicted_masks > 0.5] = 1
        predicted_masks[predicted_masks <= 0.5] = 0
        predicted_masks = predicted_masks.clamp(0, 1)
        masks.clamp(0, 1)
        self.val_step_outputs.extend(predicted_masks)
        self.val_step_targets.extend(masks)
        self.val_step_inputs.extend(images)
        return loss

    def on_validation_epoch_end(self) -> None:
        predicted_masks = torch.stack(self.val_step_outputs[-5:])
        masks = torch.stack(self.val_step_targets[-5:])
        images = torch.stack(self.val_step_inputs[-5:])
        dice_score = self.metric(preds=predicted_masks.long(), target=masks.long())  # Error
        self.log("validation dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        images_np = images.cpu().detach().numpy()
        predicted_masks_np = predicted_masks.cpu().detach().numpy()
        masks_np = masks.cpu().detach().numpy()
        table = wandb.Table(columns=["Image ID", "Validation Overlay Image", "Validation Mask", "Predicted"])
        class_labels = {
            0: "background",
            1: "tumor"
        }
        for index in range(images_np.shape[0]):
            overlay_img = wandb.Image(images_np[index].squeeze(), masks={
                "ground_truth": {
                    "mask_data": masks_np[index].squeeze(),
                    "class_labels": class_labels
                },
                "predictions": {
                    "mask_data": predicted_masks_np[index].squeeze(),
                    "class_labels": class_labels
                }
            })
            table.add_data(index, overlay_img, wandb.Image(masks_np[index].squeeze()),
                           wandb.Image(predicted_masks_np[index].squeeze()))
        wandb.log({f"validation_epoch_{self.current_epoch}": table})
        self.val_step_outputs.clear()
        self.val_step_targets.clear()
        self.val_step_inputs.clear()

    def configure_optimizers(self):
        return self.optimizer
