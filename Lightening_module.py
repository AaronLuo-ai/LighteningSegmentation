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
        output = self.model(images)
        output = torch.sigmoid(output)
        loss = self.loss_fn(output, masks)
        output[output > 0.5] = 1
        output[output <= 0.5] = 0
        output.clamp(0, 1)
        masks.clamp(0, 1)
        dice_score = self.metric(preds=output.long(), target=masks.long())    # Error
        # print(f"stage: {stage}, dice_score: {dice_score}")
        # print(f"stage {stage} loss: {loss.item()}")

        self.log(f"{stage}/dice_score", dice_score)
        self.log(f"{stage}/loss", loss.item())

        mask_img = wandb.Image(masks[0], caption="Ground Truth Mask")  # Log only the first mask for simplicity
        output_img = wandb.Image(output[0], caption="Predicted Mask")  # Log only the first predicted mask
        input_img = wandb.Image(images[0], caption="Input Image")  # Log the first input image
        table = wandb.Table(columns=["ID", "Input Image", "Ground Truth", "Prediction"])
        image_id = f"{stage}_batch{batch_idx}"
        table.add_data(image_id, input_img, mask_img, output_img)
        wandb.log({f"{stage}_predictions": table})

    def training_step(self, batch, batch_idx):
        print("stage training")
        self._common_step(batch, batch_idx, stage='training')

    def validation_step(self, batch, batch_idx):
        print("stage validation")
        self._common_step(batch, batch_idx, stage='validation')

    def configure_optimizers(self):
        return self.optimizer