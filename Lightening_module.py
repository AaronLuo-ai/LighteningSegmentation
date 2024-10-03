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
        print(f"stage: {stage}")
        print(f"batch_idx: {batch_idx}")

        output = torch.sigmoid(output)
        print("output type: ", output.dtype)
        print("masks type: ", masks.dtype)
        loss = self.loss_fn(output, masks)
        # print("loss: ", loss.item())
        # output[output > 0.5] = 1
        # output[output <= 0.5] = 0
        # masks[masks > 0.5] = 1
        # masks[masks <= 0.5] = 0
        print("output shape: ", output.shape)
        print("masks shape: ", masks.shape)
        dice_score = self.metric(preds=output, target=masks)    #Error
        self.log(f"{stage}/dice_score", dice_score)
        self.log(f"{stage}/loss", loss.item())
        mask_img = wandb.Image(masks, "ground truth masks")
        output_img = wandb.Image(output, "output masks")
        # self.log({f"{stage} ground truth masks": mask_img, f"{stage} output masks": output_img})


    def training_step(self, batch, batch_idx):
        print("stage training")
        self._common_step(batch, batch_idx, stage='training')

    def validation_step(self, batch, batch_idx):
        print("stage validation")
        self._common_step(batch, batch_idx, stage='validation')

    def configure_optimizers(self):
        return self.optimizer