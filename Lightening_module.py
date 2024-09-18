import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class Segmentation(pl.LightningModule):
    def __init__(self, model, optimizer, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        out = self.forward(image)
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats((out.sigmoid() > 0.5).long(), mask.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss)
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        stage = 'train'
        image, mask = batch
        out = self.forward(image.float())
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats((out.sigmoid() > 0.5).long(), mask.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.log(f"{stage}_IoU", iou)
        self.log(f"{stage}_loss", loss)
        # return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        stage = 'valid'
        image, mask = batch
        out = self.forward(image.float())
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats((out.sigmoid() > 0.5).long(), mask.long(), mode='binary')
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.log(f"{stage}_IoU", iou)
        self.log(f"{stage}_loss", loss)
        # return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return self.optimizer