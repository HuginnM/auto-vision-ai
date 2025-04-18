from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR

from autovisionai.configs.config import CONFIG
from autovisionai.models.unet.unet_model import Unet
from autovisionai.utils.utils import get_batch_images_and_pred_masks_in_a_grid, masks_iou


class UnetTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the UNet model.

    :param in_channels: a number of input channels.
    :param n_classes: a number of classes to predict.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.model = Unet(self.in_channels, self.n_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.training_losses = []
        self.val_outputs = []

    def training_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]], batch_idx: int
    ) -> torch.Tensor:
        images, targets = batch
        images_tensor = torch.stack(images)
        masks_tensor = torch.stack([pair["mask"] for pair in targets])

        y_hat = self.model(images_tensor)
        loss = self.criterion(y_hat, masks_tensor.to(torch.float))

        self.training_losses.append(loss.detach())
        return loss

    def on_train_epoch_end(self) -> None:
        if self.training_losses:
            loss_epoch = torch.stack(self.training_losses).mean()
            self.log("train/loss_epoch", loss_epoch.item())
            self.training_losses.clear()

    def validation_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        images, targets = batch
        images_tensor = torch.stack(images)
        masks_tensor = torch.stack([pair["mask"] for pair in targets])

        y_hat = self.model(images_tensor)
        masks_iou_score = masks_iou(masks_tensor, y_hat, self.n_classes + 1)
        loss = self.criterion(y_hat, masks_tensor.to(torch.float))
        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(y_hat, images)

        output = {"val_loss": loss, "val_iou": masks_iou_score, "val_images_and_pred_masks": imgs_grid}
        self.val_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        loss_epoch = torch.stack([o["val_loss"] for o in self.val_outputs]).mean()
        avg_iou = torch.stack([o["val_iou"] for o in self.val_outputs]).mean()

        self.log("val/loss_epoch", loss_epoch, prog_bar=True)
        self.log("val/val_iou", avg_iou, prog_bar=True)

        for idx, dict_i in enumerate(self.val_outputs):
            self.logger.experiment.add_image("Predicted masks on images", dict_i["val_images_and_pred_masks"], idx)

        self.val_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        optimizer = Adam(
            self.model.parameters(),
            lr=CONFIG["unet"]["optimizer"]["initial_lr"].get(),
            weight_decay=CONFIG["unet"]["optimizer"]["weight_decay"].get(),
        )

        lr_scheduler = StepLR(
            optimizer,
            step_size=CONFIG["unet"]["lr_scheduler"]["step_size"].get(),
            gamma=CONFIG["unet"]["lr_scheduler"]["gamma"].get(),
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
