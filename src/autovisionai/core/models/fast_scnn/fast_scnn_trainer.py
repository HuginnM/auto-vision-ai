from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR

from autovisionai.core.configs import CONFIG, LRSchedulerConfig, OptimizerConfig
from autovisionai.core.loggers.ml_logging import log_image_to_all_loggers
from autovisionai.core.models.fast_scnn.fast_scnn_model import FastSCNN
from autovisionai.core.utils.utils import get_batch_images_and_pred_masks_in_a_grid, masks_iou


class FastSCNNTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the FastSCNN model.

    :param n_classes: number of classes to predict.
    """

    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.model = FastSCNN(n_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.training_losses = []
        self.val_outputs = []

    def training_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> torch.Tensor:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: step loss.
        """
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
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Take a batch from the validation dataset and input its images into the model.
        Retrieves losses after one validation step and mask IoU score.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: dict with epoch loss value and masks IoU score and
        predicted masks for one batch step.
        """
        images, targets = batch

        images_tensor = torch.stack(images)
        masks_tensor = torch.stack([pair["mask"] for pair in targets])

        y_hat = self.model(images_tensor)
        masks_iou_score = masks_iou(masks_tensor, y_hat, self.n_classes + 1)
        loss = self.criterion(y_hat, masks_tensor.to(torch.float))

        output = {"val_loss": loss, "val_iou": masks_iou_score, "pred_masks": y_hat, "images": images_tensor}
        self.val_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        loss_epoch = torch.stack([o["val_loss"] for o in self.val_outputs]).mean()
        avg_iou = torch.stack([o["val_iou"] for o in self.val_outputs]).mean()
        pred_masks = torch.cat([o["pred_masks"] for o in self.val_outputs], dim=0)[:16]
        images = torch.cat([o["images"] for o in self.val_outputs], dim=0)[:16]

        self.log("val/loss_epoch", loss_epoch, prog_bar=True)
        self.log("val/val_iou", avg_iou, prog_bar=True)

        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(pred_masks, images)

        log_image_to_all_loggers(
            ml_loggers=self.trainer.loggers,
            tag="Predicted masks on images per epoch",
            image_tensor=imgs_grid,
            epoch=self.current_epoch,
            step=self.global_step,
        )
        self.val_outputs.clear()

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        """
        Configure the Adam optimizer and the StepLR learning rate scheduler.

        :return: a dict with the optimizer and lr_scheduler.
        """
        optim_cfg: OptimizerConfig = CONFIG.models.fast_scnn.optimizer
        lr_scheduler_cfg: LRSchedulerConfig = CONFIG.models.fast_scnn.lr_scheduler

        optimizer = Adam(
            self.model.parameters(),
            lr=optim_cfg.initial_lr,
            weight_decay=optim_cfg.weight_decay,
        )

        lr_scheduler = StepLR(
            optimizer,
            step_size=lr_scheduler_cfg.step_size,
            gamma=lr_scheduler_cfg.gamma,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
