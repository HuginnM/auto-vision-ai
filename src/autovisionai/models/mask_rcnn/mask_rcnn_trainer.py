from typing import Dict, Tuple, Union

import loguru as logger
import pytorch_lightning as pl
import torch
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

from autovisionai.configs import CONFIG, LRSchedulerConfig, OptimizerConfig
from autovisionai.loggers.ml_logging import log_image_to_all_loggers
from autovisionai.models.mask_rcnn.mask_rcnn_model import create_model
from autovisionai.utils.utils import bboxes_iou, get_batch_images_and_pred_masks_in_a_grid


class MaskRCNNTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the torchvision Mask R-CNN model.

    :param n_classes: number of classes of the Mask R-CNN (including the background).
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.model = create_model(n_classes=n_classes)
        self.training_losses = []
        self.val_outputs = []

    def _create_valid_target(self, target: Dict) -> Dict[str, torch.Tensor]:
        """
        Creates a valid target dictionary for Mask R-CNN.

        :param target: targets from datamodule batch.
        :param box: boundary box of the object on the mask.
        :return: A target dict compatible with Mask R-CNN.
        """
        return {
            "image_id": target["image_id"].to(self.device),
            "boxes": target["box"].unsqueeze(0).to(self.device),
            "masks": torch.as_tensor(target["mask"], dtype=torch.uint8, device=self.device),
            "labels": torch.tensor([1], dtype=torch.int64, device=self.device),
        }

    def _create_empty_target(self, target: Dict):
        """
        Creates an empty target dictionary for Mask R-CNN.

        :param target: targets from datamodule batch.
        :return: A target dict compatible with Mask R-CNN.
        """
        return {
            "image_id": target["image_id"].to(self.device),
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
            "masks": torch.as_tensor(target["mask"], dtype=torch.uint8, device=self.device),
            "labels": torch.zeros((0,), dtype=torch.int64, device=self.device),
        }

    def _safe_stack_pred_masks(self, preds: list, images: tuple) -> torch.Tensor:
        """
        Extracts the first predicted mask from each prediction dict.
        If no masks are present, returns an empty mask of the same shape.

        :param preds: list of prediction dicts from Mask R-CNN
        :return: torch.Tensor of shape [N, H, W]
        """
        stacked = []

        for i, pred in enumerate(preds):
            masks = pred.get("masks", None)  # mask shape is (N, 1, H-mask, W-mask)

            if masks is not None and masks.shape[0] > 0:
                # Take 0 mask here as in demostration purposes and hence shallow dataset, the train images
                # contains only 1 object. Take whole masks here to include all instances.
                stacked.append(masks[0])
            else:
                _, height, witdth = images[i].shape
                stacked.append(torch.zeros((1, height, witdth), dtype=torch.float32, device=self.device))
                logger.warning(f"[_safe_stack_pred_masks] No masks for prediction {i}, inserting empty mask.")

        return torch.stack(stacked)

    def _convert_targets_to_mask_rcnn_format(
        self, targets: Tuple[Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Converts targets from datamodule batch to Mask-RCNN format.
        :param targets: targets from datamodule batch.
        """
        mask_rcnn_targets = []

        # For Mask RCNN training, the Dataset requires to consist of
        # `masks`, `boxes`, `labels` and `image_id` keys.
        for target in targets:
            mask_rcnn_targets.append(self._create_valid_target(target))

        return tuple(mask_rcnn_targets)

    def step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], is_training: bool
    ) -> Dict[str, torch.Tensor]:
        """
        A shared step for the training_step and validation_step functions.
        Calculates the sum of losses for one batch step.
        :param batch: a batch of images and targets with annotations.
        :is_training: if False - disables grad calculation.
        :return: a dict of the losses sum and loss mask of the prediction heads per one batch step.
        """
        images, targets = batch

        images = torch.stack(images)
        valid_targets = self._convert_targets_to_mask_rcnn_format(targets)

        self.model.train()

        with torch.set_grad_enabled(is_training):
            outputs = self.model(images, valid_targets)

        loss_step = sum(outputs.values())
        loss_mask = outputs["loss_mask"]

        return {"loss": loss_step, "loss_step": loss_step.detach(), "loss_mask": loss_mask.detach()}

    def training_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.
        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict of the losses sum and loss_mask of the prediction heads for one batch step.
        """
        outputs = self.step(batch, is_training=True)

        self.log("train/loss_step", outputs["loss_step"].item())
        self.log("train/loss_mask_step", outputs["loss_mask"].item())

        self.training_losses.append(
            {"loss_step": outputs["loss_step"].detach(), "loss_mask": outputs["loss_mask"].detach()}
        )
        return outputs

    def on_train_epoch_end(self) -> None:
        """
        Calculates and logs mean total loss and loss_mask
        at the end of the training epoch with the outputs of all training steps.
        :param train_outputs: outputs of all training steps.
        :return: mean loss and loss_mask for one training epoch.
        """
        if self.training_losses:
            loss_epoch = torch.stack([d["loss_step"] for d in self.training_losses]).mean()
            loss_mask_epoch = torch.stack([d["loss_mask"] for d in self.training_losses]).mean()

            self.log("train/loss_epoch", loss_epoch.item())
            self.log("train/loss_mask_epoch", loss_mask_epoch.item())
            self.training_losses.clear()

    def validation_step(
        self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, dict]]:
        """
        Take a batch from the validation dataset and input its images into the model.
        Retrieves losses, predicted masks and IoU metric after one validation step.
        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict of the validation step losses of the prediction heads,
        intersection over union metric score and predicted masks for one batch step.
        """
        outputs = self.step(batch, is_training=False)

        images, targets = batch

        self.model.eval()
        preds = self.model(images)

        pred_masks = self._safe_stack_pred_masks(preds, images)
        images_tensor = torch.stack(images)

        targets = self._convert_targets_to_mask_rcnn_format(targets)
        bboxes_iou_score = torch.stack([bboxes_iou(t, o) for t, o in zip(targets, preds, strict=False)]).mean()

        output = {
            "val_outputs": outputs,
            "val_iou": bboxes_iou_score,
            "pred_masks": pred_masks,
            "images": images_tensor,
        }
        self.val_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        val_losses = [d["val_outputs"]["loss_step"] for d in self.val_outputs]
        val_mask_losses = [d["val_outputs"]["loss_mask"] for d in self.val_outputs]
        val_ious = [d["val_iou"] for d in self.val_outputs]

        loss_epoch = torch.stack(val_losses).mean()
        loss_mask_epoch = torch.stack(val_mask_losses).mean()
        avg_iou = torch.stack(val_ious).mean()

        pred_masks = torch.cat([o["pred_masks"] for o in self.val_outputs], dim=0)[:16]
        images = torch.cat([o["images"] for o in self.val_outputs], dim=0)[:16]

        self.log("val/loss_epoch", loss_epoch.item(), prog_bar=True)
        self.log("val/loss_mask_epoch", loss_mask_epoch.item())
        self.log("val/val_iou", avg_iou.item(), prog_bar=True)

        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(pred_masks, images, threshold=0.65)

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
        Configure the SGD optimizer and the StepLR learning rate scheduler.
        :return: a dict with the optimizer and lr_scheduler.
        """
        optim_cfg: OptimizerConfig = CONFIG.models.mask_rcnn.optimizer
        lr_scheduler_cfg: LRSchedulerConfig = CONFIG.models.mask_rcnn.lr_scheduler

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = SGD(
            params,
            lr=optim_cfg.initial_lr,
            momentum=optim_cfg.momentum,
            weight_decay=optim_cfg.weight_decay,
        )
        lr_scheduler = StepLR(
            optimizer,
            step_size=lr_scheduler_cfg.step_size,
            gamma=lr_scheduler_cfg.gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
