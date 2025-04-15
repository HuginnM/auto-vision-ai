import torch
import pytorch_lightning as pl
from typing import Dict, Tuple, Union
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

from autovisionai.configs.config import CONFIG
from autovisionai.utils.utils import bboxes_iou
from autovisionai.models.mask_rcnn.mask_rcnn_model import create_model
from autovisionai.utils.utils import get_batch_images_and_pred_masks_in_a_grid

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    @staticmethod
    def _find_bounding_box(mask):
        rows = torch.any(mask != 0, dim=1)
        cols = torch.any(mask != 0, dim=0)
        nz_rows = torch.nonzero(rows)
        nz_cols = torch.nonzero(cols)

        bbox = torch.as_tensor(
            [
                nz_rows[0][1].item(),  # First row's index
                nz_cols[0][0].item(),  # First col's index
                nz_rows[-1][1].item(),  # Last row's index
                nz_cols[-1][0],  # Last col's index
            ],
            dtype=torch.float32,
        )
        return bbox

    @staticmethod
    def _convert_targets_to_mask_rcnn_format(targets: Tuple[Dict[str, torch.Tensor]]
                                             ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Converts targets from datamodule batch to Mask-RCNN format.
        :param targets: targets from datamodule batch.
        """
        mask_rcnn_targets = []

        # For Mask RCNN training, the Dataset requires to consist of `masks`, `boxes`, `labels` and `image_id` keys.
        for target in targets:
            mask = target['mask']
            box = MaskRCNNTrainer._find_bounding_box(mask)

            mask_rcnn_target = {
                'image_id': target['image_id'],
                'boxes': box.unsqueeze(0),
                'masks': torch.as_tensor(target['mask'], dtype=torch.uint8),
                'labels': torch.tensor([1], dtype=torch.int64),
            }
            mask_rcnn_targets.append(mask_rcnn_target)

        return tuple(mask_rcnn_targets)

    def step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]], is_training: bool) -> Dict[str, torch.Tensor]:
        """
        A shared step for the training_step and validation_step functions.
        Calculates the sum of losses for one batch step.
        :param batch: a batch of images and targets with annotations.
        :is_training: if False - disables grad calculation.
        :return: a dict of the losses sum and loss mask of the prediction heads per one batch step.
        """
        images, targets = batch
        
        images = torch.stack(images)       
        targets = MaskRCNNTrainer._convert_targets_to_mask_rcnn_format(targets)
        
        self.model.train()
        
        with torch.set_grad_enabled(is_training):
            outputs = self.model(images, targets)
                
        loss_step = sum(outputs.values())
        loss_mask = outputs["loss_mask"]

        return {'loss': loss_step, 'loss_step': loss_step.detach(), 'loss_mask': loss_mask.detach()}

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                      batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.
        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict of the losses sum and loss_mask of the prediction heads for one batch step.
        """
        # Use a shared step method
        outputs = self.step(batch, is_training=True)

        self.log('train/loss_step', outputs['loss_step'].item())
        self.log('train/loss_mask_step', outputs['loss_mask'].item())
        
        self.training_losses.append({
            'loss_step': outputs['loss_step'].detach(),
            'loss_mask': outputs['loss_mask'].detach()
        })
        return outputs
    
    def on_train_epoch_end(self) -> None:
        """
        Calculates and logs mean total loss and loss_mask
        at the end of the training epoch with the outputs of all training steps.
        :param train_outputs: outputs of all training steps.
        :return: mean loss and loss_mask for one training epoch.
        """
        if self.training_losses:
            loss_epoch = torch.stack([d['loss_step'] for d in self.training_losses]).mean()
            loss_mask_epoch = torch.stack([d['loss_mask'] for d in self.training_losses]).mean()
            
            self.log('train/loss_epoch', loss_epoch.item())
            self.log('train/loss_mask_epoch', loss_mask_epoch.item())
            self.training_losses.clear()

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                        batch_idx: int) -> Dict[str, Union[torch.Tensor, dict]]:
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

        # Convert targets to Mask RCNN format
        targets = self._convert_targets_to_mask_rcnn_format(targets)
        bboxes_iou_score = torch.stack([bboxes_iou(t, o) for t, o in zip(targets, preds)]).mean()
        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(preds, images, mask_rcnn=True)
        
        output = {
            'val_outputs': outputs,
            'val_images_and_pred_masks': imgs_grid,
            'val_iou': bboxes_iou_score
        }
        self.val_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        val_losses = [d['val_outputs']['loss_step'] for d in self.val_outputs]
        val_mask_losses = [d['val_outputs']['loss_mask'] for d in self.val_outputs]
        val_ious = [d['val_iou'] for d in self.val_outputs]

        loss_epoch = torch.stack(val_losses).mean()
        loss_mask_epoch = torch.stack(val_mask_losses).mean()
        avg_iou = torch.stack(val_ious).mean()

        self.log('val/loss_epoch', loss_epoch.item(), prog_bar=True)
        self.log('val/loss_mask_epoch', loss_mask_epoch.item())
        self.log('val/val_iou', avg_iou.item(), prog_bar=True)

        for idx, d in enumerate(self.val_outputs):
            self.logger.experiment.add_image('Predicted masks on images', d['val_images_and_pred_masks'], idx)

        self.val_outputs.clear()
            
    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        """
        Configure the SGD optimizer and the StepLR learning rate scheduler.
        :return: a dict with the optimizer and lr_scheduler.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=CONFIG['mask_rcnn']['optimizer']['initial_lr'].get(),
                        momentum=CONFIG['mask_rcnn']['optimizer']['momentum'].get(),
                        weight_decay=CONFIG['mask_rcnn']['optimizer']['weight_decay'].get())
        lr_scheduler = StepLR(optimizer, step_size=CONFIG['mask_rcnn']['lr_scheduler']['step_size'].get(),
                              gamma=CONFIG['mask_rcnn']['lr_scheduler']['gamma'].get())
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
