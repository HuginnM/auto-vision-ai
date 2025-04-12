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

    @staticmethod
    def find_bounding_box(mask):
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
        ).to(accelerator)
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
            box = MaskRCNNTrainer.find_bounding_box(mask)

            mask_rcnn_target = {
                'image_id': target['image_id'],
                'boxes': box.to(accelerator),
                'masks': torch.as_tensor(target['mask'], dtype=torch.uint8),
                'labels': torch.tensor([1], dtype=torch.int64).to(accelerator),
            }
            mask_rcnn_targets.append(mask_rcnn_target)

        return tuple(mask_rcnn_targets)
