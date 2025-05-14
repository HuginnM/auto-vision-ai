import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights, MaskRCNNPredictor

from autovisionai.core.configs import CONFIG


def create_model(n_classes: int, pretrained: bool = True) -> MaskRCNNPredictor:
    """
    Creates the Mask R-CNN model based on `maskrcnn_resnet50_fpn`.

    :param n_classes: number of classes the model should predict.
    :param pretrained: an indicator to load pretrained model or not.
    :return: MaskRCNNPredictor.
    """
    # Loading an instance of segmentation model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # Getting the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replacing the pre-trained head with a new one.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=n_classes)

    # Getting the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = CONFIG.models.mask_rcnn.hidden_size

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask, dim_reduced=hidden_layer, num_classes=n_classes
    )

    return model
