import torch.nn as nn

from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights


def get_model():

    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        2
    )

    return model