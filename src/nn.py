import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

class ResNet50(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.model.fc = torch.nn.Linear(100352, 16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Loc2vec(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class TripletLossFunction(torch.nn.Module):
    def __init__(self, margin=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.margin = margin

    def forward(self,
                anchor: torch.Tensor,
                anchor_pos: torch.Tensor,
                anchor_neg: torch.Tensor
                ) -> torch.Tensor:
        """
        """
        distance_a_pos = F.pairwise_distance(anchor, anchor_pos)
        distance_a_neg = F.pairwise_distance(anchor, anchor_neg)
        distance_pos_neg = F.pairwise_distance(anchor_pos, anchor_neg)
        distance_min_neg = torch.min(distance_a_neg, distance_pos_neg)
        losses = F.relu(distance_a_pos - distance_min_neg + self.margin)
        
        np_losses = losses.cpu().data.numpy()
        np_distance_a_pos = np.mean(distance_a_pos.cpu().data.numpy())
        np_distance_a_neg = np.mean(distance_a_neg.cpu().data.numpy())
        np_min_neg_dist = np.mean(distance_min_neg.cpu().data.numpy())

        loss_log = f'MAX LOSS: {round(float(np.max(np_losses)),3)} | MEAN LOSS: {round(float(np.mean(np_losses)),3)} | (o)/(+) DIST: {round(float(np.mean(np_distance_a_pos)),3)} | (o)/(-) DIST: {round(float(np.mean(np_distance_a_neg)),3)}'

        return losses.mean(), loss_log, np_distance_a_pos, np_distance_a_neg, np_min_neg_dist
