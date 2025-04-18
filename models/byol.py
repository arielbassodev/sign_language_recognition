import torch
import sys
import torchvision
from torch import nn
import random
sys.path.insert(1, '/gpfs/scratch/acad/lsfb/cslr/cslr_cod/transforms')
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from torch.nn.modules.module import T
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
import copy
class Byol(nn.Module):

    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 512, 128)
        self.prediction_head = BYOLPredictionHead(128, 64, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self,l,r,p):
        f = self.backbone(l,r,p).flatten(start_dim=1)
        z = self.projection_head(f)
        z = self.prediction_head(z)
        return z

    def forward_momentum(self, l,r,p):
        y = self.backbone_momentum(l,r,p).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z