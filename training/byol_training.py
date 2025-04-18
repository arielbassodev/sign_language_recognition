
import sys
from torch import device
import BackboneConfig
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from lightly.loss import NTXentLoss
import torch.optim as optim
import Scheduler
from lightning.pytorch import LightningModule
import Lars
import random
sys.path.insert(1, '/gpfs/scratch/acad/lsfb/cslr/cslr_cod/transforms')
import Contrastive_data_augmentation
from Contrastive_data_augmentation import *
from lightly.loss import NegativeCosineSimilarity
import Contrastive_data_augmentation
from lightly.utils.scheduler import cosine_schedule
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
class ByolTraining():

    def __init__(self, model, epochs, train_loader, augmentation):
        self.model = model
        self.criterion = NegativeCosineSimilarity()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
        self.epochs = epochs
        self.augmentation = augmentation
        self.scheduler = Scheduler.LinearSchedulerWithWarmup(self.optimizer)
        self.train_loader = train_loader

    @staticmethod
    def plot_loss(epoch_losses):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        plt.grid(True)
        plt.legend()
        plt.savefig('Training_loss_simclr_others_essai_4.png')
        plt.show()

    def get_augmentation(self):
        abs = round(random.uniform(0.1, 5), 1)
        ord = round(random.uniform(0.1, 5), 1)
        std_1 = random.uniform(0.1, 0.5)
        std_2 = random.uniform(0.5, 0.8)
        first_rotation = DataAugmentation.Rotate(random.randint(0,30))
        second_rotation = DataAugmentation.Rotate(random.randint(45,90))
        first_translation = DataAugmentation.Translate(abs,ord)
        second_translation = DataAugmentation.Translate(random.randint(1,4), random.randint(5,6))
        vertical_flip = DataAugmentation.VFlip()
        horizontal_flip = DataAugmentation.HorizontalFlip()
        first_gaussian_noise = DataAugmentation.GaussianNoise(std_1)
        second_gaussian_noise = DataAugmentation.GaussianNoise(std_2)

        list = {"rotation":(first_rotation,second_rotation), "translation":(first_translation,second_translation),
                "vertical_flip":(vertical_flip,vertical_flip), "horizontal_flip":(horizontal_flip,horizontal_flip),
                "gaussian_noise":(first_gaussian_noise,second_gaussian_noise)}

        return list

    def generate_pairs(self, x,y,z, augmentation):
        x = x.to('cuda')
        y = y.to('cuda')
        z = z.to('cuda')
        augmentation_list = self.get_augmentation()
        first_aug, second_aug = augmentation_list[augmentation]
        x1, y1, z1 = first_aug(x, y, z)
        x2, y2, z2 = second_aug(x, y, z)
        return x1,y1,z1,x2,y2,z2

    def train(self):
        epoch_losses = []
        for epoch in range(self.epochs):
            momentum_val = cosine_schedule(epoch, self.epochs, 0.996, 1)
            a = self.optimizer.param_groups[0]['lr']
            running_loss = 0.0
            for id, feature in enumerate(tqdm(self.train_loader)):
                update_momentum(self.model, self.model.backbone_momentum, m=momentum_val)
                update_momentum(
                    self.model.projection_head, self.model.projection_head_momentum, m=momentum_val
                )
                left_hand, right_hand, pose = feature[0].to('cuda'), feature[1].to('cuda'), feature[2].to('cuda')
                x1,y1,z1, x2,y2,z2 = self.generate_pairs(left_hand, right_hand, pose, self.augmentation)
                x1 = x1.to(torch.float32)
                y1 = y1.to(torch.float32)
                z1 = z1.to(torch.float32)
                z2 = z2.to(torch.float32)
                x2 = x2.to(torch.float32)
                y2 = y2.to(torch.float32)
                p0 = self.model(x1,y1,z1)
                z0 = self.model.forward_momentum(x2,y2,z2)
                p1 = self.model(x2,y2,z2)
                z1 = self.model.forward_momentum(x1,y1,z1)
                loss =  0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
                running_loss += loss.detach()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_loss = running_loss / len(self.train_loader)
                epoch_losses.append(epoch_loss.cpu())
        self.plot_loss(epoch_losses)
        return self.model.backbone






