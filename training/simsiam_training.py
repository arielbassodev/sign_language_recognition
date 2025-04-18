
import sys
from torch import device
import BackboneConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from lightly.loss import NTXentLoss
import torch.optim as optim
import Scheduler
from lightning.pytorch import LightningModule
import Lars
from lightly.loss import NegativeCosineSimilarity

class SimSiamTraining():

    def __init__(self, model, epochs, train_loader, augmentations):
        self.model = model
        self.criterion = NegativeCosineSimilarity()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        self.epochs = epochs
        self.augmentation = augmentations
        self.scheduler =  Scheduler.LinearSchedulerWithWarmup(self.optimizer)
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

    def train(self):
        epoch_losses = []
        for epoch in range(self.epochs):
            a = self.optimizer.param_groups[0]['lr']
            self.optimizer.zero_grad()
            running_loss = 0.0
            for id, feature in enumerate(self.train_loader):
                left_hand, right_hand, pose = feature[0].to('cuda'), feature[1].to('cuda'), feature[2].to('cuda')
                z0, p0, z1, p1 = self.model.forward(left_hand, right_hand, pose, self.augmentation)
               # z1, p1 = self.model.forward(left_hand, right_hand, pose)
                loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_loss = running_loss / len(self.train_loader)
            epoch_losses.append(epoch_loss)
            # self.scheduler.step()
            # print("SimSiam Loss", epoch_loss)
        self.plot_loss(epoch_losses)
        return self.model.backbone
