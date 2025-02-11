import os
import sys
import torch
import torch.nn as nn
from click.core import F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
from tqdm import tqdm
from pytorch_metric_learning import losses
import contrastive_training
import torch.optim as optim
import BackboneConfig
import fine_tuning_trainer
import Scheduler
from lightly.loss import NTXentLoss
import matplotlib.pyplot as plt
from loader import load_data
from models import simclr, encoder

trainset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/LSFB_Dataset/lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "train",
    n_labels= 500,
    sequence_max_length=30,
    show_progress=True,
))

testset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="C:/Users/abassoma/Documents/LSFB_Dataset/lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "test",
    n_labels= 500,
    sequence_max_length=30,
    show_progress=True,
))

train_loader =  load_data.CustomDataset.build_dataset(trainset)
test_loader = load_data.CustomDataset.build_dataset(testset)
config =  BackboneConfig.BackboneConfig()
backbone = encoder.ViTModel(**vars(config)).to('cuda')
model  = simclr.SimCLR(backbone).to('cuda')
#bk = contrastive_training.ContrastiveTraining(model,1,train_loader).train()
l1 = fine_tuning_trainer.number_target(train_loader)
l2 = fine_tuning_trainer.number_target(test_loader)
print("..........................",l1,l2)

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 200)
   def forward(self, x):
        x = self.backbone(x)
        # x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
cl = classifier(backbone, 400).to('cuda')

import pytorch_lightning as L
import torchmetrics as TM
import torch
from torch import nn, optim
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = backbone
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 500
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.train_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.train_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top5_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_top3_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_top10_acc = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=10)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.train_top5_acc(logits, targets)
        self.train_top3_acc(logits, targets)
        self.train_top10_acc(logits, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_top10_acc', self.train_top5_acc, on_step=True, on_epoch=True)
        self.log('train_top3_acc', self.train_top3_acc, on_step=True, on_epoch=True)
        self.log('train_top5_acc', self.train_top10_acc, on_step=True, on_epoch=True)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_top5_acc(logits, targets)
        self.val_top3_acc(logits, targets)
        self.val_top10_acc(logits, targets)
        self.val_recall(preds, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_top5_acc', self.val_top5_acc, on_step=True, on_epoch=True)
        self.log('val_top3_acc', self.val_top3_acc, on_step=True, on_epoch=True)
        self.log('val_top10_acc', self.val_top10_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # Créez l'optimiseur
        optimizer = optim.SGD(self.model.parameters(), lr=0.001)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
my_module = Module()

trainer = L.Trainer(max_epochs=1000)
trainer.fit(
   my_module,
   train_loader,
   test_loader
)


