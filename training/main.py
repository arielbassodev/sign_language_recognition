import os
import sys
sys.path.insert(1, '/gpfs/scratch/acad/lsfb/cslr/cslr_cod/loader')
sys.path.insert(1, '/gpfs/scratch/acad/lsfb/cslr/cslr_cod/models')
sys.path.insert(1, '/gpfs/scratch/acad/lsfb/cslr/cslr_cod/training')
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lsfb_dataset import LSFBIsolConfig, LSFBIsolLandmarks
from tqdm import tqdm
from pytorch_metric_learning import losses
import contrastive_training
import torch.optim as optim
import BackboneConfig
import simsiam_training
import byol_training
import fine_tuning_trainer
import Scheduler
from lightly.loss import NTXentLoss
import matplotlib.pyplot as plt
import Reconstitution_2
import reconstitution
from pytorch_lightning.loggers import TensorBoardLogger
import load_data
import simclr, encoder, simsiam, byol
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from simsiam import SimSiam
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
#from lsfb_transfo.training.Reconstitution_2 import Reconstitution_2
from reconstitution import Reconstitution


train = LSFBIsolLandmarks(LSFBIsolConfig(
    root="/gpfs/projects/acad/lsfb/Lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "train",
    n_labels= 500,
    sequence_max_length=30,
    show_progress=False,
))

test = LSFBIsolLandmarks(LSFBIsolConfig(
    root="/gpfs/projects/acad/lsfb/Lsfb_dataset/isol",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "test",
    n_labels= 500,
    sequence_max_length=30,
    show_progress=False,
))

trainset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="/gpfs/projects/acad/lsfb/lsa64",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "train",
    n_labels= 64,
    sequence_max_length=30,
    show_progress=False,
))

testset = LSFBIsolLandmarks(LSFBIsolConfig(
    root="/gpfs/projects/acad/lsfb/lsa64",
    landmarks=('pose','left_hand', 'right_hand'),
    split = "test",
    n_labels=64,
    sequence_max_length=30,
    show_progress=False,
))

unsup_train_lsfb =  load_data.CustomDataset.build_dataset(train)
test_lsfb =  load_data.CustomDataset.build_dataset(test)
supervised_lsfb = load_data.CustomDataset.build_fine_tuning_data(train)
supervised_gsl = load_data.CustomDataset.build_fine_tuning_data(trainset)
unsupervised_gsl =  load_data.CustomDataset.build_dataset(trainset)
test_gsl = load_data.CustomDataset.build_dataset(testset)
config =  BackboneConfig.BackboneConfig()
backbone = encoder.ViTModel(**vars(config)).to('cuda')
bk_1 = byol.Byol(backbone=backbone).to('cuda')
bk   = byol_training.ByolTraining(bk_1,100, unsup_train_lsfb, "gaussian_noise").train()

class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(500,64)
        self.mlp = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.Linear(1024, 500),
        )
   def forward(self, x1,y,z1):
      x = self.backbone(x1,y,z1)
      x = self.mlp(x)
      x = self.fc(x)
      return x
cl = classifier(bk, 500).to('cuda')


import pytorch_lightning as L
import torchmetrics as TM
import torch
from torch import nn, optim
class Module(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = cl
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 64
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
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
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

        return loss

    def validation_step(self, batch, batch_idx):
        left_hand, right_hand, pose, targets = batch
        #images = torch.cat((left_hand, right_hand, pose), dim=2)
        #images = images.to(torch.float32)
        left_hand = left_hand.to(torch.float32)
        right_hand = right_hand.to(torch.float32)
        pose = pose.to(torch.float32)
        logits = self.model(left_hand, right_hand, pose)
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
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        scheduler = Scheduler.WarmupLinearScheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
logger = TensorBoardLogger(save_dir="byol_lsfb_lsa")
my_module = Module()
trainer = L.Trainer(max_epochs=1000,logger=logger)
trainer.fit(
   my_module,
   unsupervised_gsl, 
   test_gsl
)