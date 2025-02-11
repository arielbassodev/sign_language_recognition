import lsfb_dataset
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from lsfb_dataset import LSFBContConfig, LSFBIsolConfig, LSFBIsolLandmarks
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import sys
import os
import random

class CustomDataset():
   
   def __init__(self, data):
        self.data = data

   def __len__(self):
        return len(self.data)

   def __getitem__(self, idx):
        return self.data[idx]
   
   def proprocess_data(dataset):
       signs_table = []
       for data, target in dataset:
           left_hand = data['left_hand']
           right_hand = data['right_hand']
           pose = data['pose']
           sign = torch.cat((torch.Tensor(left_hand), torch.Tensor(right_hand), torch.Tensor(pose)), dim=1)

           sign_with_target = (sign, target)
           signs_table.append(sign_with_target)
       return signs_table
   
   def collate_fn(batch):
    signs, labels = zip(*batch)
    padded_signs = pad_sequence(signs, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels)
    return padded_signs, labels

   def build_dataset(dataset):
       dataset = CustomDataset.proprocess_data(dataset)
       collate_fn = CustomDataset.collate_fn
       dataloader = torch.utils.data.DataLoader(dataset,128,collate_fn=collate_fn)
       return dataloader

