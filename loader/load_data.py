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
from sign_language_tools.pose.transform import Rotation2D, translation, flip, smooth, noise, interpolate, padding, scale
import random
import numpy  as np
nan = np.nan

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
            t = target
            # sign = torch.cat((torch.Tensor(left_hand), torch.Tensor(right_hand), torch.Tensor(pose)), dim=1)
            if np.all(np.isnan(right_hand)) or np.all(np.isnan(left_hand)):
                continue
            else:
                right_hand = data['right_hand']
                ok = ~np.isnan(right_hand)
                xp = ok.ravel().nonzero()[0]
                fp = right_hand[~np.isnan(right_hand)]
                x = np.isnan(right_hand).ravel().nonzero()[0]
                right_hand[np.isnan(right_hand)] = np.interp(x, xp, fp)
                left_hand = data['left_hand']
                ok = ~np.isnan(left_hand)
                xp = ok.ravel().nonzero()[0]
                fp = left_hand[~np.isnan(left_hand)]
                x = np.isnan(left_hand).ravel().nonzero()[0]
                left_hand[np.isnan(left_hand)] = np.interp(x, xp, fp)
                sign_with_target = (left_hand, right_hand, pose, target)
                signs_table.append(sign_with_target)

        targets = np.array([row[-1] for row in signs_table])
        unique_targets = np.unique(targets)
        grouped = {}
        grouped = {int(t): [] for t in unique_targets}
        for row in signs_table:
            grouped[int(row[-1])].append(row)
        for key in grouped:
            grouped[key] = np.array(grouped[key], dtype=object)

        grouped_restrict = {}
        for key, value in grouped.items():
            n = max(1, round(len(value) / 0.3))
            grouped_restrict[key] = value[:n]
        final_restricted = [(row[0], row[1], row[2], key) for key, value in grouped_restrict.items() for row in value]
        targets_1 = np.array([row[-1] for row in signs_table])
        unique_targets_1 = np.unique(targets_1)
        targets_2 = np.array([row[-1] for row in final_restricted])
        unique_targets_2 = np.unique(targets_2)
        diff_1_not_in_2 = np.setdiff1d(unique_targets_1, unique_targets_2)

        return signs_table, final_restricted

    def collate_fn(batch):
        left_hand, right_hand, pose, labels = zip(*batch)
        left_hand_tensors = [torch.tensor(lh) for lh in left_hand]
        right_hand_tensors = [torch.tensor(lh) for lh in right_hand]
        pose_tensor = [torch.tensor(lp) for lp in pose]
        padded_lh = pad_sequence(left_hand_tensors, batch_first=True, padding_value=0.0)
        padded_rh = pad_sequence(right_hand_tensors, batch_first=True, padding_value=0.0)
        padded_pose = pad_sequence(pose_tensor, batch_first=True, padding_value=0.0)
        labels = torch.tensor(labels)
        return padded_rh, padded_lh, padded_pose, labels

    def build_dataset(dataset):
        dataset, _ = CustomDataset.proprocess_data(dataset)
        collate_fn = CustomDataset.collate_fn
        dataloader = torch.utils.data.DataLoader(dataset, 512, collate_fn=collate_fn, shuffle=True)
        return dataloader

    def build_fine_tuning_data(dataset):
        complet_dataset, dataset = CustomDataset.proprocess_data(dataset)
        collate_fn = CustomDataset.collate_fn
        dataloader = torch.utils.data.DataLoader(dataset, 512, collate_fn=collate_fn, shuffle=True)
        return dataloader

