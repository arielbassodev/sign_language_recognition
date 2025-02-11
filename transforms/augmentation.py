import random
import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F
import os
import sys
import torchvision
import random 
from sign_language_tools.pose.transform import Rotation2D, translation, flip, smooth, noise, interpolate, padding, scale

class Gaus_noise(object):
    def __init__(self, mean= 0, std = 0.05):
        self.mean = mean
        self.std = std
    def call(self, data_numpy):
        data_numpy = data_numpy.numpy()
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        output = temp + noise
        output = torch.Tensor(output)
        return output

def subSampleFlip(data_numpy, time_range=20):
    data_numpy = data_numpy.cpu().numpy()
    C, T, V, M = data_numpy.shape
    assert T >= time_range, "frames longer than data"
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range = random.sample(all_frames, time_range)
        time_range_order = sorted(time_range)
        time_range_reverse =  list(reversed(time_range_order))
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range_order, :, :] = data_numpy[:, time_range_reverse, :, :]
    x_new = torch.Tensor(x_new)
    return x_new

class Gaus_noise(object):
    def __init__(self, mean= 0, std = 0.05):
        self.mean = mean
        self.std = std
    def call(self, data_numpy):
        data_numpy = data_numpy.numpy()
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        output = temp + noise
        output = torch.Tensor(output)
        return output


class Shear(object):
    def __init__(self, s1=None, s2=None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        if self.s1 is not None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1), random.uniform(-1, 1)]

        if self.s2 is not None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1)]
        R = np.array([[1, s1_list[0]],
                      [s2_list[0], 1]])
        temp = temp.transpose([1, 0, 2, 3])
        temp = np.dot(temp, R.T)
        temp = temp.transpose([1, 0, 2, 3])
        return temp



def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    data_numpy = data_numpy.cpu().numpy()
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    
    data_numpy = torch.tensor(data_numpy)

    return data_numpy

def random_shift(data_numpy):
    data_numpy = data_numpy.cpu().numpy()
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]
    data_shift = torch.tensor(data_shift)

    return data_shift

class Translation:
    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy

    def call(self, landmarks: np.ndarray):
        landmarks[:, :, 0] += self.dx
        landmarks[:, :, 1] += self.dy
        return landmarks

class VerticalFlip:

    def call(self, landmarks: np.ndarray):
        landmarks[:, 0, :] = 1 - landmarks[:, 0, :]
        return landmarks


def apply_random_augmentation(data):
    augmentations = [ VerticalFlip().call(data), subSampleFlip(data),Translation(0.1, 0.1).call(data), Translation(-0.1, -0.1).call(data), Translation(0.1, -0.1).call(data), Translation(-0.1, 0.1).call(data)]
    augmented_version = random.choice(augmentations)
    return augmented_version

#random_move(data), subSampleFlip(data,2)