
import argparse
import sys,os

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

class UAP(nn.Module):
    def __init__(self,
                shape=(224, 224),
                num_channels=3,
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))
    def copy_weights(self,net):
        self.uap.data = net.uap.data.clone()
    def forward(self, x):
        adv_x = x + self.uap
        return adv_x
