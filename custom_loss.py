import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
from utils import one_hot

class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)


class RelavativeCrossEntropyTarget(_WeightedLoss):
    def __init__(self,):
        super(RelavativeCrossEntropyTarget, self).__init__()
    def forward(self, adv_input, orig_input,target):
        label = orig_input.argmax(dim=-1)
        loss = F.cross_entropy(adv_input,target)+ F.cross_entropy(orig_input,label)
        return loss

## define Po+Trip
def Poincare_dis(a, b):
    L2_a = torch.sum(a*a, 1)
    L2_b = torch.sum(b*b, 1)

    theta = 2 * torch.sum((a - b)*(a-b), 1) / ((1 - L2_a) * (1 - L2_b))
    x=1.0 + theta
    # torch.log(x+(x**2-1)**0.5)
    distance = torch.mean(torch.log(x+(x**2-1)**0.5))
    return distance

def Cos_dis(a, b):
    a_b = torch.abs(torch.sum(torch.mul(a, b), 1))
    L2_a = torch.sum(a*a, 1)
    L2_b = torch.sum(b*b, 1)
    distance = torch.mean(a_b / torch.sqrt(L2_a * L2_b))
    return distance

class Po_trip(nn.Module):
    def __init__(self):
        super(Po_trip, self).__init__()
    def forward(self,logits,orig_logit,target):
        label=orig_logit.argmax(dim=-1)

        labels_onehot = torch.nn.functional.one_hot(target, logits.shape[1])
        labels_true_onehot = torch.nn.functional.one_hot(label,logits.shape[1])

        loss_po = Poincare_dis(torch.clamp((labels_onehot - 0.00001), 0.0, 1.0),
            logits / torch.sum(torch.abs(logits), [1], keepdim=True))
        loss_cos = torch.clamp((Cos_dis(labels_true_onehot, logits)-Cos_dis(labels_onehot, logits) + 0.007), 0.0, 2.1)
        loss = loss_po + 0.01* loss_cos
        return loss

class target_logit_loss(nn.Module):
    def __init__(self):
        super(target_logit_loss, self).__init__()
    def forward(self,logits,labels):
        labels = labels.cuda()
        logits=torch.nn.functional.softmax(logits,dim=1)
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        logit_dists = (-1 * real)
        loss = logit_dists.sum()
        return loss
