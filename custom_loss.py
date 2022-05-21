import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
from utils import one_hot
## target loss
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


class LogitLoss(_WeightedLoss):
    def __init__(self, num_classes=1000, use_cuda=True):
        super(LogitLoss, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        # Get the logit output value
        logits = (one_hot_labels * input).max(1)[0]
        # Increase the logit value
        return torch.mean(-logits)

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

    def forward(self,logits,ori_label,target):
        labels_onehot = torch.nn.functional.one_hot(target, logits.shape[1])
        labels_true_onehot = torch.nn.functional.one_hot(ori_label,logits.shape[1])

        loss_po = Poincare_dis(logits / torch.sum(torch.abs(logits), 1, keepdim=True),
                               torch.clamp((labels_onehot - 0.00001), 0.0, 1.0))
        loss_cos = torch.clamp(Cos_dis(labels_onehot, logits) - Cos_dis(labels_true_onehot, logits) + 0.007, 0.0, 2.1)
        loss = loss_po + 0.01 * loss_cos
        return loss

import torch.nn as nn
class CrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = self.criterion(input, target)
        return loss


class AbsLogitLoss(_WeightedLoss):
    def __init__(self, num_classes=1000, use_cuda=True):
        super(AbsLogitLoss, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda

    def forward(self, input, target):
        input = input / torch.sum(torch.abs(input), 1, keepdim=True)

        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        # Get the logit output value
        logits = (one_hot_labels * input).max(1)[0]

        # logits = input.gather(1, target.unsqueeze(1)).squeeze(1)

        # Increase the logit value
        return torch.mean(-logits)


##non-target loss

class BoundedLogitLoss_neg(_WeightedLoss):
    def __init__(self, num_classes, confidence, use_cuda=False):
        super(BoundedLogitLoss_neg, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = use_cuda

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)

        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(target_logits - not_target_logits, min=-self.confidence)
        return torch.mean(logit_loss)


class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = - self.criterion(input, target)
        return loss

