import argparse
import os, copy
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, \
    IMAGENET_Train_PATH, IMAGENET_Test_PATH
from dataset_utils.voc0712 import VOCDetection
from dataset_utils.coco import CocoDetection
from utils import one_hot
from utils import TwoCropTransform, rotation
from UAP import UAP
import random
from modules.resnet import resnet50
from collections import OrderedDict
from random import sample
import torch.nn.functional as F
from opti_adam import Adam
from custom_loss import BoundedLogitLossFixedRef,LogitLoss,CrossEntropy,AbsLogitLoss
scale_size = 256
img_size = 224
import math
def keepGradUpdate(noiseData, optimizer, gradInfo, epsilon, ti=False):
    if ti:
        grad_c = F.conv2d(gradInfo.clone().unsqueeze(0), gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
        gradInfo = grad_c[0]
    betas = optimizer.param_groups[0]["betas"]
    beta1 = betas[0]
    beta2 = betas[1]
    weight_decay = optimizer.param_groups[0]["weight_decay"]
    lr = optimizer.param_groups[0]["lr"]
    bias_correction1 = 1 - betas[0]
    bias_correction2 = 1 - betas[1]
    exp_avg, exp_avg_sq = torch.zeros_like(noiseData), \
                          torch.zeros_like(noiseData)
    if weight_decay != 0:
        gradInfo.add_(weight_decay, noiseData)
    exp_avg.mul_(beta1).add_(1 - beta1, gradInfo)
    # exp_avg=exp_avg *beta1+(1-beta1)*gradInfo
    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, gradInfo, gradInfo)
    exp_avg_sq.add_(1e-8)  # to avoid possible nan in backward
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(optimizer.param_groups[0]['eps'])
    step_size = lr / bias_correction1
    new_data = noiseData.addcdiv(-step_size, exp_avg, denom)
    new_data = torch.clamp(new_data, -epsilon, epsilon)
    return new_data

def parse_arguments():
    parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
    parser.add_argument('--src', default='IN', help='Source Domain: natural images, paintings, medical scans, etc')
    parser.add_argument('--match_target', type=int, default=150, help='Target Domain samples')
    parser.add_argument('--match_dir', default='./ImageTarget', help='Path to data folder with target domain samples')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
    parser.add_argument('--iterations', type=int, default=6000, help='Number of training iterations')
    parser.add_argument('--model_nums', type=int, default=3, help='Number of training models')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')
    parser.add_argument('--model_type', type=str, default='vgg16',
                        help='Model under attack (discrimnator)')
    parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
    parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--confidence', default=0., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--di', action='store_true', help='Apply di')
    parser.add_argument('--ti', action='store_true', help='Apply ti')
    parser.add_argument('--resume', action='store_true', help='Apply resume')
    parser.add_argument('--pretrained_seed', type=int, default=0,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--loss_function', default='ce',
                        help='Used loss function for source classes: (default: cw_logit)')

    args = parser.parse_args()
    print(args)
    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    return args


def normalize(x):
    """
    Normalizes a batch of images with size (batch_size, 3, height, width)
    by mean and std dev expected by PyTorch models
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (x - mean.type_as(x)[None, :, None, None]) / std.type_as(x)[None, :, None, None]


def DI(X, p=0.3, image_width=224, image_resize=256):
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_resize, image_resize])
    return padded if torch.rand(()) < p else X


class Meta(nn.Module):
    def __init__(self, criterion, use_cuda, input_size, num_support, num_updates, epsilon, learning_rate, ti=False):
        super(Meta, self).__init__()
        self.net = UAP(shape=(input_size, input_size),
                       num_channels=3,
                       use_cuda=use_cuda)
        self.net.cuda()
        self.epsilon = epsilon
        if ti:
            self.opt = Adam(self.net.parameters(), lr=learning_rate)
        else:
            self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.num_support = num_support
        self.num_updates = num_updates

    def forward(self, train_loader, archs, args):
        running_loss = 0
        self.net.train()
        data_iterator = iter(train_loader)
        iteration = 0
        while (iteration < args.iterations):
            try:
                img, label = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                img, label = next(data_iterator)

            model_base = copy.deepcopy(self.net)
            rm_archs = sample(archs, 3)
            loss_q = 0
            for kk, arch in enumerate(rm_archs):
                ## prepare data
                img = img.cuda()
                size = int(img.shape[0] / 2)
                sup_set = img[:size]
                label_spt = label[:size]
                query_set = img[size:]
                label_qry = label[size:]
                sup_adv_before = self.net(sup_set)
                sup_adv_before = torch.clamp(sup_adv_before, 0, 1)
                # reset uap in every task
                for k, (p, q) in enumerate(zip(self.net.parameters(), model_base.parameters())):
                    # p = copy.deepcopy(q)
                    p.data = q.data.clone()
                # get temp uap
                sup_adv = self.net(sup_set)
                sup_adv = torch.clamp(sup_adv, 0, 1)

                if args.di:
                    adv_out = arch(normalize(DI(sup_adv)))
                    adv_out_before = arch(normalize(DI(sup_adv_before)))
                else:
                    adv_out = arch(normalize(sup_adv))
                    adv_out_before = arch(normalize(sup_adv_before))

                target_train = torch.ones(adv_out.shape[0], dtype=torch.int64) * args.match_target
                loss = self.criterion(adv_out, target_train.cuda()) + self.criterion(adv_out_before, target_train.cuda())
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)[0]
                # get temp uap
                self.net.uap.data = keepGradUpdate(self.net.uap.data, self.opt, grad, self.epsilon)

                # compute grad of task
                qur_adv = self.net(query_set)
                qur_adv = torch.clamp(qur_adv, 0, 1)
                if args.di:
                    qry_adv_out = arch(normalize(DI(qur_adv)))
                else:
                    qry_adv_out = arch(normalize(qur_adv))
                target_test = torch.ones(qry_adv_out.shape[0], dtype=torch.int64) * args.match_target
                loss_qry = self.criterion(qry_adv_out, target_test.cuda())
                loss_q += loss_qry

            for k, (p, q) in enumerate(zip(self.net.parameters(), model_base.parameters())):
                # p = copy.deepcopy(q)
                p.data = q.data.clone()
            loss_ = loss_q / len(rm_archs)
            self.opt.zero_grad()
            loss_.backward()
            self.opt.step()
            self.net.uap.data = torch.clamp(self.net.uap.data, -self.epsilon, self.epsilon)
            running_loss += loss_.item()
            iteration += 1

            if iteration % 100 == 99:
                print(torch.argmax(rm_archs[0](normalize(self.net(img[0:30]))), dim=1))
                print(torch.argmax(rm_archs[1](normalize(self.net(img[0:30]))), dim=1))
                print(torch.argmax(rm_archs[2](normalize(self.net(img[0:30]))), dim=1))
                print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f} \t noise: {3:.5f}'.format(0, iteration,
                                                                                           running_loss / 100,
                                                                                           self.net.uap.data.norm()))
                running_loss = 0
            if iteration % 1000 == 0 and iteration != 0:
                torch.save(self.net.state_dict(),
                           args.save_dir + '/netG_meta_{}_{}_{}_{}_di_{}_ti_{}.pth'.format(args.src,
                                                                                              iteration,
                                                                                              args.match_target,
                                                                                              args.eps,args.di,args.ti),_use_new_zipfile_serialization=False)
def main():
    args = parse_arguments()
    random.seed(args.pretrained_seed)
    np.random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed_all(args.pretrained_seed)
    #### save_dir path of trained uap
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    eps = args.eps / 255

    ### source domain model ##
    model_1 = models.__dict__['vgg16'](pretrained=True)
    model_2 = models.__dict__['resnet50'](pretrained=True)
    model_3 = models.__dict__['densenet121'](pretrained=True)

    if torch.cuda.device_count() > 1:
        model_1 = torch.nn.DataParallel(model_1, device_ids=list(range(2)))
        model_2 = torch.nn.DataParallel(model_2, device_ids=list(range(2)))
        model_3 = torch.nn.DataParallel(model_3, device_ids=list(range(2)))

    model_1 = model_1.cuda()
    model_2 = model_2.cuda()
    model_3 = model_3.cuda()

    model_1.eval()
    model_2.eval()
    model_3.eval()

    model_set = [model_1, model_2, model_3]

    if args.src == 'voc':
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(scale_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

    if args.src == 'IN':
        train_set = torchvision.datasets.ImageFolder(IMAGENET_Train_PATH,
                                                     train_transform)

    elif args.src == 'voc':
        train_set = VOCDetection(root=VOC_2012_ROOT,
                                 year="2012",
                                 image_set='train',
                                 transform=train_transform)
    elif args.src == 'coco':
        train_set = CocoDetection(root=COCO_2017_TRAIN_IMGS,
                                  annFile=COCO_2017_TRAIN_ANN,
                                  transform=train_transform)

    if args.loss_function == "logit":
        criterion = LogitLoss()
    elif args.loss_function == "logitRef":
        criterion = BoundedLogitLossFixedRef(num_classes=1000, confidence=args.confidence,
                                             use_cuda=args.use_cuda)
    elif args.loss_function == 'ce':
        criterion = CrossEntropy()
    elif args.loss_function == 'abslogit':
        criterion = AbsLogitLoss(num_classes=1000, use_cuda=args.use_cuda)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    train_size = len(train_set)
    print('Training data size:', train_size)
    learner = Meta(
        criterion=criterion,
        use_cuda=args.use_cuda,
        input_size=img_size,
        learning_rate=args.lr,
        num_support=50,
        num_updates=1,
        epsilon=eps,
    )
    learner.forward(train_loader, model_set, args)


if __name__ == '__main__':
    main()

