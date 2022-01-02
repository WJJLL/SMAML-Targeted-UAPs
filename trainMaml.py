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
from gaussian_smoothing import *
from config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, \
    IMAGENET_Train_PATH, IMAGENET_Test_PATH
from dataset_utils.voc0712 import VOCDetection
from dataset_utils.coco import CocoDetection
from utils import one_hot
from utils import TwoCropTransform, rotation
from custom_loss import  BoundedLogitLossFixedRef,Po_trip,target_logit_loss,RelavativeCrossEntropyTarget
from UAP import UAP
import random
from modules.resnet import resnet50
from collections import OrderedDict
from random import sample
import torch.nn.functional as F
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale_size = 256
img_size = 224

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

    parser.add_argument('--pretrained_seed', type=int, default=0,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--loss_function', default='ce', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg',
                                                                  'max_margin', 'triplet', 'relative_ce', 'po_trip',
                                                                  'target_logit', 'margin', 'cam_loss'],
                        help='Used loss function for source classes: (default: cw_logit)')

    args = parser.parse_args()
    print(args)
    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    return args


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t




class Meta(nn.Module):
    def __init__(self, criterion, use_cuda, input_size, num_support, num_updates, epsilon, learning_rate):
        super(Meta, self).__init__()
        self.net = UAP(shape=(input_size, input_size),
                       num_channels=3,
                       use_cuda=use_cuda)
        self.net.cuda()
        self.epsilon = epsilon
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
                img, target = next(data_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iterator = iter(train_loader)
                img, target = next(data_iterator)
            reptile_grads = {}
            num_updates = 0
            img = img.cuda()

            model_base = copy.deepcopy(self.net)
            rm_archs = sample(archs, 3)

            for arch in rm_archs:
                ##  reset weight for every task
                for k, (p, q) in enumerate(zip(self.net.parameters(), model_base.parameters())):
                    # p = copy.deepcopy(q)
                    p.data = q.data.clone()

                adv = self.net(img)
                adv = torch.clamp(adv , 0, 1)

                arch.eval()
                arch.zero_grad()
                adv_out = arch(normalize(adv))
                target = torch.ones(adv_out.shape[0], dtype=torch.int64) * args.match_target
                loss = F.cross_entropy(adv_out,target.cuda())
                self.opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                self.net.uap.data = torch.clamp(self.net.uap.data, -self.epsilon, self.epsilon)

                for j, p in enumerate(self.net.parameters()):
                    if (num_updates == 0):
                        reptile_grads[j] = [p.data.clone().detach()]
                    else:
                        reptile_grads[j].append(p.data.clone().detach())
                num_updates += 1

            for k, (p, q) in enumerate(zip(self.net.parameters(), model_base.parameters())):
                alpha = np.exp(-1.0 * ((1.0) / len(archs)))
                ll = torch.stack(reptile_grads[k])
                p.data = torch.mean(ll, 0) * (alpha) + (1 - alpha) * q.data.clone()

            self.net.uap.data = torch.clamp(self.net.uap.data, -self.epsilon, self.epsilon)
            running_loss += loss.item()
            iteration += 1

            if iteration % 100 == 99:
                print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(0, iteration, running_loss / 100))
                running_loss = 0
            if iteration % 1000 == 0 and iteration != 0:
                torch.save(self.net.state_dict(),
                           args.save_dir + '/netG_ensMaml_{}_{}_{}_{}_{}.pth'.format(args.model_type, args.src,
                                                                                      iteration,
                                                                                      args.match_target, args.eps))

def main():
    args = parse_arguments()
    random.seed(args.pretrained_seed)
    np.random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed(args.pretrained_seed)
    torch.cuda.manual_seed_all(args.pretrained_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    #### save_dir path of trained uap
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    eps = args.eps / 255

    ### source domain model ##
    model_1 = models.__dict__['vgg16'](pretrained=True)
    model_2 = models.__dict__['resnet50'](pretrained=True)
    model_3 = models.__dict__['googlenet'](pretrained=True)

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


    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()

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
