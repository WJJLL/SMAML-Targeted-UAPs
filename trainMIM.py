import argparse
import os
import numpy as np

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from gaussian_smoothing import *
from config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, IMAGENET_Train_PATH, IMAGENET_Test_PATH
from dataset_utils.voc0712 import VOCDetection
from dataset_utils.coco import CocoDetection
from utils import one_hot
from utils import TwoCropTransform, rotation
from UAP import UAP
import random
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
scale_size = 256
img_size = 224


def parse_arguments():
    parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
    parser.add_argument('--src', default='IN', help='Source Domain: natural images, paintings, medical scans, etc')
    parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
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
    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)

    print(args)
    return args



def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


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
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    model_types = ['vgg16', 'resnet50', 'googlenet']
    model_weight = 1 / args.model_nums
    models_set = []
    adv_weight = []

    for model_type in model_types:
        if model_type in model_names:
            model = models.__dict__[model_type](pretrained=True)
        else:
            assert (model_type in model_names), 'Please provide correct target model names: {}'.format(model_names)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(2)))
        model = model.cuda()
        model.eval()

        models_set.append(model)
        adv_weight.append(model_weight)

    if args.model_nums == 4:
        model = models.__dict__['alexnet'](pretrained=True)
        models_set.append(model)


    netG = UAP(shape=(img_size, img_size),
               num_channels=3,
               use_cuda=args.use_cuda)
    netG.cuda()

    optimG = optim.Adam(netG.parameters(), lr=args.lr)

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

    if args.gs:
        kernel_size = 3
        pad = 2
        sigma = 1
        kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()

    running_loss = 0
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

        img = img.cuda()
        netG.train()
        optimG.zero_grad()

        # Smoothing
        if args.gs:
            img = kernel(img)

        target = torch.ones(img.shape[0], dtype=torch.int64) * args.match_target
        target =target.cuda()

        adv_logit = 0
        for _, (net, w) in enumerate(zip(models_set, adv_weight)):
            net.zero_grad()
            adv = netG(img)
            adv = torch.clamp(adv, 0.0, 1.0)
            adv_out = net(normalize(adv))
            adv_logit = adv_logit + w * adv_out

        loss = F.cross_entropy(adv_logit, target)

        loss.backward()
        optimG.step()
        # Projection
        netG.uap.data = torch.clamp(netG.uap.data, -eps, eps)

        running_loss += loss.item()

        iteration = iteration +1

        if iteration % 100 == 99:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(0, iteration, running_loss / 100))
            running_loss = 0
        if iteration % 1000 == 0 and iteration != 0:
            torch.save(netG.state_dict(),
                       args.save_dir + '/netG_ens_{}_{}_{}_{}_{}.pth'.format(args.model_type, args.src,
                                                                                 iteration,
                                                                                 args.match_target, args.eps))

if __name__ == '__main__':
    main()





