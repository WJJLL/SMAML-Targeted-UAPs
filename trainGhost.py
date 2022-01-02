import argparse
import os
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
from config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, IMAGENET_Train_PATH, IMAGENET_Test_PATH
from dataset_utils.voc0712 import VOCDetection
from dataset_utils.coco import CocoDetection
from utils import one_hot
from utils import TwoCropTransform, rotation
from custom_loss import  BoundedLogitLossFixedRef,Po_trip,target_logit_loss,RelavativeCrossEntropyTarget
from UAP import UAP
import random
import torch.nn.functional as F
from modules.resnet import resnet50
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]




def parse_arguments():
    parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
    parser.add_argument('--src', default='coco', help='Source Domain: natural images, paintings, medical scans, etc')
    parser.add_argument('--match_target', type=int, default=24, help='Target Domain samples')
    parser.add_argument('--match_dir', default='./ImageTarget', help='Path to data folder with target domain samples')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of trainig samples/batch')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate for adam')
    parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget during training, eps')
    parser.add_argument('--model_type', type=str, default='resnet50',
                        help='Model under attack (discrimnator)')
    parser.add_argument('--gs', action='store_true', help='Apply gaussian smoothing')
    parser.add_argument('--save_dir', type=str, default='pretrained_generators', help='Directory to save generators')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--confidence', default=10., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--iterations', type=int, default=4000, help='Number of training iterations')
    parser.add_argument('--loss_function', default='ce',
                        help='Used loss function for source classes: (default: cw_logit)')
    parser.add_argument('--pretrained_seed', type=int, default=0,
                        help='Seed used in the generation process (default: 0)')

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


def smooth_one_hot(true_labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    return true_dist



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

    model = resnet50(pretrained=True)



    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(2)))


    model = model.cuda()
    model.eval()

    # Input dimensions
    if args.model_type == 'inception_v3':
        scale_size = 300
        img_size = 299
    else:
        scale_size = 256
        img_size = 224

    netG = UAP(shape=(img_size, img_size),
               num_channels=3,
               use_cuda=args.use_cuda)
    netG.cuda()

    optimG = optim.Adam(netG.parameters(), lr=args.lr)




    if args.loss_function == "logit":
        criterion = target_logit_loss()
    elif args.loss_function == "pcc":
        criterion = BoundedLogitLossFixedRef(num_classes=1000, confidence=args.confidence,
                                             use_cuda=args.use_cuda)
    elif args.loss_function == 'po_trip':
        criterion = Po_trip()
    elif args.loss_function == 'r_ce':
        criterion = RelavativeCrossEntropyTarget()


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


    iteration = 0


    running_loss = 0
    data_iterator = iter(train_loader)
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
        model.zero_grad()

        target = torch.ones(img.shape[0], dtype=torch.int64) * args.match_target

        adv = netG(img)
        adv = torch.clamp(adv, 0.0, 1.0)
        adv_logit = model(normalize(adv))

        if args.loss_function == "ce":
            loss = F.cross_entropy(adv_logit, target.cuda())
        elif args.loss_function == "r_ce" or args.loss_function == "po_trip" :
            logit = model(normalize(img))
            loss = criterion(adv_logit,logit, target.cuda())
        else:
            loss = criterion(adv_logit, target.cuda())
        loss.backward()
        optimG.step()
        # Projection
        netG.uap.data = torch.clamp(netG.uap.data, -eps, eps)
        iteration=iteration+1

        running_loss += loss.item()

        if iteration % 100 == 99:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(0, iteration, running_loss / 100))
            running_loss = 0
        if iteration % 1000 == 0 and iteration != 0:
            torch.save(netG.state_dict(),
                       args.save_dir + '/netG_ghost_{}_{}_{}_{}_{}_{}.pth'.format(args.model_type, args.src,
                                                                                  iteration,
                                                                                  args.match_target, args.eps,args.loss_function))




if __name__ == '__main__':
    main()






